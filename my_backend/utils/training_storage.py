"""
Training Results Storage Module
Handles upload/download of training results to/from Supabase Storage

This module provides functions to:
- Upload training results to Storage bucket (with optional compression)
- Download training results from Storage bucket
- Delete training results from Storage bucket
- List all results for a session

Created: 2025-10-22
Purpose: Solve timeout issues when saving large training results to database
"""

import io
import json
import gzip
import logging
import time
import pickle
from datetime import datetime
from threading import Lock
from typing import Dict, List, Optional, Any, Tuple
from shared.database.client import get_supabase_admin_client

logger = logging.getLogger(__name__)

# In-memory cache for downloaded training results
# Key: file_path, Value: (results_dict, timestamp)
_results_cache: Dict[str, Tuple[dict, float]] = {}
_cache_lock = Lock()
_CACHE_TTL = 300  # 5 minutes
_MAX_CACHE_SIZE = 10

# Per-file locks to prevent duplicate downloads (race condition fix)
_download_locks: Dict[str, Lock] = {}
_download_locks_lock = Lock()  # Lock for accessing _download_locks dict


def upload_training_results(
    session_id: str,
    results: dict,
    compress: bool = True,
    progress_callback: Optional[callable] = None
) -> Dict[str, any]:
    """
    Upload training results to Supabase Storage using pickle format (faster, smaller).

    Args:
        session_id: UUID session ID
        results: Training results dictionary containing model, data, metrics, etc.
        compress: Whether to compress with gzip (default True, recommended)
        progress_callback: Optional callback function(step: str, percent: int, message: str)
                          for reporting progress during upload stages

    Returns:
        dict: {
            'file_path': str,       # Path in bucket: "session_id/training_results_timestamp.pkl[.gz]"
            'file_size': int,       # Size in bytes
            'compressed': bool,     # Whether compressed
            'metadata': dict        # Quick-access metadata for database
        }

    Raises:
        Exception: If upload fails after 3 retries

    Example:
        >>> storage_result = upload_training_results(
        ...     session_id="abc-123",
        ...     results={"model_type": "Dense", "metrics": {...}},
        ...     compress=True,
        ...     progress_callback=lambda step, pct, msg: print(f"{step}: {pct}% - {msg}")
        ... )
        >>> print(storage_result['file_path'])
        "abc-123/training_results_20260129_130000.pkl.gz"
    """
    def emit_progress(step: str, percent: int, message: str, details: dict = None):
        """Helper to call progress callback if provided"""
        if progress_callback:
            try:
                progress_callback(step, percent, message, details)
            except Exception as e:
                logger.warning(f"Progress callback error: {e}")

    try:
        supabase = get_supabase_admin_client()

        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        # CHANGE: Use .pkl extension instead of .json
        file_name = f"training_results_{timestamp}.pkl"
        if compress:
            file_name += ".gz"
        file_path = f"{session_id}/{file_name}"

        # Stage 1: Preparing data
        emit_progress('preparing_upload', 61, 'Preparing training data for upload...')

        start_time = time.time()

        if compress:
            # Stage 2: Pickle + gzip compression
            emit_progress('compressing', 63, 'Compressing training data (pickle + gzip)...')

            buffer = io.BytesIO()
            with gzip.GzipFile(fileobj=buffer, mode='wb', compresslevel=6) as gz_file:
                # CHANGE: pickle.dump instead of json.dump - keeps numpy arrays as-is
                pickle.dump(results, gz_file, protocol=pickle.HIGHEST_PROTOCOL)

            upload_data = buffer.getvalue()
            content_type = 'application/gzip'

            serialization_time = time.time() - start_time
            compressed_size_mb = len(upload_data) / 1024 / 1024

            emit_progress('compression_done', 65, f'Compressed to {compressed_size_mb:.1f}MB (pickle)', {
                'compressed_size_mb': round(compressed_size_mb, 2),
                'serialization_time_seconds': round(serialization_time, 2),
                'action': 'compression_complete'
            })
        else:
            # Non-compressed pickle
            buffer = io.BytesIO()
            pickle.dump(results, buffer, protocol=pickle.HIGHEST_PROTOCOL)
            upload_data = buffer.getvalue()
            content_type = 'application/octet-stream'

        file_size = len(upload_data)
        file_size_mb = file_size / 1024 / 1024
        # Stage 3: Starting upload
        emit_progress('uploading', 66, f'Uploading {file_size_mb:.1f}MB to cloud storage...', {
            'file_size_mb': round(file_size_mb, 2),
            'file_size_bytes': file_size,
            'action': 'uploading'
        })
        logger.debug(f"Uploading {file_size_mb:.2f}MB to storage: {file_path}")

        max_retries = 3
        last_error = None

        for attempt in range(max_retries):
            try:
                response = supabase.storage.from_('training-results').upload(
                    path=file_path,
                    file=upload_data,
                    file_options={
                        'content-type': content_type,
                        'cache-control': '3600'
                    }
                )

                # Stage 4: Upload complete
                emit_progress('upload_complete', 69, 'Upload complete, finalizing...', {
                    'file_size_mb': round(file_size_mb, 2),
                    'file_path': file_path,
                    'action': 'upload_complete'
                })
                break

            except Exception as upload_error:
                last_error = upload_error
                if attempt < max_retries - 1:
                    logger.warning(
                        f"Upload attempt {attempt + 1}/{max_retries} failed: {upload_error}. "
                        f"Retrying..."
                    )
                    time.sleep(2 ** attempt)
                else:
                    logger.error(f"❌ All {max_retries} upload attempts failed")
                    raise upload_error

        metadata = {
            'accuracy': results.get('metrics', {}).get('accuracy'),
            'loss': results.get('metrics', {}).get('loss'),
            'epochs_completed': results.get('parameters', {}).get('EP'),
            'model_type': results.get('model_type'),
            'dataset_count': results.get('dataset_count'),
            'training_split': results.get('training_split')
        }

        # Cache results after upload to avoid re-downloading immediately after
        # (save_models_to_storage calls download right after upload)
        _set_cached_results(file_path, results)


        return {
            'file_path': file_path,
            'file_size': file_size,
            'compressed': compress,
            'metadata': metadata
        }

    except Exception as e:
        logger.error(f"❌ Failed to upload training results: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise


def _get_cached_results(file_path: str) -> Optional[dict]:
    """Get results from cache if not expired"""
    with _cache_lock:
        if file_path in _results_cache:
            cached_data, timestamp = _results_cache[file_path]
            age_seconds = time.time() - timestamp
            if age_seconds < _CACHE_TTL:

                return cached_data
            else:
                logger.warning(
                    f"⚠️ CACHE EXPIRED for {file_path} - age: {age_seconds:.1f}s exceeded TTL: {_CACHE_TTL}s. "
                    f"If this happens frequently, consider increasing _CACHE_TTL in training_storage.py"
                )
                del _results_cache[file_path]
    return None


def _set_cached_results(file_path: str, results: dict) -> None:
    """Store results in cache with LRU eviction"""
    with _cache_lock:
        # Evict oldest if at capacity
        if len(_results_cache) >= _MAX_CACHE_SIZE:
            oldest_key = min(_results_cache, key=lambda k: _results_cache[k][1])
            oldest_age = time.time() - _results_cache[oldest_key][1]
            del _results_cache[oldest_key]
            logger.warning(
                f"⚠️ CACHE FULL - Evicted {oldest_key} (age: {oldest_age:.1f}s) to make room. "
                f"Cache size: {_MAX_CACHE_SIZE}. If this happens frequently, consider increasing "
                f"_MAX_CACHE_SIZE in training_storage.py"
            )
        _results_cache[file_path] = (results, time.time())



def download_training_results(
    file_path: str,
    decompress: Optional[bool] = None
) -> dict:
    """
    Download training results from Supabase Storage.
    Auto-detects pickle vs JSON format based on file extension.

    Args:
        file_path: Path to file in storage bucket (e.g., "session_id/training_results_*.pkl.gz")
        decompress: Whether to decompress gzipped data. If None, auto-detect from file extension.

    Returns:
        dict: Training results dictionary

    Raises:
        Exception: If download or decompression fails

    Example:
        >>> results = download_training_results("abc-123/training_results_20260129_130000.pkl.gz")
        >>> print(results['model_type'])
        "Dense"
    """
    # Check cache first (quick check without lock)
    cached = _get_cached_results(file_path)
    if cached is not None:
        return cached

    # Get or create per-file lock to prevent duplicate downloads
    with _download_locks_lock:
        if file_path not in _download_locks:
            _download_locks[file_path] = Lock()
        file_lock = _download_locks[file_path]

    # Acquire per-file lock - only one thread can download this specific file
    with file_lock:
        # Double-check cache after acquiring lock (another thread may have downloaded)
        cached = _get_cached_results(file_path)
        if cached is not None:
            return cached

        # Actually download the file
        try:
            supabase = get_supabase_admin_client()

            # DEBUG LOG
            response = supabase.storage.from_('training-results').download(file_path)

            # Auto-detect format based on extension
            is_pickle = file_path.endswith('.pkl.gz') or file_path.endswith('.pkl')
            is_compressed = file_path.endswith('.gz')

            if is_pickle:
                # NEW FORMAT: Pickle
                # SECURITY NOTE: pickle.load() can execute arbitrary code.
                # This is safe here because results are only stored by authenticated users
                # via our training pipeline and retrieved from trusted Supabase storage.
                if is_compressed:
                    with gzip.GzipFile(fileobj=io.BytesIO(response)) as gz_file:
                        results = pickle.load(gz_file)
                else:
                    results = pickle.loads(response)
            else:
                # OLD FORMAT: JSON (backward compatibility)
                if decompress is None:
                    decompress = file_path.endswith('.gz')

                if decompress:
                    with gzip.GzipFile(fileobj=io.BytesIO(response)) as gz_file:
                        with io.TextIOWrapper(gz_file, encoding='utf-8') as text_wrapper:
                            results = json.load(text_wrapper)
                else:
                    results = json.loads(response.decode('utf-8'))

            # Cache results before returning
            _set_cached_results(file_path, results)

            return results

        except Exception as e:
            logger.error(f"❌ Failed to download training results from {file_path}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise


def delete_training_results(file_path: str) -> bool:
    """
    Delete training results from storage

    Args:
        file_path: Path to file in storage bucket

    Returns:
        bool: True if successful

    Raises:
        Exception: If deletion fails

    Example:
        >>> delete_training_results("abc-123/training_results_20251022_130000.json.gz")
        True
    """
    try:
        supabase = get_supabase_admin_client()

        logger.debug(f"Deleting training results: {file_path}")
        response = supabase.storage.from_('training-results').remove([file_path])

        return True

    except Exception as e:
        logger.error(f"❌ Failed to delete training results {file_path}: {e}")
        raise


def list_session_results(session_id: str) -> List[dict]:
    """
    List all training result files for a session

    Args:
        session_id: UUID session ID

    Returns:
        list: List of file metadata dicts with keys: name, id, updated_at, created_at, last_accessed_at, metadata

    Example:
        >>> files = list_session_results("abc-123")
        >>> for file in files:
        ...     print(file['name'], file.get('metadata', {}).get('size'))
    """
    try:
        supabase = get_supabase_admin_client()

        logger.debug(f"Listing training results for session: {session_id}")
        response = supabase.storage.from_('training-results').list(session_id)

        logger.debug(f"Found {len(response) if response else 0} files for session {session_id}")
        return response if response else []

    except Exception as e:
        logger.error(f"Failed to list training results for session {session_id}: {e}")
        return []


def get_storage_url(file_path: str, expires_in: int = 3600) -> str:
    """
    Get signed URL for downloading training results (for frontend/API)

    Args:
        file_path: Path to file in storage bucket
        expires_in: URL expiration time in seconds (default 1 hour)

    Returns:
        str: Signed URL for downloading file

    Raises:
        Exception: If URL generation fails

    Example:
        >>> url = get_storage_url("abc-123/training_results_20251022_130000.json.gz")
        >>> # Use this URL in frontend to download file
    """
    try:
        supabase = get_supabase_admin_client()

        response = supabase.storage.from_('training-results').create_signed_url(
            path=file_path,
            expires_in=expires_in
        )

        return response['signedURL'] if response else None

    except Exception as e:
        logger.error(f"Failed to create signed URL for {file_path}: {e}")
        raise


def fetch_training_results_with_storage(
    session_id: str,
    supabase: Optional[Any] = None,
    model_id: Optional[str] = None
) -> Optional[dict]:
    """
    Helper function to fetch training results from Storage or legacy JSONB column.
    Provides unified interface for all endpoints that need training results.

    Args:
        session_id: String session ID
        supabase: Optional Supabase client (will be created if not provided)
        model_id: Optional specific model ID (otherwise fetches most recent)

    Returns:
        dict: Training results dictionary or None if not found

    Raises:
        Exception: Only for technical errors (network, decompression, etc.)
                  Returns None for "not found" cases instead of raising.

    Example:
        >>> results = fetch_training_results_with_storage("session_123")
        >>> if results:
        ...     print(results['model_type'])
        "Dense"
    """
    try:
        from shared.database.operations import get_supabase_client, create_or_get_session_uuid

        if supabase is None:
            supabase = get_supabase_client(use_service_role=True)

        uuid_session_id = create_or_get_session_uuid(session_id)
        if not uuid_session_id:
            logger.warning(f"Session {session_id} not found")
            return None

        query = supabase.table('training_results') \
            .select('id, results_file_path, compressed, results') \
            .eq('session_id', uuid_session_id)

        if model_id:
            query = query.eq('id', model_id)
        else:
            query = query.order('created_at.desc').limit(1)

        response = query.execute()

        if not response.data:
            logger.warning(f"No training results found for session {session_id}")
            return None

        record = response.data[0]

        if record.get('results_file_path'):

            try:
                results = download_training_results(
                    file_path=record['results_file_path'],
                    decompress=record.get('compressed', False)
                )
                logger.debug(f"Successfully loaded results from storage for {session_id}")
                return results
            except Exception as download_error:
                logger.error(f"Failed to download from storage: {download_error}")

        if record.get('results'):
            logger.debug(f"Using legacy JSONB results for session {session_id}")
            return record['results']

        logger.warning(f"No results available (neither storage nor JSONB) for session {session_id}")
        return None

    except Exception as e:
        logger.error(f"Error fetching training results for session {session_id}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise
