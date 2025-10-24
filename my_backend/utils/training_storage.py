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

import json
import gzip
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from utils.supabase_client import get_supabase_admin_client

logger = logging.getLogger(__name__)


def upload_training_results(
    session_id: str,
    results: dict,
    compress: bool = True
) -> Dict[str, any]:
    """
    Upload training results to Supabase Storage

    Args:
        session_id: UUID session ID
        results: Training results dictionary containing model, data, metrics, etc.
        compress: Whether to compress with gzip (default True, recommended)

    Returns:
        dict: {
            'file_path': str,       # Path in bucket: "session_id/training_results_timestamp.json[.gz]"
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
        ...     compress=True
        ... )
        >>> print(storage_result['file_path'])
        "abc-123/training_results_20251022_130000.json.gz"
    """
    try:
        supabase = get_supabase_admin_client()

        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        file_name = f"training_results_{timestamp}.json"
        if compress:
            file_name += ".gz"
        file_path = f"{session_id}/{file_name}"

        json_str = json.dumps(results, indent=2)
        original_size = len(json_str.encode('utf-8'))

        if compress:
            logger.info(f"Compressing training results...")
            compressed_data = gzip.compress(json_str.encode('utf-8'), compresslevel=9)
            upload_data = compressed_data
            content_type = 'application/gzip'

            compression_ratio = (1 - len(compressed_data) / original_size) * 100
            logger.info(
                f"Compression complete: {original_size / 1024 / 1024:.2f}MB → "
                f"{len(compressed_data) / 1024 / 1024:.2f}MB "
                f"({compression_ratio:.1f}% reduction)"
            )
        else:
            upload_data = json_str.encode('utf-8')
            content_type = 'application/json'

        file_size = len(upload_data)
        logger.info(f"Uploading {file_size / 1024 / 1024:.2f}MB to storage: {file_path}")

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

                logger.info(f"✅ Training results uploaded successfully: {file_path}")
                break

            except Exception as upload_error:
                last_error = upload_error
                if attempt < max_retries - 1:
                    logger.warning(
                        f"Upload attempt {attempt + 1}/{max_retries} failed: {upload_error}. "
                        f"Retrying..."
                    )
                    import time
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


def download_training_results(
    file_path: str,
    decompress: Optional[bool] = None
) -> dict:
    """
    Download training results from Supabase Storage

    Args:
        file_path: Path to file in storage bucket (e.g., "session_id/training_results_*.json[.gz]")
        decompress: Whether to decompress gzipped data. If None, auto-detect from file extension.

    Returns:
        dict: Training results dictionary

    Raises:
        Exception: If download or decompression fails

    Example:
        >>> results = download_training_results("abc-123/training_results_20251022_130000.json.gz")
        >>> print(results['model_type'])
        "Dense"
    """
    try:
        supabase = get_supabase_admin_client()

        logger.info(f"Downloading training results: {file_path}")

        response = supabase.storage.from_('training-results').download(file_path)

        if decompress is None:
            decompress = file_path.endswith('.gz')

        if decompress:
            logger.info(f"Decompressing {len(response) / 1024 / 1024:.2f}MB...")
            data = gzip.decompress(response)
            logger.info(f"Decompressed to {len(data) / 1024 / 1024:.2f}MB")
        else:
            data = response

        results = json.loads(data.decode('utf-8'))

        logger.info(f"✅ Training results downloaded successfully: {file_path}")
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

        logger.info(f"Deleting training results: {file_path}")
        response = supabase.storage.from_('training-results').remove([file_path])

        logger.info(f"✅ Deleted training results: {file_path}")
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

        logger.info(f"Listing training results for session: {session_id}")
        response = supabase.storage.from_('training-results').list(session_id)

        logger.info(f"✅ Found {len(response) if response else 0} files for session {session_id}")
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
        from utils.database import get_supabase_client, create_or_get_session_uuid

        if supabase is None:
            supabase = get_supabase_client()

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
            logger.info(f"📥 Downloading training results from storage for session {session_id}")
            try:
                results = download_training_results(
                    file_path=record['results_file_path'],
                    decompress=record.get('compressed', False)
                )
                logger.info(f"✅ Successfully loaded results from storage for {session_id}")
                return results
            except Exception as download_error:
                logger.error(f"Failed to download from storage: {download_error}")

        if record.get('results'):
            logger.info(f"Using legacy JSONB results for session {session_id}")
            return record['results']

        logger.warning(f"No results available (neither storage nor JSONB) for session {session_id}")
        return None

    except Exception as e:
        logger.error(f"Error fetching training results for session {session_id}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise
