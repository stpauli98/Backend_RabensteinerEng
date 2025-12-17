"""
Upload Management Service
Business logic for file uploads, chunk processing, and CSV file management

This service handles:
- Chunked file uploads with metadata tracking
- File assembly from chunks
- CSV file CRUD operations (create, update, delete)
- Session metadata management for uploads
- File hash verification

Created: 2025-10-24
Phase 5 of training.py refactoring
"""

import os
import json
import logging
import hashlib
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

UPLOAD_BASE_DIR = os.environ.get('UPLOAD_BASE_DIR', 'uploads/file_uploads')


def generate_storage_path(session_id: str, file_name: str, bezeichnung: str = None) -> str:
    """
    Generate unique storage path for a file including bezeichnung.

    This prevents file overwrites when the same file is uploaded
    with different bezeichnung values.

    Args:
        session_id: Session UUID
        file_name: Original file name
        bezeichnung: Optional bezeichnung for uniqueness

    Returns:
        Storage path: "session_id/bezeichnung_filename" or "session_id/filename"
    """
    import re

    def sanitize_for_path(name: str) -> str:
        """Sanitize string for use in storage path."""
        if not name:
            return ""
        # Replace spaces and special chars with underscores
        sanitized = re.sub(r'[^\w\-.]', '_', name)
        # Remove multiple consecutive underscores
        sanitized = re.sub(r'_+', '_', sanitized)
        # Remove leading/trailing underscores
        return sanitized.strip('_')

    safe_filename = sanitize_for_path(file_name)

    if bezeichnung:
        safe_bezeichnung = sanitize_for_path(bezeichnung)
        return f"{session_id}/{safe_bezeichnung}_{safe_filename}"

    return f"{session_id}/{safe_filename}"


def verify_file_hash(file_data: bytes, expected_hash: str) -> bool:
    """
    Verify file integrity using SHA-256 hash.

    Args:
        file_data: Raw file bytes
        expected_hash: Expected SHA-256 hash

    Returns:
        bool: True if hash matches, False otherwise
    """
    calculated_hash = hashlib.sha256(file_data).hexdigest()
    return calculated_hash == expected_hash


def assemble_file_locally(upload_id: str, filename: str) -> str:
    """
    Assemble complete file from uploaded chunks.

    Combines all chunk files for a given filename in order,
    then deletes the individual chunks.

    Args:
        upload_id: Session/upload identifier
        filename: Name of the file to assemble

    Returns:
        str: Path to assembled file

    Raises:
        FileNotFoundError: If upload directory doesn't exist
        Exception: If assembly fails
    """
    upload_dir = os.path.join(UPLOAD_BASE_DIR, upload_id)

    if not os.path.exists(upload_dir):
        raise FileNotFoundError(f'Upload directory not found: {upload_dir}')

    metadata_path = os.path.join(upload_dir, 'metadata.json')
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f'Metadata file not found: {metadata_path}')

    with open(metadata_path, 'r') as f:
        chunk_metadata = json.load(f)

    file_chunks = [c for c in chunk_metadata if c['fileName'] == filename]

    if not file_chunks:
        raise ValueError(f'No chunks found for file: {filename}')

    file_chunks.sort(key=lambda x: x['chunkIndex'])

    assembled_file_path = os.path.join(upload_dir, filename)

    with open(assembled_file_path, 'wb') as assembled_file:
        for chunk_info in file_chunks:
            chunk_path = chunk_info['filePath']
            if os.path.exists(chunk_path):
                with open(chunk_path, 'rb') as chunk_file:
                    assembled_file.write(chunk_file.read())

    for chunk_info in file_chunks:
        chunk_path = chunk_info['filePath']
        if os.path.exists(chunk_path):
            try:
                os.remove(chunk_path)
                logger.debug(f"Deleted chunk: {chunk_path}")
            except Exception as e:
                logger.warning(f"Failed to delete chunk {chunk_path}: {str(e)}")

    logger.info(f"Assembled file: {filename} at {assembled_file_path}")

    return assembled_file_path


def upload_assembled_file_to_storage(
    session_id: str,
    filename: str,
    file_path: str,
    file_type: str,
    bezeichnung: str = None
) -> str:
    """
    Upload assembled file to Supabase Storage immediately after assembly.

    This ensures files are persisted in cloud storage before Cloud Run's
    ephemeral filesystem is cleared.

    Args:
        session_id: Session identifier
        filename: Name of the file
        file_path: Local path to assembled file
        file_type: 'input' or 'output'
        bezeichnung: File bezeichnung for unique storage path

    Returns:
        str: Storage path if successful, empty string otherwise
    """
    from shared.database.client import get_supabase_admin_client

    try:
        supabase = get_supabase_admin_client()
        bucket_name = 'aus-csv-files' if file_type == 'output' else 'csv-files'
        # Use bezeichnung in storage path to prevent overwrites
        storage_path = generate_storage_path(session_id, filename, bezeichnung)

        with open(file_path, 'rb') as f:
            file_content = f.read()

        # Try upload, handle if file already exists
        try:
            supabase.storage.from_(bucket_name).upload(
                path=storage_path,
                file=file_content,
                file_options={"content-type": "text/csv"}
            )
            logger.info(f"Uploaded {filename} to {bucket_name}/{storage_path}")
        except Exception as e:
            if "already exists" in str(e).lower() or "duplicate" in str(e).lower():
                # File exists, update instead
                supabase.storage.from_(bucket_name).update(
                    path=storage_path,
                    file=file_content,
                    file_options={"content-type": "text/csv"}
                )
                logger.info(f"Updated existing file {filename} in {bucket_name}/{storage_path}")
            else:
                raise

        return storage_path

    except Exception as e:
        logger.error(f"Failed to upload {filename} to storage: {str(e)}")
        return ""


def update_file_storage_path_in_metadata(
    session_id: str,
    filename: str,
    storage_path: str
) -> bool:
    """
    Update storagePath in session_metadata.json for a specific file.
    
    This ensures the metadata reflects the storage location so that
    finalize_session can correctly reference the files.

    Args:
        session_id: Session identifier
        filename: Name of the file
        storage_path: Storage path in Supabase Storage

    Returns:
        bool: True if successful, False otherwise
    """
    upload_dir = os.path.join(UPLOAD_BASE_DIR, session_id)
    session_metadata_path = os.path.join(upload_dir, 'session_metadata.json')

    if not os.path.exists(session_metadata_path):
        logger.warning(f"Session metadata not found: {session_metadata_path}")
        return False

    try:
        with open(session_metadata_path, 'r') as f:
            session_metadata = json.load(f)

        # Find and update the file's storagePath
        updated = False
        for file_info in session_metadata.get('files', []):
            if file_info.get('fileName') == filename:
                file_info['storagePath'] = storage_path
                updated = True
                break

        if not updated:
            logger.warning(f"File {filename} not found in session metadata")
            return False

        with open(session_metadata_path, 'w') as f:
            json.dump(session_metadata, f, indent=2)

        logger.info(f"Updated storagePath for {filename}: {storage_path}")
        return True

    except Exception as e:
        logger.error(f"Failed to update storagePath for {filename}: {str(e)}")
        return False


def save_session_metadata_locally(session_id: str, metadata: dict) -> bool:
    """
    Save session metadata to local file system.

    Args:
        session_id: Session identifier
        metadata: Metadata dictionary to save

    Returns:
        bool: True if successful

    Raises:
        Exception: If save operation fails
    """
    upload_dir = os.path.join(UPLOAD_BASE_DIR, session_id)
    os.makedirs(upload_dir, exist_ok=True)

    metadata_path = os.path.join(upload_dir, 'session_metadata.json')

    metadata['lastUpdated'] = datetime.now().isoformat()

    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Saved session metadata for session: {session_id}")
    return True


def get_session_metadata_locally(session_id: str) -> Optional[Dict]:
    """
    Retrieve session metadata from local file system.

    Args:
        session_id: Session identifier

    Returns:
        dict: Session metadata or None if not found
    """
    upload_dir = os.path.join(UPLOAD_BASE_DIR, session_id)
    metadata_path = os.path.join(upload_dir, 'session_metadata.json')

    if not os.path.exists(metadata_path):
        return None

    try:
        with open(metadata_path, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError:
        logger.error(f"Failed to parse session metadata for: {session_id}")
        return None


def update_session_metadata(session_id: str, data: Dict) -> Dict:
    """
    Update session metadata with new data.

    Merges new data with existing metadata, preserving
    existing fields unless explicitly overridden.

    Args:
        session_id: Session identifier
        data: New metadata to merge

    Returns:
        dict: Updated metadata

    Raises:
        ValueError: If session not found
    """
    existing_metadata = get_session_metadata_locally(session_id)

    if existing_metadata is None:
        raise ValueError(f'Session metadata not found for: {session_id}')

    existing_metadata.update(data)
    existing_metadata['lastUpdated'] = datetime.now().isoformat()

    save_session_metadata_locally(session_id, existing_metadata)

    return existing_metadata


def download_files_from_storage(session_id: str, files_metadata: List[Dict]) -> Tuple[str, List[str]]:
    """
    Download all session files from Supabase Storage to temp directory.

    For Cloud Run (stateless), files must be downloaded from Storage
    since local filesystem doesn't persist between instances.

    Args:
        session_id: Session identifier
        files_metadata: List of file info dicts with 'fileName', 'storagePath', 'type'

    Returns:
        Tuple of (temp_dir_path, list_of_downloaded_file_names)
    """
    import tempfile
    import shutil
    from shared.database.client import get_supabase_admin_client

    supabase = get_supabase_admin_client()

    # Create temp directory for this session
    temp_dir = tempfile.mkdtemp(prefix=f"session_{session_id}_")
    downloaded_files = []

    logger.info(f"Downloading {len(files_metadata)} files for session {session_id} to {temp_dir}")

    for file_info in files_metadata:
        file_name = file_info.get('fileName', file_info.get('file_name', ''))
        storage_path = file_info.get('storagePath', file_info.get('storage_path', ''))
        file_type = file_info.get('type', 'input')

        if not file_name:
            logger.warning(f"Missing fileName for file: {file_info}")
            continue

        if not storage_path:
            logger.warning(f"Missing storagePath for file {file_name}, skipping download")
            continue

        bucket_name = 'aus-csv-files' if file_type == 'output' else 'csv-files'
        local_path = os.path.join(temp_dir, file_name)

        try:
            logger.info(f"Downloading {bucket_name}/{storage_path} → {local_path}")
            response = supabase.storage.from_(bucket_name).download(storage_path)

            if response:
                with open(local_path, 'wb') as f:
                    f.write(response)
                downloaded_files.append(file_name)
                logger.info(f"✓ Downloaded {file_name} ({len(response)} bytes)")
            else:
                logger.warning(f"Empty response for {storage_path}")

        except Exception as e:
            logger.error(f"Failed to download {storage_path}: {str(e)}")

    logger.info(f"Downloaded {len(downloaded_files)}/{len(files_metadata)} files for session {session_id}")
    return temp_dir, downloaded_files


def cleanup_temp_dir(temp_dir: str):
    """
    Remove temporary directory and all contents.

    Should be called after verify_session_files() and calculate_n_dat_from_session()
    to clean up downloaded files.
    """
    import shutil
    try:
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            logger.info(f"Cleaned up temp dir: {temp_dir}")
    except Exception as e:
        logger.warning(f"Failed to cleanup temp dir {temp_dir}: {str(e)}")


def verify_session_files(session_id: str, metadata: Dict) -> Dict:
    """
    Verify all files exist for a session by downloading from Supabase Storage.

    For Cloud Run (stateless), files are downloaded from Storage to temp dir
    since local filesystem doesn't persist between instances.

    IMPORTANT: Files are stored directly in Supabase `files` table via frontend,
    NOT in local metadata. So we query the DB directly to get accurate file count.

    Args:
        session_id: Session identifier
        metadata: Session metadata (used as fallback, but DB is primary source)

    Returns:
        dict: {
            'valid': bool,
            'missing_files': list,
            'existing_files': list,
            'downloaded_files': list,
            'total_files': int,
            'temp_dir': str or None  # Caller must cleanup!
        }
    """
    import uuid as uuid_lib
    from shared.database.client import get_supabase_admin_client
    from shared.database.operations import create_or_get_session_uuid

    # Get UUID for the session
    try:
        uuid_lib.UUID(session_id)
        database_session_id = session_id
    except (ValueError, TypeError):
        try:
            database_session_id = create_or_get_session_uuid(session_id)
        except Exception as e:
            logger.error(f"Failed to get UUID for session {session_id}: {str(e)}")
            database_session_id = None

        if not database_session_id:
            logger.error(f"Failed to get UUID for session {session_id}")
            return {
                'valid': False,
                'missing_files': [],
                'existing_files': [],
                'downloaded_files': [],
                'total_files': 0,
                'temp_dir': None
            }

    # Query files directly from Supabase `files` table (PRIMARY SOURCE)
    files_metadata = []
    try:
        supabase = get_supabase_admin_client()
        response = supabase.table('files').select('*').eq('session_id', database_session_id).execute()

        if response.data:
            logger.info(f"Found {len(response.data)} files in DB for session {session_id}")
            for f in response.data:
                files_metadata.append({
                    'fileName': f.get('file_name'),
                    'storagePath': f.get('storage_path'),
                    'type': f.get('type', 'input')
                })
        else:
            logger.info(f"No files found in DB for session {session_id}, checking metadata fallback")
            # Fallback to metadata if DB query returns nothing
            files_metadata = metadata.get('files', [])
    except Exception as e:
        logger.error(f"Error querying files table for session {session_id}: {str(e)}")
        # Fallback to metadata on error
        files_metadata = metadata.get('files', [])

    if not files_metadata:
        logger.warning(f"No files found for session {session_id} (neither in DB nor metadata)")
        return {
            'valid': True,
            'missing_files': [],
            'existing_files': [],
            'downloaded_files': [],
            'total_files': 0,
            'temp_dir': None
        }

    logger.info(f"Processing {len(files_metadata)} files for session {session_id}")

    # Download all files from Supabase Storage to temp directory
    temp_dir, downloaded_files = download_files_from_storage(session_id, files_metadata)

    # Determine which files are missing
    expected_files = [
        f.get('fileName', f.get('file_name', ''))
        for f in files_metadata
        if f.get('fileName') or f.get('file_name')
    ]
    missing_files = [f for f in expected_files if f not in downloaded_files]

    if missing_files:
        logger.warning(f"Missing files for session {session_id}: {missing_files}")

    return {
        'valid': len(missing_files) == 0,
        'missing_files': missing_files,
        'existing_files': downloaded_files,
        'downloaded_files': downloaded_files,
        'total_files': len(files_metadata),  # Now using accurate count from DB
        'temp_dir': temp_dir  # Caller must cleanup with cleanup_temp_dir()!
    }


def process_chunk_upload(
    chunk_data: bytes,
    metadata: Dict,
    additional_data: Optional[Dict] = None
) -> Dict:
    """
    Process a single chunk upload with metadata tracking.

    Handles chunk storage, metadata updates, and file assembly
    when all chunks are received.

    Args:
        chunk_data: Raw chunk bytes
        metadata: Chunk metadata (chunkIndex, totalChunks, fileName, sessionId)
        additional_data: Optional additional data (fileMetadata, sessionInfo, etc.)

    Returns:
        dict: {
            'success': bool,
            'message': str,
            'assembled': bool (if last chunk),
            'assembled_path': str (if last chunk)
        }

    Raises:
        ValueError: If required metadata fields missing
        Exception: If chunk processing fails
    """
    required_fields = ['chunkIndex', 'totalChunks', 'fileName', 'sessionId']
    for field in required_fields:
        if field not in metadata:
            raise ValueError(f'Missing required metadata field: {field}')

    upload_id = metadata['sessionId']
    chunk_index = metadata['chunkIndex']
    total_chunks = metadata['totalChunks']
    filename = metadata['fileName']
    file_type = metadata.get('fileType', 'unknown')

    # DEBUG: Log chunk receipt with all metadata
    logger.info(f"📥 CHUNK RECEIVED: {filename} [{chunk_index+1}/{total_chunks}] type={file_type} size={len(chunk_data)} bytes session={upload_id[:8]}...")

    upload_dir = os.path.join(UPLOAD_BASE_DIR, upload_id)
    os.makedirs(upload_dir, exist_ok=True)

    chunk_filename = f"{filename}_{chunk_index}"
    chunk_path = os.path.join(upload_dir, chunk_filename)

    with open(chunk_path, 'wb') as f:
        f.write(chunk_data)

    metadata_path = os.path.join(upload_dir, 'metadata.json')
    chunk_metadata = []

    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r') as f:
                chunk_metadata = json.load(f)
        except json.JSONDecodeError:
            chunk_metadata = []

    chunk_info = {
        'chunkIndex': chunk_index,
        'totalChunks': total_chunks,
        'fileName': filename,
        'filePath': chunk_path,
        'fileType': metadata.get('fileType', 'unknown'),
        'createdAt': datetime.now().isoformat(),
        'params': additional_data or {}
    }

    existing_chunk = next((c for c in chunk_metadata if c['chunkIndex'] == chunk_index), None)
    if existing_chunk:
        existing_chunk.update(chunk_info)
    else:
        chunk_metadata.append(chunk_info)

    with open(metadata_path, 'w') as f:
        json.dump(chunk_metadata, f, indent=2)

    if chunk_index == 0 and additional_data:
        # DEBUG: Log additional_data structure for first chunk
        file_metadata = additional_data.get('fileMetadata', {})
        logger.info(f"📋 FIRST CHUNK METADATA for {filename}:")
        logger.info(f"   bezeichnung: '{file_metadata.get('bezeichnung', 'MISSING')}'")
        logger.info(f"   type: '{file_metadata.get('type', 'MISSING')}'")
        logger.info(f"   storagePath: '{file_metadata.get('storagePath', 'MISSING')}'")
        _update_session_metadata_from_chunk(upload_id, filename, additional_data)

    result = {
        'success': True,
        'message': f'Successfully received chunk {chunk_index} of {total_chunks}',
        'assembled': False
    }

    if chunk_index == total_chunks - 1:
        assembled_path = assemble_file_locally(upload_id, filename)

        file_size = os.path.getsize(assembled_path)

        # Get bezeichnung from session_metadata for unique storage path
        bezeichnung = None
        session_metadata = get_session_metadata_locally(upload_id)
        if session_metadata:
            for file_info in session_metadata.get('files', []):
                if file_info.get('fileName') == filename:
                    bezeichnung = file_info.get('bezeichnung', '')
                    break

        # CRITICAL: Upload to Supabase Storage immediately after assembly
        # This ensures files are persisted before Cloud Run ephemeral storage is cleared
        file_type = metadata.get('fileType', 'input')
        storage_path = upload_assembled_file_to_storage(
            upload_id, filename, assembled_path, file_type, bezeichnung
        )

        logger.info(f"📤 STORAGE PATH for {filename}: {storage_path} (bezeichnung='{bezeichnung}')")

        # Update session_metadata.json with storagePath
        if storage_path:
            update_file_storage_path_in_metadata(upload_id, filename, storage_path)
            logger.info(f"File {filename} uploaded to storage: {storage_path}")
        else:
            logger.warning(f"Failed to upload {filename} to storage - will retry in finalize")

        result['assembled'] = True
        result['assembled_path'] = assembled_path
        result['storage_path'] = storage_path  # Include storage path in result
        result['file_size'] = file_size
        result['message'] = f'File {filename} assembled and uploaded successfully'

        logger.info(f"File assembled: {filename}, size: {file_size / 1024:.2f} KB, storage: {storage_path}")

        if filename.endswith('.csv'):
            try:
                df = pd.read_csv(assembled_path)
                result['csv_info'] = {
                    'rows': len(df),
                    'columns': len(df.columns)
                }
                logger.info(f"CSV processed: {len(df)} rows, {len(df.columns)} columns")
            except Exception as e:
                logger.warning(f"Failed to parse CSV {filename}: {str(e)}")

    return result


def _update_session_metadata_from_chunk(
    upload_id: str,
    filename: str,
    additional_data: Dict
) -> None:
    """
    Update session metadata from first chunk's additional data.

    Internal helper function to update session metadata when
    the first chunk of a file is uploaded.

    Args:
        upload_id: Session identifier
        filename: File being uploaded
        additional_data: Additional metadata from chunk upload
    """
    upload_dir = os.path.join(UPLOAD_BASE_DIR, upload_id)
    session_metadata_path = os.path.join(upload_dir, 'session_metadata.json')

    session_metadata = {}
    if os.path.exists(session_metadata_path):
        try:
            with open(session_metadata_path, 'r') as f:
                session_metadata = json.load(f)
        except json.JSONDecodeError:
            session_metadata = {}

    if 'timeInfo' not in session_metadata:
        session_metadata['timeInfo'] = additional_data.get('timeInfo', {})
    if 'zeitschritte' not in session_metadata:
        session_metadata['zeitschritte'] = additional_data.get('zeitschritte', {})
    if 'sessionInfo' not in session_metadata:
        session_metadata['sessionInfo'] = additional_data.get('sessionInfo', {})

    if 'files' not in session_metadata:
        session_metadata['files'] = []

    file_metadata = additional_data.get('fileMetadata', {})
    if file_metadata:
        file_exists = False
        for i, existing_file in enumerate(session_metadata.get('files', [])):
            if existing_file.get('fileName') == filename:
                session_metadata['files'][i] = file_metadata
                file_exists = True
                break

        if not file_exists:
            session_metadata['files'].append(file_metadata)

    session_metadata['lastUpdated'] = datetime.now().isoformat()

    with open(session_metadata_path, 'w') as f:
        json.dump(session_metadata, f, indent=2)

    # DEBUG: Log full files list in session_metadata
    files_list = session_metadata.get('files', [])
    logger.info(f"📁 SESSION METADATA UPDATED for {upload_id}:")
    logger.info(f"   Total files in metadata: {len(files_list)}")
    for f in files_list:
        logger.info(f"   - {f.get('fileName', 'N/A')}: bezeichnung='{f.get('bezeichnung', '')}' type='{f.get('type', 'N/A')}' storagePath='{f.get('storagePath', '')[:50]}...'")


def create_csv_file_record(session_id: str, file_data: Dict) -> Dict:
    """
    Create new CSV file record in database.

    Args:
        session_id: Session identifier
        file_data: File metadata (file_name, type, bezeichnung, etc.)

    Returns:
        dict: Created file record from database

    Raises:
        ValueError: If session not found or invalid data
        Exception: If database operation fails
    """
    from shared.database.operations import get_supabase_client, create_or_get_session_uuid

    # Note: This function should receive user_id from caller for proper validation
    # For now, uses None for backward compatibility (to be fixed in caller chain)
    uuid_session_id = create_or_get_session_uuid(session_id, user_id=None)
    if not uuid_session_id:
        raise ValueError(f'Session {session_id} not found')

    required_fields = ['file_name', 'type']
    for field in required_fields:
        if field not in file_data:
            raise ValueError(f'Missing required field: {field}')

    file_record = {
        'session_id': str(uuid_session_id),
        'file_name': file_data['file_name'],
        'type': file_data['type'],
        'bezeichnung': file_data.get('bezeichnung', file_data['file_name']),
        'created_at': datetime.now().isoformat()
    }

    supabase = get_supabase_client()
    response = supabase.table('files').insert(file_record).execute()

    if not response.data or len(response.data) == 0:
        raise Exception('Failed to create file record in database')

    created_file = response.data[0]

    logger.info(f"Created CSV file record: {created_file['file_name']} (ID: {created_file['id']})")

    return created_file


def update_csv_file_record(file_id: str, file_data: Dict) -> Dict:
    """
    Update existing CSV file record in database.

    Validates UUID format and allowed fields, converts numeric fields to strings.

    Args:
        file_id: File UUID
        file_data: Updated file metadata

    Returns:
        dict: Updated file record

    Raises:
        ValueError: If file not found or invalid file_id format
        Exception: If update fails
    """
    import uuid as uuid_lib
    from shared.database.operations import get_supabase_client

    try:
        uuid_lib.UUID(file_id)
    except (ValueError, TypeError):
        raise ValueError('Invalid file ID format')

    supabase = get_supabase_client()

    allowed_fields = [
        'file_name', 'bezeichnung', 'min', 'max', 'offset', 'datenpunkte',
        'numerische_datenpunkte', 'numerischer_anteil', 'datenform',
        'datenanpassung', 'zeitschrittweite', 'zeitschrittweite_mittelwert',
        'zeitschrittweite_min', 'skalierung', 'skalierung_max', 'skalierung_min',
        'zeithorizont_start', 'zeithorizont_end', 'zeitschrittweite_transferierten_daten',
        'offset_transferierten_daten', 'mittelwertbildung_uber_den_zeithorizont',
        'utc_min', 'utc_max', 'type'
    ]

    numeric_fields = [
        'min', 'max', 'offset', 'datenpunkte', 'numerische_datenpunkte',
        'numerischer_anteil', 'zeitschrittweite', 'zeitschrittweite_mittelwert',
        'zeitschrittweite_min', 'skalierung_max', 'skalierung_min',
        'zeithorizont_start', 'zeithorizont_end', 'zeitschrittweite_transferierten_daten',
        'offset_transferierten_daten'
    ]

    update_data = {}
    for field in allowed_fields:
        if field in file_data:
            if field in numeric_fields:
                value = file_data[field]
                # Handle None, empty string, and 'null' string as SQL NULL
                if value is None or value == '' or value == 'null' or value == 'None':
                    update_data[field] = None
                else:
                    try:
                        update_data[field] = float(value)
                    except (ValueError, TypeError):
                        update_data[field] = None
            else:
                update_data[field] = file_data[field]

    if not update_data:
        raise ValueError('No valid fields to update')

    response = supabase.table('files').update(update_data).eq('id', file_id).execute()

    if not response.data or len(response.data) == 0:
        raise ValueError(f'File {file_id} not found')

    updated_file = response.data[0]

    logger.info(f"Updated CSV file record: {file_id}")

    return updated_file


def delete_csv_file_record(file_id: str) -> Dict:
    """
    Delete CSV file record from database and Supabase Storage.

    Uses reference counting for shared storage - only deletes from storage
    if no other DB records reference the same storage_path.

    Args:
        file_id: File UUID

    Returns:
        dict: {
            'deleted_file': dict,
            'message': str,
            'storage_deleted': bool
        }

    Raises:
        ValueError: If file not found or invalid file_id format
        Exception: If deletion fails
    """
    import uuid as uuid_lib
    from shared.database.operations import get_supabase_client
    from shared.database.storage import get_storage_reference_count

    try:
        uuid_lib.UUID(file_id)
    except (ValueError, TypeError):
        raise ValueError('Invalid file ID format')

    supabase = get_supabase_client()

    file_response = supabase.table('files').select('*').eq('id', file_id).execute()

    if not file_response.data or len(file_response.data) == 0:
        raise ValueError(f'File {file_id} not found')

    file_record = file_response.data[0]
    storage_path = file_record.get('storage_path', '')
    file_type = file_record.get('type', 'input')

    # Check reference count BEFORE deleting DB record (exclude current file)
    ref_count = get_storage_reference_count(storage_path, exclude_file_id=file_id) if storage_path else 0

    # Delete DB record FIRST
    delete_response = supabase.table('files').delete().eq('id', file_id).execute()

    if not delete_response.data or len(delete_response.data) == 0:
        raise Exception(f'Failed to delete file {file_id} from database')

    logger.info(f"Deleted CSV file record: {file_id} ({file_record['file_name']})")

    storage_deleted = False

    # Only delete from storage if no other records reference this storage_path
    if storage_path and ref_count == 0:
        try:
            bucket_name = 'aus-csv-files' if file_type == 'output' else 'csv-files'
            supabase.storage.from_(bucket_name).remove([storage_path])
            logger.info(f"Deleted file from storage: {bucket_name}/{storage_path} (no more references)")
            storage_deleted = True
        except Exception as storage_error:
            logger.warning(f"Could not delete file from storage: {str(storage_error)}")
    elif storage_path and ref_count > 0:
        logger.info(f"Storage file kept: {storage_path} ({ref_count} references remain)")

    return {
        'deleted_file': file_record,
        'message': f"File {file_record['file_name']} deleted successfully",
        'storage_deleted': storage_deleted,
        'shared_storage_kept': ref_count > 0
    }


def calculate_n_dat_from_session(session_id: str, temp_dir: str = None) -> int:
    """
    Calculate n_dat (total number of data samples) from CSV files.

    For shared storage, counts each DB record separately even if they share
    the same physical storage file. This means if the same CSV is uploaded
    twice with different Bezeichnung, n_dat will be doubled.

    For Cloud Run (stateless), uses temp_dir where files were downloaded
    from Supabase Storage by verify_session_files().

    Args:
        session_id: ID of the session
        temp_dir: Temp directory containing downloaded files (from verify_session_files)
                  If None, falls back to local UPLOAD_BASE_DIR (for backwards compatibility)

    Returns:
        int: Total number of data samples (n_dat)
    """
    try:
        from shared.database.operations import get_supabase_client

        # Use temp_dir if provided (Cloud Run), otherwise fall back to local dir
        if temp_dir and os.path.exists(temp_dir):
            working_dir = temp_dir
            logger.info(f"Using temp dir for n_dat calculation: {temp_dir}")
        else:
            working_dir = os.path.join(UPLOAD_BASE_DIR, session_id)
            if not os.path.exists(working_dir):
                logger.warning(f"No directory found for session {session_id}")
                return 0

        # Get DB records to count references per storage_path (for shared storage)
        supabase = get_supabase_client()
        try:
            files_response = supabase.table('files')\
                .select('id, storage_path, file_name')\
                .eq('session_id', session_id)\
                .execute()

            db_files = files_response.data if files_response.data else []

            # Group by storage_path to count how many DB records share each file
            storage_path_counts = {}
            for file_record in db_files:
                storage_path = file_record.get('storage_path', '')
                file_name = file_record.get('file_name', '')

                if storage_path:
                    if storage_path not in storage_path_counts:
                        storage_path_counts[storage_path] = {
                            'count': 0,
                            'file_name': file_name
                        }
                    storage_path_counts[storage_path]['count'] += 1

            logger.info(f"Found {len(db_files)} DB records, {len(storage_path_counts)} unique storage paths")

        except Exception as db_error:
            logger.warning(f"Could not get DB file records: {str(db_error)}, falling back to filesystem")
            storage_path_counts = {}

        total_samples = 0
        csv_files_processed = set()

        for file_name in os.listdir(working_dir):
            if file_name.lower().endswith('.csv') and os.path.isfile(os.path.join(working_dir, file_name)):
                file_path = os.path.join(working_dir, file_name)

                try:
                    df = pd.read_csv(file_path)
                    file_samples = len(df)

                    # Check if this file is shared (multiple DB records reference it)
                    # Find matching storage_path
                    multiplier = 1
                    for storage_path, info in storage_path_counts.items():
                        if file_name in storage_path or info['file_name'] == file_name:
                            multiplier = info['count']
                            break

                    # Multiply samples by number of DB records using this file
                    file_contribution = file_samples * multiplier
                    total_samples += file_contribution

                    if multiplier > 1:
                        logger.info(f"File {file_name}: {file_samples} samples × {multiplier} records = {file_contribution}")
                    else:
                        logger.info(f"File {file_name}: {file_samples} samples")

                    csv_files_processed.add(file_name)

                except Exception as e:
                    logger.error(f"Error reading CSV file {file_name}: {str(e)}")
                    continue

        if not csv_files_processed:
            logger.warning(f"No CSV files found for session {session_id}")
            return 0

        logger.info(f"Session {session_id} total n_dat: {total_samples}")
        return total_samples

    except Exception as e:
        logger.error(f"Error calculating n_dat for session {session_id}: {str(e)}")
        return 0


def cleanup_incomplete_uploads(upload_base_dir: str = None, max_age_hours: int = 24) -> int:
    """
    Clean up incomplete or old upload sessions.

    Args:
        upload_base_dir: Base directory for uploads (defaults to UPLOAD_BASE_DIR)
        max_age_hours: Maximum age for incomplete uploads

    Returns:
        int: Number of cleaned up sessions
    """
    import time
    import shutil

    if upload_base_dir is None:
        upload_base_dir = UPLOAD_BASE_DIR

    cleaned_count = 0
    current_time = time.time()
    max_age_seconds = max_age_hours * 3600

    try:
        for session_dir in Path(upload_base_dir).iterdir():
            if not session_dir.is_dir():
                continue

            dir_age = current_time - session_dir.stat().st_mtime
            if dir_age > max_age_seconds:
                finalized_marker = session_dir / '.finalized'
                if not finalized_marker.exists():
                    try:
                        shutil.rmtree(session_dir)
                        cleaned_count += 1
                        logger.info(f"Cleaned up old session: {session_dir.name}")
                    except Exception as e:
                        logger.error(f"Error cleaning up {session_dir}: {e}")

    except Exception as e:
        logger.error(f"Error during cleanup: {e}")

    return cleaned_count
