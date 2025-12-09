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


def verify_session_files(session_id: str, metadata: Dict) -> Dict:
    """
    Verify all files for a session exist and are valid.

    Checks that all files listed in metadata actually exist
    in the upload directory.

    Args:
        session_id: Session identifier
        metadata: Session metadata containing file list

    Returns:
        dict: {
            'valid': bool,
            'missing_files': list,
            'existing_files': list,
            'total_files': int
        }
    """
    upload_dir = os.path.join(UPLOAD_BASE_DIR, session_id)
    files_list = metadata.get('files', [])

    missing_files = []
    existing_files = []

    for file_info in files_list:
        file_name = file_info.get('fileName', file_info.get('file_name'))
        file_path = os.path.join(upload_dir, file_name)

        if os.path.exists(file_path):
            existing_files.append(file_name)
        else:
            missing_files.append(file_name)

    return {
        'valid': len(missing_files) == 0,
        'missing_files': missing_files,
        'existing_files': existing_files,
        'total_files': len(files_list)
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
        _update_session_metadata_from_chunk(upload_id, filename, additional_data)

    result = {
        'success': True,
        'message': f'Successfully received chunk {chunk_index} of {total_chunks}',
        'assembled': False
    }

    if chunk_index == total_chunks - 1:
        assembled_path = assemble_file_locally(upload_id, filename)

        file_size = os.path.getsize(assembled_path)

        result['assembled'] = True
        result['assembled_path'] = assembled_path
        result['file_size'] = file_size
        result['message'] = f'File {filename} assembled successfully'

        logger.info(f"File assembled: {filename}, size: {file_size / 1024:.2f} KB")

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

    logger.info(f"Updated session metadata for {upload_id} with file {filename}")


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

    Deletes file from appropriate storage bucket (based on type),
    then removes record from database.

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

    storage_deleted = False

    if storage_path:
        try:
            bucket_name = 'aus-csv-files' if file_type == 'output' else 'csv-files'
            storage_response = supabase.storage.from_(bucket_name).remove([storage_path])
            logger.info(f"Deleted file from storage: {bucket_name}/{storage_path}")
            storage_deleted = True
        except Exception as storage_error:
            logger.warning(f"Could not delete file from storage: {str(storage_error)}")

    delete_response = supabase.table('files').delete().eq('id', file_id).execute()

    if not delete_response.data or len(delete_response.data) == 0:
        raise Exception(f'Failed to delete file {file_id} from database')

    logger.info(f"Deleted CSV file record: {file_id} ({file_record['file_name']})")

    return {
        'deleted_file': file_record,
        'message': f"File {file_record['file_name']} deleted successfully",
        'storage_deleted': storage_deleted
    }
