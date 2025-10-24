"""
Session Management Service
Business logic for session lifecycle management

This service handles:
- Session initialization and finalization
- Session listing and retrieval
- Session deletion
- Time info and zeitschritte management
- CSV file listing for sessions
- Session status tracking

Created: 2025-10-24
Phase 4 of training.py refactoring
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from utils.database import get_string_id_from_uuid

logger = logging.getLogger(__name__)

# Configuration
UPLOAD_BASE_DIR = os.getenv('UPLOAD_BASE_DIR', 'uploads/file_uploads')


def initialize_session(session_id: str, time_info: Dict, zeitschritte: Dict) -> Dict:
    """
    Initialize a new upload session.

    Creates session directory and saves initial metadata.

    Args:
        session_id: Unique session identifier
        time_info: Time information dictionary
        zeitschritte: Zeitschritte dictionary

    Returns:
        dict: {
            'session_id': str,
            'message': str
        }

    Raises:
        ValueError: If session_id is missing
    """
    if not session_id:
        raise ValueError('Missing session ID')

    # Create directory for session
    upload_dir = os.path.join(UPLOAD_BASE_DIR, session_id)
    os.makedirs(upload_dir, exist_ok=True)

    # Save initial session metadata
    session_metadata = {
        'timeInfo': time_info,
        'zeitschritte': zeitschritte,
        'sessionInfo': {
            'sessionId': session_id,
            'totalFiles': 0
        },
        'lastUpdated': datetime.now().isoformat()
    }

    # Save session metadata
    session_metadata_path = os.path.join(upload_dir, 'session_metadata.json')
    with open(session_metadata_path, 'w') as f:
        json.dump(session_metadata, f, indent=2)

    # Create session in Supabase
    try:
        from utils.database import create_or_get_session_uuid
        session_uuid = create_or_get_session_uuid(session_id)
        if session_uuid:
            # Save session data to Supabase
            from api.routes.training import save_session_to_supabase
            success = save_session_to_supabase(session_id)
            if not success:
                logger.warning(f"Failed to save session {session_id} data to Supabase")
        else:
            logger.warning(f"Failed to create session UUID for {session_id}")
    except Exception as e:
        logger.error(f"Error saving session data to Supabase: {str(e)}")
        # Continue even if Supabase save fails

    return {
        'session_id': session_id,
        'message': f"Session {session_id} initialized successfully"
    }


def finalize_session(session_id: str, session_data: Dict) -> Dict:
    """
    Finalize a session after all files have been uploaded.

    Args:
        session_id: Session identifier
        session_data: Session data from request

    Returns:
        dict: {
            'session_id': str,
            'n_dat': int,
            'file_count': int,
            'message': str
        }
    """
    from api.routes.training import (
        update_session_metadata,
        verify_session_files,
        calculate_n_dat_from_session,
        save_session_metadata_locally,
        save_session_to_database
    )

    # 1. Update session metadata with finalization info
    updated_metadata = update_session_metadata(session_id, session_data)

    # 2. Verify files and update metadata
    final_metadata, file_count = verify_session_files(session_id, updated_metadata)

    # 3. Calculate n_dat (total number of data samples)
    n_dat = calculate_n_dat_from_session(session_id)
    final_metadata['n_dat'] = n_dat

    # Save updated metadata with n_dat
    save_session_metadata_locally(session_id, final_metadata)

    # 4. Save session data to Supabase
    try:
        success = save_session_to_database(session_id, n_dat, file_count)
        if not success:
            logger.warning(f"Failed to save session {session_id} to database, but continuing")
    except Exception as e:
        logger.error(f"Error saving session {session_id} to database: {str(e)}")
        # Continue even if database save fails

    return {
        'session_id': session_id,
        'n_dat': n_dat,
        'file_count': file_count,
        'message': f"Session {session_id} finalized successfully"
    }


def get_sessions_list(user_id: str = None, limit: int = 50) -> List[Dict]:
    """
    List all available training sessions with detailed metadata.

    Queries sessions with file counts, zeitschritte, time_info, and file details.
    Uses complex SQL JOINs with fallback strategies.

    Args:
        user_id: Optional user ID to filter sessions
        limit: Maximum number of sessions to return (default: 50)

    Returns:
        List of session dictionaries with complete metadata
    """
    from utils.database import get_supabase_client

    supabase = get_supabase_client()
    if not supabase:
        raise Exception('Database connection not available')

    # Complex query with JOINs to get all session data
    sessions_query = """
    SELECT
        sm.string_session_id as session_id,
        s.session_name,
        s.created_at,
        s.updated_at,
        s.finalized,
        s.file_count,
        s.n_dat,
        (SELECT COUNT(*) FROM files f WHERE f.session_id = s.id) as actual_file_count,
        (SELECT COUNT(*) FROM zeitschritte z WHERE z.session_id = s.id) as zeitschritte_count,
        (SELECT COUNT(*) FROM time_info ti WHERE ti.session_id = s.id) as time_info_count,
        (SELECT json_agg(
            json_build_object(
                'id', f.id,
                'type', f.type,
                'file_name', f.file_name,
                'bezeichnung', f.bezeichnung
            )
        ) FROM files f WHERE f.session_id = s.id) as files_info
    FROM session_mappings sm
    JOIN sessions s ON sm.uuid_session_id = s.id
    ORDER BY s.created_at DESC
    LIMIT %s
    """

    try:
        # Try RPC for complex query
        response = supabase.rpc('execute_sql', {
            'sql_query': sessions_query,
            'params': [limit]
        }).execute()

        if response.data:
            return _format_sessions_response(response.data)

    except Exception as e:
        logger.warning(f"Complex query failed, using fallback: {str(e)}")

    # Fallback: manual JOIN with multiple queries
    sessions_response = supabase.table('session_mappings').select(
        'string_session_id, uuid_session_id, created_at'
    ).order('created_at', desc=True).limit(limit).execute()

    sessions_data = []
    for session_mapping in sessions_response.data:
        session_uuid = session_mapping['uuid_session_id']
        session_id = session_mapping['string_session_id']

        # Get session details
        session_response = supabase.table('sessions').select('*').eq('id', session_uuid).execute()
        session_details = session_response.data[0] if session_response.data else {}

        # Get file details and count
        files_response = supabase.table('files').select(
            'id, type, file_name, bezeichnung'
        ).eq('session_id', session_uuid).execute()
        files_info = files_response.data if files_response.data else []
        file_count = len(files_info)

        # Get zeitschritte count
        zeit_response = supabase.table('zeitschritte').select(
            'id', count='exact'
        ).eq('session_id', session_uuid).execute()
        zeitschritte_count = zeit_response.count or 0

        # Get time_info count
        time_response = supabase.table('time_info').select(
            'id', count='exact'
        ).eq('session_id', session_uuid).execute()
        time_info_count = time_response.count or 0

        sessions_data.append({
            'session_id': session_id,
            'session_name': session_details.get('session_name'),
            'created_at': session_mapping.get('created_at') or session_details.get('created_at'),
            'updated_at': session_details.get('updated_at'),
            'finalized': session_details.get('finalized', False),
            'file_count': session_details.get('file_count', 0),
            'n_dat': session_details.get('n_dat', 0),
            'actual_file_count': file_count,
            'zeitschritte_count': zeitschritte_count,
            'time_info_count': time_info_count,
            'files_info': files_info
        })

    return _format_sessions_response(sessions_data)


def _format_sessions_response(sessions_data: List[Dict]) -> List[Dict]:
    """
    Format raw session data for frontend consumption.

    Args:
        sessions_data: Raw session data from database

    Returns:
        Formatted session list
    """
    sessions = []
    for session_data in sessions_data:
        session_info = {
            'sessionId': session_data['session_id'],
            'sessionName': session_data.get('session_name'),
            'createdAt': session_data['created_at'],
            'lastUpdated': session_data.get('updated_at') or session_data['created_at'],
            'fileCount': session_data.get('actual_file_count', session_data.get('file_count', 0)),
            'finalized': session_data.get('finalized', False),
            'nDat': session_data.get('n_dat'),
            'zeitschritteCount': session_data.get('zeitschritte_count', 0),
            'timeInfoCount': session_data.get('time_info_count', 0),
            'filesInfo': session_data.get('files_info') or []
        }
        sessions.append(session_info)

    logger.info(f"Formatted {len(sessions)} sessions for response")
    return sessions


def get_session_info(session_id: str) -> Dict:
    """
    Get session information from local files.

    Args:
        session_id: Session identifier

    Returns:
        dict: Session metadata

    Raises:
        FileNotFoundError: If session directory or metadata doesn't exist
    """
    upload_dir = os.path.join(UPLOAD_BASE_DIR, session_id)

    if not os.path.exists(upload_dir):
        raise FileNotFoundError(f"Session {session_id} not found")

    metadata_path = os.path.join(upload_dir, 'session_metadata.json')

    if not os.path.exists(metadata_path):
        # Return basic info if metadata doesn't exist
        return {
            'sessionId': session_id,
            'status': 'incomplete',
            'message': 'Session metadata not found'
        }

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    return metadata


def get_session_from_database(session_id: str) -> Dict:
    """
    Get session information from Supabase database.

    Args:
        session_id: Session identifier (string format like session_XXX_YYY)

    Returns:
        dict: Session data from database

    Raises:
        ValueError: If session not found
    """
    from utils.database import create_or_get_session_uuid, get_supabase_client

    # Get UUID for session
    uuid_session_id = create_or_get_session_uuid(session_id)
    if not uuid_session_id:
        raise ValueError(f'Session {session_id} not found in database')

    supabase = get_supabase_client()

    # Query session from database
    response = supabase.table('sessions').select('*').eq('id', str(uuid_session_id)).execute()

    if not response.data or len(response.data) == 0:
        raise ValueError(f'Session {session_id} not found in database')

    return response.data[0]


def delete_session(session_id: str) -> Dict:
    """
    Delete a session and all associated data.

    Args:
        session_id: Session identifier

    Returns:
        dict: {
            'deleted_files': int,
            'deleted_db_records': int,
            'message': str
        }

    Raises:
        ValueError: If session not found
    """
    from utils.database import create_or_get_session_uuid, get_supabase_client
    import shutil

    # Get UUID for session
    uuid_session_id = create_or_get_session_uuid(session_id)
    if not uuid_session_id:
        raise ValueError(f'Session {session_id} not found')

    supabase = get_supabase_client()
    deleted_files = 0
    deleted_db_records = 0

    # 1. Delete local files
    upload_dir = os.path.join(UPLOAD_BASE_DIR, session_id)
    if os.path.exists(upload_dir):
        shutil.rmtree(upload_dir)
        deleted_files = 1
        logger.info(f"Deleted local directory for session {session_id}")

    # 2. Delete from database tables
    try:
        # Delete from training_results
        supabase.table('training_results').delete().eq('session_id', str(uuid_session_id)).execute()
        deleted_db_records += 1

        # Delete from files
        supabase.table('files').delete().eq('session_id', str(uuid_session_id)).execute()
        deleted_db_records += 1

        # Delete from time_info
        supabase.table('time_info').delete().eq('session_id', str(uuid_session_id)).execute()
        deleted_db_records += 1

        # Delete from zeitschritte
        supabase.table('zeitschritte').delete().eq('session_id', str(uuid_session_id)).execute()
        deleted_db_records += 1

        # Delete from sessions
        supabase.table('sessions').delete().eq('id', str(uuid_session_id)).execute()
        deleted_db_records += 1

        # Delete from session_mappings
        supabase.table('session_mappings').delete().eq('uuid_session_id', str(uuid_session_id)).execute()
        deleted_db_records += 1

        logger.info(f"Deleted database records for session {session_id}")
    except Exception as e:
        logger.error(f"Error deleting database records: {str(e)}")
        # Continue even if some deletions fail

    return {
        'deleted_files': deleted_files,
        'deleted_db_records': deleted_db_records,
        'message': f"Session {session_id} deleted successfully"
    }


def delete_all_sessions(confirm: bool = False) -> Dict:
    """
    Delete all sessions from database, storage, and local storage.

    Args:
        confirm: Must be True to proceed with deletion

    Returns:
        dict: {
            'summary': {'database_records_deleted', 'storage_files_deleted', 'local_directories_deleted', 'local_files_deleted'},
            'details': {'initial_counts', 'deleted_counts', 'storage_deleted', 'local_deleted'},
            'warnings': [] (optional),
            'message': str
        }

    Raises:
        ValueError: If confirmation is not provided
    """
    from utils.database import get_supabase_client
    import shutil

    if not confirm:
        raise ValueError('Confirmation required. To delete all sessions, pass confirm=True')

    logger.warning("🚨 DELETE ALL SESSIONS operation initiated!")

    supabase = get_supabase_client()

    # Count current data before deletion
    initial_counts = {}
    database_errors = []

    try:
        tables_to_check = [
            'training_results',
            'training_visualizations',
            'files',
            'zeitschritte',
            'time_info',
            'session_mappings',
            'sessions'
        ]

        for table in tables_to_check:
            try:
                count_response = supabase.table(table).select('id', count='exact').execute()
                initial_counts[table] = count_response.count
                logger.info(f"Found {count_response.count} records in {table}")
            except Exception as e:
                logger.warning(f"Could not count records in {table}: {str(e)}")
                initial_counts[table] = 'unknown'

    except Exception as e:
        logger.error(f"Error getting initial counts: {str(e)}")

    # 1. Delete all CSV files from Supabase Storage
    storage_deleted = {'csv-files': 0, 'aus-csv-files': 0}

    try:
        files_response = supabase.table('files').select('storage_path, type').execute()

        for file_data in files_response.data:
            storage_path = file_data.get('storage_path')
            file_type = file_data.get('type', 'input')

            if storage_path:
                try:
                    bucket_name = 'aus-csv-files' if file_type == 'output' else 'csv-files'
                    supabase.storage.from_(bucket_name).remove([storage_path])
                    storage_deleted[bucket_name] += 1
                    logger.info(f"Deleted from storage: {bucket_name}/{storage_path}")
                except Exception as storage_error:
                    logger.warning(f"Could not delete from storage {storage_path}: {str(storage_error)}")
                    database_errors.append(f"Storage: {storage_path} - {str(storage_error)}")

    except Exception as e:
        logger.error(f"Error deleting from storage: {str(e)}")
        database_errors.append(f"Storage deletion error: {str(e)}")

    # 2. Delete from database tables (order matters due to foreign keys)
    tables_to_delete = [
        'training_results',
        'training_visualizations',
        'evaluation_tables',
        'files',
        'zeitschritte',
        'time_info',
        'session_mappings',
        'sessions'
    ]

    deleted_counts = {}

    for table in tables_to_delete:
        try:
            response = supabase.table(table).delete().neq('id', '00000000-0000-0000-0000-000000000000').execute()
            deleted_count = len(response.data) if response.data else 0
            deleted_counts[table] = deleted_count

            if deleted_count > 0:
                logger.info(f"✅ Deleted {deleted_count} records from {table}")
            else:
                logger.info(f"ℹ️  No records to delete from {table}")

        except Exception as table_error:
            logger.error(f"Error deleting from {table}: {str(table_error)}")
            database_errors.append(f"Table {table}: {str(table_error)}")
            deleted_counts[table] = 'error'

    # 3. Delete all local session directories
    local_deleted = {'directories': 0, 'files': 0}
    local_errors = []

    try:
        if os.path.exists(UPLOAD_BASE_DIR):
            for item in os.listdir(UPLOAD_BASE_DIR):
                item_path = os.path.join(UPLOAD_BASE_DIR, item)

                # Only delete directories that look like session directories
                if os.path.isdir(item_path) and item.startswith('session_'):
                    try:
                        # Count files before deletion
                        file_count = sum([len(files) for r, d, files in os.walk(item_path)])
                        local_deleted['files'] += file_count

                        # Delete directory and all contents
                        shutil.rmtree(item_path)
                        local_deleted['directories'] += 1
                        logger.info(f"🗂️  Deleted local directory: {item_path} ({file_count} files)")

                    except Exception as dir_error:
                        logger.error(f"Error deleting directory {item_path}: {str(dir_error)}")
                        local_errors.append(f"Directory {item}: {str(dir_error)}")
        else:
            logger.info("📁 Local upload directory does not exist")

    except Exception as e:
        logger.error(f"Error during local cleanup: {str(e)}")
        local_errors.append(f"Local cleanup error: {str(e)}")

    # Prepare result
    all_errors = database_errors + local_errors
    total_database_records = sum([count for count in deleted_counts.values() if isinstance(count, int)])
    total_storage_files = sum(storage_deleted.values())

    result = {
        'summary': {
            'database_records_deleted': total_database_records,
            'storage_files_deleted': total_storage_files,
            'local_directories_deleted': local_deleted['directories'],
            'local_files_deleted': local_deleted['files']
        },
        'details': {
            'initial_counts': initial_counts,
            'deleted_counts': deleted_counts,
            'storage_deleted': storage_deleted,
            'local_deleted': local_deleted
        },
        'message': 'All sessions deletion completed'
    }

    if all_errors:
        result['warnings'] = all_errors
        result['message'] += ' (with some warnings)'
        logger.warning(f"Completed with {len(all_errors)} warnings")
    else:
        logger.info("✅ All sessions deleted successfully without errors")

    return result


def update_session_name(session_id: str, new_name: str) -> Dict:
    """
    Update session name in the database.

    Args:
        session_id: Session identifier
        new_name: New session name

    Returns:
        dict: {
            'session_id': str,
            'session_name': str,
            'message': str
        }

    Raises:
        ValueError: If validation fails or session not found
    """
    from utils.database import update_session_name as db_update_session_name
    from utils.database import ValidationError, SessionNotFoundError

    # Validate inputs
    if not session_id:
        raise ValueError('sessionId is required')

    if not new_name:
        raise ValueError('sessionName is required')

    if not isinstance(new_name, str):
        raise ValueError('sessionName must be a string')

    # Trim and validate session name
    new_name = new_name.strip()
    if len(new_name) == 0:
        raise ValueError('sessionName cannot be empty')

    if len(new_name) > 255:
        raise ValueError('sessionName too long (max 255 characters)')

    # Update in database
    try:
        result = db_update_session_name(session_id, new_name)
        return {
            'session_id': session_id,
            'session_name': new_name,
            'message': 'Session name updated successfully'
        }
    except (ValidationError, SessionNotFoundError) as e:
        raise ValueError(str(e))


def save_time_info_data(session_id: str, time_info: Dict) -> bool:
    """
    Save time information for a session.

    Args:
        session_id: Session identifier
        time_info: Time information dictionary

    Returns:
        bool: True if successful

    Raises:
        ValueError: If validation fails
    """
    from utils.database import save_time_info

    # Validate session_id format
    if not session_id or not isinstance(session_id, str):
        raise ValueError('Invalid session_id format')

    success = save_time_info(session_id, time_info)

    if not success:
        raise Exception('Failed to save time info')

    return True


def save_zeitschritte_data(session_id: str, zeitschritte: Dict) -> bool:
    """
    Save zeitschritte for a session.

    Args:
        session_id: Session identifier
        zeitschritte: Zeitschritte dictionary

    Returns:
        bool: True if successful

    Raises:
        ValueError: If validation fails
    """
    from utils.database import save_zeitschritte

    # Validate inputs
    if not session_id or not isinstance(session_id, str):
        raise ValueError('Invalid session_id format')

    if not isinstance(zeitschritte, dict):
        raise ValueError('Invalid zeitschritte format - must be a dictionary')

    success = save_zeitschritte(session_id, zeitschritte)

    if not success:
        raise Exception('Failed to save zeitschritte')

    return True


def get_time_info_data(session_id: str) -> Dict:
    """
    Get time information for a session.

    Args:
        session_id: Session identifier

    Returns:
        dict: Time info data

    Raises:
        ValueError: If session not found
    """
    from utils.database import create_or_get_session_uuid, get_supabase_client

    # Get UUID for session
    uuid_session_id = create_or_get_session_uuid(session_id)
    if not uuid_session_id:
        raise ValueError(f'Session {session_id} not found')

    supabase = get_supabase_client()

    # Query time_info from database
    response = supabase.table('time_info').select('*').eq('session_id', str(uuid_session_id)).execute()

    if not response.data or len(response.data) == 0:
        raise ValueError(f'No time info found for session {session_id}')

    return response.data[0]


def get_zeitschritte_data(session_id: str) -> Dict:
    """
    Get zeitschritte for a session.

    Args:
        session_id: Session identifier

    Returns:
        dict: Zeitschritte data

    Raises:
        ValueError: If session not found
    """
    from utils.database import create_or_get_session_uuid, get_supabase_client

    # Get UUID for session
    uuid_session_id = create_or_get_session_uuid(session_id)
    if not uuid_session_id:
        raise ValueError(f'Session {session_id} not found')

    supabase = get_supabase_client()

    # Query zeitschritte from database
    response = supabase.table('zeitschritte').select('*').eq('session_id', str(uuid_session_id)).execute()

    if not response.data or len(response.data) == 0:
        raise ValueError(f'No zeitschritte found for session {session_id}')

    return response.data[0]


def get_csv_files_for_session(session_id: str, file_type: str = None) -> List[Dict]:
    """
    Get list of CSV files for a session from database.

    Args:
        session_id: Session identifier
        file_type: Optional filter for file type ('input' or 'output')

    Returns:
        List of file dictionaries

    Raises:
        ValueError: If session not found
    """
    from utils.database import create_or_get_session_uuid, get_supabase_client

    # Get UUID for session
    uuid_session_id = create_or_get_session_uuid(session_id)
    if not uuid_session_id:
        raise ValueError(f'Session {session_id} not found')

    supabase = get_supabase_client()

    # Query files from database
    query = supabase.table('files').select('*').eq('session_id', str(uuid_session_id))
    
    # Apply file type filter if provided
    if file_type:
        query = query.eq('type', file_type)
    
    response = query.execute()

    return response.data if response.data else []


def get_session_status(session_id: str) -> Dict:
    """
    Get detailed status of a training session.

    Args:
        session_id: Session identifier

    Returns:
        dict: Session status information

    Raises:
        ValueError: If session not found
    """
    from utils.database import create_or_get_session_uuid, get_supabase_client

    # Get UUID for session
    uuid_session_id = create_or_get_session_uuid(session_id)
    if not uuid_session_id:
        raise ValueError(f'Session {session_id} not found')

    supabase = get_supabase_client()

    # Query session from database
    response = supabase.table('sessions').select('*').eq('id', str(uuid_session_id)).execute()

    if not response.data or len(response.data) == 0:
        raise ValueError(f'Session {session_id} not found in database')

    session_data = response.data[0]

    # Query training results to get status
    training_response = supabase.table('training_results').select('*').eq('session_id', str(uuid_session_id)).execute()

    training_status = 'not_started'
    training_data = None

    if training_response.data and len(training_response.data) > 0:
        training_data = training_response.data[0]
        if training_data.get('success'):
            training_status = 'completed'
        elif training_data.get('error'):
            training_status = 'failed'
        else:
            training_status = 'in_progress'

    return {
        'session_id': session_id,
        'uuid': str(uuid_session_id),
        'session_name': session_data.get('session_name', session_id),
        'training_status': training_status,
        'created_at': session_data.get('created_at'),
        'updated_at': session_data.get('updated_at'),
        'n_dat': session_data.get('n_dat'),
        'training_data': training_data
    }


def get_upload_status(session_id: str) -> Dict:
    """
    Get upload progress status for a session.

    Checks:
    - Session existence in database
    - Local file directory existence
    - Upload progress based on files

    Args:
        session_id: Session identifier (UUID or string ID)

    Returns:
        dict: {
            'status': 'error'|'pending'|'processing'|'completed',
            'progress': int (0-100),
            'message': str
        }

    Raises:
        ValueError: If session not found (404)
    """
    from utils.database import get_supabase_client, create_or_get_session_uuid
    import uuid as uuid_lib

    # Check if provided ID is UUID or string ID
    try:
        uuid_lib.UUID(session_id)
        # It's a UUID, get the string ID for local file access
        string_session_id = get_string_id_from_uuid(session_id)
        if not string_session_id:
            raise ValueError('Session mapping not found for UUID')
    except (ValueError, TypeError):
        # It's a string ID, use it directly
        string_session_id = session_id

    # Check session exists in database
    supabase = get_supabase_client()
    if supabase:
        session_uuid = create_or_get_session_uuid(string_session_id)
        if not session_uuid:
            raise ValueError('Session mapping not found')

        # Get session from database
        session_response = supabase.table('sessions').select('*').eq('id', session_uuid).execute()
        if not session_response.data or len(session_response.data) == 0:
            raise ValueError('Session not found in database')

        session_data = session_response.data[0]

        # Check local files exist
        upload_dir = os.path.join(UPLOAD_BASE_DIR, string_session_id)
        if not os.path.exists(upload_dir):
            # Session exists in DB but no local files yet
            return {
                'status': 'pending',
                'progress': 0,
                'message': 'Session exists but no files uploaded yet',
                'finalized': session_data.get('finalized', False)
            }
    else:
        # No Supabase client, check local only
        upload_dir = os.path.join(UPLOAD_BASE_DIR, string_session_id)
        if not os.path.exists(upload_dir):
            raise ValueError('Session not found')

    # Directory exists, check session metadata
    upload_dir = os.path.join(UPLOAD_BASE_DIR, string_session_id)
    session_metadata_path = os.path.join(upload_dir, 'session_metadata.json')

    if not os.path.exists(session_metadata_path):
        return {
            'status': 'pending',
            'progress': 10,
            'message': 'Session initialized but no metadata found'
        }

    with open(session_metadata_path, 'r') as f:
        session_metadata = json.load(f)

    # Check if finalized
    if session_metadata.get('finalized', False):
        return {
            'status': 'completed',
            'progress': 100,
            'message': 'Session completed successfully'
        }

    # Check upload progress
    metadata_path = os.path.join(upload_dir, 'metadata.json')
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            chunks_metadata = json.load(f)

        # Calculate progress based on uploaded files
        total_files = session_metadata.get('sessionInfo', {}).get('totalFiles', 0)
        if total_files > 0:
            # Count unique files
            unique_files = {chunk.get('fileName') for chunk in chunks_metadata if chunk.get('fileName')}
            progress = (len(unique_files) / total_files) * 90  # 90% for upload, 10% for finalization

            return {
                'status': 'processing',
                'progress': int(progress),
                'message': f'Uploading files: {len(unique_files)}/{total_files}'
            }

    # No metadata.json, session just initialized
    return {
        'status': 'pending',
        'progress': 5,
        'message': 'Session initialized, waiting for files'
    }


def create_database_session(session_id: str, session_name: str = None) -> str:
    """
    Create a new session in Supabase database and return UUID.

    Args:
        session_id: String session identifier
        session_name: Optional session name

    Returns:
        str: UUID session identifier

    Raises:
        ValueError: If session creation fails
    """
    from utils.database import create_or_get_session_uuid

    if not session_id:
        raise ValueError('Missing session ID')

    uuid_session_id = create_or_get_session_uuid(session_id)

    if not uuid_session_id:
        raise ValueError('Failed to create session in database')

    # Optionally update session name
    if session_name:
        try:
            update_session_name(session_id, session_name)
        except Exception as e:
            logger.warning(f"Failed to update session name: {str(e)}")

    return str(uuid_session_id)


def get_session_uuid(session_id: str) -> str:
    """
    Get UUID for a session identifier.

    Args:
        session_id: String session identifier

    Returns:
        str: UUID session identifier

    Raises:
        ValueError: If session not found
    """
    from utils.database import create_or_get_session_uuid

    uuid_session_id = create_or_get_session_uuid(session_id)

    if not uuid_session_id:
        raise ValueError(f'Session {session_id} not found')

    return str(uuid_session_id)
