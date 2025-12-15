"""
Session lifecycle management functions.

This module handles the complete session lifecycle including loading metadata,
saving to database, finalizing sessions, and managing session names.
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List

from .config import DomainDefaults, TableNames
from .exceptions import (
    DatabaseError,
    SessionNotFoundError,
    ValidationError,
    ConfigurationError
)
from .validators import validate_session_id
from .session import get_supabase_client, get_session_uuid
from .persistence import save_time_info, save_zeitschritte
from .batch import prepare_file_batch_data, batch_upsert_files
from .storage import save_csv_file_content


logger = logging.getLogger(__name__)


def load_session_metadata(session_id: str) -> Optional[Dict[str, Any]]:
    """Load session metadata from filesystem.

    Args:
        session_id: Session ID for directory structure

    Returns:
        dict: Session metadata or None if loading fails

    Raises:
        FileNotFoundError: If session directory or metadata file not found
        DatabaseError: If JSON parsing fails
    """
    upload_base_dir = 'uploads/file_uploads'
    session_dir = os.path.join(upload_base_dir, session_id)

    if not os.path.exists(session_dir):
        logger.error(f"Session directory not found: {session_dir}")
        return None

    metadata_path = os.path.join(session_dir, 'session_metadata.json')

    if not os.path.exists(metadata_path):
        logger.error(f"Session metadata file not found: {metadata_path}")
        return None

    try:
        with open(metadata_path, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in session metadata: {str(e)}")
        raise DatabaseError(f"Invalid session metadata format: {str(e)}")
    except Exception as e:
        logger.error(f"Error loading session metadata: {str(e)}")
        return None


def save_metadata_to_database(database_session_id: str, metadata: Dict[str, Any]) -> bool:
    """Save time info and zeitschritte metadata to database.

    Args:
        database_session_id: UUID format session ID
        metadata: Session metadata dictionary

    Returns:
        bool: True if all saves successful

    Raises:
        DatabaseError: If database operations fail
        ValidationError: If metadata validation fails
    """
    if 'timeInfo' in metadata:
        save_time_info(database_session_id, metadata['timeInfo'])

    if 'zeitschritte' in metadata:
        save_zeitschritte(database_session_id, metadata['zeitschritte'])

    return True


def save_files_to_database(
    database_session_id: str,
    session_id: str,
    metadata: dict
) -> bool:
    """Save file info and content to database and storage using batch operations.

    Args:
        database_session_id: UUID format session ID
        session_id: Original session ID for directory structure
        metadata: Session metadata dictionary

    Returns:
        bool: True if successful

    Raises:
        DatabaseError: If database operations fail
        ConfigurationError: If Supabase client not available
    """
    if 'files' not in metadata or not isinstance(metadata['files'], list):
        return True

    supabase = get_supabase_client()

    upload_base_dir = 'uploads/file_uploads'
    session_dir = os.path.join(upload_base_dir, session_id)

    # Prepare batch data
    batch_data = prepare_file_batch_data(database_session_id, metadata['files'])
    if not batch_data:
        logger.warning("No valid file data to upsert")
        return True

    # Batch upsert files
    upserted_uuids = batch_upsert_files(supabase, database_session_id, batch_data)
    if not upserted_uuids:
        raise DatabaseError("Batch file upsert failed - no files were processed")

    # Build UUID mapping for storage uploads
    uuid_map = {}
    for data, uuid_val in zip(batch_data, upserted_uuids):
        file_name = data.get('file_name', '')
        if file_name:
            uuid_map[file_name] = {
                'uuid': uuid_val,
                'type': data.get('type', DomainDefaults.FILE_TYPE)
            }

    # Upload files to storage
    upload_failures = []
    for file_info in metadata['files']:
        file_name = file_info.get('fileName', '')
        if file_name not in uuid_map:
            logger.warning(f"File {file_name} not found in UUID mapping, skipping upload")
            continue

        file_uuid = uuid_map[file_name]['uuid']
        file_type = uuid_map[file_name]['type']
        file_path = os.path.join(session_dir, file_name)

        if os.path.exists(file_path):
            try:
                save_csv_file_content(file_uuid, database_session_id, file_name, file_path, file_type)
            except Exception as e:
                logger.error(f"Error uploading {file_name}: {str(e)}")
                upload_failures.append(file_name)
        else:
            logger.warning(f"CSV file not found locally: {file_path}")
            upload_failures.append(file_name)

    if upload_failures:
        logger.warning(f"Some file uploads failed: {upload_failures}, but metadata was saved")

    return True


def finalize_session(
    database_session_id: str,
    n_dat: Optional[int] = None,
    file_count: Optional[int] = None
) -> bool:
    """Update sessions table with finalization data.

    Args:
        database_session_id: UUID format session ID
        n_dat: Total number of data samples (optional)
        file_count: Number of files in the session (optional)

    Returns:
        bool: True if successful

    Raises:
        DatabaseError: If database operation fails
        ConfigurationError: If Supabase client not available
    """
    if n_dat is None and file_count is None:
        return True

    session_update_data = {
        "finalized": True,
        "updated_at": datetime.now().isoformat()
    }
    if n_dat is not None:
        session_update_data["n_dat"] = n_dat
    if file_count is not None:
        session_update_data["file_count"] = file_count

    supabase = get_supabase_client()

    try:
        session_response = supabase.table(TableNames.SESSIONS)\
            .update(session_update_data)\
            .eq("id", database_session_id)\
            .execute()

        if hasattr(session_response, 'error') and session_response.error:
            raise DatabaseError(f"Error updating sessions table: {session_response.error}")

        return True

    except DatabaseError:
        raise
    except Exception as e:
        raise DatabaseError(f"Error updating sessions table: {str(e)}")


def update_session_name(session_id: str, session_name: str, user_id: str = None) -> bool:
    """Update session name in the sessions table.

    Args:
        session_id: ID of the session (can be string or UUID)
        session_name: New name for the session
        user_id: User ID to validate session ownership (required for security)

    Returns:
        bool: True if successful

    Raises:
        ValidationError: If input parameters are invalid
        ConfigurationError: If Supabase client is not available
        SessionNotFoundError: If session cannot be found
        PermissionError: If session doesn't belong to the user
        DatabaseError: If database operations fail
    """
    if not validate_session_id(session_id):
        raise ValidationError(f"Invalid session_id format: {session_id}")

    if not session_name or not isinstance(session_name, str):
        raise ValidationError(f"Invalid session_name: {session_name}")

    session_name = session_name.strip()
    if len(session_name) == 0:
        raise ValidationError("Session name cannot be empty")

    if len(session_name) > 255:
        raise ValidationError("Session name too long (max 255 characters)")

    supabase = get_supabase_client()
    database_session_id = get_session_uuid(session_id)

    try:
        # Validate session ownership if user_id provided
        session_query = supabase.table(TableNames.SESSIONS)\
            .select("id, user_id")\
            .eq("id", database_session_id)
        existing = session_query.execute()

        if not existing.data or len(existing.data) == 0:
            raise SessionNotFoundError(f"Session {database_session_id} not found")

        # Verify session belongs to user (if user_id provided)
        if user_id and existing.data[0].get('user_id') != user_id:
            logger.warning(
                f"User {user_id} attempted to rename session {database_session_id} "
                f"owned by {existing.data[0].get('user_id')}"
            )
            raise PermissionError(f"Session {session_id} does not belong to user")

        response = supabase.table(TableNames.SESSIONS).update({
            "session_name": session_name,
            "updated_at": datetime.now().isoformat()
        }).eq("id", database_session_id).execute()

        if hasattr(response, 'error') and response.error:
            raise DatabaseError(f"Error updating session name: {response.error}")

        return True

    except (DatabaseError, SessionNotFoundError, ValidationError, PermissionError):
        raise
    except Exception as e:
        raise DatabaseError(f"Error updating session name: {str(e)}")


def save_session_to_supabase(
    session_id: str,
    n_dat: int = None,
    file_count: int = None
) -> bool:
    """Save all session data to Supabase.

    This is the main orchestration function that:
    1. Converts session ID to UUID
    2. Loads metadata from filesystem
    3. Saves time_info and zeitschritte
    4. Batch upserts file metadata
    5. Uploads CSV files to storage
    6. Finalizes the session

    Args:
        session_id: ID of the session (string format)
        n_dat: Total number of data samples (optional)
        file_count: Number of files in the session (optional)

    Returns:
        bool: True if successful

    Raises:
        SessionNotFoundError: If session cannot be found
        DatabaseError: If database operations fail
        ConfigurationError: If Supabase client not available
    """
    database_session_id = get_session_uuid(session_id)

    metadata = load_session_metadata(session_id)
    if not metadata:
        raise SessionNotFoundError(f"Could not load metadata for session {session_id}")

    # Save metadata (time_info, zeitschritte)
    try:
        save_metadata_to_database(database_session_id, metadata)
    except Exception as e:
        logger.warning(f"Some metadata failed to save: {str(e)}, continuing with files...")

    # Save files (batch upsert + storage upload)
    try:
        save_files_to_database(database_session_id, session_id, metadata)
    except Exception as e:
        logger.warning(f"Some files failed to save: {str(e)}, continuing with finalization...")

    # Finalize session
    try:
        finalize_session(database_session_id, n_dat, file_count)
    except Exception as e:
        logger.warning(f"Session finalization failed: {str(e)}, but core data was saved")

    return True


# Aliases for backward compatibility
_load_session_metadata = load_session_metadata
_save_metadata_to_database = save_metadata_to_database
_save_files_to_database = save_files_to_database
_finalize_session = finalize_session
