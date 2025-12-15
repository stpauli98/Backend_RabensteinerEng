"""
File storage operations for Supabase Storage.

This module handles uploading and managing CSV files in Supabase Storage buckets.
"""

import os
import logging
import uuid
from typing import Optional, List, Dict, Any

from .config import BucketNames, TableNames
from .exceptions import StorageError, ConfigurationError
from .validators import sanitize_filename
from .session import get_supabase_client


logger = logging.getLogger(__name__)


def save_csv_file_content(
    file_id: str,
    session_id: str,
    file_name: str,
    file_path: str,
    file_type: str
) -> bool:
    """Save CSV file content to Supabase Storage.

    Args:
        file_id: ID of the file (from files table)
        session_id: ID of the session
        file_name: Name of the file
        file_path: Path to the file on local filesystem
        file_type: Type of the file ('input' or 'output')

    Returns:
        bool: True if successful

    Raises:
        StorageError: If file upload fails
        ConfigurationError: If Supabase client is not available
        FileNotFoundError: If local file does not exist
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    supabase = get_supabase_client()

    bucket_name = BucketNames.get_bucket_for_type(file_type)

    try:
        with open(file_path, 'rb') as f:
            file_content = f.read()
    except IOError as e:
        raise StorageError(f"Could not read file {file_path}: {str(e)}")

    storage_path = f"{session_id}/{sanitize_filename(file_name)}"

    try:
        # Check if file already exists in bucket
        try:
            existing_files = supabase.storage.from_(bucket_name).list(session_id)
            file_exists = any(f['name'] == file_name for f in existing_files)

            if file_exists:
                storage_response = supabase.storage.from_(bucket_name).update(
                    path=storage_path,
                    file=file_content,
                    file_options={"content-type": "text/csv"}
                )
            else:
                storage_response = supabase.storage.from_(bucket_name).upload(
                    path=storage_path,
                    file=file_content,
                    file_options={"content-type": "text/csv"}
                )
        except Exception as list_error:
            logger.warning(f"Could not check if file exists, attempting upload: {str(list_error)}")
            storage_response = supabase.storage.from_(bucket_name).upload(
                path=storage_path,
                file=file_content,
                file_options={"content-type": "text/csv"}
            )

        # Update files table with storage path
        try:
            uuid_obj = uuid.UUID(file_id)
            valid_file_id = str(uuid_obj)

            supabase.table(TableNames.FILES).update({
                "storage_path": storage_path
            }).eq("id", valid_file_id).execute()

        except (ValueError, TypeError, AttributeError) as e:
            logger.warning(f"Could not update files table - invalid file_id: {file_id}, error: {str(e)}")

        return True

    except Exception as storage_error:
        error_msg = str(storage_error).lower()
        # If file already exists, consider it a success
        if "already exists" in error_msg:
            logger.info(f"File {storage_path} already exists in bucket {bucket_name}")
            return True
        raise StorageError(f"Error uploading file to storage: {str(storage_error)}")


def delete_csv_file_content(session_id: str, file_name: str, file_type: str) -> bool:
    """Delete CSV file content from Supabase Storage.

    Args:
        session_id: ID of the session
        file_name: Name of the file
        file_type: Type of the file ('input' or 'output')

    Returns:
        bool: True if successful

    Raises:
        StorageError: If file deletion fails
        ConfigurationError: If Supabase client is not available
    """
    supabase = get_supabase_client()
    bucket_name = BucketNames.get_bucket_for_type(file_type)
    storage_path = f"{session_id}/{sanitize_filename(file_name)}"

    try:
        supabase.storage.from_(bucket_name).remove([storage_path])
        return True
    except Exception as e:
        raise StorageError(f"Error deleting file from storage: {str(e)}")


def get_csv_file_url(session_id: str, file_name: str, file_type: str, expiry: int = 3600) -> Optional[str]:
    """Get a signed URL for downloading a CSV file from Supabase Storage.

    Args:
        session_id: ID of the session
        file_name: Name of the file
        file_type: Type of the file ('input' or 'output')
        expiry: URL expiry time in seconds (default: 1 hour)

    Returns:
        str: Signed URL for the file, or None if not found

    Raises:
        StorageError: If URL generation fails
        ConfigurationError: If Supabase client is not available
    """
    supabase = get_supabase_client()
    bucket_name = BucketNames.get_bucket_for_type(file_type)
    storage_path = f"{session_id}/{sanitize_filename(file_name)}"

    try:
        response = supabase.storage.from_(bucket_name).create_signed_url(
            path=storage_path,
            expires_in=expiry
        )
        return response.get('signedURL') if response else None
    except Exception as e:
        raise StorageError(f"Error generating signed URL: {str(e)}")


def check_file_exists_by_hash(session_id: str, file_hash: str) -> Optional[Dict[str, Any]]:
    """Check if a file with the given hash already exists in the session.

    This is used for deduplication - to avoid uploading the same file twice.

    Args:
        session_id: UUID of the session
        file_hash: SHA-256 hash of the file content

    Returns:
        dict: File info (id, storage_path) if exists, None otherwise
    """
    if not file_hash:
        return None

    supabase = get_supabase_client()

    try:
        response = supabase.table(TableNames.FILES)\
            .select("id, storage_path, file_name, bezeichnung")\
            .eq("session_id", session_id)\
            .eq("file_hash", file_hash)\
            .limit(1)\
            .execute()

        if response.data and len(response.data) > 0:
            return response.data[0]
        return None

    except Exception as e:
        logger.warning(f"Error checking file hash: {str(e)}")
        return None


def cleanup_orphan_files(session_id: str, valid_file_ids: List[str]) -> int:
    """Delete files from session that are no longer in the valid list.

    This is used during finalization to clean up any orphan files
    that were uploaded but later removed from the session.

    Args:
        session_id: UUID of the session
        valid_file_ids: List of file IDs that should remain

    Returns:
        int: Number of deleted files
    """
    supabase = get_supabase_client()

    try:
        # Get all files for this session
        all_files_response = supabase.table(TableNames.FILES)\
            .select("id, storage_path, type")\
            .eq("session_id", session_id)\
            .execute()

        if not all_files_response.data:
            return 0

        deleted_count = 0
        valid_ids_set = set(valid_file_ids)

        for file_record in all_files_response.data:
            file_id = file_record.get('id')
            if file_id not in valid_ids_set:
                # Delete from Storage
                storage_path = file_record.get('storage_path')
                file_type = file_record.get('type', 'input')

                if storage_path:
                    try:
                        bucket_name = BucketNames.get_bucket_for_type(file_type)
                        supabase.storage.from_(bucket_name).remove([storage_path])
                    except Exception as storage_err:
                        logger.warning(f"Could not delete file from storage: {storage_err}")

                # Delete from database
                try:
                    supabase.table(TableNames.FILES)\
                        .delete()\
                        .eq("id", file_id)\
                        .execute()
                    deleted_count += 1
                except Exception as db_err:
                    logger.error(f"Could not delete file record {file_id}: {db_err}")

        logger.info(f"Cleaned up {deleted_count} orphan files from session {session_id}")
        return deleted_count

    except Exception as e:
        logger.error(f"Error during orphan cleanup: {str(e)}")
        return 0
