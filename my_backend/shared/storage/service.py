"""Supabase Storage service for persistent file storage on Cloud Run.

This module provides persistent file storage using Supabase Storage,
solving the stateless instance problem on Google Cloud Run where
temp files are lost between requests.

Usage:
    from shared.storage import storage_service

    # Upload a file
    file_id = storage_service.upload_csv(user_id, csv_content, original_filename)

    # Get download URL
    url = storage_service.get_download_url(file_id)

    # Download file content
    content = storage_service.download_csv(file_id)

    # Delete file
    storage_service.delete_file(file_id)
"""

import logging
import uuid
from datetime import datetime
from typing import Optional, Dict, Any
from shared.database.client import get_supabase_admin_client

logger = logging.getLogger(__name__)

# Bucket name for processed CSV files
BUCKET_NAME = "processed-files"

# File expiry time (24 hours in seconds)
FILE_EXPIRY_SECONDS = 86400


class StorageService:
    """Service for managing file storage in Supabase Storage."""

    def __init__(self):
        """Initialize storage service."""
        self._client = None
        self._bucket_verified = False

    @property
    def client(self):
        """Get Supabase client (lazy initialization)."""
        if self._client is None:
            self._client = get_supabase_admin_client()
        return self._client

    def _ensure_bucket_exists(self) -> bool:
        """
        Ensure the storage bucket exists, create if not.

        Returns:
            True if bucket exists or was created successfully
        """
        if self._bucket_verified:
            return True

        try:
            # Try to get bucket info
            buckets = self.client.storage.list_buckets()
            bucket_names = [b.name for b in buckets]

            if BUCKET_NAME not in bucket_names:
                # Create bucket with public access for downloads
                self.client.storage.create_bucket(
                    BUCKET_NAME,
                    options={
                        "public": False,  # Private bucket, use signed URLs
                        "file_size_limit": 52428800,  # 50MB limit
                        "allowed_mime_types": ["text/csv", "text/plain", "application/octet-stream"]
                    }
                )
                logger.info(f"Created storage bucket: {BUCKET_NAME}")

            self._bucket_verified = True
            return True

        except Exception as e:
            logger.error(f"Failed to ensure bucket exists: {e}")
            return False

    def upload_csv(
        self,
        user_id: str,
        csv_content: str,
        original_filename: str = "processed.csv",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Upload CSV content to Supabase Storage.

        Args:
            user_id: User ID for organizing files
            csv_content: CSV content as string
            original_filename: Original filename for reference
            metadata: Optional metadata to store with file

        Returns:
            File ID if successful, None if failed
        """
        try:
            if not self._ensure_bucket_exists():
                logger.error("Bucket not available for upload")
                return None

            # Generate unique file ID
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            unique_id = str(uuid.uuid4())[:8]
            file_id = f"{timestamp}_{unique_id}"

            # Storage path: user_id/file_id.csv
            storage_path = f"{user_id}/{file_id}.csv"

            # Convert string to bytes
            csv_bytes = csv_content.encode('utf-8')

            # Upload file
            result = self.client.storage.from_(BUCKET_NAME).upload(
                path=storage_path,
                file=csv_bytes,
                file_options={
                    "content-type": "text/csv",
                    "cache-control": "3600",
                    "upsert": "false"
                }
            )

            logger.info(f"Uploaded file to storage: {storage_path}")

            # Return composite file_id that includes user_id for retrieval
            return f"{user_id}/{file_id}"

        except Exception as e:
            logger.error(f"Failed to upload CSV to storage: {e}")
            return None

    def get_download_url(self, file_id: str, expires_in: int = 3600) -> Optional[str]:
        """
        Get a signed download URL for a file.

        Args:
            file_id: File ID from upload_csv (format: user_id/file_id)
            expires_in: URL expiry time in seconds (default: 1 hour)

        Returns:
            Signed URL if successful, None if failed
        """
        try:
            storage_path = f"{file_id}.csv"

            # Create signed URL for private bucket
            result = self.client.storage.from_(BUCKET_NAME).create_signed_url(
                path=storage_path,
                expires_in=expires_in
            )

            if result and 'signedURL' in result:
                return result['signedURL']

            logger.error(f"Failed to create signed URL: {result}")
            return None

        except Exception as e:
            logger.error(f"Failed to get download URL: {e}")
            return None

    def download_csv(self, file_id: str) -> Optional[str]:
        """
        Download CSV content from storage.

        Args:
            file_id: File ID from upload_csv (format: user_id/file_id)

        Returns:
            CSV content as string if successful, None if failed
        """
        try:
            storage_path = f"{file_id}.csv"

            # Download file
            response = self.client.storage.from_(BUCKET_NAME).download(storage_path)

            if response:
                return response.decode('utf-8')

            return None

        except Exception as e:
            logger.error(f"Failed to download CSV from storage: {e}")
            return None

    def delete_file(self, file_id: str) -> bool:
        """
        Delete a file from storage.

        Args:
            file_id: File ID from upload_csv (format: user_id/file_id)

        Returns:
            True if deleted successfully, False otherwise
        """
        try:
            storage_path = f"{file_id}.csv"

            self.client.storage.from_(BUCKET_NAME).remove([storage_path])
            logger.info(f"Deleted file from storage: {storage_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete file from storage: {e}")
            return False

    def file_exists(self, file_id: str) -> bool:
        """
        Check if a file exists in storage.

        Args:
            file_id: File ID to check

        Returns:
            True if file exists, False otherwise
        """
        try:
            storage_path = f"{file_id}.csv"

            # Try to get file info by listing
            user_id = file_id.split('/')[0] if '/' in file_id else ''
            files = self.client.storage.from_(BUCKET_NAME).list(user_id)

            file_name = storage_path.split('/')[-1]
            return any(f['name'] == file_name for f in files)

        except Exception as e:
            logger.error(f"Failed to check file existence: {e}")
            return False

    def cleanup_old_files(self, user_id: str, max_age_hours: int = 24) -> int:
        """
        Clean up old files for a user.

        Args:
            user_id: User ID to clean up files for
            max_age_hours: Maximum file age in hours

        Returns:
            Number of files deleted
        """
        try:
            files = self.client.storage.from_(BUCKET_NAME).list(user_id)

            if not files:
                return 0

            deleted_count = 0
            current_time = datetime.now()

            for file_info in files:
                # Parse file creation time from name (format: YYYYMMDD_HHMMSS_uuid.csv)
                try:
                    file_name = file_info['name']
                    date_part = file_name.split('_')[0]
                    time_part = file_name.split('_')[1]

                    file_time = datetime.strptime(f"{date_part}_{time_part}", '%Y%m%d_%H%M%S')
                    age_hours = (current_time - file_time).total_seconds() / 3600

                    if age_hours > max_age_hours:
                        storage_path = f"{user_id}/{file_name}"
                        self.client.storage.from_(BUCKET_NAME).remove([storage_path])
                        deleted_count += 1
                        logger.info(f"Cleaned up old file: {storage_path}")

                except (ValueError, IndexError):
                    # Skip files with unexpected naming format
                    continue

            return deleted_count

        except Exception as e:
            logger.error(f"Failed to cleanup old files: {e}")
            return 0


# Global singleton instance
storage_service = StorageService()
