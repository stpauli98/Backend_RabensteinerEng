"""Supabase Storage service for temporary chunk storage.

This module provides temporary chunk storage using Supabase Storage,
enabling multi-instance Cloud Run deployments where chunks from the same
upload can arrive at different instances.

Chunks are automatically cleaned up:
1. Immediately after successful processing (explicit delete)
2. Via scheduled cleanup job for abandoned/failed uploads (1 hour expiry)

Supports metadata storage for uploads that need to persist parameters
(e.g., load_data.py which stores delimiter, timezone, selected_columns, etc.)

Usage:
    from shared.storage.chunk_service import chunk_storage_service

    # Upload a chunk
    chunk_storage_service.upload_chunk(upload_id, chunk_index, chunk_bytes)

    # For uploads with metadata (load_data.py):
    chunk_storage_service.save_upload_metadata(upload_id, total_chunks, parameters)
    metadata = chunk_storage_service.get_upload_metadata(upload_id)

    # Check if all chunks received
    chunks = chunk_storage_service.list_chunks(upload_id)
    if len(chunks) == total_chunks:
        content = chunk_storage_service.download_all_chunks(upload_id, total_chunks)
        # Process content...
        chunk_storage_service.delete_upload_chunks(upload_id)
"""

import json
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any
from shared.database.client import get_supabase_admin_client

logger = logging.getLogger(__name__)

# Bucket name for temporary chunk storage
CHUNK_BUCKET_NAME = "temp-chunks"

# Chunk expiry time (1 hour in seconds)
CHUNK_EXPIRY_SECONDS = 3600


class ChunkStorageService:
    """Service for managing temporary chunk storage in Supabase Storage."""

    def __init__(self):
        """Initialize chunk storage service."""
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
        Ensure the temp-chunks bucket exists, create if not.

        Returns:
            True if bucket exists or was created successfully
        """
        if self._bucket_verified:
            return True

        try:
            # Try to create bucket - if it already exists, that's fine
            try:
                self.client.storage.create_bucket(
                    CHUNK_BUCKET_NAME,
                    options={
                        "public": False,
                        "file_size_limit": 52428800,  # 50MB per chunk
                        "allowed_mime_types": ["application/octet-stream"]
                    }
                )
                logger.info(f"Created chunk storage bucket: {CHUNK_BUCKET_NAME}")
            except Exception as create_error:
                # Bucket already exists - this is expected
                if 'Duplicate' in str(create_error) or 'already exists' in str(create_error).lower():
                    logger.debug(f"Chunk bucket already exists: {CHUNK_BUCKET_NAME}")
                else:
                    raise create_error

            self._bucket_verified = True
            return True

        except Exception as e:
            logger.error(f"Failed to ensure chunk bucket exists: {e}")
            return False

    def upload_chunk(self, upload_id: str, chunk_index: int, data: bytes) -> bool:
        """
        Upload a single chunk to Supabase Storage.

        Args:
            upload_id: Unique upload identifier
            chunk_index: Zero-based chunk index
            data: Chunk data as bytes

        Returns:
            True if upload successful, False otherwise
        """
        try:
            if not self._ensure_bucket_exists():
                logger.error("Chunk bucket not available for upload")
                return False

            # Storage path: upload_id/chunk_XXXX.part
            storage_path = f"{upload_id}/chunk_{chunk_index:04d}.part"

            self.client.storage.from_(CHUNK_BUCKET_NAME).upload(
                path=storage_path,
                file=data,
                file_options={
                    "content-type": "application/octet-stream",
                    "upsert": "true"  # Allow re-upload for retry scenarios
                }
            )

            logger.debug(f"Uploaded chunk: {storage_path} ({len(data)} bytes)")
            return True

        except Exception as e:
            logger.error(f"Failed to upload chunk {chunk_index} for {upload_id}: {e}")
            return False

    def list_chunks(self, upload_id: str) -> List[str]:
        """
        List all chunks for an upload.

        Args:
            upload_id: Unique upload identifier

        Returns:
            List of chunk filenames (e.g., ['chunk_0000.part', 'chunk_0001.part'])
        """
        try:
            if not self._ensure_bucket_exists():
                return []

            files = self.client.storage.from_(CHUNK_BUCKET_NAME).list(upload_id)

            if not files:
                return []

            # Filter for .part files only (exclude any metadata files)
            chunk_files = [f['name'] for f in files if f['name'].endswith('.part')]
            return sorted(chunk_files)

        except Exception as e:
            logger.error(f"Failed to list chunks for {upload_id}: {e}")
            return []

    def download_chunk(self, upload_id: str, chunk_index: int) -> Optional[bytes]:
        """
        Download a single chunk.

        Args:
            upload_id: Unique upload identifier
            chunk_index: Zero-based chunk index

        Returns:
            Chunk data as bytes if successful, None otherwise
        """
        try:
            storage_path = f"{upload_id}/chunk_{chunk_index:04d}.part"
            response = self.client.storage.from_(CHUNK_BUCKET_NAME).download(storage_path)
            return response

        except Exception as e:
            logger.error(f"Failed to download chunk {chunk_index} for {upload_id}: {e}")
            return None

    def download_all_chunks(self, upload_id: str, total_chunks: int) -> Optional[bytes]:
        """
        Download and combine all chunks into a single byte sequence.

        Args:
            upload_id: Unique upload identifier
            total_chunks: Expected total number of chunks

        Returns:
            Combined chunk data as bytes if successful, None otherwise
        """
        try:
            combined = bytearray()

            for i in range(total_chunks):
                chunk_data = self.download_chunk(upload_id, i)
                if chunk_data is None:
                    logger.error(f"Missing chunk {i} for {upload_id}")
                    return None
                combined.extend(chunk_data)
                logger.debug(f"Downloaded chunk {i}/{total_chunks} for {upload_id}")

            logger.debug(f"Combined {total_chunks} chunks for {upload_id}: {len(combined)} bytes total")
            return bytes(combined)

        except Exception as e:
            logger.error(f"Failed to download all chunks for {upload_id}: {e}")
            return None

    def download_all_chunks_as_string(
        self,
        upload_id: str,
        total_chunks: int,
        encoding: str = 'utf-8'
    ) -> Optional[str]:
        """
        Download and combine all chunks, returning as decoded string.

        Args:
            upload_id: Unique upload identifier
            total_chunks: Expected total number of chunks
            encoding: Text encoding (default: utf-8)

        Returns:
            Combined chunk data as string if successful, None otherwise
        """
        try:
            combined_bytes = self.download_all_chunks(upload_id, total_chunks)
            if combined_bytes is None:
                return None

            # Try primary encoding first, fallback to alternatives
            for enc in [encoding, 'latin-1', 'cp1252', 'iso-8859-1']:
                try:
                    return combined_bytes.decode(enc)
                except UnicodeDecodeError:
                    continue

            # Last resort: ignore errors
            return combined_bytes.decode(encoding, errors='ignore')

        except Exception as e:
            logger.error(f"Failed to decode chunks for {upload_id}: {e}")
            return None

    def delete_upload_chunks(self, upload_id: str) -> int:
        """
        Delete all chunks and metadata for an upload.

        Args:
            upload_id: Unique upload identifier

        Returns:
            Number of files deleted (chunks + metadata)
        """
        try:
            chunks = self.list_chunks(upload_id)

            # Build list of paths to delete (chunks + metadata)
            paths_to_delete = [f"{upload_id}/{chunk}" for chunk in chunks]

            # Also delete metadata file if exists
            paths_to_delete.append(f"{upload_id}/_metadata.json")

            if not paths_to_delete:
                return 0

            self.client.storage.from_(CHUNK_BUCKET_NAME).remove(paths_to_delete)
            logger.debug(f"Deleted {len(paths_to_delete)} files for upload {upload_id[:8]}...")

            return len(paths_to_delete)

        except Exception as e:
            logger.error(f"Failed to delete chunks for {upload_id}: {e}")
            return 0

    def cleanup_expired_uploads(self, max_age_hours: float = 1.0) -> int:
        """
        Clean up expired chunk uploads (for abandoned/failed uploads).

        This method lists all upload folders and deletes those older than max_age_hours.

        Args:
            max_age_hours: Maximum upload age in hours (default: 1 hour)

        Returns:
            Number of upload folders cleaned up
        """
        try:
            if not self._ensure_bucket_exists():
                return 0

            # List root of bucket to get upload folders
            folders = self.client.storage.from_(CHUNK_BUCKET_NAME).list()
            if not folders:
                return 0

            deleted_count = 0
            current_time = datetime.now()

            for folder_info in folders:
                try:
                    folder_name = folder_info.get('name', '')
                    if not folder_name:
                        continue

                    # Check folder modification time
                    # Supabase returns updated_at or created_at
                    updated_at = folder_info.get('updated_at') or folder_info.get('created_at')
                    if not updated_at:
                        # If no timestamp, list contents to check chunk timestamps
                        chunks = self.client.storage.from_(CHUNK_BUCKET_NAME).list(folder_name)
                        if chunks:
                            # Use first chunk's timestamp
                            chunk_updated = chunks[0].get('updated_at') or chunks[0].get('created_at')
                            if chunk_updated:
                                updated_at = chunk_updated

                    if updated_at:
                        # Parse ISO format timestamp
                        if isinstance(updated_at, str):
                            folder_time = datetime.fromisoformat(updated_at.replace('Z', '+00:00'))
                            folder_time = folder_time.replace(tzinfo=None)  # Remove timezone for comparison
                        else:
                            folder_time = updated_at

                        age_hours = (current_time - folder_time).total_seconds() / 3600

                        if age_hours > max_age_hours:
                            # Delete all chunks in this folder
                            deleted = self.delete_upload_chunks(folder_name)
                            if deleted > 0:
                                deleted_count += 1
                                logger.info(f"Cleaned up expired upload: {folder_name[:8]}... ({age_hours:.1f}h old)")

                except Exception as folder_error:
                    logger.debug(f"Error processing folder {folder_name}: {folder_error}")
                    continue

            return deleted_count

        except Exception as e:
            logger.error(f"Failed to cleanup expired uploads: {e}")
            return 0

    def get_upload_chunk_count(self, upload_id: str) -> int:
        """
        Get the number of chunks currently uploaded for an upload.

        Args:
            upload_id: Unique upload identifier

        Returns:
            Number of chunks uploaded
        """
        return len(self.list_chunks(upload_id))

    # ========================================================================
    # Metadata Operations (for load_data.py which needs to store parameters)
    # ========================================================================

    def save_upload_metadata(
        self,
        upload_id: str,
        total_chunks: int,
        parameters: Dict[str, Any]
    ) -> bool:
        """
        Save upload metadata to _metadata.json file in Supabase Storage.

        Used by load_data.py to persist upload parameters (delimiter, timezone,
        selected_columns, etc.) across multi-instance Cloud Run deployments.

        Args:
            upload_id: Unique upload identifier
            total_chunks: Total number of chunks expected
            parameters: Upload parameters dict (delimiter, timezone, etc.)

        Returns:
            True if saved successfully, False otherwise
        """
        try:
            if not self._ensure_bucket_exists():
                logger.error("Chunk bucket not available for metadata save")
                return False

            metadata = {
                "total_chunks": total_chunks,
                "parameters": parameters,
                "created_at": datetime.utcnow().isoformat()
            }

            storage_path = f"{upload_id}/_metadata.json"
            metadata_bytes = json.dumps(metadata, ensure_ascii=False).encode('utf-8')

            # Use application/octet-stream since bucket only allows this mime type
            self.client.storage.from_(CHUNK_BUCKET_NAME).upload(
                path=storage_path,
                file=metadata_bytes,
                file_options={
                    "content-type": "application/octet-stream",
                    "upsert": "true"
                }
            )

            logger.debug(f"Saved metadata for upload {upload_id[:8]}...")
            return True

        except Exception as e:
            logger.error(f"Failed to save metadata for {upload_id}: {e}")
            return False

    def get_upload_metadata(self, upload_id: str) -> Optional[Dict[str, Any]]:
        """
        Get upload metadata from _metadata.json file.

        Args:
            upload_id: Unique upload identifier

        Returns:
            Metadata dict with 'total_chunks', 'parameters', 'created_at' or None
        """
        try:
            storage_path = f"{upload_id}/_metadata.json"
            response = self.client.storage.from_(CHUNK_BUCKET_NAME).download(storage_path)

            if response:
                return json.loads(response.decode('utf-8'))
            return None

        except Exception as e:
            # Log at debug level since metadata may not exist for all uploads
            logger.debug(f"No metadata found for {upload_id}: {e}")
            return None

    def update_upload_metadata(
        self,
        upload_id: str,
        updates: Dict[str, Any]
    ) -> bool:
        """
        Update existing upload metadata with new values.

        Args:
            upload_id: Unique upload identifier
            updates: Dict of fields to update

        Returns:
            True if updated successfully, False otherwise
        """
        try:
            existing = self.get_upload_metadata(upload_id)
            if existing is None:
                logger.error(f"Cannot update metadata - not found for {upload_id}")
                return False

            # Merge updates
            existing.update(updates)

            # Re-save
            storage_path = f"{upload_id}/_metadata.json"
            metadata_bytes = json.dumps(existing, ensure_ascii=False).encode('utf-8')

            # Use application/octet-stream since bucket only allows this mime type
            self.client.storage.from_(CHUNK_BUCKET_NAME).upload(
                path=storage_path,
                file=metadata_bytes,
                file_options={
                    "content-type": "application/octet-stream",
                    "upsert": "true"
                }
            )

            logger.debug(f"Updated metadata for upload {upload_id[:8]}...")
            return True

        except Exception as e:
            logger.error(f"Failed to update metadata for {upload_id}: {e}")
            return False


# Global singleton instance
chunk_storage_service = ChunkStorageService()
