"""
Local filesystem chunk storage service.

Provides temporary chunk storage using local filesystem instead of Supabase Storage.
This is faster and more reliable for Cloud Run with Session Affinity enabled.

Chunks are automatically cleaned up:
1. Via TTL expiry (1 hour by default)
2. Immediately after successful processing (explicit delete)

Usage:
    from domains.processing.services.local_chunk_service import local_chunk_service

    # Upload a chunk
    local_chunk_service.upload_chunk(upload_id, chunk_index, chunk_bytes)

    # Check if all chunks received
    chunks = local_chunk_service.list_chunks(upload_id)
    if len(chunks) == total_chunks:
        content = local_chunk_service.download_all_chunks(upload_id, total_chunks)
        # Process content...
        local_chunk_service.delete_upload_chunks(upload_id)
"""

import os
import json
import shutil
import tempfile
import logging
from datetime import datetime, timedelta
from collections import OrderedDict
from typing import Optional, Dict, List, Any
from threading import Lock

logger = logging.getLogger(__name__)

# Chunk directory in system temp folder
CHUNK_DIR = os.path.join(tempfile.gettempdir(), 'processing_chunks')
os.makedirs(CHUNK_DIR, exist_ok=True)


class LocalChunkService:
    """Local filesystem chunk storage with TTL cleanup."""

    def __init__(self, ttl_hours: float = 1.0):
        """
        Initialize local chunk service.

        Args:
            ttl_hours: Time-to-live for uploads in hours (default: 1 hour)
        """
        self._uploads: OrderedDict[str, Dict] = OrderedDict()
        self._lock = Lock()
        self._ttl = timedelta(hours=ttl_hours)

    def get_chunk_dir(self, upload_id: str) -> str:
        """
        Get chunk directory path for an upload.

        Args:
            upload_id: Unique upload identifier

        Returns:
            Path to chunk directory
        """
        chunk_dir = os.path.join(CHUNK_DIR, upload_id)
        os.makedirs(chunk_dir, exist_ok=True)
        return chunk_dir

    def upload_chunk(self, upload_id: str, chunk_index: int, data: bytes) -> bool:
        """
        Save a chunk to local filesystem.

        Args:
            upload_id: Unique upload identifier
            chunk_index: Zero-based chunk index
            data: Chunk data as bytes

        Returns:
            True if successful, False otherwise
        """
        try:
            # Cleanup expired uploads periodically
            self._cleanup_expired()

            chunk_dir = self.get_chunk_dir(upload_id)
            chunk_path = os.path.join(chunk_dir, f"chunk_{chunk_index:04d}.part")

            with open(chunk_path, 'wb') as f:
                f.write(data)

            # Track upload in memory
            with self._lock:
                if upload_id not in self._uploads:
                    self._uploads[upload_id] = {
                        'created_at': datetime.now(),
                        'chunks': set()
                    }
                self._uploads[upload_id]['chunks'].add(chunk_index)

            logger.debug(f"Saved chunk {chunk_index} for {upload_id[:8]}... ({len(data)} bytes)")
            return True

        except Exception as e:
            logger.error(f"Failed to save chunk {chunk_index} for {upload_id}: {e}")
            return False

    def list_chunks(self, upload_id: str) -> List[str]:
        """
        List all chunks for an upload.

        Args:
            upload_id: Unique upload identifier

        Returns:
            List of chunk filenames
        """
        chunk_dir = os.path.join(CHUNK_DIR, upload_id)
        if not os.path.exists(chunk_dir):
            return []
        return sorted([f for f in os.listdir(chunk_dir) if f.endswith('.part')])

    def get_upload_chunk_count(self, upload_id: str) -> int:
        """
        Get number of chunks uploaded.

        Args:
            upload_id: Unique upload identifier

        Returns:
            Number of chunks
        """
        return len(self.list_chunks(upload_id))

    def download_all_chunks(self, upload_id: str, total_chunks: int) -> Optional[bytes]:
        """
        Combine all chunks into a single bytes object.

        Args:
            upload_id: Unique upload identifier
            total_chunks: Expected total number of chunks

        Returns:
            Combined chunk data as bytes, or None on error
        """
        try:
            chunk_dir = os.path.join(CHUNK_DIR, upload_id)
            combined = bytearray()

            for i in range(total_chunks):
                chunk_path = os.path.join(chunk_dir, f"chunk_{i:04d}.part")
                if not os.path.exists(chunk_path):
                    logger.error(f"Missing chunk {i} for {upload_id}")
                    return None
                with open(chunk_path, 'rb') as f:
                    combined.extend(f.read())

            logger.debug(f"Combined {total_chunks} chunks for {upload_id[:8]}...: {len(combined)} bytes")
            return bytes(combined)

        except Exception as e:
            logger.error(f"Failed to combine chunks for {upload_id}: {e}")
            return None

    def download_all_chunks_as_string(
        self,
        upload_id: str,
        total_chunks: int,
        encoding: str = 'utf-8'
    ) -> Optional[str]:
        """
        Combine all chunks and return as decoded string.

        Args:
            upload_id: Unique upload identifier
            total_chunks: Expected total number of chunks
            encoding: Text encoding (default: utf-8)

        Returns:
            Combined chunk data as string, or None on error
        """
        data = self.download_all_chunks(upload_id, total_chunks)
        if data is None:
            return None

        # Detect encoding from BOM (Byte Order Mark)
        detected_encoding = encoding
        if data.startswith(b'\xff\xfe'):
            detected_encoding = 'utf-16-le'
            data = data[2:]  # Remove BOM
        elif data.startswith(b'\xfe\xff'):
            detected_encoding = 'utf-16-be'
            data = data[2:]  # Remove BOM
        elif data.startswith(b'\xef\xbb\xbf'):
            detected_encoding = 'utf-8'
            data = data[3:]  # Remove UTF-8 BOM

        # Try detected encoding first
        encodings_to_try = [detected_encoding]
        if detected_encoding != 'utf-8':
            encodings_to_try.append('utf-8')
        encodings_to_try.extend(['latin-1', 'cp1252', 'iso-8859-1'])

        for enc in encodings_to_try:
            try:
                result = data.decode(enc)
                logger.debug(f"Successfully decoded with {enc}")
                return result
            except (UnicodeDecodeError, LookupError):
                continue

        # Last resort: ignore errors
        logger.warning(f"Could not decode with any encoding, using utf-8 with errors='ignore'")
        return data.decode('utf-8', errors='ignore')

    def delete_upload_chunks(self, upload_id: str) -> int:
        """
        Delete all chunks for an upload.

        Args:
            upload_id: Unique upload identifier

        Returns:
            Number of files deleted
        """
        try:
            chunk_dir = os.path.join(CHUNK_DIR, upload_id)
            if os.path.exists(chunk_dir):
                files = os.listdir(chunk_dir)
                file_count = len(files)

                # Use retry logic for race conditions where files are being added
                # while we try to delete
                for attempt in range(3):
                    try:
                        shutil.rmtree(chunk_dir)
                        break
                    except OSError as e:
                        if attempt == 2:  # Last attempt
                            # Force delete individual files then directory
                            for f in os.listdir(chunk_dir):
                                try:
                                    os.remove(os.path.join(chunk_dir, f))
                                except OSError:
                                    pass
                            try:
                                os.rmdir(chunk_dir)
                            except OSError:
                                pass
                        else:
                            import time
                            time.sleep(0.1)  # Brief wait before retry

                with self._lock:
                    if upload_id in self._uploads:
                        del self._uploads[upload_id]

                logger.debug(f"Deleted {file_count} chunks for {upload_id[:8]}...")
                return file_count
            return 0

        except Exception as e:
            logger.error(f"Failed to delete chunks for {upload_id}: {e}")
            return 0

    # === METADATA OPERATIONS (for load_data.py compatibility) ===

    def save_upload_metadata(
        self,
        upload_id: str,
        total_chunks: int,
        parameters: Dict[str, Any]
    ) -> bool:
        """
        Save upload metadata to JSON file.

        Args:
            upload_id: Unique upload identifier
            total_chunks: Total number of chunks expected
            parameters: Upload parameters dict

        Returns:
            True if saved successfully, False otherwise
        """
        try:
            chunk_dir = self.get_chunk_dir(upload_id)
            metadata_path = os.path.join(chunk_dir, '_metadata.json')

            metadata = {
                'total_chunks': total_chunks,
                'parameters': parameters,
                'created_at': datetime.now().isoformat()
            }

            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False)

            logger.debug(f"Saved metadata for {upload_id[:8]}...")
            return True

        except Exception as e:
            logger.error(f"Failed to save metadata for {upload_id}: {e}")
            return False

    def get_upload_metadata(self, upload_id: str) -> Optional[Dict[str, Any]]:
        """
        Get upload metadata from JSON file.

        Args:
            upload_id: Unique upload identifier

        Returns:
            Metadata dict or None if not found
        """
        try:
            metadata_path = os.path.join(CHUNK_DIR, upload_id, '_metadata.json')
            if not os.path.exists(metadata_path):
                return None

            with open(metadata_path, 'r', encoding='utf-8') as f:
                return json.load(f)

        except Exception as e:
            logger.debug(f"No metadata for {upload_id}: {e}")
            return None

    def _cleanup_expired(self):
        """Clean up expired uploads based on TTL."""
        now = datetime.now()

        with self._lock:
            expired = [
                uid for uid, data in self._uploads.items()
                if now - data['created_at'] > self._ttl
            ]

        for uid in expired:
            self.delete_upload_chunks(uid)
            logger.info(f"Cleaned up expired upload: {uid[:8]}...")

    def cleanup_all_expired(self) -> int:
        """
        Clean up all expired uploads (public method for scheduled cleanup).

        Returns:
            Number of uploads cleaned up
        """
        # Also scan filesystem for orphaned uploads not in memory
        count = 0
        now = datetime.now()

        try:
            if os.path.exists(CHUNK_DIR):
                for upload_id in os.listdir(CHUNK_DIR):
                    upload_dir = os.path.join(CHUNK_DIR, upload_id)
                    if os.path.isdir(upload_dir):
                        # Check directory modification time
                        mtime = datetime.fromtimestamp(os.path.getmtime(upload_dir))
                        if now - mtime > self._ttl:
                            self.delete_upload_chunks(upload_id)
                            count += 1
                            logger.info(f"Cleaned up orphaned upload: {upload_id[:8]}...")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

        return count

    # ========== PROCESSED RESULTS STORAGE ==========

    def save_processed_result(
        self,
        file_id: str,
        csv_content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Save processed CSV result to local filesystem.

        Args:
            file_id: Unique file identifier
            csv_content: CSV content as string
            metadata: Optional metadata dict

        Returns:
            True if saved successfully
        """
        try:
            result_dir = os.path.join(CHUNK_DIR, '_results', file_id)
            os.makedirs(result_dir, exist_ok=True)

            # Save CSV content
            csv_path = os.path.join(result_dir, 'result.csv')
            with open(csv_path, 'w', encoding='utf-8') as f:
                f.write(csv_content)

            # Save metadata
            if metadata:
                meta_path = os.path.join(result_dir, 'metadata.json')
                metadata['created_at'] = datetime.now().isoformat()
                with open(meta_path, 'w') as f:
                    json.dump(metadata, f)

            logger.debug(f"Saved processed result: {file_id[:8]}... ({len(csv_content)} chars)")
            return True

        except Exception as e:
            logger.error(f"Failed to save processed result {file_id}: {e}")
            return False

    def get_processed_result(self, file_id: str) -> Optional[str]:
        """
        Get processed CSV result from local filesystem.

        Args:
            file_id: Unique file identifier

        Returns:
            CSV content as string, or None if not found
        """
        try:
            csv_path = os.path.join(CHUNK_DIR, '_results', file_id, 'result.csv')
            if not os.path.exists(csv_path):
                return None

            with open(csv_path, 'r', encoding='utf-8') as f:
                return f.read()

        except Exception as e:
            logger.error(f"Failed to get processed result {file_id}: {e}")
            return None

    def delete_processed_result(self, file_id: str) -> bool:
        """
        Delete processed result from local filesystem.

        Args:
            file_id: Unique file identifier

        Returns:
            True if deleted successfully
        """
        try:
            result_dir = os.path.join(CHUNK_DIR, '_results', file_id)
            if os.path.exists(result_dir):
                shutil.rmtree(result_dir)
                logger.debug(f"Deleted processed result: {file_id[:8]}...")
                return True
            return False

        except Exception as e:
            logger.error(f"Failed to delete processed result {file_id}: {e}")
            return False

    def cleanup_old_processed_results(self, max_age_hours: int = 24) -> int:
        """
        Clean up old processed results from local filesystem.

        Args:
            max_age_hours: Maximum age in hours before cleanup (default 24h)

        Returns:
            Number of results cleaned up
        """
        count = 0
        now = datetime.now()
        max_age = timedelta(hours=max_age_hours)

        try:
            results_dir = os.path.join(CHUNK_DIR, '_results')
            if not os.path.exists(results_dir):
                return 0

            for file_id in os.listdir(results_dir):
                result_path = os.path.join(results_dir, file_id)
                if os.path.isdir(result_path):
                    # Check directory modification time
                    mtime = datetime.fromtimestamp(os.path.getmtime(result_path))
                    if now - mtime > max_age:
                        shutil.rmtree(result_path)
                        count += 1
                        logger.info(f"Cleaned up old processed result: {file_id[:16]}...")

        except Exception as e:
            logger.error(f"Error during processed results cleanup: {e}")

        return count


# Singleton instance
local_chunk_service = LocalChunkService()
