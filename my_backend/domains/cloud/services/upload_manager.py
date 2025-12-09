"""
Cloud Upload Manager
Thread-safe upload session manager with TTL cleanup
"""
import os
import shutil
import logging
from datetime import datetime, timedelta
from collections import OrderedDict
from threading import Lock

from domains.cloud.config import CHUNK_DIR, MAX_ACTIVE_UPLOADS

logger = logging.getLogger(__name__)


class UploadManager:
    """Thread-safe upload session manager with TTL cleanup."""

    def __init__(self, max_size=MAX_ACTIVE_UPLOADS, ttl_hours=1):
        self.uploads = OrderedDict()
        self.max_size = max_size
        self.ttl = timedelta(hours=ttl_hours)
        self.lock = Lock()

    def __getitem__(self, upload_id: str):
        """Dictionary-style access for backward compatibility."""
        return self.get(upload_id)

    def __setitem__(self, upload_id: str, data: dict):
        """Dictionary-style assignment for backward compatibility."""
        self.add(upload_id, data)

    def __contains__(self, upload_id: str):
        """Check if upload_id exists."""
        return self.contains(upload_id)

    def __delitem__(self, upload_id: str):
        """Dictionary-style deletion for backward compatibility."""
        self.remove(upload_id)

    def add(self, upload_id: str, data: dict):
        """Add or update upload session with timestamp."""
        with self.lock:
            self.cleanup_expired()

            if len(self.uploads) >= self.max_size and upload_id not in self.uploads:
                oldest_id, oldest_data = self.uploads.popitem(last=False)
                logger.warning(f"Upload capacity reached. Removed oldest upload: {oldest_id}")
                try:
                    chunk_dir = os.path.join(CHUNK_DIR, oldest_id)
                    if os.path.exists(chunk_dir):
                        shutil.rmtree(chunk_dir)
                except Exception as e:
                    logger.error(f"Error cleaning up removed upload {oldest_id}: {e}")

            self.uploads[upload_id] = {
                'data': data,
                'created_at': datetime.now()
            }
            self.uploads.move_to_end(upload_id)

    def get(self, upload_id: str) -> dict:
        """Get upload session data."""
        with self.lock:
            self.cleanup_expired()
            if upload_id in self.uploads:
                self.uploads.move_to_end(upload_id)
                return self.uploads[upload_id]['data']
            return None

    def remove(self, upload_id: str):
        """Remove upload session."""
        with self.lock:
            if upload_id in self.uploads:
                del self.uploads[upload_id]

    def clear(self):
        """Clear all upload sessions."""
        with self.lock:
            self.uploads.clear()

    def contains(self, upload_id: str) -> bool:
        """Check if upload session exists."""
        with self.lock:
            self.cleanup_expired()
            return upload_id in self.uploads

    def cleanup_expired(self):
        """Remove expired upload sessions."""
        now = datetime.now()
        expired = [
            uid for uid, data in self.uploads.items()
            if now - data['created_at'] > self.ttl
        ]
        for uid in expired:
            logger.info(f"Removing expired upload session: {uid}")
            del self.uploads[uid]
            try:
                chunk_dir = os.path.join(CHUNK_DIR, uid)
                if os.path.exists(chunk_dir):
                    shutil.rmtree(chunk_dir)
            except Exception as e:
                logger.error(f"Error cleaning up expired upload {uid}: {e}")


# Global upload manager instance
upload_manager = UploadManager()
chunk_uploads = upload_manager  # Alias for backward compatibility
