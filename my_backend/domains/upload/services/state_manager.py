"""
Upload State Manager
Thread-safe state management for chunked uploads and temporary files
"""
import os
import time
import threading
from typing import Dict, List, Optional, Any, Tuple

from domains.upload.config import UPLOAD_EXPIRY_SECONDS


class UploadStateManager:
    """
    Thread-safe state manager for upload chunks and temporary files.

    Replaces global dict-based state with proper encapsulation and thread safety.
    Handles both chunked uploads and temporary file storage.
    """

    def __init__(self):
        """Initialize state manager with thread locks."""
        self._chunk_storage: Dict[str, Dict[str, Any]] = {}
        self._temp_files: Dict[str, Dict[str, Any]] = {}
        self._chunk_lock = threading.Lock()
        self._temp_lock = threading.Lock()

    # ========================================================================
    # Chunk Storage Operations
    # ========================================================================

    def create_upload(
        self,
        upload_id: str,
        total_chunks: int,
        parameters: Dict[str, Any]
    ) -> None:
        """
        Create new upload session.

        Args:
            upload_id: Unique upload identifier
            total_chunks: Total number of chunks expected
            parameters: Upload parameters (delimiter, timezone, etc.)
        """
        with self._chunk_lock:
            self._chunk_storage[upload_id] = {
                'chunks': {},
                'total_chunks': total_chunks,
                'received_chunks': 0,
                'last_activity': time.time(),
                'parameters': parameters
            }

    def upload_exists(self, upload_id: str) -> bool:
        """Check if upload session exists."""
        with self._chunk_lock:
            return upload_id in self._chunk_storage

    def get_upload(self, upload_id: str) -> Optional[Dict[str, Any]]:
        """
        Get upload session data.

        Args:
            upload_id: Upload identifier

        Returns:
            Upload data dict or None if not found
        """
        with self._chunk_lock:
            return self._chunk_storage.get(upload_id)

    def store_chunk(
        self,
        upload_id: str,
        chunk_index: int,
        chunk_data: bytes
    ) -> None:
        """
        Store uploaded chunk.

        Args:
            upload_id: Upload identifier
            chunk_index: Index of the chunk
            chunk_data: Chunk binary data
        """
        with self._chunk_lock:
            if upload_id in self._chunk_storage:
                self._chunk_storage[upload_id]['chunks'][chunk_index] = chunk_data
                self._chunk_storage[upload_id]['received_chunks'] += 1
                self._chunk_storage[upload_id]['last_activity'] = time.time()

    def update_total_chunks(self, upload_id: str, total_chunks: int) -> None:
        """Update total chunks count (handles retries)."""
        with self._chunk_lock:
            if upload_id in self._chunk_storage:
                current = self._chunk_storage[upload_id]['total_chunks']
                self._chunk_storage[upload_id]['total_chunks'] = max(current, total_chunks)

    def get_chunk_progress(self, upload_id: str) -> Tuple[int, int]:
        """
        Get upload progress.

        Returns:
            Tuple of (received_chunks, total_chunks). Returns (0, 0) if upload not found.
        """
        with self._chunk_lock:
            if upload_id in self._chunk_storage:
                data = self._chunk_storage[upload_id]
                return (data['received_chunks'], data['total_chunks'])
        return (0, 0)

    def is_upload_complete(self, upload_id: str) -> bool:
        """Check if all chunks have been received."""
        with self._chunk_lock:
            if upload_id in self._chunk_storage:
                data = self._chunk_storage[upload_id]
                return data['received_chunks'] == data['total_chunks']
        return False

    def get_chunks(self, upload_id: str) -> Optional[Dict[int, bytes]]:
        """
        Get chunks dictionary.

        Returns:
            Dictionary mapping chunk indices to chunk data, or None if upload not found
        """
        with self._chunk_lock:
            if upload_id not in self._chunk_storage:
                return None

            return self._chunk_storage[upload_id]['chunks']

    def get_parameters(self, upload_id: str) -> Optional[Dict[str, Any]]:
        """Get upload parameters."""
        with self._chunk_lock:
            if upload_id in self._chunk_storage:
                return self._chunk_storage[upload_id]['parameters']
        return None

    def delete_upload(self, upload_id: str) -> bool:
        """
        Delete upload session.

        Returns:
            True if deleted, False if not found
        """
        with self._chunk_lock:
            if upload_id in self._chunk_storage:
                del self._chunk_storage[upload_id]
                return True
        return False

    def cleanup_expired_uploads(self, expiry_seconds: int = UPLOAD_EXPIRY_SECONDS) -> int:
        """
        Clean up expired uploads.

        Args:
            expiry_seconds: Expiration time in seconds

        Returns:
            Number of uploads cleaned up
        """
        current_time = time.time()
        expired_ids = []

        with self._chunk_lock:
            for upload_id, data in list(self._chunk_storage.items()):
                last_activity = data.get('last_activity', 0)
                if current_time - last_activity > expiry_seconds:
                    expired_ids.append(upload_id)

            for upload_id in expired_ids:
                del self._chunk_storage[upload_id]

        return len(expired_ids)

    # ========================================================================
    # Temporary File Operations
    # ========================================================================

    def store_temp_file(
        self,
        file_id: str,
        file_path: str,
        file_name: str
    ) -> None:
        """
        Store temporary file information.

        Args:
            file_id: Unique file identifier
            file_path: Path to temporary file
            file_name: Original file name
        """
        with self._temp_lock:
            self._temp_files[file_id] = {
                'path': file_path,
                'fileName': file_name,
                'timestamp': time.time()
            }

    def get_temp_file(self, file_id: str) -> Optional[Dict[str, Any]]:
        """
        Get temporary file information.

        Returns:
            File info dict or None if not found
        """
        with self._temp_lock:
            return self._temp_files.get(file_id)

    def delete_temp_file(self, file_id: str) -> bool:
        """
        Delete temporary file record.

        Returns:
            True if deleted, False if not found
        """
        with self._temp_lock:
            if file_id in self._temp_files:
                del self._temp_files[file_id]
                return True
        return False

    # ========================================================================
    # Utility Methods
    # ========================================================================

    def get_all_upload_ids(self) -> List[str]:
        """Get list of all active upload IDs."""
        with self._chunk_lock:
            return list(self._chunk_storage.keys())

    def clear_all(self, storage_type: str = None) -> None:
        """Clear all state (for testing purposes).

        Args:
            storage_type: Optional. If 'chunk', clears only chunk storage.
                         If 'temp', clears only temp files.
                         If None, clears both.
        """
        if storage_type is None or storage_type == 'chunk':
            with self._chunk_lock:
                self._chunk_storage.clear()

        if storage_type is None or storage_type == 'temp':
            with self._temp_lock:
                self._temp_files.clear()


# Global state manager instance
_upload_state = UploadStateManager()


# ============================================================================
# StateProxy for backward compatibility
# ============================================================================

class _StateProxy:
    """Proxy to maintain backward compatibility while using UploadStateManager."""

    def __init__(self, state_manager: UploadStateManager, storage_type: str):
        self._state = state_manager
        self._type = storage_type

    def __getitem__(self, key: str) -> Dict[str, Any]:
        if self._type == 'chunk':
            result = self._state.get_upload(key)
            if result is None:
                raise KeyError(key)
            return result
        else:  # temp
            result = self._state.get_temp_file(key)
            if result is None:
                raise KeyError(key)
            return result

    def __setitem__(self, key: str, value: Dict[str, Any]) -> None:
        if self._type == 'chunk':
            params = value.get('parameters', {})
            total_chunks = value.get('total_chunks', 0)
            self._state.create_upload(key, total_chunks, params)

            # If chunks are provided (for testing), store them
            if 'chunks' in value:
                for chunk_idx, chunk_data in value['chunks'].items():
                    self._state.store_chunk(key, chunk_idx, chunk_data)

            # Update fields that may be provided (for testing)
            with self._state._chunk_lock:
                if key in self._state._chunk_storage:
                    if 'received_chunks' in value:
                        self._state._chunk_storage[key]['received_chunks'] = value['received_chunks']
                    if 'last_activity' in value:
                        self._state._chunk_storage[key]['last_activity'] = value['last_activity']
        else:  # temp
            path = value.get('path', '')
            fileName = value.get('fileName', '')
            self._state.store_temp_file(key, path, fileName)

    def __delitem__(self, key: str) -> None:
        if self._type == 'chunk':
            self._state.delete_upload(key)
        else:  # temp
            self._state.delete_temp_file(key)

    def __contains__(self, key: str) -> bool:
        if self._type == 'chunk':
            return self._state.upload_exists(key)
        else:  # temp
            return self._state.get_temp_file(key) is not None

    def get(self, key: str, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def items(self):
        if self._type == 'chunk':
            return [(uid, self._state.get_upload(uid)) for uid in self._state.get_all_upload_ids()]
        return []

    def clear(self) -> None:
        """Clear all items from storage."""
        self._state.clear_all(self._type)

    def __len__(self) -> int:
        """Return number of items in storage."""
        if self._type == 'chunk':
            return len(self._state.get_all_upload_ids())
        else:  # temp
            with self._state._temp_lock:
                return len(self._state._temp_files)


# Legacy-compatible global proxies
chunk_storage = _StateProxy(_upload_state, 'chunk')
temp_files = _StateProxy(_upload_state, 'temp')


def cleanup_old_uploads() -> None:
    """
    Clean up old incomplete uploads and expired temp files.

    Removes upload chunks that haven't been active within UPLOAD_EXPIRY_SECONDS.
    Also removes temp files older than 30 minutes.
    Should be called periodically to prevent memory leaks.
    """
    _upload_state.cleanup_expired_uploads()

    # Cleanup expired temp files (older than 30 minutes)
    current_time = time.time()
    expired_files = []

    for file_id, file_info in list(temp_files.items()):
        if current_time - file_info.get('timestamp', 0) > 1800:  # 30 minutes
            expired_files.append(file_id)

    for file_id in expired_files:
        try:
            file_path = temp_files[file_id].get('path')
            if file_path and os.path.exists(file_path):
                os.unlink(file_path)
            del temp_files[file_id]
        except Exception:
            pass
