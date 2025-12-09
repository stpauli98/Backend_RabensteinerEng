"""
Adjustments State Manager
Thread-safe state management for adjustment chunks and cached data
"""
import time
import logging
from typing import Dict, Any, Optional, List

from domains.adjustments.config import (
    CHUNK_BUFFER_TIMEOUT,
    ADJUSTMENT_CHUNKS_TIMEOUT,
    STORED_DATA_TIMEOUT,
    INFO_CACHE_TIMEOUT
)

logger = logging.getLogger(__name__)

# Global state dictionaries
adjustment_chunks: Dict[str, Dict[str, Any]] = {}
adjustment_chunks_timestamps: Dict[str, float] = {}

chunk_buffer: Dict[str, Dict[int, str]] = {}
chunk_buffer_timestamps: Dict[str, float] = {}

stored_data: Dict[str, Any] = {}
stored_data_timestamps: Dict[str, float] = {}

info_df_cache: Dict[str, Dict[str, Any]] = {}
info_df_cache_timestamps: Dict[str, float] = {}


def cleanup_expired_chunk_buffers() -> int:
    """Remove chunk buffers older than CHUNK_BUFFER_TIMEOUT"""
    current_time = time.time()
    expired_uploads = []

    for upload_id, timestamp in chunk_buffer_timestamps.items():
        if current_time - timestamp > CHUNK_BUFFER_TIMEOUT:
            expired_uploads.append(upload_id)

    for upload_id in expired_uploads:
        if upload_id in chunk_buffer:
            del chunk_buffer[upload_id]
        if upload_id in chunk_buffer_timestamps:
            del chunk_buffer_timestamps[upload_id]

    return len(expired_uploads)


def cleanup_expired_adjustment_chunks() -> int:
    """Remove adjustment chunks older than ADJUSTMENT_CHUNKS_TIMEOUT"""
    current_time = time.time()
    expired_uploads = []

    for upload_id, timestamp in adjustment_chunks_timestamps.items():
        if current_time - timestamp > ADJUSTMENT_CHUNKS_TIMEOUT:
            expired_uploads.append(upload_id)

    for upload_id in expired_uploads:
        if upload_id in adjustment_chunks:
            del adjustment_chunks[upload_id]
        if upload_id in adjustment_chunks_timestamps:
            del adjustment_chunks_timestamps[upload_id]

    return len(expired_uploads)


def cleanup_expired_stored_data() -> int:
    """Remove stored data older than STORED_DATA_TIMEOUT"""
    current_time = time.time()
    expired_files = []

    for filename, timestamp in stored_data_timestamps.items():
        if current_time - timestamp > STORED_DATA_TIMEOUT:
            expired_files.append(filename)

    for filename in expired_files:
        if filename in stored_data:
            del stored_data[filename]
        if filename in stored_data_timestamps:
            del stored_data_timestamps[filename]

    return len(expired_files)


def cleanup_expired_info_cache() -> int:
    """Remove info cache entries older than INFO_CACHE_TIMEOUT"""
    current_time = time.time()
    expired_files = []

    for filename, timestamp in info_df_cache_timestamps.items():
        if current_time - timestamp > INFO_CACHE_TIMEOUT:
            expired_files.append(filename)

    for filename in expired_files:
        if filename in info_df_cache:
            del info_df_cache[filename]
        if filename in info_df_cache_timestamps:
            del info_df_cache_timestamps[filename]

    return len(expired_files)


def cleanup_all_expired_data() -> int:
    """Run all cleanup functions and return total cleaned items"""
    total = 0
    total += cleanup_expired_chunk_buffers()
    total += cleanup_expired_adjustment_chunks()
    total += cleanup_expired_stored_data()
    total += cleanup_expired_info_cache()
    return total


def get_file_info_from_cache(filename: str, upload_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Helper function to retrieve file info from cache with fallback

    Args:
        filename: Filename to lookup
        upload_id: Upload ID for upload-specific cache

    Returns:
        File info dict or None if not found
    """
    if upload_id and upload_id in adjustment_chunks:
        file_info_cache_local = adjustment_chunks[upload_id].get('file_info_cache', {})
        file_info = file_info_cache_local.get(filename)
        if file_info:
            return file_info

    return info_df_cache.get(filename)


def check_files_need_methods(
    filenames: List[str],
    time_step: float,
    offset: float,
    methods: Dict[str, Any],
    file_info_cache_local: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Fast batch check if files need processing methods

    Args:
        filenames: List of filenames to check
        time_step: Requested time step size
        offset: Requested offset
        methods: Dictionary of methods per filename
        file_info_cache_local: Upload-specific cache (Cloud Run compatible)

    Returns:
        List of files needing methods with their info, or empty list if all OK
    """
    from domains.adjustments.config import VALID_METHODS

    files_needing_methods = []

    for filename in filenames:
        file_info = None
        if file_info_cache_local:
            file_info = file_info_cache_local.get(filename)
        if not file_info:
            file_info = info_df_cache.get(filename)
        if not file_info:
            logger.warning(f"File {filename} not found in cache")
            continue

        file_time_step = file_info['timestep']
        file_offset = file_info['offset']

        requested_offset = offset
        if file_time_step > 0 and requested_offset >= file_time_step:
            requested_offset = requested_offset % file_time_step

        needs_processing = file_time_step != time_step or file_offset != requested_offset

        if needs_processing:
            method_info = methods.get(filename, {})
            method = method_info.get('method', '').strip() if isinstance(method_info, dict) else ''
            has_valid_method = method and method in VALID_METHODS

            if not has_valid_method:
                files_needing_methods.append({
                    "filename": filename,
                    "current_timestep": file_time_step,
                    "requested_timestep": time_step,
                    "current_offset": file_offset,
                    "requested_offset": requested_offset,
                    "valid_methods": list(VALID_METHODS)
                })

    return files_needing_methods
