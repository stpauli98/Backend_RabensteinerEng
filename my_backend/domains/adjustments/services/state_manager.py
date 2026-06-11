"""
Adjustments State Manager
Thread-safe state management for adjustment chunks and cached data
"""
import threading
import time
import logging
from typing import Dict, Any, Optional, List

from domains.adjustments.debug_log import dlog, _short

from domains.adjustments.config import (
    CHUNK_BUFFER_TIMEOUT,
    ADJUSTMENT_CHUNKS_TIMEOUT,
    STORED_DATA_TIMEOUT,
    INFO_CACHE_TIMEOUT
)

logger = logging.getLogger(__name__)

# Module-level lock for anomaly-state mutations. Flask + gevent (used by
# Socket.IO without Redis) can context-switch between dict reads/writes; this
# guards init/get/set/reset against interleaved concurrent requests.
_anomaly_state_lock = threading.RLock()

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


# ---------------------------------------------------------------------------
# Anomaly detection pipeline state
# ---------------------------------------------------------------------------
# Lives inside the existing `adjustment_chunks[upload_id]` dict under the
# `anomaly` key, so it shares the same eviction TTL as the rest of the upload.
# Schema:
#   {
#     "user_id": str,                     # binds the session to its owner
#     "lang": "en" | "de",                # last requested language
#     "filename": str,                    # CSV filename within UPLOAD_FOLDER/{upload_id}
#     "file_path": str,                   # absolute path on disk
#     "original_df": pd.DataFrame,        # the loaded CSV (datetime + numeric)
#     "processed_df": pd.DataFrame,       # output after running the pipeline
#     "dt_avg": pd.Timedelta,             # mean time step
#     "params": Dict,                     # last-validated `par` dict
#     "pipeline_status": PipelineStatus,  # enum below
#     "intermediate": {                   # transient state needed to resume after pause
#         "stl_result": Any | None,
#         "lstm_results_df": pd.DataFrame | None,
#     },
#     "plots": Dict[str, Any],            # accumulated plot data per phase
#   }


class PipelineStatus:
    IDLE = "idle"
    LOADED = "loaded"
    RUNNING = "running"
    AWAITING_STL_THRESHOLD = "awaiting_stl_threshold"
    APPLYING_STL = "applying_stl"
    AWAITING_LSTM_THRESHOLD = "awaiting_lstm_threshold"
    APPLYING_LSTM = "applying_lstm"
    COMPLETE = "complete"
    ERROR = "error"


def init_anomaly_state(upload_id: str, user_id: str, lang: str) -> Optional[Dict[str, Any]]:
    """
    Create or refresh the anomaly state container for an upload session.

    Returns the state dict on success. Returns None if the session already
    exists and is owned by a different user (handler should respond 404 to
    avoid leaking ownership).
    """
    with _anomaly_state_lock:
        if upload_id not in adjustment_chunks:
            adjustment_chunks[upload_id] = {}
        adjustment_chunks_timestamps[upload_id] = time.time()

        session = adjustment_chunks[upload_id]
        existing = session.get("anomaly")

        if existing is not None and existing.get("user_id") != user_id:
            # Ownership mismatch — refuse to leak or overwrite victim's state.
            return None

        if existing is None:
            session["anomaly"] = {
                "user_id": user_id,
                "lang": lang,
                "filename": None,
                "file_path": None,
                "original_df": None,
                "processed_df": None,
                "dt_avg": None,
                "params": None,
                "pipeline_status": PipelineStatus.IDLE,
                "intermediate": {"stl_result": None, "lstm_results_df": None},
                "plots": {},
            }
        else:
            # Update lang on every interaction; keep ownership immutable.
            existing["lang"] = lang
        return session["anomaly"]


def get_anomaly_state(upload_id: str, user_id: str) -> Optional[Dict[str, Any]]:
    """
    Return the anomaly state for an upload, BUT only if owned by `user_id`.
    Returns None if upload is unknown or owned by someone else.
    """
    with _anomaly_state_lock:
        session = adjustment_chunks.get(upload_id)
        if session is None:
            return None
        anomaly = session.get("anomaly")
        if anomaly is None:
            return None
        if anomaly.get("user_id") != user_id:
            return None
        # Touch timestamp so active sessions don't expire mid-pipeline
        adjustment_chunks_timestamps[upload_id] = time.time()
        return anomaly


def is_adjustment_owner(upload_id: str, user_id: str) -> bool:
    """True iff the plain-adjustment session exists and is owned by user_id."""
    session = adjustment_chunks.get(upload_id)
    return session is not None and session.get("user_id") == user_id


def set_pipeline_status(upload_id: str, user_id: str, status: str) -> bool:
    """Update pipeline status; returns False if session unknown / not owned."""
    with _anomaly_state_lock:
        state = get_anomaly_state(upload_id, user_id)
        if state is None:
            return False
        old = state.get("pipeline_status")
        state["pipeline_status"] = status
        dlog("PIPELINE_STATUS",
             upload=_short(upload_id),
             user=_short(user_id),
             old=old,
             new=status)
        return True


_PIPELINE_RUNNING_STATES = frozenset({
    PipelineStatus.RUNNING,
    PipelineStatus.APPLYING_STL,
    PipelineStatus.APPLYING_LSTM,
})

# If a session has been in a "running" state longer than this, treat it as
# stale (the client likely crashed / disconnected) and let the next request
# acquire the pipeline. Without this guard a single network drop wedges the
# session forever until the 1 h TTL eviction.
_PIPELINE_STALE_AFTER_S = 60.0


def try_acquire_pipeline(upload_id: str, user_id: str) -> bool:
    """
    Atomically claim the pipeline for this user. Returns True if acquired
    (caller may run pipeline phases), False if another run is already in
    progress for the same upload. Prevents double-click / concurrent /start
    or /stl-threshold races.

    Stale running states (>60 s) are auto-released so a transient client
    failure can't permanently lock a session.
    """
    with _anomaly_state_lock:
        state = adjustment_chunks.get(upload_id, {}).get("anomaly")
        if state is None or state.get("user_id") != user_id:
            dlog("ACQUIRE_DENIED", upload=_short(upload_id), reason="state_missing_or_owner_mismatch")
            return False
        if state.get("pipeline_status") in _PIPELINE_RUNNING_STATES:
            last_run_at = state.get("running_since", 0.0)
            if time.time() - last_run_at < _PIPELINE_STALE_AFTER_S:
                dlog("ACQUIRE_DENIED", upload=_short(upload_id), reason="pipeline_busy",
                     status=state.get("pipeline_status"))
                return False
            # Stale — force-release and re-acquire below.
            logger.warning(
                "try_acquire_pipeline: forcing stale lock release for upload_id=%s "
                "(was in %s for >%ss)",
                upload_id,
                state.get("pipeline_status"),
                _PIPELINE_STALE_AFTER_S,
            )
            dlog("ACQUIRE_STALE_RELEASE", upload=_short(upload_id),
                 old_status=state.get("pipeline_status"))
        dlog("ACQUIRE_OK", upload=_short(upload_id), user=_short(user_id))
        state["pipeline_status"] = PipelineStatus.RUNNING
        state["running_since"] = time.time()
        adjustment_chunks_timestamps[upload_id] = time.time()
        return True


def reset_anomaly_intermediate(upload_id: str, user_id: str) -> bool:
    """Wipe transient pipeline state when iterating with `/use-processed`."""
    with _anomaly_state_lock:
        state = get_anomaly_state(upload_id, user_id)
        if state is None:
            return False
        state["intermediate"] = {"stl_result": None, "lstm_results_df": None}
        state["pipeline_status"] = PipelineStatus.LOADED
        state["plots"] = {}
        dlog("INTERMEDIATE_RESET", upload=_short(upload_id))
        return True


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
