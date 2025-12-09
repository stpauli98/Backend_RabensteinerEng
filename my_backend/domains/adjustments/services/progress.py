"""
Progress Tracking for Adjustments Domain
Real-time progress updates via WebSocket with ETA calculation
"""
import time
import logging
from typing import Optional, Dict, Any

from core.app_factory import socketio
from domains.adjustments.config import DATAFRAME_CHUNK_SIZE, SOCKETIO_CHUNK_DELAY

logger = logging.getLogger(__name__)


class ProgressStages:
    """Constants for Socket.IO progress tracking"""
    # Upload phase
    FILE_COMBINATION = 25
    FILE_ANALYSIS = 28
    FILE_COMPLETE = 30

    # Processing phase
    PARAMETER_PROCESSING = 5
    DATA_PROCESSING_START = 0
    FILE_PROCESSING_START = 10
    FILE_PROCESSING_END = 95
    COMPLETION = 100

    @staticmethod
    def calculate_file_progress(file_index: int, total_files: int, start: int = 10, end: int = 95) -> float:
        """Calculate progress percentage for file processing"""
        if total_files == 0:
            return start
        return start + (file_index / total_files) * (end - start)


class ProgressTracker:
    """
    Progress tracking with ETA based on two phases:
    1. Processing phase (pandas) - slow, calculated per file
    2. Streaming phase (chunks) - fast, calculated upfront
    """

    # Constants for streaming time estimation
    # Based on: emit time (~0.02s) + SOCKETIO_CHUNK_DELAY (0.1s) = ~0.12s per chunk
    ESTIMATED_TIME_PER_CHUNK = 0.13  # seconds per chunk

    def __init__(self, upload_id: str, total_files: int = 0, total_rows: int = 0):
        self.upload_id = upload_id
        self.start_time = time.time()
        self.total_files = total_files
        self.total_rows = total_rows

        # File-level tracking
        self.files_processed = 0
        self.rows_processed = 0
        self.file_start_times: Dict[str, float] = {}
        self.file_durations: list = []
        self.file_rows: Dict[str, int] = {}

        # Current file tracking
        self.current_file: Optional[str] = None
        self.current_file_rows = 0

        self.last_emit_time = 0
        self.emit_interval = 0.5

        # Minimum time for stable ETA
        self.min_elapsed_for_eta = 2.0  # seconds

        # === Phase 1: Processing (pandas) ===
        self.processing_times: list = []
        self.current_processing_start: Optional[float] = None

        # === Phase 2: Streaming (chunks) ===
        self.current_streaming_start: Optional[float] = None
        self.chunks_sent = 0
        self.total_chunks_for_file = 0

        # Total estimated streaming time for all files
        self.total_estimated_streaming_time = 0
        self.completed_streaming_time = 0
        self.current_file_streaming_estimate = 0

        # Current phase: 'processing' or 'streaming'
        self.current_phase: Optional[str] = None

        # ETA tracking - keeps last value to prevent increase
        self.last_eta: Optional[int] = None

    def estimate_streaming_time(self, row_count: int) -> float:
        """Calculate estimated streaming time for a file with given row count"""
        total_chunks = (row_count + DATAFRAME_CHUNK_SIZE - 1) // DATAFRAME_CHUNK_SIZE
        return total_chunks * self.ESTIMATED_TIME_PER_CHUNK

    def set_file_rows(self, file_rows_dict: Dict[str, int]) -> None:
        """Set row count per file and calculate total streaming time"""
        self.file_rows = file_rows_dict
        self.total_estimated_streaming_time = sum(
            self.estimate_streaming_time(rows) for rows in file_rows_dict.values()
        )

    def start_file(self, filename: str, row_count: int) -> None:
        """Start processing a new file"""
        self.current_file = filename
        self.current_file_rows = row_count
        self.file_start_times[filename] = time.time()
        self.file_rows[filename] = row_count
        # Start processing phase
        self.current_processing_start = time.time()
        self.current_phase = 'processing'

    def start_streaming(self, filename: str, total_chunks: int) -> None:
        """Start streaming phase for a file"""
        # End processing phase
        if self.current_processing_start:
            proc_time = time.time() - self.current_processing_start
            self.processing_times.append(proc_time)
            self.current_processing_start = None

        # Start streaming phase
        self.current_streaming_start = time.time()
        self.total_chunks_for_file = total_chunks
        self.chunks_sent = 0
        self.current_phase = 'streaming'

        # Calculate estimated time for this streaming
        self.current_file_streaming_estimate = total_chunks * self.ESTIMATED_TIME_PER_CHUNK

    def chunk_sent(self) -> None:
        """Record a sent chunk"""
        self.chunks_sent += 1

    def complete_file(self, filename: str) -> None:
        """Complete file processing"""
        # Add completed streaming time
        if self.current_streaming_start:
            actual_stream_time = time.time() - self.current_streaming_start
            self.completed_streaming_time += actual_stream_time
            self.current_streaming_start = None

        if filename in self.file_start_times:
            duration = time.time() - self.file_start_times[filename]
            self.file_durations.append(duration)
            self.files_processed += 1
            self.rows_processed += self.file_rows.get(filename, 0)
            self.current_file = None
            self.current_file_rows = 0

        self.current_phase = None
        self.chunks_sent = 0
        self.total_chunks_for_file = 0
        self.current_file_streaming_estimate = 0

    def calculate_eta(self, current_progress: float) -> Optional[int]:
        """
        ETA calculation based on phases.

        Streaming time is known upfront (based on chunk count).
        Processing time is estimated based on average of previous files.

        ETA = remaining_processing + remaining_streaming
        """
        elapsed = time.time() - self.start_time

        # Wait minimum time for stable estimate
        if elapsed < self.min_elapsed_for_eta:
            return None

        if current_progress <= 0:
            return None

        # === 1. Remaining STREAMING time ===
        remaining_streaming = self.total_estimated_streaming_time - self.completed_streaming_time

        # If in streaming phase, calculate remaining for current file
        if self.current_phase == 'streaming' and self.total_chunks_for_file > 0:
            chunks_remaining = self.total_chunks_for_file - self.chunks_sent
            current_file_remaining = chunks_remaining * self.ESTIMATED_TIME_PER_CHUNK

            # Subtract estimated time for this file (included in total)
            # and add actual remaining
            remaining_streaming -= self.current_file_streaming_estimate
            remaining_streaming += current_file_remaining

        remaining_streaming = max(0, remaining_streaming)

        # === 2. Remaining PROCESSING time (estimate) ===
        files_remaining = self.total_files - self.files_processed
        if self.current_file:
            files_remaining -= 1  # Current file is in progress

        remaining_processing = 0

        # For remaining files (not including current)
        if files_remaining > 0 and self.processing_times:
            avg_processing = sum(self.processing_times) / len(self.processing_times)
            remaining_processing = files_remaining * avg_processing

        # For current file if in processing phase
        if self.current_phase == 'processing' and self.current_processing_start:
            time_in_processing = time.time() - self.current_processing_start
            if self.processing_times:
                avg_processing = sum(self.processing_times) / len(self.processing_times)
                remaining_processing += max(0, avg_processing - time_in_processing)
            else:
                # No data - assume remaining equals elapsed
                remaining_processing += time_in_processing

        # === 3. Total ETA = remaining processing + remaining streaming ===
        raw_eta = int(remaining_processing + remaining_streaming)

        # If progress is 100%, ETA is 0
        if current_progress >= 100:
            raw_eta = 0

        # If no data, use fallback formula
        if self.total_estimated_streaming_time == 0 and not self.processing_times:
            progress_ratio = current_progress / 100.0
            remaining_ratio = 1.0 - progress_ratio
            if progress_ratio > 0:
                raw_eta = int(elapsed * (remaining_ratio / progress_ratio))
            else:
                return None

        # ETA should never increase
        if self.last_eta is not None:
            raw_eta = min(raw_eta, self.last_eta)

        self.last_eta = raw_eta
        return max(0, raw_eta)

    def emit(
        self,
        progress: float,
        message_key: str,
        step: str,
        phase: str,
        message_params: Optional[Dict[str, Any]] = None,
        detail_key: Optional[str] = None,
        detail_params: Optional[Dict[str, Any]] = None,
        force: bool = False
    ) -> None:
        """Send progress update with ETA and i18n support"""
        current_time = time.time()

        if not force and (current_time - self.last_emit_time) < self.emit_interval:
            return

        payload = {
            'uploadId': self.upload_id,
            'progress': int(progress),
            'messageKey': message_key,
            'step': step,
            'phase': phase,
            'status': 'processing'
        }

        if message_params:
            payload['messageParams'] = message_params

        if detail_key:
            payload['detailKey'] = detail_key
        if detail_params:
            payload['detailParams'] = detail_params

        # Add file tracking info
        if self.total_files > 0:
            if self.current_file:
                payload['currentFile'] = self.files_processed + 1
                payload['currentFileName'] = self.current_file
            else:
                payload['currentFile'] = min(self.files_processed, self.total_files)

            payload['totalFiles'] = self.total_files

        # Calculate and add ETA
        eta = self.calculate_eta(progress)
        if eta is not None:
            payload['eta'] = eta
            payload['etaFormatted'] = self.format_time(eta)

        try:
            socketio.emit('processing_progress', payload, room=self.upload_id)
            self.last_emit_time = current_time

            # Focused log: File X/Y | Progress% | ETA
            file_info = f"File {payload.get('currentFile', '?')}/{payload.get('totalFiles', '?')}"
            eta_info = payload.get('etaFormatted', 'N/A')
            logger.info(f"=> {file_info} | {int(progress)}% | ETA: {eta_info} | key: {message_key}")
        except Exception as e:
            logger.error(f"Error emitting progress: {e}")

    @staticmethod
    def format_time(seconds: Optional[int]) -> str:
        """Format seconds into readable format"""
        if seconds is None:
            return "Estimating..."
        if seconds < 60:
            return f"{seconds}s"
        elif seconds < 3600:
            minutes = seconds // 60
            secs = seconds % 60
            return f"{minutes}m {secs}s"
        else:
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            return f"{hours}h {minutes}m"


def emit_progress(
    upload_id: str,
    progress: float,
    message_key: str,
    step: str,
    phase: str,
    message_params: Optional[Dict[str, Any]] = None,
    detail_key: Optional[str] = None,
    detail_params: Optional[Dict[str, Any]] = None
) -> None:
    """
    Emit Socket.IO progress update with error handling and i18n support
    """
    try:
        data = {
            'uploadId': upload_id,
            'progress': progress,
            'messageKey': message_key,
            'step': step,
            'phase': phase,
            'status': 'processing'
        }
        if message_params:
            data['messageParams'] = message_params
        if detail_key:
            data['detailKey'] = detail_key
        if detail_params:
            data['detailParams'] = detail_params
        socketio.emit('processing_progress', data, room=upload_id)
    except Exception as e:
        logger.error(f"Failed to emit progress for {upload_id}: {e}")


def emit_file_result(
    upload_id: str,
    filename: str,
    result_data: list,
    info_record: Dict[str, Any],
    file_index: int,
    total_files: int,
    tracker: Optional[ProgressTracker] = None
) -> None:
    """
    Emit file processing result via SocketIO with chunking for large datasets
    """
    try:
        # Calculate progress range for this file
        file_start_progress = ProgressStages.FILE_PROCESSING_START + (file_index / total_files) * (ProgressStages.FILE_PROCESSING_END - ProgressStages.FILE_PROCESSING_START)
        file_end_progress = ProgressStages.FILE_PROCESSING_START + ((file_index + 1) / total_files) * (ProgressStages.FILE_PROCESSING_END - ProgressStages.FILE_PROCESSING_START)
        file_progress_range = file_end_progress - file_start_progress

        if len(result_data) <= DATAFRAME_CHUNK_SIZE:
            # Small file - no streaming phase, single emit
            if tracker:
                tracker.start_streaming(filename, 1)
                tracker.chunk_sent()
            socketio.emit('file_result', {
                'uploadId': upload_id,
                'filename': filename,
                'info_record': info_record,
                'dataframe_chunk': result_data,
                'fileIndex': file_index,
                'totalFiles': total_files,
                'chunked': False
            }, room=upload_id)
            logger.info(f"Emitted single file_result for {filename} ({len(result_data)} rows)")
        else:
            total_chunks = (len(result_data) + DATAFRAME_CHUNK_SIZE - 1) // DATAFRAME_CHUNK_SIZE

            # Start streaming phase in tracker
            if tracker:
                tracker.start_streaming(filename, total_chunks)

            socketio.emit('file_result', {
                'uploadId': upload_id,
                'filename': filename,
                'info_record': info_record,
                'dataframe_chunk': [],
                'fileIndex': file_index,
                'totalFiles': total_files,
                'chunked': True,
                'totalChunks': total_chunks
            }, room=upload_id)

            for chunk_idx in range(total_chunks):
                start_idx = chunk_idx * DATAFRAME_CHUNK_SIZE
                end_idx = min((chunk_idx + 1) * DATAFRAME_CHUNK_SIZE, len(result_data))
                chunk = result_data[start_idx:end_idx]

                socketio.emit('dataframe_chunk', {
                    'uploadId': upload_id,
                    'filename': filename,
                    'chunk': chunk,
                    'chunkIndex': chunk_idx,
                    'totalChunks': total_chunks,
                    'fileIndex': file_index
                }, room=upload_id)

                # Record sent chunk and emit progress
                if tracker:
                    tracker.chunk_sent()
                    chunk_progress = (chunk_idx + 1) / total_chunks
                    smooth_progress = file_start_progress + (chunk_progress * file_progress_range * 0.8)
                    tracker.emit(
                        smooth_progress,
                        'streaming_chunk',
                        'data_streaming',
                        'data_processing',
                        message_params={'current': chunk_idx + 1, 'total': total_chunks, 'filename': filename},
                        detail_key='detail_streaming_progress',
                        detail_params={'fileNum': file_index + 1, 'totalFiles': total_files, 'percentage': int(chunk_progress * 100)}
                    )

                time.sleep(SOCKETIO_CHUNK_DELAY)

                logger.info(f"Emitted chunk {chunk_idx + 1}/{total_chunks} for {filename} ({len(chunk)} rows)")

            socketio.emit('dataframe_complete', {
                'uploadId': upload_id,
                'filename': filename,
                'totalChunks': total_chunks,
                'totalRows': len(result_data)
            }, room=upload_id)
            logger.info(f" Dataframe streaming complete for {filename} ({total_chunks} chunks, {len(result_data)} rows)")

    except Exception as e:
        logger.error(f"Failed to emit file_result for {filename}: {e}")
        emit_file_error(upload_id, filename, str(e))


def emit_file_error(upload_id: str, filename: str, error_message: str) -> None:
    """
    Emit file processing error via SocketIO
    """
    try:
        socketio.emit('file_error', {
            'uploadId': upload_id,
            'filename': filename,
            'error': error_message
        }, room=upload_id)
        logger.error(f"Emitted file_error for {filename}: {error_message}")
    except Exception as e:
        logger.error(f"Failed to emit file_error for {filename}: {e}")
