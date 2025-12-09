"""Unified Progress Tracking module for WebSocket-based progress updates.

This module provides a comprehensive progress tracking system with ETA calculations
for various processing phases. It supports both file-based and row-based progress
tracking with intelligent rate-limiting and WebSocket emission.

Usage:
    from shared.progress import ProgressTracker

    tracker = ProgressTracker(upload_id='abc123', socketio=socketio, file_size_bytes=1024000)
    tracker.start_phase('parsing')
    tracker.emit('parsing', 50, 'parsing_csv', message_params={'rowCount': 1000})
    tracker.end_phase('parsing')
"""

import time
import logging
from typing import Dict, Optional, Any

logger = logging.getLogger(__name__)


class ProgressTracker:
    """
    Unified progress tracking with ETA for all processing phases.

    Features:
    - File size-based ETA estimation
    - Row-based progress tracking
    - Per-step ETA calculations
    - WebSocket rate-limiting
    - Phase duration tracking

    Payload structure for WebSocket emissions:
    {
        'uploadId': str,           # Upload ID for WebSocket room
        'step': str,               # Current phase name
        'progress': int,           # Progress percentage (0-100)
        'messageKey': str,         # i18n key for frontend translation
        'status': str,             # Status (processing, completed, error)
        'currentStep': int,        # Current step number
        'totalSteps': int,         # Total number of steps
        'eta': int,                # ETA in seconds
        'etaFormatted': str,       # Formatted ETA (e.g., "2m 30s")
        'totalRows': int,          # Total rows to process (optional)
        'processedRows': int       # Rows processed so far (optional)
    }

    Benchmark times per MB (100k rows / ~5MB):
    - validation: ~0.02s/MB
    - parsing: ~0.10s/MB
    - datetime: ~0.15s/MB
    - utc: ~0.04s/MB
    - build: ~0.20s/MB
    - streaming: ~1.0s/MB (calculated real-time)
    """

    # Benchmark coefficients (seconds per MB)
    PHASE_TIME_PER_MB = {
        'validation': 0.02,
        'parsing': 0.10,
        'datetime': 0.15,
        'utc': 0.04,
        'build': 0.20,
        'streaming': 1.0
    }

    def __init__(
        self,
        upload_id: str,
        socketio=None,
        file_size_bytes: int = 0,
        total_items: Optional[int] = None,
        total_chunks: Optional[int] = None,
        emit_interval: float = 0.3
    ):
        """
        Initialize the progress tracker.

        Args:
            upload_id: Unique identifier for the upload/session
            socketio: Flask-SocketIO instance for emitting events
            file_size_bytes: Total file size in bytes for ETA estimation
            total_items: Total number of items to process (optional)
            total_chunks: Total number of chunks for chunked uploads (optional)
            emit_interval: Minimum interval between emissions in seconds
        """
        self.upload_id = upload_id
        self.socketio = socketio
        self.start_time = time.time()
        self.phase_start_times: Dict[str, float] = {}
        self.phase_durations: Dict[str, float] = {}

        # File size for ETA estimation
        self.file_size_bytes = file_size_bytes
        self.file_size_mb = file_size_bytes / (1024 * 1024) if file_size_bytes > 0 else 1
        self.total_chunks = total_chunks
        self.total_items = total_items

        # Estimated total time based on file size
        self.estimated_total_time = self._estimate_total_time()

        # Rate limiting
        self.last_emit_time = 0
        self.emit_interval = emit_interval

        # Step tracking for frontend
        self.current_step = 0
        self.total_steps = 0

        # Row tracking
        self.total_rows = 0

        # Per-step ETA tracking
        self.current_step_start: Optional[float] = None
        self.current_step_rows = 0
        self.current_step_processed = 0
        self.min_calibration_rows = 1000  # Wait for 1000 rows for stable estimate

    def _estimate_total_time(self) -> float:
        """Estimate total processing time based on file size."""
        total = 0
        for phase, time_per_mb in self.PHASE_TIME_PER_MB.items():
            total += time_per_mb * self.file_size_mb
        return max(total, 2.0)  # Minimum 2 seconds

    def set_total_rows(self, rows: int) -> None:
        """Set the number of rows and update time estimate."""
        self.total_rows = rows
        streaming_time = rows * 0.00005
        self.estimated_total_time = (
            self.PHASE_TIME_PER_MB['validation'] * self.file_size_mb +
            self.PHASE_TIME_PER_MB['parsing'] * self.file_size_mb +
            self.PHASE_TIME_PER_MB['datetime'] * self.file_size_mb +
            self.PHASE_TIME_PER_MB['utc'] * self.file_size_mb +
            self.PHASE_TIME_PER_MB['build'] * self.file_size_mb +
            streaming_time
        )

    def start_phase(self, phase_name: str) -> None:
        """Mark the start of a new processing phase."""
        self.phase_start_times[phase_name] = time.time()
        logger.debug(f"Started phase: {phase_name}")

    def end_phase(self, phase_name: str) -> None:
        """End a phase and record its actual duration."""
        if phase_name in self.phase_start_times:
            duration = time.time() - self.phase_start_times[phase_name]
            self.phase_durations[phase_name] = duration
            logger.info(f"Phase '{phase_name}' completed in {duration:.2f}s")

    def start_step(self, total_rows: int) -> None:
        """Start a new step - reset ETA tracking for this step."""
        self.current_step_start = time.time()
        self.current_step_rows = total_rows
        self.current_step_processed = 0

    def update_step_progress(self, processed_count: int) -> None:
        """Update the number of processed rows in the current step."""
        self.current_step_processed = processed_count

    def calculate_step_eta(self) -> Optional[int]:
        """
        Calculate ETA for the current step only.

        Returns:
            ETA in seconds, or None if insufficient data
        """
        if self.current_step_start is None:
            return None

        if self.current_step_rows <= 0:
            return None

        # Wait for calibration period
        if self.current_step_processed < self.min_calibration_rows:
            return None

        elapsed = time.time() - self.current_step_start
        if elapsed <= 0:
            return None

        # Calculate based on processed items
        rows_per_second = self.current_step_processed / elapsed
        if rows_per_second <= 0:
            return None

        remaining_rows = self.current_step_rows - self.current_step_processed
        eta_seconds = int(remaining_rows / rows_per_second)

        # Limit to reasonable values
        return max(0, min(eta_seconds, 3600))  # Max 1 hour

    def calculate_eta_for_progress(self, current_progress: float) -> int:
        """
        Calculate ETA based on current progress and estimated total time.

        For fast phases, uses file size-based estimation.
        For slower phases (elapsed > 1s), uses linear extrapolation.

        Args:
            current_progress: Current progress (0-100)

        Returns:
            ETA in seconds
        """
        if current_progress <= 0:
            return int(self.estimated_total_time)

        if current_progress >= 100:
            return 0

        elapsed = time.time() - self.start_time
        remaining_progress = 100 - current_progress

        # If more than 1 second elapsed, use linear extrapolation
        if elapsed > 1.0:
            time_per_percent = elapsed / current_progress
            eta_seconds = int(remaining_progress * time_per_percent)
        else:
            # For fast phases, use file size-based estimation
            eta_seconds = int(self.estimated_total_time * remaining_progress / 100)

        # Limit to reasonable values
        return max(0, min(eta_seconds, 3600))  # Max 1 hour

    def emit(
        self,
        step: str,
        progress: float,
        message_key: str,
        eta_seconds: Optional[int] = None,
        force: bool = False,
        processed_rows: Optional[int] = None,
        message_params: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Send progress update via WebSocket with i18n key for translation.

        Args:
            step: Phase name (validating, parsing, datetime, utc, build, streaming, complete, error)
            progress: Percentage (0-100)
            message_key: i18n key for frontend translation
            eta_seconds: ETA in seconds (optional, calculated if not provided)
            force: Ignore rate limiting
            processed_rows: Number of processed rows (optional)
            message_params: Additional parameters for the message (e.g., {'rowCount': 1000})
        """
        current_time = time.time()

        # Rate limiting - don't send more frequently than emit_interval unless force=True
        if not force and (current_time - self.last_emit_time) < self.emit_interval:
            return

        payload = {
            'uploadId': self.upload_id,
            'step': step,
            'progress': int(progress),
            'messageKey': message_key,
            'status': 'processing' if step not in ['complete', 'error'] else ('completed' if step == 'complete' else 'error')
        }

        # Add message parameters if present
        if message_params:
            payload['messageParams'] = message_params

        # Add currentStep and totalSteps if set
        if self.total_steps > 0:
            payload['currentStep'] = self.current_step
            payload['totalSteps'] = self.total_steps

        # Add totalRows and processedRows if available
        if self.total_rows > 0:
            payload['totalRows'] = self.total_rows
        if processed_rows is not None:
            payload['processedRows'] = processed_rows

        # Add ETA
        if eta_seconds is not None:
            payload['eta'] = eta_seconds
            payload['etaFormatted'] = self.format_time(eta_seconds)
        else:
            # Try step-based ETA first, fall back to progress-based
            step_eta = self.calculate_step_eta()
            if step_eta is not None:
                payload['eta'] = step_eta
                payload['etaFormatted'] = self.format_time(step_eta)
            else:
                progress_eta = self.calculate_eta_for_progress(progress)
                payload['eta'] = progress_eta
                payload['etaFormatted'] = self.format_time(progress_eta)

        try:
            if self.socketio:
                self.socketio.emit('upload_progress', payload, room=self.upload_id)
            self.last_emit_time = current_time
            logger.debug(f"Emitted progress: step={step}, progress={progress}%")
        except Exception as e:
            logger.warning(f"Failed to emit progress update: {e}")

    def emit_complete(self, message_key: str = 'processing_complete') -> None:
        """Emit completion status."""
        self.emit('complete', 100, message_key, eta_seconds=0, force=True)

    def emit_error(self, error_message: str, error_key: str = 'processing_error') -> None:
        """Emit error status."""
        self.emit('error', 0, error_key, force=True, message_params={'error': error_message})

    @staticmethod
    def format_time(seconds: Optional[int]) -> str:
        """Format seconds into a human-readable format."""
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

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of all phase durations."""
        total_elapsed = time.time() - self.start_time
        return {
            'total_elapsed': round(total_elapsed, 2),
            'phase_durations': {k: round(v, 2) for k, v in self.phase_durations.items()},
            'file_size_mb': round(self.file_size_mb, 2),
            'total_rows': self.total_rows
        }
