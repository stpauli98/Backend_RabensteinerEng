"""
Progress tracking service for real-time processing updates.
Unified ProgressTracker used by both first_processing and data_processing.
"""
import time
import logging
from core.app_factory import socketio
from domains.processing.config import EMIT_INTERVAL, MIN_CALIBRATION_ROWS

logger = logging.getLogger(__name__)


class ProgressTracker:
    """
    Progress tracking with per-step ETA calculation.
    ETA is calculated only for the current step - precise and stable.
    """

    def __init__(self, upload_id, total_items=None, file_size_bytes=None, total_chunks=None):
        self.upload_id = upload_id
        self.start_time = time.time()
        self.phase_start_times = {}
        self.phase_durations = {}

        self.file_size_bytes = file_size_bytes
        self.total_chunks = total_chunks
        self.total_items = total_items

        self.last_emit_time = 0
        self.emit_interval = EMIT_INTERVAL

        # Per-step ETA tracking
        self.current_step_start = None
        self.current_step_rows = 0
        self.current_step_processed = 0
        self.min_calibration_rows = MIN_CALIBRATION_ROWS

        # Step tracking for frontend
        self.current_step = 0
        self.total_steps = 0

    def start_phase(self, phase_name):
        """Mark the start of a new processing phase"""
        self.phase_start_times[phase_name] = time.time()

    def end_phase(self, phase_name):
        """End phase and record actual duration"""
        if phase_name in self.phase_start_times:
            duration = time.time() - self.phase_start_times[phase_name]
            self.phase_durations[phase_name] = duration
            logger.info(f"Phase '{phase_name}' completed in {duration:.2f}s")

    def start_step(self, total_rows):
        """Start a new step - reset ETA tracking for this step"""
        self.current_step_start = time.time()
        self.current_step_rows = total_rows
        self.current_step_processed = 0

    def update_step_progress(self, processed_count):
        """Update the number of processed rows in current step"""
        self.current_step_processed = processed_count

    def calculate_step_eta(self):
        """
        Calculate ETA for CURRENT step only.
        Returns None if not enough data for estimation.
        """
        if not self.current_step_start or self.current_step_rows == 0:
            return None

        elapsed = time.time() - self.current_step_start
        processed = self.current_step_processed

        # Wait for min_calibration_rows before giving ETA
        if processed < self.min_calibration_rows:
            return None

        remaining = self.current_step_rows - processed
        if remaining <= 0:
            return 0

        # Linear prediction
        time_per_row = elapsed / processed
        eta_seconds = int(remaining * time_per_row)

        return eta_seconds

    def emit(self, step, progress, message_key, eta_seconds=None, force=False, message_params=None):
        """
        Send progress update with translation key.

        Args:
            step: Processing phase (chunk_assembly, parsing, processing, streaming, etc.)
            progress: Progress percentage (0-100)
            message_key: Translation key for frontend (e.g., 'fp_parsing_start')
            eta_seconds: ETA in seconds (optional)
            force: Ignore rate limiting
            message_params: Additional message parameters (e.g., {'count': 150, 'total': 1000})
        """
        current_time = time.time()

        # Rate limiting
        if not force and (current_time - self.last_emit_time) < self.emit_interval:
            return

        # Determine status based on step
        if step == 'complete':
            status = 'completed'
        elif step == 'error':
            status = 'error'
        else:
            status = 'processing'

        payload = {
            'uploadId': self.upload_id,
            'step': step,
            'progress': int(progress),
            'messageKey': message_key,
            'status': status
        }

        # Add message parameters if present
        if message_params:
            payload['messageParams'] = message_params

        # Add currentStep and totalSteps for processing/cleaning phases
        if step in ('processing', 'cleaning') and self.total_steps > 0:
            payload['currentStep'] = self.current_step
            payload['totalSteps'] = self.total_steps

        # Add ETA if provided or calculate for current step
        if eta_seconds is not None:
            payload['eta'] = eta_seconds
            payload['etaFormatted'] = self.format_time(eta_seconds)
        else:
            # Try to calculate ETA for current step
            step_eta = self.calculate_step_eta()
            if step_eta is not None:
                payload['eta'] = step_eta
                payload['etaFormatted'] = self.format_time(step_eta)

        try:
            socketio.emit('processing_progress', payload, room=self.upload_id)
            self.last_emit_time = current_time
            eta_text = f" (ETA: {payload.get('etaFormatted', 'N/A')})" if 'eta' in payload else ""
            step_info = f" [{self.current_step}/{self.total_steps}]" if self.total_steps > 0 else ""
            params_text = f" {message_params}" if message_params else ""
            logger.info(f"Progress: {progress}%{step_info} - {message_key}{params_text}{eta_text}")
        except Exception as e:
            logger.error(f"Error emitting progress: {e}")

    @staticmethod
    def format_time(seconds):
        """Format seconds to human-readable format"""
        if seconds is None:
            return "Procjenjujem..."
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
