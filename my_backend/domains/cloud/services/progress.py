"""
Cloud Progress Tracking
Progress tracking for cloud operations with ETA and i18n support
"""
import time
import logging

from core.app_factory import socketio

logger = logging.getLogger(__name__)


class CloudProgressTracker:
    """
    Progress tracking for cloud operations with ETA and i18n support.
    Uses the same pattern as first_processing.py.
    """

    def __init__(self, upload_id):
        self.upload_id = upload_id
        self.start_time = time.time()
        self.last_emit_time = 0
        self.emit_interval = 0.5  # Emit every 500ms

    def calculate_eta(self, progress):
        """Calculate remaining time based on current progress."""
        if progress <= 0:
            return None

        elapsed = time.time() - self.start_time
        if elapsed < 1:  # Wait at least 1 second
            return None

        progress_ratio = progress / 100.0
        remaining_ratio = 1.0 - progress_ratio

        if progress_ratio > 0:
            eta_seconds = int(elapsed * (remaining_ratio / progress_ratio))
            return min(eta_seconds, 3600)  # Max 1 hour
        return None

    def emit(self, step, progress, message_key, message_params=None, force=False):
        """
        Send progress update via SocketIO.

        Args:
            step: Processing phase (assembling, parsing, validating, processing, etc.)
            progress: Progress percentage (0-100)
            message_key: Key for i18n translation on frontend
            message_params: Additional parameters for the message
            force: Ignore rate limiting
        """
        current_time = time.time()

        # Rate limiting
        if not force and (current_time - self.last_emit_time) < self.emit_interval:
            return

        # Determine status
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

        # Add ETA for processing steps
        if status == 'processing':
            eta = self.calculate_eta(progress)
            if eta is not None:
                payload['eta'] = eta
                payload['etaFormatted'] = self.format_time(eta)

        try:
            socketio.emit('processing_progress', payload, room=self.upload_id)
            self.last_emit_time = current_time
            eta_text = f" (ETA: {payload.get('etaFormatted', 'N/A')})" if 'eta' in payload else ""
            logger.info(f"=> Cloud Progress: {progress}% - {message_key}{eta_text}")
        except Exception as e:
            logger.error(f"Error emitting cloud progress: {e}")

    @staticmethod
    def format_time(seconds):
        """Format seconds into readable format."""
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
