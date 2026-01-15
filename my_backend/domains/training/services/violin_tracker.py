"""
Progress tracker for violin plot generation.
Emits WebSocket events to existing frontend progress UI.

Created: 2025-12-08
Part of violin plot generation progress tracking implementation.
"""
import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)


class ViolinProgressTracker:
    """
    Tracks progress during violin plot generation and emits updates via WebSocket.
    Uses existing 'dataset_progress' event that frontend already listens to.

    Progress phases:
    - Download (0-30%): Downloading CSV files from Supabase
    - Parsing (30-50%): Reading and processing CSV data
    - Plot Generation (50-95%): Creating violin plots with matplotlib/seaborn
    - Finalization (95-100%): Saving to database
    """

    def __init__(self, socketio, session_id: str):
        """
        Initialize progress tracker.

        Args:
            socketio: Flask-SocketIO instance for emitting events
            session_id: Session identifier for room targeting
        """
        self.socketio = socketio
        self.session_id = session_id
        self.room = f"training_{session_id}"  # Uses existing room format
        self.current_progress = 0
        self.start_time = None  # For ETA calculation

    def emit(self, progress: int, step: str, status: str = 'processing'):
        """
        Emit progress update to frontend via existing event.

        Args:
            progress: Progress percentage (0-100)
            step: Human-readable step description
            status: Status string ('processing', 'completed', 'error')
        """
        self.current_progress = progress

        # Start timing on first emit
        if self.start_time is None:
            self.start_time = time.time()

        # Calculate ETA
        eta_seconds = None
        if progress > 0 and progress < 100:
            elapsed = time.time() - self.start_time
            eta_seconds = int((elapsed / progress) * (100 - progress))

        if self.socketio:
            try:
                self.socketio.emit('dataset_status_update', {
                    'session_id': self.session_id,
                    'progress': progress,
                    'step': step,
                    'status': status,
                    'message': step,
                    'eta_seconds': eta_seconds
                }, room=self.room)
                logger.info(f"Progress emit: {progress}% - {step} (ETA: {eta_seconds}s)")
            except Exception as e:
                logger.error(f"Failed to emit progress: {str(e)}")

    # =========================================================================
    # Phase 1: Download (0-30%)
    # =========================================================================

    def start(self):
        """Emit initial progress."""
        self.emit(5, "violin.progress.starting")

    def downloading_input(self):
        """Emit when starting input CSV download."""
        self.emit(10, "violin.progress.downloadingInput")

    def downloading_output(self):
        """Emit when starting output CSV download."""
        self.emit(25, "violin.progress.downloadingOutput")

    def download_complete(self):
        """Emit when all downloads are complete."""
        self.emit(30, "violin.progress.downloadComplete")

    # =========================================================================
    # Phase 2: Parsing (30-50%)
    # =========================================================================

    def parsing_files(self):
        """Emit when starting CSV parsing."""
        self.emit(40, "violin.progress.parsingFiles")

    def parsing_complete(self):
        """Emit when parsing is complete."""
        self.emit(50, "violin.progress.parsingComplete")

    # =========================================================================
    # Phase 3: Plot Generation (50-95%)
    # =========================================================================

    def generating_input_plot(self):
        """Emit when starting input violin plot generation."""
        self.emit(55, "violin.progress.generatingInput")

    def input_plot_complete(self):
        """Emit when input plot is complete."""
        self.emit(65, "violin.progress.inputComplete")

    def generating_time_plot(self):
        """Emit when starting time violin plot generation."""
        self.emit(70, "violin.progress.generatingTime")

    def time_plot_complete(self):
        """Emit when time plot is complete."""
        self.emit(80, "violin.progress.timeComplete")

    def generating_output_plot(self):
        """Emit when starting output violin plot generation."""
        self.emit(85, "violin.progress.generatingOutput")

    def output_plot_complete(self):
        """Emit when output plot is complete."""
        self.emit(95, "violin.progress.outputComplete")

    # =========================================================================
    # Phase 4: Finalization (95-100%)
    # =========================================================================

    def saving_to_database(self):
        """Emit when saving visualizations to database."""
        self.emit(97, "violin.progress.saving")

    def complete(self):
        """Emit completion status."""
        self.emit(100, "violin.progress.complete", "completed")

    def error(self, message: str):
        """
        Emit error status.

        Args:
            message: Error message to display
        """
        self.emit(self.current_progress, f"violin.progress.error", "error")
