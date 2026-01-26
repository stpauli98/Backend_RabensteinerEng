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
    - Download (0-25%): Downloading CSV files from Supabase
    - Parsing (25-40%): Reading and processing CSV data
    - Plot Generation (40-75%): Creating violin plots with matplotlib/seaborn
    - Dataset Calculation (75-95%): Calculating n_dat (training samples count)
    - Finalization (95-100%): Complete
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
    # Phase 1: Download (0-25%)
    # =========================================================================

    def start(self):
        """Emit initial progress."""
        self.emit(5, "violin.progress.starting")

    def downloading_input(self):
        """Emit when starting input CSV download."""
        self.emit(10, "violin.progress.downloadingInput")

    def downloading_output(self):
        """Emit when starting output CSV download."""
        self.emit(20, "violin.progress.downloadingOutput")

    def download_complete(self):
        """Emit when all downloads are complete."""
        self.emit(25, "violin.progress.downloadComplete")

    # =========================================================================
    # Phase 2: Parsing (25-40%)
    # =========================================================================

    def parsing_files(self):
        """Emit when starting CSV parsing."""
        self.emit(30, "violin.progress.parsingFiles")

    def parsing_complete(self):
        """Emit when parsing is complete."""
        self.emit(40, "violin.progress.parsingComplete")

    # =========================================================================
    # Phase 3: Plot Generation (40-75%)
    # =========================================================================

    def generating_input_plot(self):
        """Emit when starting input violin plot generation."""
        self.emit(45, "violin.progress.generatingInput")

    def input_plot_complete(self):
        """Emit when input plot is complete."""
        self.emit(55, "violin.progress.inputComplete")

    def generating_time_plot(self):
        """Emit when starting time violin plot generation."""
        self.emit(58, "violin.progress.generatingTime")

    def time_plot_complete(self):
        """Emit when time plot is complete."""
        self.emit(65, "violin.progress.timeComplete")

    def generating_output_plot(self):
        """Emit when starting output violin plot generation."""
        self.emit(68, "violin.progress.generatingOutput")

    def output_plot_complete(self):
        """Emit when output plot is complete."""
        self.emit(75, "violin.progress.outputComplete")

    # =========================================================================
    # Phase 4: Dataset Count Calculation (75-95%)
    # =========================================================================

    def calculating_dataset_count(self):
        """Emit when starting n_dat calculation."""
        self.emit(80, "violin.progress.calculatingDatasets")

    def dataset_count_complete(self, n_dat: int = 0):
        """Emit when n_dat calculation is complete."""
        self.emit(95, "violin.progress.datasetsComplete")

    # =========================================================================
    # Phase 5: Finalization (95-100%)
    # =========================================================================

    def saving_to_database(self):
        """Emit when saving violin plots to database."""
        self.emit(97, "violin.progress.savingToDatabase")

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
