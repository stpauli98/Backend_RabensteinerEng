"""
Progress tracker for violin plot generation.
Emits WebSocket events to existing frontend progress UI.
Also persists status to database for page refresh recovery.

Created: 2025-12-08
Part of violin plot generation progress tracking implementation.
"""
import logging
import time
from typing import Optional
from shared.database.operations import get_supabase_client

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

    def __init__(self, socketio, session_id: str, uuid_session_id: str = None):
        """
        Initialize progress tracker.

        Args:
            socketio: Flask-SocketIO instance for emitting events
            session_id: Session identifier for room targeting
            uuid_session_id: UUID version of session_id for database operations
        """
        self.socketio = socketio
        self.session_id = session_id
        self.uuid_session_id = uuid_session_id
        self.room = f"training_{session_id}"  # Uses existing room format
        self.current_progress = 0
        self.start_time = None  # For ETA calculation
        self._last_db_update = 0  # Throttle DB updates

    def emit(self, progress: int, step: str, status: str = 'processing'):
        """
        Emit progress update to frontend via existing event.
        Also persists to database for page refresh recovery.

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

        # Emit via Socket.IO
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
            except Exception as e:
                logger.error(f"Failed to emit progress: {str(e)}")

        # Persist to database (throttled to avoid too many writes)
        current_time = time.time()
        should_update_db = (
            current_time - self._last_db_update > 5 or  # At least 5 seconds between updates
            progress == 0 or  # Always update on start
            progress >= 100 or  # Always update on complete
            status in ['completed', 'error']  # Always update on status change
        )

        if should_update_db and self.uuid_session_id:
            self._persist_to_database(progress, step, status)

    def _persist_to_database(self, progress: int, step: str, status: str):
        """
        Persist dataset generation progress to training_progress table.
        This allows the frontend to restore state on page refresh.
        """
        try:
            supabase = get_supabase_client(use_service_role=True)

            # Map status to database status
            db_status = 'running' if status == 'processing' else status

            data = {
                'session_id': str(self.uuid_session_id),
                'status': db_status,
                'overall_progress': progress,
                'current_step': f'Dataset generation: {step}',
                'total_steps': 7,
                'completed_steps': int(progress / 14),  # Approximate
                'updated_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
            }

            # Upsert to training_progress table (specify conflict column for proper upsert)
            supabase.table('training_progress').upsert(data, on_conflict='session_id').execute()
            self._last_db_update = time.time()

        except Exception as e:
            logger.error(f"Failed to persist dataset progress to DB: {str(e)}")

    def cleanup_database_entry(self):
        """Remove the progress entry when dataset generation completes."""
        if self.uuid_session_id:
            try:
                supabase = get_supabase_client(use_service_role=True)
                supabase.table('training_progress').delete().eq('session_id', str(self.uuid_session_id)).execute()
            except Exception as e:
                logger.error(f"Failed to cleanup dataset progress entry: {str(e)}")

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
    # Phase 4: Dataset Count Calculation (80-95%) - Granular tracking
    # =========================================================================

    def calculating_dataset_count(self):
        """Emit when starting n_dat calculation."""
        self.emit(80, "violin.progress.calculatingDatasets")

    def ndat_loading_data(self):
        """Emit when loading data for n_dat calculation."""
        self.emit(82, "violin.progress.ndatLoading")

    def ndat_transforming(self):
        """Emit when transforming data for n_dat."""
        self.emit(85, "violin.progress.ndatTransforming")

    def ndat_creating_arrays(self):
        """Emit when creating training arrays (slowest step)."""
        self.emit(88, "violin.progress.ndatCreatingArrays")

    def ndat_arrays_complete(self):
        """Emit when training arrays are created."""
        self.emit(93, "violin.progress.ndatArraysComplete")

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
        """Emit completion status and cleanup database entry."""
        self.emit(100, "violin.progress.complete", "completed")
        # Cleanup the progress entry so it doesn't interfere with training status
        self.cleanup_database_entry()

    def error(self, message: str):
        """
        Emit error status.

        Args:
            message: Error message to display
        """
        self.emit(self.current_progress, f"violin.progress.error", "error")
