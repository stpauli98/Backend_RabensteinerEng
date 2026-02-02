"""
Progress tracker for model training.
Emits WebSocket events to existing frontend progress UI.
Also persists status to database for page refresh recovery.

Includes heartbeat mechanism to detect orphaned training sessions.

Created: 2026-01-30
Part of model training progress tracking implementation.
"""
import logging
import time
import threading
from typing import Optional, Any
from shared.database.operations import get_supabase_client

logger = logging.getLogger(__name__)

# Debug flag - set to True to trace all emit calls
DEBUG_TRACKER = True

# Heartbeat configuration
HEARTBEAT_INTERVAL_SECONDS = 10  # Send heartbeat every 10 seconds
STALE_THRESHOLD_SECONDS = 60    # Consider training stale after 60 seconds without heartbeat


def cleanup_stale_training_progress():
    """
    Clean up orphaned training_progress entries on backend startup.

    Called during app initialization to remove entries where:
    - status is 'running'
    - updated_at is older than STALE_THRESHOLD_SECONDS

    This handles cases where backend crashed/restarted during training.
    """
    try:
        supabase = get_supabase_client(use_service_role=True)

        # Calculate cutoff time
        cutoff_time = time.strftime(
            '%Y-%m-%dT%H:%M:%SZ',
            time.gmtime(time.time() - STALE_THRESHOLD_SECONDS)
        )

        # Find stale entries
        result = supabase.table('training_progress') \
            .select('session_id, status, overall_progress, updated_at') \
            .eq('status', 'running') \
            .lt('updated_at', cutoff_time) \
            .execute()

        if result.data:
            logger.info(f"[HEARTBEAT] Found {len(result.data)} stale training_progress entries")
            for entry in result.data:
                session_id = entry.get('session_id')
                updated_at = entry.get('updated_at')
                progress = entry.get('overall_progress', 0)

                logger.warning(
                    f"[HEARTBEAT] Cleaning up orphaned training: "
                    f"session={session_id}, progress={progress}%, last_update={updated_at}"
                )

                # Delete the stale entry
                supabase.table('training_progress').delete().eq(
                    'session_id', session_id
                ).execute()

            logger.info(f"[HEARTBEAT] Cleaned up {len(result.data)} orphaned training entries")
        else:
            logger.info("[HEARTBEAT] No stale training_progress entries found")

    except Exception as e:
        logger.error(f"[HEARTBEAT] Failed to cleanup stale entries: {str(e)}")


class TrainingProgressTracker:
    """
    Tracks progress during model training and emits updates via WebSocket.
    Uses existing 'training_progress' event that frontend already listens to.

    Progress phases:
    - Training execution (0-50%): Keras model training with epoch updates
    - Post-training (50-100%): Evaluation, results upload, model saving

    Also persists to database for page refresh recovery.
    Includes heartbeat mechanism to detect orphaned sessions.
    """

    def __init__(
        self,
        socketio: Any,
        session_id: str,
        uuid_session_id: str = None,
        total_epochs: int = 100,
        model_name: str = "Dense"
    ):
        """
        Initialize progress tracker.

        Args:
            socketio: Flask-SocketIO instance for emitting events
            session_id: Session identifier for room targeting
            uuid_session_id: UUID version of session_id for database operations
            total_epochs: Total number of training epochs
            model_name: Name of the model being trained
        """
        self.socketio = socketio
        self.session_id = session_id
        self.uuid_session_id = uuid_session_id
        self.room = f"training_{session_id}"
        self.total_epochs = total_epochs
        self.model_name = model_name
        self.current_progress = 0
        self.start_time = None
        self._last_db_update = 0  # Throttle DB updates
        self._current_epoch = 0
        self._current_loss = 0.0
        self._current_val_loss = 0.0
        self._current_step = "Initializing..."

        # Heartbeat mechanism
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._stop_heartbeat = threading.Event()
        self._is_active = False

    def start_heartbeat(self):
        """Start the heartbeat thread to keep training_progress entry fresh."""
        if self._heartbeat_thread is not None and self._heartbeat_thread.is_alive():
            logger.warning("[HEARTBEAT] Heartbeat thread already running")
            return

        self._stop_heartbeat.clear()
        self._is_active = True
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            daemon=True,
            name=f"heartbeat_{self.session_id[:8]}"
        )
        self._heartbeat_thread.start()
        logger.info(f"[HEARTBEAT] Started heartbeat for session {self.session_id}")

    def stop_heartbeat(self):
        """Stop the heartbeat thread."""
        self._is_active = False
        self._stop_heartbeat.set()

        if self._heartbeat_thread is not None:
            self._heartbeat_thread.join(timeout=2)
            self._heartbeat_thread = None

        logger.info(f"[HEARTBEAT] Stopped heartbeat for session {self.session_id}")

    def _heartbeat_loop(self):
        """Background thread that sends heartbeats to database."""
        while not self._stop_heartbeat.is_set() and self._is_active:
            try:
                self._send_heartbeat()
            except Exception as e:
                logger.error(f"[HEARTBEAT] Error sending heartbeat: {str(e)}")

            # Wait for interval or until stopped
            self._stop_heartbeat.wait(timeout=HEARTBEAT_INTERVAL_SECONDS)

    def _send_heartbeat(self):
        """Send a single heartbeat update to database."""
        if not self.uuid_session_id:
            return

        try:
            supabase = get_supabase_client(use_service_role=True)

            # Only update the timestamp - minimal write
            data = {
                'session_id': str(self.uuid_session_id),
                'updated_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
            }

            supabase.table('training_progress').upsert(
                data, on_conflict='session_id'
            ).execute()

        except Exception as e:
            logger.error(f"[HEARTBEAT] Failed to send heartbeat: {str(e)}")

    def emit(
        self,
        progress: int,
        step: str,
        status: str = 'processing',
        extra_data: Optional[dict] = None
    ):
        """
        Emit progress update to frontend via existing event.
        Also persists to database for page refresh recovery.

        Args:
            progress: Progress percentage (0-100)
            step: Human-readable step description
            status: Status string ('processing', 'completed', 'error')
            extra_data: Optional dict with additional data (epoch, loss, etc.)
        """
        self.current_progress = progress
        self._current_step = step

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
                event_data = {
                    'session_id': self.session_id,
                    'progress_percent': progress,
                    'step': step,
                    'current_step': step,
                    'status': status,
                    'message': step,
                    'eta_seconds': eta_seconds,
                    'phase': 'training_execution',
                    'model_name': self.model_name
                }
                if extra_data:
                    event_data.update(extra_data)

                if DEBUG_TRACKER:
                    logger.info(f"[DEBUG_TRACKER] 📤 EMIT training_progress: progress={progress}%, status={status}, step={step[:50]}, room={self.room}")

                self.socketio.emit('training_progress', event_data, room=self.room)
            except Exception as e:
                logger.error(f"[TRAINING_TRACKER] Failed to emit progress: {str(e)}")

        # Persist to database (throttled to avoid too many writes)
        current_time = time.time()
        should_update_db = (
            current_time - self._last_db_update > 3 or  # At least 3 seconds between updates
            progress == 0 or  # Always update on start
            progress >= 100 or  # Always update on complete
            status in ['completed', 'error']  # Always update on status change
        )

        if should_update_db and self.uuid_session_id:
            self._persist_to_database(progress, step, status, extra_data)

    def _persist_to_database(
        self,
        progress: int,
        step: str,
        status: str,
        extra_data: Optional[dict] = None
    ):
        """
        Persist model training progress to training_progress table.
        This allows the frontend to restore state on page refresh.
        """
        try:
            supabase = get_supabase_client(use_service_role=True)

            # Map status to database status
            # NOTE: Database check constraint only allows: 'running', 'completed', 'error'
            db_status = 'running' if status == 'processing' else status

            # Build model_progress data for restoration
            model_progress = {
                'epoch': self._current_epoch,
                'total_epochs': self.total_epochs,
                'loss': self._current_loss,
                'val_loss': self._current_val_loss,
                'model_name': self.model_name,
                'phase': 'training_execution'
            }
            if extra_data:
                model_progress.update(extra_data)

            data = {
                'session_id': str(self.uuid_session_id),
                'status': db_status,
                'overall_progress': progress,
                'current_step': f'Model training: {step}',
                'total_steps': self.total_epochs,
                'completed_steps': self._current_epoch,
                'model_progress': model_progress,
                'updated_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
            }

            # Upsert to training_progress table
            supabase.table('training_progress').upsert(
                data, on_conflict='session_id'
            ).execute()
            self._last_db_update = time.time()

        except Exception as e:
            logger.error(f"[TRAINING_TRACKER] Failed to persist training progress to DB: {str(e)}")

    def cleanup_database_entry(self):
        """Remove the progress entry when training completes."""
        # Stop heartbeat first
        self.stop_heartbeat()

        if self.uuid_session_id:
            try:
                supabase = get_supabase_client(use_service_role=True)
                supabase.table('training_progress').delete().eq(
                    'session_id', str(self.uuid_session_id)
                ).execute()
            except Exception as e:
                logger.error(
                    f"[TRAINING_TRACKER] Failed to cleanup training progress entry: {str(e)}"
                )

    # =========================================================================
    # Phase 1: Training Execution (0-50%)
    # =========================================================================

    def training_started(self):
        """Emit when training starts. Also starts heartbeat."""
        logger.info(f"[TRAINING_TRACKER] Training started for session {self.session_id}")

        # Start heartbeat to keep entry fresh
        self.start_heartbeat()

        self.emit(
            1,
            f"Starting {self.model_name} training...",
            'processing',
            {'total_epochs': self.total_epochs}
        )

    def epoch_update(
        self,
        epoch: int,
        loss: float,
        val_loss: float,
        eta_seconds: Optional[int] = None
    ):
        """
        Emit progress update for epoch completion.
        Called by SocketIOProgressCallback on each epoch end.

        Args:
            epoch: Current epoch number (1-indexed)
            loss: Training loss
            val_loss: Validation loss
            eta_seconds: Estimated time remaining
        """
        self._current_epoch = epoch
        self._current_loss = loss
        self._current_val_loss = val_loss

        # Training execution is 0-50% of total progress
        # Map epoch progress to 1-50%
        epoch_progress = (epoch / self.total_epochs) * 49 + 1
        progress = int(min(epoch_progress, 50))

        step = f"Epoch {epoch}/{self.total_epochs} - loss: {loss:.4f}, val_loss: {val_loss:.4f}"

        self.emit(
            progress,
            step,
            'processing',
            {
                'epoch': epoch,
                'total_epochs': self.total_epochs,
                'loss': loss,
                'val_loss': val_loss,
                'eta_seconds': eta_seconds
            }
        )

    def training_complete(self):
        """Emit when Keras training is done (before post-training phases)."""
        if DEBUG_TRACKER:
            logger.info(f"[DEBUG_TRACKER] 🏁 training_complete() called - Keras training finished, starting post-training")
        logger.info(
            f"[TRAINING_TRACKER] Keras training completed for session {self.session_id}"
        )
        self.emit(
            50,
            f"{self.model_name} training completed, starting evaluation...",
            'processing'
        )

    # =========================================================================
    # Phase 2: Post-Training (50-100%)
    # =========================================================================

    def evaluating_model(self):
        """Emit when evaluating model performance."""
        if DEBUG_TRACKER:
            logger.info(f"[DEBUG_TRACKER] 📊 evaluating_model() called - 55%")
        self.emit(55, "Evaluating model performance...")

    def preparing_results(self):
        """Emit when preparing results for storage."""
        if DEBUG_TRACKER:
            logger.info(f"[DEBUG_TRACKER] 📦 preparing_results() called - 60%")
        self.emit(60, "Preparing results for storage...")

    def uploading_results(self):
        """Emit when uploading training results to storage."""
        if DEBUG_TRACKER:
            logger.info(f"[DEBUG_TRACKER] ⬆️ uploading_results() called - 65%")
        self.emit(65, "Uploading training results...")

    def upload_progress(self, sub_progress: int, message: str):
        """
        Emit granular progress during upload (65-75%).

        Args:
            sub_progress: Sub-progress 0-100 within upload phase
            message: Status message
        """
        # Map 0-100 to 65-75
        progress = 65 + int((sub_progress / 100) * 10)
        self.emit(progress, message)

    def results_uploaded(self):
        """Emit when results are uploaded."""
        self.emit(75, "Results uploaded successfully")

    def saving_metadata(self):
        """Emit when saving training metadata to database."""
        self.emit(80, "Saving metadata to database...")

    def metadata_saved(self):
        """Emit when metadata is saved."""
        self.emit(85, "Metadata saved to database")

    def uploading_models(self):
        """Emit when uploading trained models to storage."""
        self.emit(90, "Uploading trained models...")

    def models_uploaded(self, model_count: int):
        """Emit when models are uploaded."""
        self.emit(95, f"Uploaded {model_count} model(s)")

    def complete(self):
        """Emit completion status and cleanup database entry."""
        logger.info(f"[TRAINING_TRACKER] Training completed for session {self.session_id}")
        self.emit(100, "Training completed successfully", "completed")
        # Cleanup the progress entry so it doesn't interfere with next training
        self.cleanup_database_entry()

    def error(self, message: str):
        """
        Emit error status and stop heartbeat.

        Args:
            message: Error message to display
        """
        logger.error(f"[TRAINING_TRACKER] Training error for session {self.session_id}: {message}")

        # Stop heartbeat on error
        self.stop_heartbeat()

        self.emit(self.current_progress, f"Error: {message}", "error")
