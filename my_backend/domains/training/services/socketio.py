"""
SocketIO Progress Callback for Keras Training

This module provides a Keras callback that emits real-time training progress
via SocketIO, enabling frontend to display live epoch updates, loss values,
and ETA estimates.
"""

import time
import logging
from typing import Optional, Any, List, TYPE_CHECKING

logger = logging.getLogger(__name__)

# Debug flag - set to True to trace Socket.IO emissions
DEBUG_SOCKETIO = True

# Type checking import for TrainingProgressTracker
if TYPE_CHECKING:
    from domains.training.services.training_tracker import TrainingProgressTracker

# Check if TensorFlow is available
try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logger.warning("TensorFlow not available - SocketIOProgressCallback will be disabled")


if TENSORFLOW_AVAILABLE:
    class SocketIOProgressCallback(tf.keras.callbacks.Callback):
        """
        Keras callback that emits training progress via SocketIO.

        Emits 'training_metrics' event on each epoch end with:
        - epoch: current epoch number (1-indexed)
        - total_epochs: total number of epochs
        - loss: training loss
        - val_loss: validation loss
        - eta_seconds: estimated time remaining
        - eta_formatted: human-readable ETA string
        - model_name: name of the model being trained

        Also emits 'training_progress' events for training lifecycle:
        - on_train_begin: status='training_started'
        - on_train_end: status='training_completed'

        Optionally persists progress to database via TrainingProgressTracker
        for page refresh recovery.
        """

        def __init__(
            self,
            socketio: Any,
            session_id: str,
            total_epochs: int,
            model_name: str = "Dense",
            progress_tracker: Optional['TrainingProgressTracker'] = None
        ):
            """
            Initialize the SocketIO progress callback.

            Args:
                socketio: SocketIO instance for emitting events
                session_id: Training session ID (used for room targeting)
                total_epochs: Total number of epochs for training
                model_name: Name of the model being trained (e.g., "Dense", "CNN", "LSTM")
                progress_tracker: Optional TrainingProgressTracker for database persistence
            """
            super().__init__()
            self.socketio = socketio
            self.session_id = session_id
            self.total_epochs = total_epochs
            self.model_name = model_name
            self.room = f"training_{session_id}"
            self.epoch_start_time: Optional[float] = None
            self.epoch_times: List[float] = []
            self.training_start_time: Optional[float] = None
            self.progress_tracker = progress_tracker

        def on_train_begin(self, logs=None):
            """Emit training started event."""
            self.training_start_time = time.time()

            # Persist to database via tracker if available
            if self.progress_tracker:
                self.progress_tracker.training_started()

            if self.socketio:
                try:
                    self.socketio.emit('training_progress', {
                        'session_id': self.session_id,
                        'status': 'training_started',
                        'message': f'Training {self.model_name} model...',
                        'progress_percent': 0,
                        'phase': 'training_execution',
                        'model_name': self.model_name,
                        'total_epochs': self.total_epochs
                    }, room=self.room)
                    logger.info(f"🚀 Training started for session {self.session_id} - {self.model_name} ({self.total_epochs} epochs)")
                except Exception as e:
                    logger.error(f"Failed to emit training_started: {e}")

        def on_epoch_begin(self, epoch, logs=None):
            """Track epoch start time for ETA calculation."""
            self.epoch_start_time = time.time()
            self._current_epoch = epoch  # Track for batch progress

        def on_batch_end(self, batch, logs=None):
            """Emit batch-level progress (throttled to every N batches)."""
            logs = logs or {}

            # Throttle: emit only every 10 batches to reduce network traffic
            if batch % 10 != 0:
                return

            if self.socketio:
                try:
                    # Get total batches from params set during fit()
                    total_batches = self.params.get('steps', 0) or self.params.get('samples', 0) // max(self.params.get('batch_size', 1), 1)

                    self.socketio.emit('training_batch_progress', {
                        'session_id': self.session_id,
                        'batch': batch,
                        'total_batches': total_batches,
                        'loss': float(logs.get('loss', 0)),
                        'root_mean_squared_error': float(logs.get('root_mean_squared_error', 0)) if 'root_mean_squared_error' in logs else None,
                        'model_name': self.model_name,
                        'current_epoch': getattr(self, '_current_epoch', 0) + 1,
                        'total_epochs': self.total_epochs
                    }, room=self.room)
                except Exception as e:
                    logger.debug(f"Failed to emit batch progress: {e}")

        def on_epoch_end(self, epoch, logs=None):
            """Emit epoch metrics via SocketIO."""
            logs = logs or {}

            # Calculate epoch duration and ETA
            epoch_duration = time.time() - self.epoch_start_time if self.epoch_start_time else 0
            self.epoch_times.append(epoch_duration)

            # Use weighted average (recent epochs weighted more heavily)
            if len(self.epoch_times) > 1:
                weights = list(range(1, len(self.epoch_times) + 1))
                avg_epoch_time = sum(t * w for t, w in zip(self.epoch_times, weights)) / sum(weights)
            else:
                avg_epoch_time = epoch_duration

            remaining_epochs = self.total_epochs - (epoch + 1)
            eta_seconds = int(avg_epoch_time * remaining_epochs)

            # Calculate progress percentage (0-50% range for Keras training)
            # Post-training phases use 50-100%
            progress_percent = int(((epoch + 1) / self.total_epochs) * 50)

            # Format ETA string
            if eta_seconds >= 3600:
                eta_formatted = f"{eta_seconds // 3600}h {(eta_seconds % 3600) // 60}m"
            elif eta_seconds >= 60:
                eta_formatted = f"{eta_seconds // 60}m {eta_seconds % 60}s"
            else:
                eta_formatted = f"{eta_seconds}s"

            # Persist to database via tracker for page refresh recovery
            if self.progress_tracker:
                self.progress_tracker.epoch_update(
                    epoch=epoch + 1,
                    loss=float(logs.get('loss', 0)),
                    val_loss=float(logs.get('val_loss', 0)),
                    eta_seconds=eta_seconds
                )

            if self.socketio:
                try:
                    # Emit training_metrics event (frontend already listens to this!)
                    self.socketio.emit('training_metrics', {
                        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                        'session_id': self.session_id,
                        'metrics_type': 'real_time',
                        'epoch': epoch + 1,
                        'total_epochs': self.total_epochs,
                        'loss': float(logs.get('loss', 0)),
                        'val_loss': float(logs.get('val_loss', 0)),
                        'accuracy': float(logs.get('accuracy', 0)) if 'accuracy' in logs else None,
                        'val_accuracy': float(logs.get('val_accuracy', 0)) if 'val_accuracy' in logs else None,
                        'root_mean_squared_error': float(logs.get('root_mean_squared_error', 0)) if 'root_mean_squared_error' in logs else None,
                        'val_root_mean_squared_error': float(logs.get('val_root_mean_squared_error', 0)) if 'val_root_mean_squared_error' in logs else None,
                        'model_name': self.model_name,
                        'eta_seconds': eta_seconds,
                        'eta_formatted': eta_formatted,
                        'progress_percent': progress_percent,
                        'epoch_duration': round(epoch_duration, 2)
                    }, room=self.room)

                    logger.info(
                        f"📊 Epoch {epoch + 1}/{self.total_epochs} - "
                        f"loss: {logs.get('loss', 0):.4f}, "
                        f"val_loss: {logs.get('val_loss', 0):.4f}, "
                        f"ETA: {eta_formatted}"
                    )
                except Exception as e:
                    logger.error(f"Failed to emit training_metrics: {e}")

        def on_train_end(self, logs=None):
            """Emit training completed event."""
            if DEBUG_SOCKETIO:
                logger.info(f"[DEBUG_SOCKETIO] 🏁 on_train_end() CALLED - Keras training finished!")

            total_duration = time.time() - self.training_start_time if self.training_start_time else 0

            # Format total duration
            if total_duration >= 3600:
                duration_formatted = f"{int(total_duration // 3600)}h {int((total_duration % 3600) // 60)}m {int(total_duration % 60)}s"
            elif total_duration >= 60:
                duration_formatted = f"{int(total_duration // 60)}m {int(total_duration % 60)}s"
            else:
                duration_formatted = f"{int(total_duration)}s"

            if DEBUG_SOCKETIO:
                logger.info(f"[DEBUG_SOCKETIO] ⏱️ Training duration: {duration_formatted}")
                logger.info(f"[DEBUG_SOCKETIO] 🔄 Calling progress_tracker.training_complete()...")

            # Notify tracker that Keras training is complete (post-training phases follow)
            if self.progress_tracker:
                self.progress_tracker.training_complete()

            if DEBUG_SOCKETIO:
                logger.info(f"[DEBUG_SOCKETIO] 📤 Emitting 'model_training_completed' status...")

            if self.socketio:
                try:
                    self.socketio.emit('training_progress', {
                        'session_id': self.session_id,
                        'status': 'model_training_completed',
                        'message': f'{self.model_name} training completed in {duration_formatted}',
                        'progress_percent': 50,  # Keras done = 50%, post-training uses 50-100%
                        'phase': 'training_execution',
                        'model_name': self.model_name,
                        'epochs_completed': len(self.epoch_times),
                        'total_epochs': self.total_epochs,
                        'training_duration': round(total_duration, 2),
                        'training_duration_formatted': duration_formatted
                    }, room=self.room)
                    logger.info(f"✅ {self.model_name} training completed in {duration_formatted} ({len(self.epoch_times)} epochs)")

                    if DEBUG_SOCKETIO:
                        logger.info(f"[DEBUG_SOCKETIO] ✅ 'model_training_completed' emitted successfully")
                        logger.info(f"[DEBUG_SOCKETIO] ⏰ on_train_end() RETURNING - control goes back to runner...")
                except Exception as e:
                    logger.error(f"Failed to emit training_completed: {e}")

else:
    # Dummy class when TensorFlow is not available
    class SocketIOProgressCallback:
        """Dummy callback when TensorFlow is not available."""
        
        def __init__(self, *args, **kwargs):
            logger.warning("SocketIOProgressCallback is disabled (TensorFlow not available)")
