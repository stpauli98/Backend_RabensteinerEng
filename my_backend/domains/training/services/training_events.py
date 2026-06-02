"""Helpers for constructing and dispatching training Socket.IO events.

Centralises the room convention (clients join ``training_<session_id>`` in
core/socketio_handlers.py) so emit sites cannot drift back to the raw
``session_id`` room, which silently drops events.
"""
from typing import Any, Dict, Optional


def training_room(session_id: str) -> str:
    """The Socket.IO room a client joins for a training session."""
    return f"training_{session_id}"


def emit_training_error(socketio: Any, session_id: str, error: str) -> None:
    """Emit a ``training_error`` event to the room the client actually joined."""
    if not socketio:
        return
    socketio.emit(
        "training_error",
        {
            "session_id": session_id,
            "status": "failed",
            "error": error,
        },
        room=training_room(session_id),
    )


def build_active_training_progress_event(progress: Dict[str, Any], session_id: str) -> Dict[str, Any]:
    """Build the ``training_progress`` payload for an in-flight run.

    Uses ``training_started`` (already running) rather than ``training_starting``
    so a reconnect does not visually regress a near-complete run back to "starting".
    """
    model_progress: Dict[str, Any] = progress.get("model_progress", {}) or {}
    return {
        "session_id": session_id,
        "status": "training_started",
        "phase": model_progress.get("phase", "training_execution"),
        "progress_percent": progress.get("overall_progress", 0),
        "message": progress.get("current_step", "Training in progress..."),
        "epoch": model_progress.get("epoch", 0),
        "total_epochs": model_progress.get("total_epochs", 100),
        "loss": model_progress.get("loss", 0),
        "val_loss": model_progress.get("val_loss", 0),
        "model_name": model_progress.get("model_name", "Dense"),
    }
