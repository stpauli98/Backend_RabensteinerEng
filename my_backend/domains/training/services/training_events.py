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
