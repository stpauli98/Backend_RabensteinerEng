from unittest.mock import MagicMock
from domains.training.services.training_events import emit_training_error


def test_emit_training_error_targets_the_joined_room():
    socketio = MagicMock()
    emit_training_error(socketio, "sess_abc", "boom")

    socketio.emit.assert_called_once()
    args, kwargs = socketio.emit.call_args
    assert args[0] == "training_error"
    assert args[1]["session_id"] == "sess_abc"
    assert args[1]["status"] == "failed"
    assert args[1]["error"] == "boom"
    # Must match the room the client joined: f"training_{session_id}"
    assert kwargs["room"] == "training_sess_abc"


def test_emit_training_error_is_noop_without_socketio():
    # Should not raise when socketio is None
    emit_training_error(None, "sess_abc", "boom")
