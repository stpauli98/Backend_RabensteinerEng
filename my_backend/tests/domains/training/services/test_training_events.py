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


from domains.training.services.training_events import build_active_training_progress_event


def test_active_training_event_uses_running_status_and_real_progress():
    progress_row = {
        "overall_progress": 80,
        "current_step": "Saving results",
        "model_progress": {
            "phase": "post_training",
            "epoch": 5,
            "total_epochs": 5,
            "loss": 0.1,
            "val_loss": 0.2,
            "model_name": "LSTM",
        },
    }
    event = build_active_training_progress_event(progress_row, "sess_abc")

    # Must NOT regress an in-flight run to 'training_starting'
    assert event["status"] == "training_started"
    assert event["progress_percent"] == 80
    assert event["phase"] == "post_training"
    assert event["session_id"] == "sess_abc"
    assert event["model_name"] == "LSTM"
