"""FIX-4 (Bug 3): training_tracker DB-constraint mismatch regression.

The training_progress table has CHECK constraint:

    status IN ('idle', 'running', 'completed', 'failed', 'abandoned')

Pre-fix, ``TrainingProgressTracker._persist_to_database`` wrote
``status='error'`` directly (e.g. when ``self.error()`` was called by the
orchestrator's failure path), triggering Postgres 23514 violations on
every error path and silently losing the error state.

These tests pin the mapping behaviour: 'processing' → 'running',
'error' → 'failed', terminal statuses pass through unchanged.
"""

from unittest.mock import MagicMock, patch

import pytest

from domains.training.services.training_tracker import TrainingProgressTracker


@pytest.fixture
def tracker():
    """Build a tracker without spinning up the heartbeat thread."""
    t = TrainingProgressTracker(
        socketio=None,
        session_id="test_session",
        uuid_session_id="a1b2c3d4-e5f6-4a7b-8c9d-0e1f2a3b4c5d",
        total_epochs=10,
        model_name="Dense",
    )
    return t


def _captured_upsert_payload(supabase_mock):
    """Pull the dict passed to .upsert() on the most recent call.

    The chain is supabase.table('training_progress').upsert(data, ...).
    """
    table_mock = supabase_mock.table.return_value
    upsert_calls = table_mock.upsert.call_args_list
    assert upsert_calls, "Expected at least one upsert call"
    # First positional arg of the most recent call is the row dict.
    return upsert_calls[-1].args[0]


def test_persist_maps_error_status_to_failed(tracker):
    """FIX-4 Bug 3: status='error' must be written as 'failed' (canonical DB name)."""
    fake_supabase = MagicMock()
    with patch(
        'domains.training.services.training_tracker.get_supabase_client',
        return_value=fake_supabase,
    ):
        tracker._persist_to_database(
            progress=45,
            step="Model training failed",
            status="error",
        )

    payload = _captured_upsert_payload(fake_supabase)
    assert payload['status'] == 'failed', (
        f"FIX-4 Bug 3: 'error' must be mapped to 'failed' before DB write "
        f"(CHECK constraint rejects 'error'). Got payload: {payload}"
    )


def test_persist_maps_processing_status_to_running(tracker):
    """'processing' is the tracker's in-flight signal; DB stores it as 'running'."""
    fake_supabase = MagicMock()
    with patch(
        'domains.training.services.training_tracker.get_supabase_client',
        return_value=fake_supabase,
    ):
        tracker._persist_to_database(
            progress=10,
            step="Epoch 1/10",
            status="processing",
        )

    payload = _captured_upsert_payload(fake_supabase)
    assert payload['status'] == 'running'


@pytest.mark.parametrize("terminal_status", ['completed', 'failed', 'idle', 'abandoned', 'running'])
def test_persist_passes_through_canonical_db_statuses(tracker, terminal_status):
    """Canonical DB statuses (the CHECK-constraint members) must not be remapped."""
    fake_supabase = MagicMock()
    with patch(
        'domains.training.services.training_tracker.get_supabase_client',
        return_value=fake_supabase,
    ):
        tracker._persist_to_database(
            progress=100 if terminal_status == 'completed' else 50,
            step="done",
            status=terminal_status,
        )

    payload = _captured_upsert_payload(fake_supabase)
    assert payload['status'] == terminal_status, (
        f"Canonical status '{terminal_status}' must pass through unchanged. "
        f"Got payload: {payload}"
    )


def test_persist_never_writes_legacy_error_string(tracker):
    """No code path may write the literal string 'error' to training_progress.status.

    Defends against future regressions where a caller passes status='error'
    and the mapping is bypassed.
    """
    fake_supabase = MagicMock()
    with patch(
        'domains.training.services.training_tracker.get_supabase_client',
        return_value=fake_supabase,
    ):
        # Run the full emit() path the orchestrator triggers on failure.
        tracker.emit(
            progress=42,
            step="Boom",
            status="error",
        )

    payload = _captured_upsert_payload(fake_supabase)
    assert payload['status'] != 'error', (
        "training_progress.status='error' violates the DB CHECK constraint "
        "and was historically lost via 23514. Must be 'failed' instead."
    )
    assert payload['status'] == 'failed'
