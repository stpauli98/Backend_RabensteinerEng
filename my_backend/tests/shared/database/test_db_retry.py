"""retry_database_operation must retry transient connection drops (not just DNS
timeouts). Supabase Free tier closes idle pooled connections, so the first call
after an idle period raises httpcore.RemoteProtocolError("Server disconnected") —
which previously fell through to an immediate 500 instead of being retried.
"""
import pytest

from shared.database.session import (
    retry_database_operation,
    _is_transient_db_error,
)
from shared.database.exceptions import DatabaseError


def test_classifier_marks_connection_drops_transient():
    assert _is_transient_db_error("Server disconnected") is True
    assert _is_transient_db_error("httpcore.RemoteProtocolError: Server disconnected") is True
    assert _is_transient_db_error("Connection reset by peer") is True
    assert _is_transient_db_error("Lookup timed out") is True
    assert _is_transient_db_error("read timeout") is True


def test_classifier_marks_deterministic_errors_non_transient():
    assert _is_transient_db_error("duplicate key value violates unique constraint") is False
    assert _is_transient_db_error("permission denied for table sessions") is False
    assert _is_transient_db_error("null value in column violates not-null") is False


def test_retries_then_succeeds_on_server_disconnected():
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise Exception("Server disconnected")
        return "ok"

    # initial_delay=0 keeps the test fast (backoff = 0 * 2**attempt = 0)
    result = retry_database_operation(flaky, max_retries=5, initial_delay=0)
    assert result == "ok"
    assert calls["n"] == 3


def test_non_transient_raises_immediately_without_retry():
    calls = {"n": 0}

    def deterministic():
        calls["n"] += 1
        raise Exception("duplicate key value violates unique constraint")

    with pytest.raises(DatabaseError):
        retry_database_operation(deterministic, max_retries=5, initial_delay=0)
    assert calls["n"] == 1  # not retried


def test_transient_exhausts_retries_then_raises():
    calls = {"n": 0}

    def always_drops():
        calls["n"] += 1
        raise Exception("Server disconnected")

    with pytest.raises(DatabaseError):
        retry_database_operation(always_drops, max_retries=2, initial_delay=0)
    assert calls["n"] == 3  # initial try + 2 retries
