"""Tests for validate_training_session_format — accepts session_<uuid> + bare UUID."""

import pytest
from flask import Flask

from shared.validators.uuid import validate_training_session_format


@pytest.fixture
def app_ctx():
    """jsonify() needs an app context to return a Response."""
    app = Flask(__name__)
    with app.app_context():
        yield


def test_bare_uuid_is_valid(app_ctx):
    assert validate_training_session_format("a1b2c3d4-e5f6-4a7b-8c9d-0e1f2a3b4c5d") is None


def test_session_prefixed_uuid_is_valid(app_ctx):
    assert validate_training_session_format("session_a1b2c3d4-e5f6-4a7b-8c9d-0e1f2a3b4c5d") is None


def test_garbage_returns_400_bad_uuid(app_ctx):
    result = validate_training_session_format("not-a-uuid-at-all")
    assert result is not None
    response, status = result
    assert status == 400
    body = response.get_json()
    assert body["success"] is False
    assert body["code"] == "BAD_UUID"
    assert "session_id" in body["error"]


def test_session_prefix_with_garbage_returns_400(app_ctx):
    result = validate_training_session_format("session_zzzzzzzz-zzzz-zzzz-zzzz-zzzzzzzzzzzz")
    assert result is not None
    _, status = result
    assert status == 400


def test_empty_string_returns_400(app_ctx):
    assert validate_training_session_format("")[1] == 400


def test_none_returns_400(app_ctx):
    assert validate_training_session_format(None)[1] == 400
