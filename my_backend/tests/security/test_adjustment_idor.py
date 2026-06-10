"""Regression: the plain data-adjustment session must be owner-scoped.

Mirrors tests/security/test_cross_tenant_session.py — unit-level, Flask
request context populated manually; no external deps required.
"""
from __future__ import annotations
import pytest
from flask import Flask, g

from domains.adjustments.services.state_manager import (
    adjustment_chunks,
    is_adjustment_owner,
)

USER_A = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
USER_B = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"


@pytest.fixture(autouse=True)
def _clean_state():
    adjustment_chunks.clear()
    yield
    adjustment_chunks.clear()


def test_is_adjustment_owner():
    adjustment_chunks["u1"] = {"user_id": USER_B, "params": {}, "dataframes": {}}
    assert is_adjustment_owner("u1", USER_B) is True
    assert is_adjustment_owner("u1", USER_A) is False
    assert is_adjustment_owner("missing", USER_A) is False


def _unwrap(fn):
    """Strip @require_auth / @require_subscription so we exercise the handler
    body directly (mirrors test_cross_tenant_session.py's __wrapped__ use).
    Both decorators use functools.wraps, so walk the __wrapped__ chain.
    """
    while hasattr(fn, "__wrapped__"):
        fn = fn.__wrapped__
    return fn


def test_adjust_data_rejects_cross_user():
    from domains.adjustments.api.adjustments import adjust_data
    adjust_data = _unwrap(adjust_data)
    adjustment_chunks["u1"] = {"user_id": USER_B, "params": {}, "dataframes": {"f.csv": object()}}
    app = Flask(__name__)
    with app.test_request_context(json={"upload_id": "u1"}):
        g.user_id = USER_A  # attacker
        resp, status = adjust_data()
    assert status == 404
