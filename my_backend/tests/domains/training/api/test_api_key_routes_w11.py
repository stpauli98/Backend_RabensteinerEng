"""Endpoint-level behaviour tests for training api_key_routes (W11-A T7).

Covers the W11-BE1 rate-limit decorator and W11-BE2 UUID guard for
the 3 endpoints in api_key_routes.py. Note: the DELETE /api-keys/<key_id>
endpoint takes a row id (NOT a session_id) so it MUST NOT reject the
parameter with BAD_UUID — verified by an explicit test below.

Test pattern mirrors tests/domains/training/api/test_session_routes.py
(T6).
"""
import os

# Force testing mode (1000/min rate limit) BEFORE any app import.
os.environ.setdefault('FLASK_ENV', 'testing')

import importlib
from functools import wraps
from unittest.mock import patch, MagicMock

import pytest
from flask import Flask, g


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _stub_require_auth(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        g.user_id = 'user-id-123'
        g.user_email = 'test@example.com'
        g.access_token = 'test-token'
        return f(*args, **kwargs)
    return wrapper


def _stub_require_subscription(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        g.subscription = {'id': 'sub-1'}
        g.plan = {
            'name': 'pro',
            'max_processing_jobs_per_month': 100,
            'max_training_runs_per_month': 100,
            'total_compute_hours': 0,
        }
        g.usage = {
            'processing_jobs_count': 0,
            'processing_count': 0,
            'training_runs_count': 0,
        }
        return f(*args, **kwargs)
    return wrapper


def _stub_check_processing_limit(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        return f(*args, **kwargs)
    return wrapper


def _stub_check_training_limit(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        return f(*args, **kwargs)
    return wrapper


def _build_app_with_stubs():
    from core.rate_limits import limiter

    with patch('shared.auth.jwt.require_auth', side_effect=_stub_require_auth), \
         patch('shared.auth.subscription.require_subscription', side_effect=_stub_require_subscription), \
         patch('shared.auth.subscription.check_processing_limit', side_effect=_stub_check_processing_limit), \
         patch('shared.auth.subscription.check_training_limit', side_effect=_stub_check_training_limit):
        import domains.training.api.common as common_module
        importlib.reload(common_module)
        import domains.training.api.api_key_routes as api_key_routes
        importlib.reload(api_key_routes)

        app = Flask(__name__)
        app.config['TESTING'] = True
        limiter.init_app(app)
        app.register_blueprint(api_key_routes.bp, url_prefix='/api/training')

    # Restore the real decorators on the module so subsequent imports get the
    # production behaviour.
    import domains.training.api.common as common_module
    importlib.reload(common_module)
    import domains.training.api.api_key_routes as api_key_routes
    importlib.reload(api_key_routes)
    return app


@pytest.fixture
def client():
    app = _build_app_with_stubs()
    with app.test_client() as c:
        yield c


def _auth_headers():
    return {'Authorization': 'Bearer test-token'}


# ---------------------------------------------------------------------------
# W11-BE1: rate-limit decorator presence
# ---------------------------------------------------------------------------

def _has_rate_limit(fn) -> bool:
    """Walk decorator chain to find Flask-Limiter's marker."""
    current = fn
    while current is not None:
        if hasattr(current, '_rate_limits'):
            return True
        if hasattr(current, '__dict__') and '__wrapper-limiter-instance' in current.__dict__:
            return True
        current = getattr(current, '__wrapped__', None)
    return False


@pytest.mark.parametrize("handler_name", [
    "generate_api_key",
    "list_api_keys",
    "revoke_api_key",
])
def test_handler_has_rate_limit(handler_name):
    """Every routed handler must declare @limiter.limit(training_limit_string)."""
    import domains.training.api.api_key_routes as api_key_routes
    importlib.reload(api_key_routes)

    handler = getattr(api_key_routes, handler_name, None)
    assert handler is not None, f"{handler_name} not exported from api_key_routes"
    assert _has_rate_limit(handler), (
        f"{handler_name} missing @limiter.limit decorator. "
        "Apply @limiter.limit(training_limit_string) between @bp.route and @require_auth."
    )


# ---------------------------------------------------------------------------
# W11-BE2: malformed session_id path param → 400 BAD_UUID before DB hit
# ---------------------------------------------------------------------------

def test_generate_api_key_invalid_uuid_returns_400_bad_uuid(client):
    """POST /api-keys/<sid> rejects malformed UUID with 400 BAD_UUID."""
    resp = client.post(
        '/api/training/api-keys/not-a-uuid',
        headers=_auth_headers(),
        json={'name': 'test-key'},
    )
    assert resp.status_code == 400, (
        f"Expected 400 BAD_UUID, got {resp.status_code}: {resp.get_data(as_text=True)}"
    )
    body = resp.get_json()
    assert body is not None
    assert body.get('success') is False
    assert body.get('code') == 'BAD_UUID', f"Expected BAD_UUID, got: {body}"


def test_list_api_keys_invalid_uuid_returns_400_bad_uuid(client):
    """GET /api-keys/<sid> rejects malformed UUID with 400 BAD_UUID."""
    resp = client.get(
        '/api/training/api-keys/not-a-uuid',
        headers=_auth_headers(),
    )
    assert resp.status_code == 400
    body = resp.get_json()
    assert body is not None
    assert body.get('code') == 'BAD_UUID'


# ---------------------------------------------------------------------------
# /api-keys/<key_id> DELETE — key_id is NOT a session_id, do NOT BAD_UUID it
# ---------------------------------------------------------------------------

def test_revoke_api_key_arbitrary_key_id_not_rejected_as_bad_uuid(client):
    """DELETE /api-keys/<key_id> path param is a row id, not a session UUID.

    The handler must NOT short-circuit with BAD_UUID. It should reach the
    Supabase lookup (which we mock to return empty) and then surface a 404.
    """
    import domains.training.api.api_key_routes as api_key_routes

    fake_supabase = MagicMock()
    # .select(...).eq(...).eq(...).is_(...).limit(...).execute()
    fake_supabase.table.return_value.select.return_value.eq.return_value.eq.return_value.is_.return_value.limit.return_value.execute.return_value.data = []

    with patch.object(api_key_routes, 'get_supabase_client', return_value=fake_supabase):
        resp = client.delete(
            '/api/training/api-keys/some-key-id-that-is-not-a-uuid',
            headers=_auth_headers(),
        )

    body = resp.get_json() or {}
    assert body.get('code') != 'BAD_UUID', (
        f"DELETE /api-keys/<key_id> must not reject key_id with BAD_UUID: {body}"
    )
    # Should be 404 (key not found) — confirms route progressed past the guard
    assert resp.status_code == 404, (
        f"Expected 404 (key not found), got {resp.status_code}: {body}"
    )
    assert body.get('code') == 'API_KEY_NOT_FOUND', f"Expected API_KEY_NOT_FOUND, got: {body}"
