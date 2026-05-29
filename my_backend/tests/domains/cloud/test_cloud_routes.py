"""Integration tests for /api/cloud/* routes — security hardening (W10-BE).

Verifies rate limit decorator presence, UUID guard, auth gate, and
processing-quota gate.  Tests run against a minimal Flask app with only the
cloud blueprint registered, so no Docker, Supabase, or Stripe connection is
required.
"""
import os

# Force testing mode (1000/min rate limit) BEFORE any app import.
os.environ.setdefault('FLASK_ENV', 'testing')

from unittest.mock import patch, MagicMock
from functools import wraps

import pytest
from flask import Flask, jsonify, g


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _stub_require_auth(f):
    """Drop-in replacement for @require_auth that sets g.user_id."""
    @wraps(f)
    def wrapper(*args, **kwargs):
        g.user_id = 'user-id-123'
        g.user_email = 'test@example.com'
        g.access_token = 'test-token'
        return f(*args, **kwargs)
    return wrapper


def _stub_require_subscription(f):
    """Drop-in replacement for @require_subscription that sets g.plan."""
    @wraps(f)
    def wrapper(*args, **kwargs):
        g.subscription = {'id': 'sub-1'}
        g.plan = {
            'name': 'pro',
            'max_processing_jobs_per_month': 100,
            'total_compute_hours': 0,
        }
        g.usage = {'processing_jobs_count': 0, 'processing_count': 0}
        return f(*args, **kwargs)
    return wrapper


def _stub_check_processing_limit(f):
    """Drop-in replacement for @check_processing_limit — always allows through."""
    @wraps(f)
    def wrapper(*args, **kwargs):
        return f(*args, **kwargs)
    return wrapper


def _build_app_with_stubs():
    """Create a Flask app with cloud blueprint loaded using auth/sub stubs."""
    from core.rate_limits import limiter
    import importlib

    with patch('shared.auth.jwt.require_auth', side_effect=_stub_require_auth), \
         patch('shared.auth.subscription.require_subscription', side_effect=_stub_require_subscription), \
         patch('shared.auth.subscription.check_processing_limit', side_effect=_stub_check_processing_limit):
        import domains.cloud.api.cloud as cloud_module
        importlib.reload(cloud_module)
        _app = Flask(__name__)
        _app.config['TESTING'] = True
        limiter.init_app(_app)
        _app.register_blueprint(cloud_module.bp, url_prefix='/api/cloud')

    # Reload with real decorators after app is built so module is restored.
    importlib.reload(cloud_module)
    return _app


def _build_app_with_real_auth():
    """Create a Flask app with the real auth decorators (no stubs)."""
    from core.rate_limits import limiter
    import importlib
    import domains.cloud.api.cloud as cloud_module
    importlib.reload(cloud_module)

    _app = Flask(__name__)
    _app.config['TESTING'] = True
    limiter.init_app(_app)
    _app.register_blueprint(cloud_module.bp, url_prefix='/api/cloud')
    return _app


@pytest.fixture
def client():
    _app = _build_app_with_stubs()
    with _app.test_client() as c:
        yield c


@pytest.fixture
def client_real_auth():
    _app = _build_app_with_real_auth()
    with _app.test_client() as c:
        yield c


# ---------------------------------------------------------------------------
# W10-BE4: UUID guard on /clouddata
# ---------------------------------------------------------------------------

def test_clouddata_returns_400_for_bad_uuid(client):
    """W10-BE4: malformed UUID must 400 BAD_UUID before any DB hit."""
    response = client.post(
        '/api/cloud/clouddata',
        json={'session_id': 'not-a-uuid'},
    )
    assert response.status_code == 400, (
        f"Expected 400, got {response.status_code}. "
        "validate_uuid_format may not be applied to /clouddata."
    )
    body = response.get_json()
    assert body.get('success') is False
    assert body.get('code') == 'BAD_UUID', (
        f"Expected code=BAD_UUID, got: {body}. "
        "UUID guard appears absent from /clouddata handler."
    )


# ---------------------------------------------------------------------------
# Auth gate: missing Authorization returns 401
# ---------------------------------------------------------------------------

def test_missing_authorization_returns_401(client_real_auth):
    """W10-BE: missing auth header returns 401 from require_auth."""
    response = client_real_auth.post(
        '/api/cloud/clouddata',
        json={'session_id': '4633c88e-36fb-446d-a17e-90374359875c'},
    )
    assert response.status_code == 401, (
        f"Expected 401 from auth decorator, got {response.status_code}. "
        f"Body: {response.get_json()}"
    )
    body = response.get_json()
    # Auth decorator returns {'error': '...'} — any 401 body with 'error' key is acceptable.
    assert body is not None
    assert 'error' in body


# ---------------------------------------------------------------------------
# W10-BE2: /clouddata must have @check_processing_limit
# ---------------------------------------------------------------------------

def test_clouddata_enforces_processing_limit():
    """W10-BE2: /clouddata must have @check_processing_limit applied.

    Verifies the decorator is present by inspecting the route source.
    """
    import inspect
    import importlib
    import domains.cloud.api.cloud as cloud_module
    importlib.reload(cloud_module)

    src = inspect.getsource(cloud_module.clouddata)
    assert 'check_processing_limit' in src, (
        "@check_processing_limit decoration not found in /clouddata route source. "
        "Add @check_processing_limit between @require_subscription and the view function."
    )


# ---------------------------------------------------------------------------
# W10-BE1: Rate limiter decorator wired to all 4 routes
# ---------------------------------------------------------------------------

def test_rate_limit_decorator_applied_to_all_cloud_routes():
    """W10-BE1: @limiter.limit(cloud_limit_string) must be on all 4 cloud routes."""
    import importlib
    import domains.cloud.api.cloud as cloud_module
    importlib.reload(cloud_module)

    from core.rate_limits import limiter
    from flask import Flask

    _app = Flask(__name__)
    _app.config['TESTING'] = True
    limiter.init_app(_app)
    _app.register_blueprint(cloud_module.bp, url_prefix='/api/cloud')

    routes = {
        'upload_chunk': cloud_module.upload_chunk,
        'complete_redirect': cloud_module.complete_redirect,
        'clouddata': cloud_module.clouddata,
        'interpolate_chunked': cloud_module.interpolate_chunked,
    }
    for name, fn in routes.items():
        found_limit = _has_rate_limit(fn)
        assert found_limit, (
            f"@limiter.limit not found on {name}. "
            "Apply @limiter.limit(cloud_limit_string) between @bp.route and @require_auth."
        )


def _has_rate_limit(fn) -> bool:
    """Walk decorator chain to find Flask-Limiter's _rate_limits marker."""
    current = fn
    while current is not None:
        if hasattr(current, '_rate_limits'):
            return True
        if hasattr(current, '__dict__'):
            if any('limit' in str(k).lower() or 'rate' in str(k).lower()
                   for k in current.__dict__):
                return True
        current = getattr(current, '__wrapped__', None)
    return False
