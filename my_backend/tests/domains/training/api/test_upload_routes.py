"""Endpoint-level behaviour tests for training upload_routes.

Covers the W11-BE1 rate-limit decorator, W11-BE2 UUID guard, and W11-BE4
standardized error contract for the 6 upload endpoints.

Test pattern modelled on tests/domains/cloud/test_cloud_routes.py — stubs
the three auth decorators in shared.auth before importing the route
module so the blueprint can be exercised without Supabase/JWT calls.
"""
import os

# Force testing mode (1000/min rate limit) BEFORE any app import.
os.environ.setdefault('FLASK_ENV', 'testing')

import importlib
from functools import wraps
from unittest.mock import patch

import pytest
from flask import Flask, g


# ---------------------------------------------------------------------------
# Helpers (mirrors tests/domains/cloud/test_cloud_routes.py)
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
            'total_compute_hours': 0,
        }
        g.usage = {'processing_jobs_count': 0, 'processing_count': 0}
        return f(*args, **kwargs)
    return wrapper


def _stub_check_processing_limit(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        return f(*args, **kwargs)
    return wrapper


def _build_app_with_stubs():
    from core.rate_limits import limiter

    with patch('shared.auth.jwt.require_auth', side_effect=_stub_require_auth), \
         patch('shared.auth.subscription.require_subscription', side_effect=_stub_require_subscription), \
         patch('shared.auth.subscription.check_processing_limit', side_effect=_stub_check_processing_limit):
        # `common` re-exports the decorators, so reload common first then upload_routes
        # so each picks up the stubbed decorators.
        import domains.training.api.common as common_module
        importlib.reload(common_module)
        import domains.training.api.upload_routes as upload_routes
        importlib.reload(upload_routes)

        app = Flask(__name__)
        app.config['TESTING'] = True
        limiter.init_app(app)
        app.register_blueprint(upload_routes.bp, url_prefix='/api/training')

    # Restore the real decorators on the module so subsequent imports get the
    # production behaviour.
    import domains.training.api.common as common_module
    importlib.reload(common_module)
    import domains.training.api.upload_routes as upload_routes
    importlib.reload(upload_routes)
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
    """Walk decorator chain to find Flask-Limiter's marker.

    Flask-Limiter attaches one of two markers depending on bind state:
    - ``_rate_limits`` (list of Limit objects) when limiter is bound to an app
    - ``__wrapper-limiter-instance`` (always present, even pre-bind)
    Either is sufficient to confirm the decorator was applied.
    """
    current = fn
    while current is not None:
        if hasattr(current, '_rate_limits'):
            return True
        if hasattr(current, '__dict__') and '__wrapper-limiter-instance' in current.__dict__:
            return True
        current = getattr(current, '__wrapped__', None)
    return False


@pytest.mark.parametrize("handler_name", [
    "upload_chunk",
    "get_csv_files_endpoint",
    "create_csv_file",
    "update_csv_file",
    "delete_csv_file",
    "instant_upload",
])
def test_handler_has_rate_limit(handler_name):
    """Every routed handler must declare @limiter.limit(training_limit_string)."""
    import domains.training.api.upload_routes as upload_routes
    importlib.reload(upload_routes)

    handler = getattr(upload_routes, handler_name, None)
    assert handler is not None, f"{handler_name} not exported from upload_routes"
    assert _has_rate_limit(handler), (
        f"{handler_name} missing @limiter.limit decorator. "
        "Apply @limiter.limit(training_limit_string) between @bp.route and @require_auth."
    )


# ---------------------------------------------------------------------------
# W11-BE2: malformed session_id path param → 400 BAD_UUID before DB hit
# ---------------------------------------------------------------------------

def test_csv_files_get_invalid_uuid_returns_400(client):
    resp = client.get(
        '/api/training/csv-files/not-a-uuid',
        headers=_auth_headers(),
    )
    assert resp.status_code == 400, (
        f"Expected 400 BAD_UUID, got {resp.status_code}: {resp.get_data(as_text=True)}"
    )
    body = resp.get_json()
    assert body is not None
    assert body.get('success') is False
    assert body.get('code') == 'BAD_UUID', (
        f"Expected code=BAD_UUID, got: {body}"
    )


def test_csv_files_get_session_prefix_uuid_passes_validation(client):
    """session_<uuid> form should NOT be rejected by validator.

    Downstream code may 404/500 (no Supabase mock here), but the validator
    itself must accept this format.
    """
    resp = client.get(
        '/api/training/csv-files/session_a1b2c3d4-e5f6-4a7b-8c9d-0e1f2a3b4c5d',
        headers=_auth_headers(),
    )
    body = resp.get_json() or {}
    assert body.get('code') != 'BAD_UUID', (
        f"session_<uuid> form was rejected by UUID validator: {body}"
    )


# ---------------------------------------------------------------------------
# W11-BE4: error contract — every error response must have 'code'
# ---------------------------------------------------------------------------

def test_upload_chunk_missing_body_returns_structured_error(client):
    """POST /upload-chunk with no chunk part → 400 with success/code/error keys."""
    resp = client.post('/api/training/upload-chunk', headers=_auth_headers())
    body = resp.get_json()
    assert body is not None, f"Response body must be JSON, got: {resp.get_data(as_text=True)}"
    assert body.get('success') is False
    assert 'code' in body, (
        f"Error response must include machine-readable 'code'. Got: {body}"
    )
    assert isinstance(body['code'], str)


def test_instant_upload_missing_file_returns_structured_error(client):
    """POST /instant-upload with no file → 400 with code=MISSING_FILE."""
    resp = client.post('/api/training/instant-upload', headers=_auth_headers())
    body = resp.get_json()
    assert body is not None
    assert body.get('success') is False
    assert 'code' in body
    # Should be MISSING_FILE per the standard mapping.
    assert body['code'] == 'MISSING_FILE', (
        f"Expected code=MISSING_FILE for missing file part, got: {body}"
    )


def test_create_csv_file_missing_body_returns_structured_error(client):
    """POST /csv-files with no JSON body → 400 with code=MISSING_BODY."""
    resp = client.post(
        '/api/training/csv-files',
        headers=_auth_headers(),
        json=None,  # empty/none body
    )
    body = resp.get_json()
    assert body is not None
    assert body.get('success') is False
    assert 'code' in body
    assert body['code'] in ('MISSING_BODY', 'BAD_REQUEST'), (
        f"Expected MISSING_BODY/BAD_REQUEST for empty body, got: {body}"
    )


def test_update_csv_file_missing_body_returns_structured_error(client):
    """PUT /csv-files/<id> with no body → 400 with code in error contract."""
    resp = client.put(
        '/api/training/csv-files/file-123',
        headers=_auth_headers(),
        json=None,
    )
    body = resp.get_json()
    assert body is not None
    assert body.get('success') is False
    assert 'code' in body
