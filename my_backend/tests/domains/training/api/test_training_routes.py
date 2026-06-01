"""Endpoint-level behaviour tests for training_routes.

Covers the W11-BE1 rate-limit decorator, W11-BE2 UUID guard, W11-BE4
standardized error contract, and the W11-BE5 /status information-leak
fix for the 7 endpoints in training_routes.py.

Test pattern mirrors tests/domains/training/api/test_upload_routes.py
(stub the auth decorators, importlib.reload, walk decorator chain for
rate-limit marker).
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
# Helpers (mirrors tests/domains/training/api/test_upload_routes.py)
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
        # `common` re-exports the decorators, so reload common first then training_routes
        # so each picks up the stubbed decorators.
        import domains.training.api.common as common_module
        importlib.reload(common_module)
        import domains.training.api.training_routes as training_routes
        importlib.reload(training_routes)

        app = Flask(__name__)
        app.config['TESTING'] = True
        limiter.init_app(app)
        app.register_blueprint(training_routes.bp, url_prefix='/api/training')

    # Restore the real decorators on the module so subsequent imports get the
    # production behaviour.
    import domains.training.api.common as common_module
    importlib.reload(common_module)
    import domains.training.api.training_routes as training_routes
    importlib.reload(training_routes)
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
    "generate_datasets",
    "train_models",
    "get_training_status",
    "get_results_summary",
    "get_training_results",
    "get_training_results_details",
    "download_training_arrays",
])
def test_handler_has_rate_limit(handler_name):
    """Every routed handler must declare @limiter.limit(training_limit_string)."""
    import domains.training.api.training_routes as training_routes
    importlib.reload(training_routes)

    handler = getattr(training_routes, handler_name, None)
    assert handler is not None, f"{handler_name} not exported from training_routes"
    assert _has_rate_limit(handler), (
        f"{handler_name} missing @limiter.limit decorator. "
        "Apply @limiter.limit(training_limit_string) between @bp.route and @require_auth."
    )


# ---------------------------------------------------------------------------
# W11-BE2: malformed session_id path param → 400 BAD_UUID before DB hit
# ---------------------------------------------------------------------------

def test_status_invalid_uuid_returns_400(client):
    resp = client.get(
        '/api/training/status/not-a-uuid',
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


def test_results_summary_invalid_uuid_returns_400(client):
    resp = client.get(
        '/api/training/results-summary/not-a-uuid',
        headers=_auth_headers(),
    )
    assert resp.status_code == 400
    body = resp.get_json()
    assert body.get('code') == 'BAD_UUID'


def test_download_arrays_invalid_uuid_returns_400(client):
    resp = client.get(
        '/api/training/download-arrays/not-a-uuid',
        headers=_auth_headers(),
    )
    assert resp.status_code == 400
    body = resp.get_json()
    assert body.get('code') == 'BAD_UUID'


def test_status_session_prefix_uuid_passes_validation(client):
    """session_<uuid> form should NOT be rejected by validator.

    Downstream code may 404/500 (no Supabase mock here), but the validator
    itself must accept this format.
    """
    resp = client.get(
        '/api/training/status/session_a1b2c3d4-e5f6-4a7b-8c9d-0e1f2a3b4c5d',
        headers=_auth_headers(),
    )
    body = resp.get_json() or {}
    assert body.get('code') != 'BAD_UUID', (
        f"session_<uuid> form was rejected by UUID validator: {body}"
    )


# ---------------------------------------------------------------------------
# W11-BE4: error contract — every error response must have 'code'
# ---------------------------------------------------------------------------

def test_generate_datasets_missing_body_returns_structured_error(client):
    """POST /generate-datasets/<sid> with no JSON body → 400 with code in error contract."""
    sid = "session_a1b2c3d4-e5f6-4a7b-8c9d-0e1f2a3b4c5d"
    resp = client.post(
        f'/api/training/generate-datasets/{sid}',
        headers=_auth_headers(),
        json=None,
    )
    body = resp.get_json()
    assert body is not None, f"Response body must be JSON, got: {resp.get_data(as_text=True)}"
    assert body.get('success') is False
    assert 'code' in body, (
        f"Error response must include machine-readable 'code'. Got: {body}"
    )
    assert isinstance(body['code'], str)


def test_train_models_missing_body_returns_structured_error(client):
    """POST /train-models/<sid> with no JSON body → 400 with code in error contract."""
    sid = "session_a1b2c3d4-e5f6-4a7b-8c9d-0e1f2a3b4c5d"
    resp = client.post(
        f'/api/training/train-models/{sid}',
        headers=_auth_headers(),
        json=None,
    )
    body = resp.get_json()
    assert body is not None
    assert body.get('success') is False
    assert 'code' in body


# ---------------------------------------------------------------------------
# W11-BE5: /status 500 path MUST NOT leak message or session_id
# ---------------------------------------------------------------------------

def test_status_500_does_not_leak_message_or_session_id(client):
    """W11-BE5: /status response body must not contain internal exception text
    nor echo the session_id back to the caller."""
    import domains.training.api.training_routes as training_routes

    sid = "session_a1b2c3d4-e5f6-4a7b-8c9d-0e1f2a3b4c5d"

    # Patch get_supabase_client (called from inside the route's try: block)
    # so the route's `except Exception` branch fires with a known marker.
    with patch.object(
        training_routes,
        'get_supabase_client',
        side_effect=Exception("internal stacktrace details that must not leak"),
    ):
        resp = client.get(
            f"/api/training/status/{sid}",
            headers=_auth_headers(),
        )

    assert resp.status_code == 500, (
        f"Expected 500, got {resp.status_code}: {resp.get_data(as_text=True)}"
    )
    body = resp.get_json()
    assert body is not None
    assert body.get('success') is False
    assert body.get('code') == 'INTERNAL_ERROR', (
        f"Expected code=INTERNAL_ERROR, got: {body}"
    )
    # The raw exception text must NOT appear anywhere in the user-facing payload.
    assert "internal stacktrace" not in (body.get('error') or "")
    # W11-BE5 critical assertions:
    assert 'message' not in body, (
        f"W11-BE5: response MUST NOT leak internal exception text via 'message' key. Got: {body}"
    )
    assert 'session_id' not in body, (
        f"W11-BE5: response MUST NOT echo session_id back to client. Got: {body}"
    )


# ---------------------------------------------------------------------------
# W11-BE5/BE7: 500 path on other handlers also matches INTERNAL_ERROR shape
# ---------------------------------------------------------------------------

def test_results_summary_500_returns_internal_error_shape(client):
    """A downstream exception on /results-summary must not leak text."""
    import domains.training.api.training_routes as training_routes

    sid = "session_a1b2c3d4-e5f6-4a7b-8c9d-0e1f2a3b4c5d"

    with patch.object(
        training_routes,
        'get_supabase_client',
        side_effect=Exception("internal stacktrace details that must not leak"),
    ):
        resp = client.get(
            f"/api/training/results-summary/{sid}",
            headers=_auth_headers(),
        )

    assert resp.status_code == 500
    body = resp.get_json()
    assert body is not None
    assert body.get('success') is False
    assert body.get('code') == 'INTERNAL_ERROR'
    assert "internal stacktrace" not in (body.get('error') or "")
