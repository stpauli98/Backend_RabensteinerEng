"""Endpoint-level behaviour tests for training session_routes.

Covers the W11-BE1 rate-limit decorator, W11-BE2 UUID guard, W11-BE4
standardized error contract, and the W11-BE5 information-leak fix for
the 15 endpoints in session_routes.py.

Test pattern mirrors tests/domains/training/api/test_model_routes.py
(T5) and tests/domains/training/api/test_training_routes.py (T4).
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
# Helpers (mirrors tests/domains/training/api/test_model_routes.py)
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
        # `common` re-exports the decorators, so reload common first then session_routes
        # so each picks up the stubbed decorators.
        import domains.training.api.common as common_module
        importlib.reload(common_module)
        import domains.training.api.session_routes as session_routes
        importlib.reload(session_routes)

        app = Flask(__name__)
        app.config['TESTING'] = True
        limiter.init_app(app)
        app.register_blueprint(session_routes.bp, url_prefix='/api/training')

    # Restore the real decorators on the module so subsequent imports get the
    # production behaviour.
    import domains.training.api.common as common_module
    importlib.reload(common_module)
    import domains.training.api.session_routes as session_routes
    importlib.reload(session_routes)
    return app


@pytest.fixture
def client():
    """Build the test app and bypass the FIX-1 ownership check on
    /get-time-info and /get-zeitschritte. Ownership semantics are covered
    by test_idor_multi_user.py; this file targets the W11-BE error
    contract."""
    app = _build_app_with_stubs()
    import domains.training.api.session_routes as session_routes
    with patch.object(session_routes, 'create_or_get_session_uuid',
                      return_value='a1b2c3d4-e5f6-4a7b-8c9d-0e1f2a3b4c5d'), \
         patch.object(session_routes, 'assert_session_ownership',
                      return_value='a1b2c3d4-e5f6-4a7b-8c9d-0e1f2a3b4c5d'):
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
    "init_session",
    "finalize_session_endpoint",
    "list_sessions",
    "get_session_endpoint",
    "get_session_from_database_endpoint",
    "session_status",
    "delete_session_endpoint",
    "delete_all_sessions_endpoint",
    "create_database_session_endpoint",
    "get_session_uuid_endpoint",
    "save_time_info_endpoint",
    "save_zeitschritte_endpoint",
    "get_time_info_endpoint",
    "get_zeitschritte_endpoint",
    "change_session_name_endpoint",
])
def test_handler_has_rate_limit(handler_name):
    """Every routed handler must declare @limiter.limit(training_limit_string)."""
    import domains.training.api.session_routes as session_routes
    importlib.reload(session_routes)

    handler = getattr(session_routes, handler_name, None)
    assert handler is not None, f"{handler_name} not exported from session_routes"
    assert _has_rate_limit(handler), (
        f"{handler_name} missing @limiter.limit decorator. "
        "Apply @limiter.limit(training_limit_string) between @bp.route and @require_auth."
    )


# ---------------------------------------------------------------------------
# W11-BE2: malformed session_id path param → 400 BAD_UUID before DB hit
# ---------------------------------------------------------------------------

def test_get_session_invalid_uuid_returns_400(client):
    """GET /session/<sid> rejects malformed UUID with 400 BAD_UUID."""
    resp = client.get(
        '/api/training/session/not-a-uuid',
        headers=_auth_headers(),
    )
    assert resp.status_code == 400, (
        f"Expected 400 BAD_UUID, got {resp.status_code}: {resp.get_data(as_text=True)}"
    )
    body = resp.get_json()
    assert body is not None
    assert body.get('success') is False
    assert body.get('code') == 'BAD_UUID', f"Expected code=BAD_UUID, got: {body}"


def test_get_session_database_invalid_uuid_returns_400(client):
    """GET /session/<sid>/database rejects malformed UUID with 400 BAD_UUID."""
    resp = client.get(
        '/api/training/session/not-a-uuid/database',
        headers=_auth_headers(),
    )
    assert resp.status_code == 400
    body = resp.get_json()
    assert body is not None
    assert body.get('code') == 'BAD_UUID'


def test_delete_session_invalid_uuid_returns_400(client):
    """POST /session/<sid>/delete rejects malformed UUID with 400 BAD_UUID."""
    resp = client.post(
        '/api/training/session/not-a-uuid/delete',
        headers=_auth_headers(),
    )
    assert resp.status_code == 400
    body = resp.get_json()
    assert body is not None
    assert body.get('code') == 'BAD_UUID'


def test_save_time_info_invalid_uuid_in_body_returns_400(client):
    """POST /save-time-info rejects malformed UUID in body with 400 BAD_UUID."""
    resp = client.post(
        '/api/training/save-time-info',
        headers=_auth_headers(),
        json={'sessionId': 'not-a-uuid', 'timeInfo': {'jahr': True}},
    )
    assert resp.status_code == 400, (
        f"Expected 400 BAD_UUID, got {resp.status_code}: {resp.get_data(as_text=True)}"
    )
    body = resp.get_json()
    assert body is not None
    assert body.get('success') is False
    assert body.get('code') == 'BAD_UUID'


def test_session_name_change_invalid_uuid_in_body_returns_400(client):
    """POST /session-name-change rejects malformed UUID in body with 400 BAD_UUID."""
    resp = client.post(
        '/api/training/session-name-change',
        headers=_auth_headers(),
        json={'sessionId': 'not-a-uuid', 'sessionName': 'foo'},
    )
    assert resp.status_code == 400
    body = resp.get_json()
    assert body is not None
    assert body.get('code') == 'BAD_UUID'


def test_session_prefix_uuid_passes_validation_on_get_session(client):
    """session_<uuid> form must NOT be rejected by validator.

    Downstream may 500 (no Supabase mock), but validator must accept.
    """
    sid = "session_a1b2c3d4-e5f6-4a7b-8c9d-0e1f2a3b4c5d"
    resp = client.get(
        f'/api/training/session/{sid}',
        headers=_auth_headers(),
    )
    body = resp.get_json() or {}
    assert body.get('code') != 'BAD_UUID', (
        f"session_<uuid> form was rejected by UUID validator: {body}"
    )


# ---------------------------------------------------------------------------
# W11-BE4/BE5/BE7: 500 path returns INTERNAL_ERROR shape, no exception text leak
# ---------------------------------------------------------------------------

def test_list_sessions_500_returns_internal_error_shape(client):
    """A downstream exception on /list-sessions must not leak text."""
    import domains.training.api.session_routes as session_routes

    with patch.object(
        session_routes,
        'get_sessions_list',
        side_effect=Exception("internal stacktrace details that must not leak"),
    ):
        resp = client.get(
            '/api/training/list-sessions',
            headers=_auth_headers(),
        )

    assert resp.status_code == 500, (
        f"Expected 500, got {resp.status_code}: {resp.get_data(as_text=True)}"
    )
    body = resp.get_json()
    assert body is not None
    assert body.get('success') is False
    assert body.get('code') == 'INTERNAL_ERROR', f"Expected INTERNAL_ERROR, got: {body}"
    assert "internal stacktrace" not in (body.get('error') or "")
    assert "internal stacktrace" not in (body.get('message') or "")
    assert "internal stacktrace" not in (body.get('suggestion') or "")


def test_get_time_info_500_returns_internal_error_shape(client):
    """A downstream exception on /get-time-info/<sid> must not leak text."""
    import domains.training.api.session_routes as session_routes

    sid = "a1b2c3d4-e5f6-4a7b-8c9d-0e1f2a3b4c5d"

    with patch.object(
        session_routes,
        'get_time_info_data',
        side_effect=Exception("internal stacktrace details that must not leak"),
    ):
        resp = client.get(
            f'/api/training/get-time-info/{sid}',
            headers=_auth_headers(),
        )

    assert resp.status_code == 500
    body = resp.get_json()
    assert body is not None
    assert body.get('success') is False
    assert body.get('code') == 'INTERNAL_ERROR'
    assert "internal stacktrace" not in (body.get('error') or "")
    assert "internal stacktrace" not in (body.get('message') or "")


def test_save_time_info_missing_body_returns_400_missing_body(client):
    """POST /save-time-info with no JSON body → 400 MISSING_BODY."""
    resp = client.post(
        '/api/training/save-time-info',
        headers=_auth_headers(),
        data='',
        content_type='application/json',
    )
    body = resp.get_json()
    assert body is not None
    assert body.get('success') is False
    assert 'code' in body
    # Either MISSING_BODY (no body) or BAD_REQUEST (JSON parse) is acceptable
    assert body.get('code') in ('MISSING_BODY', 'BAD_REQUEST'), (
        f"Expected MISSING_BODY/BAD_REQUEST, got: {body}"
    )


# ---------------------------------------------------------------------------
# W11-BE6: FORBIDDEN 403 when session ownership is violated
# ---------------------------------------------------------------------------

def test_get_session_database_ownership_violation_returns_403_forbidden(client):
    """SessionOwnershipError on /session/<sid>/database must return 403 FORBIDDEN."""
    import domains.training.api.session_routes as session_routes
    from shared.auth.ownership import SessionOwnershipError

    sid = "a1b2c3d4-e5f6-4a7b-8c9d-0e1f2a3b4c5d"

    with patch.object(
        session_routes,
        'resolve_session_id',
        return_value=(sid, sid),
    ), patch.object(
        session_routes,
        'assert_session_ownership',
        side_effect=SessionOwnershipError(f"session {sid} not owned by user-id-123"),
    ):
        resp = client.get(
            f'/api/training/session/{sid}/database',
            headers=_auth_headers(),
        )

    assert resp.status_code == 403, (
        f"Expected 403 FORBIDDEN, got {resp.status_code}: {resp.get_data(as_text=True)}"
    )
    body = resp.get_json()
    assert body is not None
    assert body.get('success') is False
    assert body.get('code') == 'FORBIDDEN', f"Expected FORBIDDEN, got: {body}"


# ---------------------------------------------------------------------------
# /delete-all-sessions: CONFIRMATION_REQUIRED when confirm flag missing
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# /create-database-session: conditional UUID guard
# ---------------------------------------------------------------------------

def test_create_database_session_omitted_sessionId_passes_validation(client):
    """When body omits sessionId, route should NOT reject with BAD_UUID
    (service auto-generates the id)."""
    import domains.training.api.session_routes as session_routes

    with patch.object(
        session_routes,
        'create_database_session',
        return_value='auto-generated-uuid',
    ):
        r = client.post(
            "/api/training/create-database-session",
            headers=_auth_headers(),
            json={},
        )
    body = r.get_json() or {}
    assert body.get('code') != 'BAD_UUID', "empty body must not trigger BAD_UUID"


def test_create_database_session_malformed_sessionId_returns_400_bad_uuid(client):
    """When body PROVIDES a malformed sessionId, route MUST return BAD_UUID."""
    r = client.post(
        "/api/training/create-database-session",
        headers=_auth_headers(),
        json={"sessionId": "not-a-uuid-at-all"},
    )
    assert r.status_code == 400
    body = r.get_json()
    assert body['code'] == 'BAD_UUID'


def test_delete_all_sessions_without_confirm_returns_confirmation_required(client):
    """POST /delete-all-sessions without confirm_delete_all=True must return
    400 CONFIRMATION_REQUIRED via the ValueError raised by the service layer.
    """
    resp = client.post(
        '/api/training/delete-all-sessions',
        headers=_auth_headers(),
        json={},
    )

    assert resp.status_code == 400, (
        f"Expected 400, got {resp.status_code}: {resp.get_data(as_text=True)}"
    )
    body = resp.get_json()
    assert body is not None
    assert body.get('success') is False
    assert body.get('code') == 'CONFIRMATION_REQUIRED', (
        f"Expected CONFIRMATION_REQUIRED, got: {body}"
    )
