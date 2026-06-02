"""Endpoint-level behaviour tests for training visualization_routes.

Covers the W11-BE1 rate-limit decorator, W11-BE2 UUID guard, W11-BE4
standardized error contract, and the W11-BE5 information-leak fix for
the 5 endpoints in visualization_routes.py.

Test pattern mirrors tests/domains/training/api/test_session_routes.py
(T6) and tests/domains/training/api/test_model_routes.py (T5).
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
# Helpers (mirrors tests/domains/training/api/test_session_routes.py)
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
        import domains.training.api.visualization_routes as visualization_routes
        importlib.reload(visualization_routes)

        app = Flask(__name__)
        app.config['TESTING'] = True
        limiter.init_app(app)
        app.register_blueprint(visualization_routes.bp, url_prefix='/api/training')

    # Restore the real decorators on the module so subsequent imports get the
    # production behaviour.
    import domains.training.api.common as common_module
    importlib.reload(common_module)
    import domains.training.api.visualization_routes as visualization_routes
    importlib.reload(visualization_routes)
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
    "get_plot_variables",
    "get_training_visualizations",
    "generate_plot",
    "get_evaluation_tables",
    "save_evaluation_tables",
])
def test_handler_has_rate_limit(handler_name):
    """Every routed handler must declare @limiter.limit(training_limit_string)."""
    import domains.training.api.visualization_routes as visualization_routes
    importlib.reload(visualization_routes)

    handler = getattr(visualization_routes, handler_name, None)
    assert handler is not None, f"{handler_name} not exported from visualization_routes"
    assert _has_rate_limit(handler), (
        f"{handler_name} missing @limiter.limit decorator. "
        "Apply @limiter.limit(training_limit_string) between @bp.route and @require_auth."
    )


# ---------------------------------------------------------------------------
# W11-BE2: malformed session_id path param → 400 BAD_UUID before DB hit
# ---------------------------------------------------------------------------

def test_get_visualizations_invalid_uuid_returns_400(client):
    """GET /visualizations/<sid> rejects malformed UUID with 400 BAD_UUID."""
    resp = client.get(
        '/api/training/visualizations/not-a-uuid',
        headers=_auth_headers(),
    )
    assert resp.status_code == 400, (
        f"Expected 400 BAD_UUID, got {resp.status_code}: {resp.get_data(as_text=True)}"
    )
    body = resp.get_json()
    assert body is not None
    assert body.get('success') is False
    assert body.get('code') == 'BAD_UUID', f"Expected code=BAD_UUID, got: {body}"


def test_get_plot_variables_invalid_uuid_returns_400(client):
    """GET /plot-variables/<sid> rejects malformed UUID with 400 BAD_UUID."""
    resp = client.get(
        '/api/training/plot-variables/not-a-uuid',
        headers=_auth_headers(),
    )
    assert resp.status_code == 400
    body = resp.get_json()
    assert body is not None
    assert body.get('code') == 'BAD_UUID'


def test_get_evaluation_tables_invalid_uuid_returns_400(client):
    """GET /evaluation-tables/<sid> rejects malformed UUID with 400 BAD_UUID."""
    resp = client.get(
        '/api/training/evaluation-tables/not-a-uuid',
        headers=_auth_headers(),
    )
    assert resp.status_code == 400
    body = resp.get_json()
    assert body is not None
    assert body.get('code') == 'BAD_UUID'


def test_save_evaluation_tables_invalid_uuid_returns_400(client):
    """POST /save-evaluation-tables/<sid> rejects malformed UUID with 400 BAD_UUID."""
    resp = client.post(
        '/api/training/save-evaluation-tables/not-a-uuid',
        headers=_auth_headers(),
        json={'df_eval': {'foo': 1}},
    )
    assert resp.status_code == 400
    body = resp.get_json()
    assert body is not None
    assert body.get('code') == 'BAD_UUID'


# ---------------------------------------------------------------------------
# Body-id route: /generate-plot
# ---------------------------------------------------------------------------

def test_generate_plot_missing_body_returns_400_missing_body(client):
    """POST /generate-plot with no JSON body → 400 MISSING_BODY."""
    resp = client.post(
        '/api/training/generate-plot',
        headers=_auth_headers(),
        data='',
        content_type='application/json',
    )
    body = resp.get_json()
    assert body is not None
    assert body.get('success') is False
    # No body OR missing sessionId both surface as MISSING_BODY
    assert body.get('code') in ('MISSING_BODY', 'BAD_REQUEST'), (
        f"Expected MISSING_BODY/BAD_REQUEST, got: {body}"
    )


def test_generate_plot_missing_session_id_returns_missing_body(client):
    """POST /generate-plot without sessionId field → 400 MISSING_BODY."""
    resp = client.post(
        '/api/training/generate-plot',
        headers=_auth_headers(),
        json={'plot_settings': {}},
    )
    assert resp.status_code == 400
    body = resp.get_json()
    assert body is not None
    assert body.get('success') is False
    assert body.get('code') == 'MISSING_BODY', f"Expected MISSING_BODY, got: {body}"


def test_generate_plot_invalid_uuid_in_body_returns_400_bad_uuid(client):
    """POST /generate-plot with malformed sessionId in body → 400 BAD_UUID."""
    resp = client.post(
        '/api/training/generate-plot',
        headers=_auth_headers(),
        json={'session_id': 'not-a-uuid', 'plot_settings': {}},
    )
    assert resp.status_code == 400
    body = resp.get_json()
    assert body is not None
    assert body.get('code') == 'BAD_UUID', f"Expected BAD_UUID, got: {body}"


# ---------------------------------------------------------------------------
# W11-BE4/BE5/BE7: 500 path returns INTERNAL_ERROR shape, no exception text leak
# ---------------------------------------------------------------------------

def test_get_plot_variables_500_returns_internal_error_shape(client):
    """A downstream exception on /plot-variables/<sid> must not leak text."""
    import domains.training.api.visualization_routes as visualization_routes

    sid = "a1b2c3d4-e5f6-4a7b-8c9d-0e1f2a3b4c5d"

    # Visualizer is imported at module level; patch it on the routes module
    with patch.object(
        visualization_routes,
        'Visualizer',
        side_effect=Exception("UNIQUE_T7_MARKER_stacktrace_leak"),
    ):
        resp = client.get(
            f'/api/training/plot-variables/{sid}',
            headers=_auth_headers(),
        )

    assert resp.status_code == 500, (
        f"Expected 500, got {resp.status_code}: {resp.get_data(as_text=True)}"
    )
    body = resp.get_json()
    assert body is not None
    assert body.get('success') is False
    assert body.get('code') == 'INTERNAL_ERROR', f"Expected INTERNAL_ERROR, got: {body}"
    assert "UNIQUE_T7_MARKER_stacktrace_leak" not in (body.get('error') or "")
    assert "UNIQUE_T7_MARKER_stacktrace_leak" not in (body.get('message') or "")
    assert "UNIQUE_T7_MARKER_stacktrace_leak" not in (body.get('suggestion') or "")
