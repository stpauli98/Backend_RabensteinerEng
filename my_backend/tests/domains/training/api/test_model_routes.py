"""Endpoint-level behaviour tests for training model_routes.

Covers the W11-BE1 rate-limit decorator, W11-BE2 UUID guard, W11-BE4
standardized error contract, and the W11-BE5 information-leak fix for
the 7 endpoints in model_routes.py.

Test pattern mirrors tests/domains/training/api/test_training_routes.py
(T4) and tests/domains/training/api/test_upload_routes.py (T3).
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
# Helpers (mirrors tests/domains/training/api/test_training_routes.py)
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
        # `common` re-exports the decorators, so reload common first then model_routes
        # so each picks up the stubbed decorators.
        import domains.training.api.common as common_module
        importlib.reload(common_module)
        import domains.training.api.model_routes as model_routes
        importlib.reload(model_routes)

        app = Flask(__name__)
        app.config['TESTING'] = True
        limiter.init_app(app)
        app.register_blueprint(model_routes.bp, url_prefix='/api/training')

    # Restore the real decorators on the module so subsequent imports get the
    # production behaviour.
    import domains.training.api.common as common_module
    importlib.reload(common_module)
    import domains.training.api.model_routes as model_routes
    importlib.reload(model_routes)
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
    "get_scalers",
    "download_scalers_as_save_files",
    "scale_input_data",
    "save_model",
    "list_models_database",
    "download_model_h5",
    "predict_with_model",
])
def test_handler_has_rate_limit(handler_name):
    """Every routed handler must declare @limiter.limit(training_limit_string)."""
    import domains.training.api.model_routes as model_routes
    importlib.reload(model_routes)

    handler = getattr(model_routes, handler_name, None)
    assert handler is not None, f"{handler_name} not exported from model_routes"
    assert _has_rate_limit(handler), (
        f"{handler_name} missing @limiter.limit decorator. "
        "Apply @limiter.limit(training_limit_string) between @bp.route and @require_auth."
    )


# ---------------------------------------------------------------------------
# W11-BE2: malformed session_id path param → 400 BAD_UUID before DB hit
# ---------------------------------------------------------------------------

def test_predict_invalid_uuid_returns_400(client):
    resp = client.post(
        '/api/training/predict/not-a-uuid',
        headers=_auth_headers(),
        json={'model_filename': 'm.h5', 'input_data': [{}]},
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


def test_list_models_invalid_uuid_returns_400(client):
    resp = client.get(
        '/api/training/list-models-database/not-a-uuid',
        headers=_auth_headers(),
    )
    assert resp.status_code == 400
    body = resp.get_json()
    assert body is not None
    assert body.get('code') == 'BAD_UUID'


def test_get_scalers_invalid_uuid_returns_400(client):
    """Smoke test: malformed UUID rejected on /scalers/<sid> GET."""
    resp = client.get(
        '/api/training/scalers/not-a-uuid',
        headers=_auth_headers(),
    )
    assert resp.status_code == 400
    body = resp.get_json()
    assert body is not None
    assert body.get('code') == 'BAD_UUID'


def test_save_model_session_prefix_uuid_passes_validation(client):
    """session_<uuid> form must NOT be rejected by validator.

    Downstream may 500 (no Supabase mock), but validator must accept.
    """
    sid = "session_a1b2c3d4-e5f6-4a7b-8c9d-0e1f2a3b4c5d"
    resp = client.post(
        f'/api/training/save-model/{sid}',
        headers=_auth_headers(),
    )
    body = resp.get_json() or {}
    assert body.get('code') != 'BAD_UUID', (
        f"session_<uuid> form was rejected by UUID validator: {body}"
    )


# ---------------------------------------------------------------------------
# W11-BE4: error contract — every error response must have 'code'
# ---------------------------------------------------------------------------

def test_save_model_missing_body_returns_structured_error(client):
    """POST /save-model/<sid> exercises the standardized error contract.

    /save-model does not require a JSON body, but if the underlying call
    fails it must still emit {success, code, error}. We assert here that
    when triggered with no body the response — whatever shape — includes a
    'code' key per the W11-BE4 contract.
    """
    sid = "session_a1b2c3d4-e5f6-4a7b-8c9d-0e1f2a3b4c5d"
    import domains.training.api.model_routes as model_routes

    # Force the route into its error branch by raising a generic exception
    # from inside the try: block (no Supabase here).
    with patch.object(
        model_routes,
        'save_models_to_storage',
        side_effect=Exception("downstream save failure marker"),
    ):
        resp = client.post(
            f'/api/training/save-model/{sid}',
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


def test_predict_missing_body_returns_structured_error(client):
    """POST /predict/<sid> with no JSON body → 400 with code in error contract."""
    sid = "session_a1b2c3d4-e5f6-4a7b-8c9d-0e1f2a3b4c5d"
    resp = client.post(
        f'/api/training/predict/{sid}',
        headers=_auth_headers(),
        json=None,
    )
    body = resp.get_json()
    assert body is not None
    assert body.get('success') is False
    assert 'code' in body


# ---------------------------------------------------------------------------
# W11-BE5/BE7: 500 path returns INTERNAL_ERROR shape, no exception text leak
# ---------------------------------------------------------------------------

def test_save_model_500_returns_internal_error_shape(client):
    """A downstream exception on /save-model must not leak text.

    /save-model uses the MODEL_SAVE_ERROR code (storage-upload-specific
    500) per the W11-A T5 mapping; the shape contract is identical to
    INTERNAL_ERROR (success=False, code=string, no exception leak).
    """
    import domains.training.api.model_routes as model_routes

    sid = "session_a1b2c3d4-e5f6-4a7b-8c9d-0e1f2a3b4c5d"

    with patch.object(
        model_routes,
        'save_models_to_storage',
        side_effect=Exception("internal stacktrace details that must not leak"),
    ):
        resp = client.post(
            f"/api/training/save-model/{sid}",
            headers=_auth_headers(),
        )

    assert resp.status_code == 500, (
        f"Expected 500, got {resp.status_code}: {resp.get_data(as_text=True)}"
    )
    body = resp.get_json()
    assert body is not None
    assert body.get('success') is False
    # /save-model uses MODEL_SAVE_ERROR (storage-save-specific) per task mapping.
    assert body.get('code') == 'MODEL_SAVE_ERROR', (
        f"Expected code=MODEL_SAVE_ERROR, got: {body}"
    )
    # The raw exception text must NOT appear anywhere in the user-facing payload.
    assert "internal stacktrace" not in (body.get('error') or "")
    assert "internal stacktrace" not in (body.get('message') or "")
    assert "internal stacktrace" not in (body.get('suggestion') or "")


def test_predict_500_returns_internal_error_shape(client):
    """A downstream exception on /predict must not leak text.

    PredictionService is imported lazily inside the route. We patch the
    import target so the route's generic except branch fires. The error
    code is PREDICTION_ERROR (predict-specific 500) per the W11-A T5
    code-mapping table; the shape contract is identical to INTERNAL_ERROR.
    """
    import domains.training.api.model_routes as model_routes  # noqa: F401

    sid = "session_a1b2c3d4-e5f6-4a7b-8c9d-0e1f2a3b4c5d"

    class _BoomService:
        def __init__(self, *_, **__):
            pass

        def predict(self, *_, **__):
            raise Exception("internal stacktrace details that must not leak")

    with patch(
        'domains.training.services.prediction_service.PredictionService',
        new=_BoomService,
    ):
        resp = client.post(
            f"/api/training/predict/{sid}",
            headers=_auth_headers(),
            json={
                'model_filename': 'best_model.h5',
                'input_data': [{'feature1': 1.5}],
            },
        )

    assert resp.status_code == 500, (
        f"Expected 500, got {resp.status_code}: {resp.get_data(as_text=True)}"
    )
    body = resp.get_json()
    assert body is not None
    assert body.get('success') is False
    # /predict uses PREDICTION_ERROR (predict-specific) per task mapping;
    # other handlers use INTERNAL_ERROR.
    assert body.get('code') == 'PREDICTION_ERROR', (
        f"Expected code=PREDICTION_ERROR, got: {body}"
    )
    assert "internal stacktrace" not in (body.get('error') or "")
    assert "internal stacktrace" not in (body.get('message') or "")
    assert "internal stacktrace" not in (body.get('suggestion') or "")


# ---------------------------------------------------------------------------
# Storage download 404 vs 500 split (mirrors T4 polish on /download-arrays).
# Pre-fix, ValueError → 404 only; we must also ensure non-not-found
# exceptions cleanly route to INTERNAL_ERROR 500.
# ---------------------------------------------------------------------------

def test_download_model_h5_unexpected_exception_returns_500_internal_error(client):
    """Unexpected exceptions in /download-model-h5 must return 500, not 404."""
    import domains.training.api.model_routes as model_routes

    sid = "session_a1b2c3d4-e5f6-4a7b-8c9d-0e1f2a3b4c5d"

    with patch.object(
        model_routes,
        'download_model_file',
        side_effect=RuntimeError("simulated storage outage"),
    ):
        resp = client.get(
            f"/api/training/download-model-h5/{sid}?filename=best_model.h5",
            headers=_auth_headers(),
        )

    assert resp.status_code == 500, (
        f"Expected 500 INTERNAL_ERROR, got {resp.status_code}: {resp.get_data(as_text=True)}"
    )
    body = resp.get_json()
    assert body is not None
    assert body.get('success') is False
    assert body.get('code') == 'INTERNAL_ERROR'
    assert 'simulated storage outage' not in (body.get('error') or '')


def test_download_model_h5_file_not_found_returns_404_model_not_found(client):
    """FileNotFoundError on /download-model-h5 must return 404 MODEL_NOT_FOUND."""
    import domains.training.api.model_routes as model_routes

    sid = "session_a1b2c3d4-e5f6-4a7b-8c9d-0e1f2a3b4c5d"

    with patch.object(
        model_routes,
        'download_model_file',
        side_effect=FileNotFoundError("not in storage"),
    ):
        resp = client.get(
            f"/api/training/download-model-h5/{sid}?filename=best_model.h5",
            headers=_auth_headers(),
        )

    assert resp.status_code == 404
    body = resp.get_json()
    assert body is not None
    assert body.get('success') is False
    assert body.get('code') == 'MODEL_NOT_FOUND'


def test_download_scalers_file_not_found_returns_404_scaler_not_found(client):
    """FileNotFoundError on /scalers/<sid>/download must return 404 SCALER_NOT_FOUND."""
    import domains.training.api.model_routes as model_routes

    sid = "session_a1b2c3d4-e5f6-4a7b-8c9d-0e1f2a3b4c5d"

    with patch.object(
        model_routes,
        'create_scaler_download_package',
        side_effect=FileNotFoundError("scalers not in storage"),
    ):
        resp = client.get(
            f"/api/training/scalers/{sid}/download",
            headers=_auth_headers(),
        )

    assert resp.status_code == 404
    body = resp.get_json()
    assert body is not None
    assert body.get('success') is False
    assert body.get('code') == 'SCALER_NOT_FOUND'


def test_download_scalers_unexpected_exception_returns_500_internal_error(client):
    """Non-not-found exceptions on /scalers/<sid>/download must return 500."""
    import domains.training.api.model_routes as model_routes

    sid = "session_a1b2c3d4-e5f6-4a7b-8c9d-0e1f2a3b4c5d"

    with patch.object(
        model_routes,
        'create_scaler_download_package',
        side_effect=RuntimeError("simulated storage outage"),
    ):
        resp = client.get(
            f"/api/training/scalers/{sid}/download",
            headers=_auth_headers(),
        )

    assert resp.status_code == 500
    body = resp.get_json()
    assert body is not None
    assert body.get('code') == 'INTERNAL_ERROR'
    assert 'simulated storage outage' not in (body.get('error') or '')
