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
    """Build the test app and bypass the FIX-1 ownership check on
    /train-models. Ownership semantics are covered by
    test_idor_multi_user.py; this file targets the W11-BE error contract."""
    app = _build_app_with_stubs()
    import domains.training.api.training_routes as training_routes
    with patch.object(training_routes, 'create_or_get_session_uuid',
                      return_value='a1b2c3d4-e5f6-4a7b-8c9d-0e1f2a3b4c5d'), \
         patch.object(training_routes, 'assert_session_ownership',
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


# ---------------------------------------------------------------------------
# Code-review follow-up: /download-arrays must distinguish 404 (file not
# found) from 500 (unexpected server failure). Pre-fix, ALL exceptions
# collapsed into RESULTS_NOT_FOUND 404, hiding genuine server failures
# (Supabase outage, network errors, etc.) behind a misleading 404.
# ---------------------------------------------------------------------------

def test_download_arrays_unexpected_exception_returns_500_internal_error(client):
    """Unexpected exceptions in /download-arrays must return INTERNAL_ERROR 500, not RESULTS_NOT_FOUND 404."""
    import domains.training.api.training_routes as training_routes

    sid = "session_a1b2c3d4-e5f6-4a7b-8c9d-0e1f2a3b4c5d"

    # `create_or_get_session_uuid` runs inside the route's try block before any
    # storage call — patching it with a generic RuntimeError simulates a
    # non-file-not-found failure (e.g., Supabase outage). The handler MUST
    # route this to the INTERNAL_ERROR / 500 branch, NOT to RESULTS_NOT_FOUND.
    with patch.object(
        training_routes,
        'create_or_get_session_uuid',
        side_effect=RuntimeError("simulated DB outage"),
    ):
        r = client.get(
            f"/api/training/download-arrays/{sid}",
            headers=_auth_headers(),
        )

    assert r.status_code == 500, (
        f"Expected 500 INTERNAL_ERROR, got {r.status_code}: {r.get_data(as_text=True)}"
    )
    body = r.get_json()
    assert body is not None
    assert body.get('success') is False
    assert body.get('code') == 'INTERNAL_ERROR', (
        f"Expected code=INTERNAL_ERROR, got: {body}"
    )
    # Internal exception text must NOT leak into the user-facing payload.
    assert 'simulated DB outage' not in (body.get('error') or '')


# ---------------------------------------------------------------------------
# FIX-4: /status 200 path leak fix + contradictory-fields fix
#
# Bug 1: the 200 success path always returned session_id + message
#        (W11-BE5 only hardened the 500 path).
# Bug 2: when training_results.status='failed', the response overlaid
#        progress=100 / current_step='Training completed' /
#        message='Training completed successfully' — five contradictory
#        signals in one payload.
# ---------------------------------------------------------------------------

class _FakeQuery:
    """Minimal chainable Supabase query stub.

    The real builder returns ``self`` from select/eq/order/limit and only
    materialises on ``.execute()``. We capture which table was hit so a
    single fake can serve both the training_progress and training_results
    branches with different payloads.
    """

    def __init__(self, table_name, table_responses):
        self._table = table_name
        self._responses = table_responses  # {'training_progress': [...], 'training_results': [...]}

    def select(self, *_args, **_kwargs):
        return self

    def eq(self, *_args, **_kwargs):
        return self

    def order(self, *_args, **_kwargs):
        return self

    def limit(self, *_args, **_kwargs):
        return self

    def execute(self):
        class _R:
            pass
        r = _R()
        r.data = self._responses.get(self._table, [])
        return r


class _FakeSupabase:
    """Stub for get_supabase_client(...).table('foo').select(...).execute().

    Accepts a dict of {table_name: [row, ...]} keyed payloads.
    """

    def __init__(self, table_responses):
        self._table_responses = table_responses

    def table(self, name):
        return _FakeQuery(name, self._table_responses)


def test_status_200_success_response_minimal_shape(client):
    """FIX-4 (Bug 1): /status 200 must return ONLY status/progress/current_step/completed_at.
    NO session_id, NO message, NO leak of internal state."""
    import domains.training.api.training_routes as training_routes

    sid = "session_a1b2c3d4-e5f6-4a7b-8c9d-0e1f2a3b4c5d"
    fake = _FakeSupabase({
        'training_progress': [{
            'status': 'completed',
            'overall_progress': 100,
            'current_step': 'Model training: done',
            'started_at': '2026-06-01T12:00:00Z',
            'completed_at': '2026-06-01T12:05:00Z',
            'updated_at': '2026-06-01T12:05:00Z',
        }],
    })

    with patch.object(training_routes, 'get_supabase_client', return_value=fake):
        resp = client.get(f"/api/training/status/{sid}", headers=_auth_headers())

    assert resp.status_code == 200, (
        f"Expected 200, got {resp.status_code}: {resp.get_data(as_text=True)}"
    )
    body = resp.get_json()
    assert 'session_id' not in body, (
        f"FIX-4 Bug 1: 200 response MUST NOT echo session_id. Got: {body}"
    )
    assert 'message' not in body, (
        f"FIX-4 Bug 1: 200 response MUST NOT carry decorative message. Got: {body}"
    )
    # Required minimal shape:
    assert body.get('status') == 'completed'
    assert body.get('progress') == 100
    assert body.get('current_step') == 'Model training: done'
    assert body.get('completed_at') == '2026-06-01T12:05:00Z'


def test_status_failed_run_does_not_show_success_message(client):
    """FIX-4 (Bug 2): when the latest training_results row is 'failed', the
    response must NOT overlay 'Training completed successfully' nor progress=100."""
    import domains.training.api.training_routes as training_routes

    sid = "session_a1b2c3d4-e5f6-4a7b-8c9d-0e1f2a3b4c5d"
    # No live training_progress row → fall back to archived training_results.
    fake = _FakeSupabase({
        'training_progress': [],
        'training_results': [{
            'status': 'failed',
            'completed_at': None,
        }],
    })

    with patch.object(training_routes, 'get_supabase_client', return_value=fake):
        resp = client.get(f"/api/training/status/{sid}", headers=_auth_headers())

    assert resp.status_code == 200
    body = resp.get_json()
    assert body['status'] == 'failed', (
        f"Archived failed run must surface as status='failed'. Got: {body}"
    )
    # MUST NOT contain success-flavoured text/values.
    assert 'message' not in body
    assert body.get('current_step') != 'Training completed', (
        f"Failed run must not claim 'Training completed'. Got: {body}"
    )
    assert body.get('progress') != 100, (
        f"Failed run must not report progress=100. Got: {body}"
    )
    assert 'session_id' not in body


def test_status_live_progress_takes_precedence_over_archived_results(client):
    """FIX-4 (Bug 2 corollary): training_progress (live) is the source of
    truth. When both tables have data, the live row must win — otherwise
    a stale training_results row from a PREVIOUS run leaks into the
    current run's status."""
    import domains.training.api.training_routes as training_routes

    sid = "session_a1b2c3d4-e5f6-4a7b-8c9d-0e1f2a3b4c5d"
    fake = _FakeSupabase({
        'training_progress': [{
            'status': 'running',
            'overall_progress': 45,
            'current_step': 'Model training: epoch 23/50',
            'completed_at': None,
        }],
        # Even if a previous successful run left behind a row here,
        # the live progress row must be picked.
        'training_results': [{
            'status': 'completed',
            'completed_at': '2026-05-01T00:00:00Z',
        }],
    })

    with patch.object(training_routes, 'get_supabase_client', return_value=fake):
        resp = client.get(f"/api/training/status/{sid}", headers=_auth_headers())

    assert resp.status_code == 200
    body = resp.get_json()
    assert body['status'] == 'running'
    assert body['progress'] == 45
    assert body['current_step'] == 'Model training: epoch 23/50'
    assert body['completed_at'] is None


def test_status_not_found_response_minimal_shape(client):
    """FIX-4 (Bug 1): not_found branch also returns minimal shape (no session_id, no message)."""
    import domains.training.api.training_routes as training_routes

    sid = "session_a1b2c3d4-e5f6-4a7b-8c9d-0e1f2a3b4c5d"
    fake = _FakeSupabase({'training_progress': [], 'training_results': []})

    with patch.object(training_routes, 'get_supabase_client', return_value=fake):
        resp = client.get(f"/api/training/status/{sid}", headers=_auth_headers())

    assert resp.status_code == 200
    body = resp.get_json()
    assert body['status'] == 'not_found'
    assert 'session_id' not in body
    assert 'message' not in body
    # current_step + completed_at + progress are always present in the contract.
    assert 'progress' in body
    assert 'current_step' in body
    assert 'completed_at' in body


# ---------------------------------------------------------------------------
# FIX-5 (Bug 1): /train-models concurrency guard
#
# Pre-fix: two parallel POSTs spawned two threading.Thread workers that
# raced on the shared Supabase HTTP/2 pool (RST_STREAM kills both threads)
# AND each ran increment_training_count → user double-billed.
# Post-fix: a heartbeat-fresh training_progress row with status='running'
# forces the second call to 409 TRAINING_IN_PROGRESS BEFORE the counter
# increment and BEFORE the thread spawn.
# ---------------------------------------------------------------------------

def test_train_models_rejects_when_already_running(client):
    """FIX-5: concurrent training launches must be rejected with 409 TRAINING_IN_PROGRESS."""
    import domains.training.api.training_routes as training_routes
    sid = "session_a1b2c3d4-e5f6-4a7b-8c9d-0e1f2a3b4c5d"

    # is_training_in_flight is imported into training_routes from lifecycle —
    # patching the route-module binding is the correct seam.
    with patch.object(training_routes, 'is_training_in_flight', return_value=True), \
         patch.object(training_routes, 'increment_training_count') as mock_increment, \
         patch.object(training_routes, 'threading') as mock_threading:
        r = client.post(
            f"/api/training/train-models/{sid}",
            headers=_auth_headers(),
            json={'model_parameters': {'MODE': 'Linear'}, 'training_split': {}},
        )

    assert r.status_code == 409, (
        f"Expected 409 TRAINING_IN_PROGRESS, got {r.status_code}: {r.get_data(as_text=True)}"
    )
    body = r.get_json()
    assert body is not None
    assert body.get('success') is False
    assert body.get('code') == 'TRAINING_IN_PROGRESS'
    # User must NOT be billed when the request is rejected.
    mock_increment.assert_not_called()
    # No thread must be spawned on the rejected path.
    mock_threading.Thread.assert_not_called()


def test_train_models_accepts_when_not_in_flight(client):
    """FIX-5: when no live training, the request proceeds past the 409 guard."""
    import domains.training.api.training_routes as training_routes
    sid = "session_a1b2c3d4-e5f6-4a7b-8c9d-0e1f2a3b4c5d"

    with patch.object(training_routes, 'is_training_in_flight', return_value=False), \
         patch.object(training_routes, 'increment_training_count'), \
         patch.object(training_routes, 'threading'), \
         patch.object(training_routes, 'run_model_training_async'):
        r = client.post(
            f"/api/training/train-models/{sid}",
            headers=_auth_headers(),
            json={'model_parameters': {'MODE': 'Linear'}, 'training_split': {}},
        )

    # The 409 guard must NOT fire when no run is live. Any other status
    # (200 success, 500 from a downstream stub gap) is acceptable here —
    # we're only asserting the FIX-5 guard didn't intercept.
    assert r.status_code != 409, (
        f"FIX-5 guard fired when no training is in flight. body={r.get_data(as_text=True)}"
    )
    body = r.get_json() or {}
    assert body.get('code') != 'TRAINING_IN_PROGRESS'


def test_train_models_accepts_when_previous_run_stale(client):
    """FIX-5: a stale heartbeat (worker crashed) must NOT lock the endpoint forever.

    is_training_in_flight enforces the heartbeat window — when it returns
    False (window expired), the route must accept the new request and
    NOT 409.
    """
    import domains.training.api.training_routes as training_routes
    sid = "session_a1b2c3d4-e5f6-4a7b-8c9d-0e1f2a3b4c5d"

    # Stale heartbeat → helper returns False → guard does not fire.
    with patch.object(training_routes, 'is_training_in_flight', return_value=False), \
         patch.object(training_routes, 'increment_training_count'), \
         patch.object(training_routes, 'threading'), \
         patch.object(training_routes, 'run_model_training_async'):
        r = client.post(
            f"/api/training/train-models/{sid}",
            headers=_auth_headers(),
            json={'model_parameters': {'MODE': 'Linear'}, 'training_split': {}},
        )

    assert r.status_code != 409
    body = r.get_json() or {}
    assert body.get('code') != 'TRAINING_IN_PROGRESS'


# ---------------------------------------------------------------------------
# FIX-5 (Bug 1 unit): is_training_in_flight heartbeat-window logic
#
# Unit-level coverage of the helper itself so the route tests above can
# remain a thin mock-based contract assertion.
# ---------------------------------------------------------------------------

def test_is_training_in_flight_returns_true_for_fresh_running_row():
    """status='running' + recent updated_at → True."""
    from datetime import datetime, timezone, timedelta
    import domains.training.services.lifecycle as lifecycle

    fake_recent = (datetime.now(timezone.utc) - timedelta(seconds=30)).isoformat()
    fake = _FakeSupabase({
        'training_progress': [{'status': 'running', 'updated_at': fake_recent}],
    })
    with patch.object(lifecycle, 'get_supabase_client', return_value=fake):
        assert lifecycle.is_training_in_flight(
            'a1b2c3d4-e5f6-4a7b-8c9d-0e1f2a3b4c5d'
        ) is True


def test_is_training_in_flight_returns_false_for_stale_running_row():
    """status='running' but updated_at older than window → False (worker crashed)."""
    from datetime import datetime, timezone, timedelta
    import domains.training.services.lifecycle as lifecycle

    fake_stale = (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat()
    fake = _FakeSupabase({
        'training_progress': [{'status': 'running', 'updated_at': fake_stale}],
    })
    with patch.object(lifecycle, 'get_supabase_client', return_value=fake):
        assert lifecycle.is_training_in_flight(
            'a1b2c3d4-e5f6-4a7b-8c9d-0e1f2a3b4c5d'
        ) is False


def test_is_training_in_flight_returns_false_when_no_row():
    """No training_progress row at all → False (never trained / cleaned up)."""
    import domains.training.services.lifecycle as lifecycle

    fake = _FakeSupabase({'training_progress': []})
    with patch.object(lifecycle, 'get_supabase_client', return_value=fake):
        assert lifecycle.is_training_in_flight(
            'a1b2c3d4-e5f6-4a7b-8c9d-0e1f2a3b4c5d'
        ) is False


def test_is_training_in_flight_returns_false_for_completed_status():
    """status='completed' → False even with a fresh updated_at."""
    from datetime import datetime, timezone, timedelta
    import domains.training.services.lifecycle as lifecycle

    fake_recent = (datetime.now(timezone.utc) - timedelta(seconds=30)).isoformat()
    fake = _FakeSupabase({
        'training_progress': [{'status': 'completed', 'updated_at': fake_recent}],
    })
    with patch.object(lifecycle, 'get_supabase_client', return_value=fake):
        assert lifecycle.is_training_in_flight(
            'a1b2c3d4-e5f6-4a7b-8c9d-0e1f2a3b4c5d'
        ) is False


def test_is_training_in_flight_does_not_fail_closed_on_db_error():
    """DB outage → False (do NOT permanently lock /train-models + /delete).

    See lifecycle.py module docstring: a fail-closed design would lock every
    session's lifecycle endpoints whenever Supabase blips. The training
    thread itself will surface DB outages via its own write path.
    """
    import domains.training.services.lifecycle as lifecycle

    with patch.object(
        lifecycle, 'get_supabase_client',
        side_effect=Exception('simulated supabase outage'),
    ):
        assert lifecycle.is_training_in_flight(
            'a1b2c3d4-e5f6-4a7b-8c9d-0e1f2a3b4c5d'
        ) is False
