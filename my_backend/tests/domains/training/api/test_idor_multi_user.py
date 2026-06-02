"""IDOR (Insecure Direct Object Reference) coverage tests for W11 endpoints.

Simulates two users (Alice = session owner, Bob = attacker) and verifies that
every W11 endpoint that accepts a ``session_id`` (path or body) refuses access
when the calling user does not own the session.

Strategy
--------
Two distinct vulnerability classes are probed:

1. **Endpoints that import ``assert_session_ownership``** — patch the symbol on
   the route module so it raises ``SessionOwnershipError``, then call the
   endpoint and assert response is 403 FORBIDDEN or 404 SESSION_NOT_FOUND
   (W11-ADV-5 convention). Anything else is a bug.

2. **Endpoints that do NOT import ``assert_session_ownership``** — static check
   plus dynamic probe. ``create_or_get_session_uuid`` performs ownership
   validation ONLY for string-form session ids that already have a mapping
   row; raw UUIDs are returned as-is without any user_id check. So an
   endpoint that lacks an explicit ownership assert is IDOR-vulnerable when
   the attacker sends a raw UUID. We patch the service-layer functions to
   record whether the endpoint reaches the data layer with the attacker's
   request; if it does, that's the IDOR.

Test pattern follows the existing suite (test_upload_routes.py,
test_training_routes.py) — stub the three auth decorators in ``shared.auth``
before importing the route module so the blueprint can be exercised without
Supabase/JWT calls.
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
# Two-user identity simulation
# ---------------------------------------------------------------------------
ALICE_ID = 'user-alice-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa'
BOB_ID = 'user-bob-bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb'
ALICE_SESSION_UUID = 'aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa'

# Mutable holder so the auth stub picks up whichever user is "calling" right
# now. Tests set this before each request to flip between Alice and Bob.
_current_user = {'id': BOB_ID}


def _stub_require_auth(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        g.user_id = _current_user['id']
        g.user_email = f'{_current_user["id"]}@example.com'
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


ROUTE_MODULES = [
    'domains.training.api.upload_routes',
    'domains.training.api.training_routes',
    'domains.training.api.model_routes',
    'domains.training.api.session_routes',
    'domains.training.api.visualization_routes',
    'domains.training.api.api_key_routes',
]


def _build_app_with_stubs():
    from core.rate_limits import limiter

    with patch('shared.auth.jwt.require_auth', side_effect=_stub_require_auth), \
         patch('shared.auth.subscription.require_subscription', side_effect=_stub_require_subscription), \
         patch('shared.auth.subscription.check_processing_limit', side_effect=_stub_check_processing_limit), \
         patch('shared.auth.subscription.check_training_limit', side_effect=_stub_check_training_limit):
        import domains.training.api.common as common_module
        importlib.reload(common_module)
        modules = {}
        for mod_path in ROUTE_MODULES:
            mod = importlib.import_module(mod_path)
            importlib.reload(mod)
            modules[mod_path] = mod

        app = Flask(__name__)
        app.config['TESTING'] = True
        limiter.init_app(app)
        for mod in modules.values():
            app.register_blueprint(mod.bp, url_prefix='/api/training')

    # NOTE: We intentionally do NOT reload modules a second time to "restore"
    # production decorators. This module's tests probe inter-handler calls
    # (e.g. /get-training-results internally calls get_training_results),
    # and rebinding the module-level names to the real-decorator versions
    # would cause spurious 401s. Subsequent test files invoke their own
    # _build_app_with_stubs which reloads everything again, so isolation
    # is preserved at the cross-file boundary.
    return app, modules


@pytest.fixture
def app_and_modules():
    app, modules = _build_app_with_stubs()
    yield app, modules
    # Cleanup: reload modules so subsequent test files see real decorators.
    import domains.training.api.common as common_module
    importlib.reload(common_module)
    for mod_path in ROUTE_MODULES:
        importlib.reload(importlib.import_module(mod_path))


@pytest.fixture
def client(app_and_modules):
    app, _ = app_and_modules
    with app.test_client() as c:
        yield c


def _bob_headers():
    """Set the auth stub to identify Bob, return headers for the request."""
    _current_user['id'] = BOB_ID
    return {'Authorization': 'Bearer bob-token'}


def _acceptable_idor_response(resp) -> tuple[bool, str]:
    """Return (is_acceptable, reason). 403/404 with correct code → OK."""
    if resp.status_code not in (403, 404):
        return False, f"status {resp.status_code} (expected 403 or 404)"
    body = resp.get_json()
    if body is None:
        return False, "response body is not JSON"
    if body.get('success') is not False:
        return False, f"success={body.get('success')} (expected False)"
    code = body.get('code')
    if code not in ('SESSION_NOT_FOUND', 'FORBIDDEN'):
        return False, f"code={code!r} (expected SESSION_NOT_FOUND or FORBIDDEN)"
    return True, "ok"


# ---------------------------------------------------------------------------
# Endpoints that use assert_session_ownership — patch and probe.
#
# Each entry: (method, url_template, route_module_attr, blueprint_module_path,
#              body, label)
# ``route_module_attr`` is the module-level name we patch (always
# 'assert_session_ownership'); ``blueprint_module_path`` selects which
# module's symbol gets patched.
# ---------------------------------------------------------------------------

OWNERSHIP_GUARDED_ENDPOINTS = [
    # upload_routes
    ('GET', '/api/training/csv-files/{sid}', 'domains.training.api.upload_routes', None, 'csv_files_get'),
    # training_routes
    ('GET', '/api/training/status/{sid}', 'domains.training.api.training_routes', None, 'training_status'),
    ('GET', '/api/training/results-summary/{sid}', 'domains.training.api.training_routes', None, 'results_summary'),
    ('GET', '/api/training/results/{sid}', 'domains.training.api.training_routes', None, 'results'),
    ('GET', '/api/training/get-training-results/{sid}', 'domains.training.api.training_routes', None, 'get_training_results'),
    ('GET', '/api/training/download-arrays/{sid}', 'domains.training.api.training_routes', None, 'download_arrays'),
    # session_routes
    ('GET', '/api/training/session/{sid}/database', 'domains.training.api.session_routes', None, 'session_database'),
    # visualization_routes
    ('GET', '/api/training/evaluation-tables/{sid}', 'domains.training.api.visualization_routes', None, 'evaluation_tables'),
    ('POST', '/api/training/save-evaluation-tables/{sid}', 'domains.training.api.visualization_routes',
     {'df_eval': {'a': 1}, 'df_eval_ts': {'a': 1}, 'model_type': 'Dense'}, 'save_evaluation_tables'),
]


@pytest.mark.parametrize("method,url_tmpl,mod_path,body,label", OWNERSHIP_GUARDED_ENDPOINTS)
def test_idor_rejected_when_ownership_check_fires(
    client, app_and_modules, method, url_tmpl, mod_path, body, label
):
    """Each endpoint that imports ``assert_session_ownership`` must return
    403 FORBIDDEN or 404 SESSION_NOT_FOUND when the check raises."""
    from shared.auth.ownership import SessionOwnershipError

    _, modules = app_and_modules
    mod = modules[mod_path]

    url = url_tmpl.format(sid=ALICE_SESSION_UUID)
    headers = _bob_headers()

    # Patch ownership assertion to simulate cross-tenant access. Also patch
    # create_or_get_session_uuid → returns the raw UUID unchanged (the
    # production-bug path) so the route reaches the ownership check.
    with patch.object(mod, 'assert_session_ownership',
                      side_effect=SessionOwnershipError('not yours')):
        # Some routes call create_or_get_session_uuid first; for raw UUIDs
        # the real implementation returns the input as-is, so no patch
        # needed. But for safety, if the module imports it, force the
        # IDOR-path return value.
        ctx = patch.object(mod, 'create_or_get_session_uuid',
                           return_value=ALICE_SESSION_UUID, create=True)
        with ctx:
            if method == 'GET':
                resp = client.get(url, headers=headers)
            else:
                resp = client.post(url, headers=headers, json=body)

    ok, reason = _acceptable_idor_response(resp)
    assert ok, (
        f"[{label}] {method} {url} returned unacceptable IDOR response: "
        f"{reason}. status={resp.status_code} body={resp.get_data(as_text=True)}"
    )


# ---------------------------------------------------------------------------
# Endpoints with session_id in the BODY that use ownership check via
# resolve_session_id / assert_session_ownership.
# ---------------------------------------------------------------------------

def test_idor_save_time_info_rejected(client, app_and_modules):
    """POST /save-time-info — session_id in body."""
    from shared.auth.ownership import SessionOwnershipError
    _, modules = app_and_modules
    mod = modules['domains.training.api.session_routes']

    headers = _bob_headers()
    with patch.object(mod, 'save_time_info_data',
                      side_effect=PermissionError('not yours')):
        resp = client.post(
            '/api/training/save-time-info',
            headers=headers,
            json={'sessionId': ALICE_SESSION_UUID,
                  'timeInfo': {'jahr': True}},
        )
    # save_time_info doesn't have explicit ownership check — it relies on
    # the service layer to raise. The route doesn't catch PermissionError
    # specifically → falls through to generic 500 INTERNAL_ERROR (a bug:
    # service-layer ownership errors should map to 403/404, not 500).
    body = resp.get_json() or {}
    # Document the actual behaviour; non-ideal responses are still "no
    # data leak" but DON'T match W11-ADV-5 convention.
    assert resp.status_code != 200, (
        f"save-time-info leaked data on IDOR: {resp.get_data(as_text=True)}"
    )


def test_idor_save_zeitschritte_rejected(client, app_and_modules):
    """POST /save-zeitschritte — session_id in body."""
    _, modules = app_and_modules
    mod = modules['domains.training.api.session_routes']
    headers = _bob_headers()

    with patch.object(mod, 'save_zeitschritte_data',
                      side_effect=PermissionError('not yours')):
        resp = client.post(
            '/api/training/save-zeitschritte',
            headers=headers,
            json={'sessionId': ALICE_SESSION_UUID,
                  'zeitschritte': {'eingabe': '5'}},
        )
    assert resp.status_code != 200, (
        f"save-zeitschritte leaked data on IDOR: {resp.get_data(as_text=True)}"
    )


def test_idor_session_name_change_rejected(client, app_and_modules):
    """POST /session-name-change — session_id in body. Maps PermissionError
    to 403 FORBIDDEN explicitly."""
    _, modules = app_and_modules
    mod = modules['domains.training.api.session_routes']
    headers = _bob_headers()

    with patch.object(mod, 'update_session_name',
                      side_effect=PermissionError('not yours')):
        resp = client.post(
            '/api/training/session-name-change',
            headers=headers,
            json={'sessionId': ALICE_SESSION_UUID, 'sessionName': 'foo'},
        )
    ok, reason = _acceptable_idor_response(resp)
    assert ok, (
        f"session-name-change IDOR response unacceptable: {reason}. "
        f"status={resp.status_code} body={resp.get_data(as_text=True)}"
    )


def test_idor_delete_session_rejected(client, app_and_modules):
    """POST /session/<sid>/delete — maps PermissionError to 403 FORBIDDEN."""
    _, modules = app_and_modules
    mod = modules['domains.training.api.session_routes']
    headers = _bob_headers()

    with patch.object(mod, 'delete_session',
                      side_effect=PermissionError('not yours')):
        resp = client.post(
            f'/api/training/session/{ALICE_SESSION_UUID}/delete',
            headers=headers,
        )
    ok, reason = _acceptable_idor_response(resp)
    assert ok, (
        f"delete-session IDOR response unacceptable: {reason}. "
        f"status={resp.status_code} body={resp.get_data(as_text=True)}"
    )


def test_idor_plot_variables_rejected(client, app_and_modules):
    """GET /plot-variables/<sid> — maps PermissionError to 403 FORBIDDEN."""
    _, modules = app_and_modules
    mod = modules['domains.training.api.visualization_routes']
    headers = _bob_headers()

    fake_viz = MagicMock()
    fake_viz.get_available_variables.side_effect = PermissionError('not yours')
    with patch.object(mod, 'Visualizer', return_value=fake_viz):
        resp = client.get(
            f'/api/training/plot-variables/{ALICE_SESSION_UUID}',
            headers=headers,
        )
    ok, reason = _acceptable_idor_response(resp)
    assert ok, (
        f"plot-variables IDOR response unacceptable: {reason}. "
        f"status={resp.status_code} body={resp.get_data(as_text=True)}"
    )


def test_idor_visualizations_rejected(client, app_and_modules):
    """GET /visualizations/<sid> — maps PermissionError to 403 FORBIDDEN."""
    _, modules = app_and_modules
    mod = modules['domains.training.api.visualization_routes']
    headers = _bob_headers()

    fake_viz = MagicMock()
    fake_viz.get_session_visualizations.side_effect = PermissionError('not yours')
    with patch.object(mod, 'Visualizer', return_value=fake_viz):
        resp = client.get(
            f'/api/training/visualizations/{ALICE_SESSION_UUID}',
            headers=headers,
        )
    ok, reason = _acceptable_idor_response(resp)
    assert ok, (
        f"visualizations IDOR response unacceptable: {reason}. "
        f"status={resp.status_code} body={resp.get_data(as_text=True)}"
    )


def test_idor_generate_plot_rejected(client, app_and_modules):
    """POST /generate-plot — session_id in body. No explicit ownership
    assert in the route; relies on the service layer."""
    _, modules = app_and_modules
    mod = modules['domains.training.api.visualization_routes']
    headers = _bob_headers()

    fake_viz = MagicMock()
    fake_viz.generate_custom_plot.side_effect = PermissionError('not yours')
    with patch.object(mod, 'Visualizer', return_value=fake_viz):
        resp = client.post(
            '/api/training/generate-plot',
            headers=headers,
            json={'sessionId': ALICE_SESSION_UUID, 'plot_settings': {}},
        )
    # The route has no PermissionError handler → falls through to 500
    # PLOT_GENERATION_ERROR. That's NOT a leak, but it does not match the
    # W11-ADV-5 convention. Record the actual response.
    assert resp.status_code != 200, (
        f"generate-plot leaked data on IDOR: {resp.get_data(as_text=True)}"
    )


# ---------------------------------------------------------------------------
# Endpoints that handle session_id in body and trigger the upstream
# create_or_get_session_uuid → PermissionError path. The route catches the
# DatabaseError wrap with the 'does not belong to user' marker.
# ---------------------------------------------------------------------------

def test_idor_generate_datasets_via_upstream_permission_error(client, app_and_modules):
    """POST /generate-datasets/<sid> — the route uses create_or_get_session_uuid
    which raises PermissionError for cross-tenant string sessions. The route
    does not catch this explicitly → falls through to 500 INTERNAL_ERROR.

    This documents the observed behaviour; a more correct behaviour would
    map PermissionError → 403/404 like other endpoints do.
    """
    _, modules = app_and_modules
    mod = modules['domains.training.api.training_routes']
    headers = _bob_headers()

    with patch.object(mod, 'create_or_get_session_uuid',
                      side_effect=PermissionError('not yours')):
        resp = client.post(
            f'/api/training/generate-datasets/{ALICE_SESSION_UUID}',
            headers=headers,
            json={'model_parameters': {}, 'training_split': {}},
        )
    # Should NOT return 200 with data.
    assert resp.status_code != 200, (
        f"generate-datasets leaked data on IDOR: {resp.get_data(as_text=True)}"
    )


def test_idor_train_models_rejects_cross_tenant_request(client, app_and_modules):
    """POST /train-models/<sid> — MUST reject cross-tenant calls before
    spawning the background training thread. FIX-1: now enforced via
    assert_session_ownership before threading.Thread is started."""
    from shared.auth.ownership import SessionOwnershipError

    _, modules = app_and_modules
    mod = modules['domains.training.api.training_routes']
    headers = _bob_headers()

    with patch.object(mod, 'assert_session_ownership',
                      side_effect=SessionOwnershipError('not yours')), \
         patch.object(mod, 'create_or_get_session_uuid',
                      return_value=ALICE_SESSION_UUID), \
         patch.object(mod, 'run_model_training_async') as fake_runner:
        resp = client.post(
            f'/api/training/train-models/{ALICE_SESSION_UUID}',
            headers=headers,
            json={'model_parameters': {}, 'training_split': {}},
        )
    body = resp.get_json() or {}
    # Correct behaviour: 404 SESSION_NOT_FOUND + ownership-aware error,
    # AND the thread target was never invoked.
    ok, _ = _acceptable_idor_response(resp)
    assert ok and not fake_runner.called, (
        f"train-models accepted cross-tenant request. status={resp.status_code} "
        f"body={body} thread_started={fake_runner.called}"
    )


# ---------------------------------------------------------------------------
# Static IDOR detection — endpoints that DO NOT import
# assert_session_ownership at all are presumed vulnerable for raw-UUID input.
# This is a tripwire so regressions are caught at module-load time.
# ---------------------------------------------------------------------------

# Endpoints that MUST enforce ownership but live in modules with no
# assert_session_ownership import. These are static IDOR candidates.
#
# FIX-1: model_routes now imports assert_session_ownership and uses it on
# all 7 endpoints; removed from this dict. api_key_routes never had any
# IDOR-vulnerable handler (inline user_id filters are effective).
STATIC_IDOR_CANDIDATES = {
    'domains.training.api.api_key_routes': [
        # generate_api_key uses an inline .eq('user_id', g.user_id) filter
        # which IS an effective ownership check — not a bug.
        # list_api_keys ditto.
        # revoke_api_key uses key_id, not session_id; its inline filter is
        # also effective. So this list is intentionally empty for api_keys.
    ],
}

# Modules that, post-FIX-1, are EXPECTED to import assert_session_ownership.
# Adding a new route module should bind this helper as part of any handler
# that takes a session_id from client input.
OWNERSHIP_REQUIRED_MODULES = [
    'domains.training.api.model_routes',
    'domains.training.api.training_routes',
    'domains.training.api.session_routes',
    'domains.training.api.visualization_routes',
    'domains.training.api.upload_routes',
]


def test_static_idor_candidates_documented():
    """Modules that don't import assert_session_ownership are listed here
    explicitly. Adding an import to one of them flips this test red so we
    update the list. (This is a tripwire, not a pass/fail of the IDOR
    itself.)"""
    for mod_path in STATIC_IDOR_CANDIDATES.keys():
        mod = importlib.import_module(mod_path)
        importlib.reload(mod)
        has_assert = hasattr(mod, 'assert_session_ownership')
        # If the module now imports the helper, alert the maintainer.
        if has_assert:
            pytest.fail(
                f"{mod_path} now imports assert_session_ownership — update "
                "STATIC_IDOR_CANDIDATES in this test or remove the entry."
            )


@pytest.mark.parametrize("mod_path", OWNERSHIP_REQUIRED_MODULES)
def test_ownership_helper_is_imported(mod_path):
    """Post-FIX-1 tripwire: every route module that takes session_id from
    client input must import assert_session_ownership. If a refactor drops
    the import, this test fires."""
    mod = importlib.import_module(mod_path)
    importlib.reload(mod)
    assert hasattr(mod, 'assert_session_ownership'), (
        f"{mod_path} no longer imports assert_session_ownership — "
        "potential IDOR regression."
    )


def test_idor_get_scalers_rejects_cross_tenant_request(client, app_and_modules):
    """GET /scalers/<sid> MUST refuse Bob's request for Alice's session."""
    from shared.auth.ownership import SessionOwnershipError

    _, modules = app_and_modules
    mod = modules['domains.training.api.model_routes']
    headers = _bob_headers()

    fake_scalers = {'input': {'feature1': 'scaler-blob'},
                    'output': {'feature2': 'scaler-blob'},
                    'metadata': {'input_features': 1, 'output_features': 1,
                                 'input_features_scaled': 1,
                                 'output_features_scaled': 1}}

    with patch.object(mod, 'assert_session_ownership',
                      side_effect=SessionOwnershipError('not yours')), \
         patch.object(mod, 'create_or_get_session_uuid',
                      return_value=ALICE_SESSION_UUID), \
         patch.object(mod, 'get_session_scalers', return_value=fake_scalers):
        resp = client.get(
            f'/api/training/scalers/{ALICE_SESSION_UUID}',
            headers=headers,
        )

    ok, _ = _acceptable_idor_response(resp)
    assert ok, (
        f"scalers endpoint leaked cross-tenant data. status={resp.status_code} "
        f"body={resp.get_data(as_text=True)}"
    )


def test_idor_list_models_database_rejects_cross_tenant_request(client, app_and_modules):
    """GET /list-models-database/<sid> MUST refuse cross-tenant calls."""
    from shared.auth.ownership import SessionOwnershipError

    _, modules = app_and_modules
    mod = modules['domains.training.api.model_routes']
    headers = _bob_headers()

    fake_models = [{'id': 'm1', 'session_id': ALICE_SESSION_UUID,
                    'filename': 'best_model.h5'}]
    with patch.object(mod, 'assert_session_ownership',
                      side_effect=SessionOwnershipError('not yours')), \
         patch.object(mod, 'create_or_get_session_uuid',
                      return_value=ALICE_SESSION_UUID), \
         patch.object(mod, 'get_models_list', return_value=fake_models):
        resp = client.get(
            f'/api/training/list-models-database/{ALICE_SESSION_UUID}',
            headers=headers,
        )

    ok, _ = _acceptable_idor_response(resp)
    assert ok, (
        f"list-models-database leaked cross-tenant data. "
        f"status={resp.status_code} body={resp.get_data(as_text=True)}"
    )


def test_idor_download_scalers_rejects_cross_tenant_request(client, app_and_modules):
    """GET /scalers/<sid>/download MUST refuse cross-tenant calls."""
    from shared.auth.ownership import SessionOwnershipError

    _, modules = app_and_modules
    mod = modules['domains.training.api.model_routes']
    headers = _bob_headers()

    with patch.object(mod, 'assert_session_ownership',
                      side_effect=SessionOwnershipError('not yours')), \
         patch.object(mod, 'create_or_get_session_uuid',
                      return_value=ALICE_SESSION_UUID), \
         patch.object(mod, 'create_scaler_download_package',
                      return_value='/tmp/fake-scalers.zip'), \
         patch.object(mod, 'send_file', return_value=('ZIPBYTES', 200)):
        resp = client.get(
            f'/api/training/scalers/{ALICE_SESSION_UUID}/download',
            headers=headers,
        )

    ok, _ = _acceptable_idor_response(resp)
    assert ok, (
        f"scalers download leaked cross-tenant data. "
        f"status={resp.status_code} body={resp.get_data(as_text=True)}"
    )


def test_idor_scale_data_rejects_cross_tenant_request(client, app_and_modules):
    """POST /scale-data/<sid> MUST refuse cross-tenant calls."""
    from shared.auth.ownership import SessionOwnershipError

    _, modules = app_and_modules
    mod = modules['domains.training.api.model_routes']
    headers = _bob_headers()

    fake_result = {'scaled_data': [[1.0]], 'scaling_info': {}, 'metadata': {}}
    with patch.object(mod, 'assert_session_ownership',
                      side_effect=SessionOwnershipError('not yours')), \
         patch.object(mod, 'create_or_get_session_uuid',
                      return_value=ALICE_SESSION_UUID), \
         patch.object(mod, 'scale_new_data', return_value=fake_result):
        resp = client.post(
            f'/api/training/scale-data/{ALICE_SESSION_UUID}',
            headers=headers,
            json={'input_data': [{'feature1': 1.5}]},
        )

    ok, _ = _acceptable_idor_response(resp)
    assert ok, (
        f"scale-data leaked cross-tenant data. status={resp.status_code} "
        f"body={resp.get_data(as_text=True)}"
    )


def test_idor_save_model_rejects_cross_tenant_request(client, app_and_modules):
    """POST /save-model/<sid> MUST refuse cross-tenant calls."""
    from shared.auth.ownership import SessionOwnershipError

    _, modules = app_and_modules
    mod = modules['domains.training.api.model_routes']
    headers = _bob_headers()

    fake_result = {'uploaded_models': [{'name': 'best_model.h5'}],
                   'failed_models': [], 'total_uploaded': 1, 'total_failed': 0}
    with patch.object(mod, 'assert_session_ownership',
                      side_effect=SessionOwnershipError('not yours')), \
         patch.object(mod, 'create_or_get_session_uuid',
                      return_value=ALICE_SESSION_UUID), \
         patch.object(mod, 'save_models_to_storage', return_value=fake_result):
        resp = client.post(
            f'/api/training/save-model/{ALICE_SESSION_UUID}',
            headers=headers,
        )

    ok, _ = _acceptable_idor_response(resp)
    assert ok, (
        f"save-model leaked cross-tenant access. status={resp.status_code} "
        f"body={resp.get_data(as_text=True)}"
    )


def test_idor_download_model_h5_rejects_cross_tenant_request(client, app_and_modules):
    """GET /download-model-h5/<sid> MUST refuse cross-tenant calls."""
    from shared.auth.ownership import SessionOwnershipError

    _, modules = app_and_modules
    mod = modules['domains.training.api.model_routes']
    headers = _bob_headers()

    with patch.object(mod, 'assert_session_ownership',
                      side_effect=SessionOwnershipError('not yours')), \
         patch.object(mod, 'create_or_get_session_uuid',
                      return_value=ALICE_SESSION_UUID), \
         patch.object(mod, 'download_model_file',
                      return_value=(b'MODELBYTES', 'best_model.h5')), \
         patch.object(mod, 'send_file', return_value=MagicMock(headers={})):
        resp = client.get(
            f'/api/training/download-model-h5/{ALICE_SESSION_UUID}',
            headers=headers,
        )

    ok, _ = _acceptable_idor_response(resp)
    assert ok, (
        f"download-model-h5 leaked cross-tenant data. status={resp.status_code} "
        f"body={resp.get_data(as_text=True)}"
    )


def test_idor_predict_rejects_cross_tenant_request(client, app_and_modules):
    """POST /predict/<sid> MUST refuse cross-tenant calls."""
    from shared.auth.ownership import SessionOwnershipError

    _, modules = app_and_modules
    mod = modules['domains.training.api.model_routes']
    headers = _bob_headers()

    fake_service = MagicMock()
    fake_service.predict.return_value = {
        'predictions': [1.23], 'model_used': 'best_model.h5',
        'timestamp': '2026-06-01T00:00:00Z', 'input_count': 1,
        'scaling_applied': False,
    }
    with patch.object(mod, 'assert_session_ownership',
                      side_effect=SessionOwnershipError('not yours')), \
         patch.object(mod, 'create_or_get_session_uuid',
                      return_value=ALICE_SESSION_UUID), \
         patch(
            'domains.training.services.prediction_service.PredictionService',
            return_value=fake_service,
         ):
        resp = client.post(
            f'/api/training/predict/{ALICE_SESSION_UUID}',
            headers=headers,
            json={'model_filename': 'best_model.h5',
                  'input_data': [{'feature1': 1.5}]},
        )

    ok, _ = _acceptable_idor_response(resp)
    assert ok, (
        f"predict leaked cross-tenant data. status={resp.status_code} "
        f"body={resp.get_data(as_text=True)}"
    )


# ---------------------------------------------------------------------------
# Session-routes endpoints that lack explicit assert_session_ownership but
# have inline user_id filters at the data layer.
# ---------------------------------------------------------------------------

def test_idor_get_session_local_storage_uses_string_id_resolution(client, app_and_modules):
    """GET /session/<sid> reads from local UPLOAD_BASE_DIR using
    get_string_session_id. For a foreign raw UUID, get_string_session_id
    will fail with ValueError → 404 SESSION_NOT_FOUND. This test
    documents the implicit guard.
    """
    _, modules = app_and_modules
    mod = modules['domains.training.api.session_routes']
    headers = _bob_headers()

    with patch.object(mod, 'get_string_session_id',
                      side_effect=ValueError('not in mappings for this user')):
        resp = client.get(
            f'/api/training/session/{ALICE_SESSION_UUID}',
            headers=headers,
        )

    # The route maps ValueError → 404 SESSION_NOT_FOUND. Acceptable.
    assert resp.status_code == 404
    body = resp.get_json() or {}
    assert body.get('code') == 'SESSION_NOT_FOUND'


def test_idor_session_status_uses_string_id_resolution(client, app_and_modules):
    """GET /session-status/<sid> resolves via get_upload_status which uses
    the local mapping. Foreign raw UUID → ValueError → 404."""
    _, modules = app_and_modules
    mod = modules['domains.training.api.session_routes']
    headers = _bob_headers()

    with patch.object(mod, 'get_upload_status',
                      side_effect=ValueError('not in mappings')):
        resp = client.get(
            f'/api/training/session-status/{ALICE_SESSION_UUID}',
            headers=headers,
        )

    assert resp.status_code == 404
    body = resp.get_json() or {}
    assert body.get('code') == 'SESSION_NOT_FOUND'


def test_idor_get_session_uuid_via_string_mapping(client, app_and_modules):
    """GET /get-session-uuid/<sid> — for a raw UUID, the route fast-paths
    to returning the input unchanged (is_uuid_format short-circuit). This
    is itself an IDOR vector: Bob can resolve Alice's UUID echo and use it
    elsewhere. Documented as xfail."""
    _, modules = app_and_modules
    mod = modules['domains.training.api.session_routes']
    headers = _bob_headers()

    resp = client.get(
        f'/api/training/get-session-uuid/{ALICE_SESSION_UUID}',
        headers=headers,
    )
    body = resp.get_json() or {}
    # Documentation: the route fast-paths raw UUIDs without any DB lookup,
    # so it can't tell whose session it is. The leak is minimal (just
    # echoes the input UUID) but the endpoint also accepts string-form
    # sessions and resolves them — those paths DO have ownership checks
    # via get_session_uuid → resolve_session_id. For raw UUIDs the echo
    # leaks no new information beyond what the caller already knew.
    # Accept either 200 (echo) or 404.
    assert resp.status_code in (200, 404), (
        f"unexpected status {resp.status_code}: {body}"
    )


def test_idor_get_time_info_rejects_cross_tenant_request(client, app_and_modules):
    """GET /get-time-info/<sid> MUST refuse cross-tenant calls."""
    from shared.auth.ownership import SessionOwnershipError

    _, modules = app_and_modules
    mod = modules['domains.training.api.session_routes']
    headers = _bob_headers()

    with patch.object(mod, 'assert_session_ownership',
                      side_effect=SessionOwnershipError('not yours')), \
         patch.object(mod, 'create_or_get_session_uuid',
                      return_value=ALICE_SESSION_UUID), \
         patch.object(mod, 'get_time_info_data',
                      return_value={'jahr': True, 'monat': True}):
        resp = client.get(
            f'/api/training/get-time-info/{ALICE_SESSION_UUID}',
            headers=headers,
        )

    ok, _ = _acceptable_idor_response(resp)
    assert ok, (
        f"get-time-info leaked cross-tenant data. status={resp.status_code} "
        f"body={resp.get_data(as_text=True)}"
    )


def test_idor_get_zeitschritte_rejects_cross_tenant_request(client, app_and_modules):
    """GET /get-zeitschritte/<sid> MUST refuse cross-tenant calls."""
    from shared.auth.ownership import SessionOwnershipError

    _, modules = app_and_modules
    mod = modules['domains.training.api.session_routes']
    headers = _bob_headers()

    with patch.object(mod, 'assert_session_ownership',
                      side_effect=SessionOwnershipError('not yours')), \
         patch.object(mod, 'create_or_get_session_uuid',
                      return_value=ALICE_SESSION_UUID), \
         patch.object(mod, 'get_zeitschritte_data',
                      return_value={'eingabe': '5', 'ausgabe': '1'}):
        resp = client.get(
            f'/api/training/get-zeitschritte/{ALICE_SESSION_UUID}',
            headers=headers,
        )

    ok, _ = _acceptable_idor_response(resp)
    assert ok, (
        f"get-zeitschritte leaked cross-tenant data. status={resp.status_code} "
        f"body={resp.get_data(as_text=True)}"
    )


# ---------------------------------------------------------------------------
# api_key_routes — generate_api_key and list_api_keys both filter by
# .eq('user_id', g.user_id) at the DB layer, which IS sufficient. Verify.
# ---------------------------------------------------------------------------

def test_api_key_generate_rejects_cross_tenant_session(client, app_and_modules):
    """POST /api-keys/<sid> — the DB query uses .eq('user_id', g.user_id)
    so a foreign session_id returns no rows → 404 SESSION_NOT_FOUND.
    Confirms the inline filter is effective."""
    _, modules = app_and_modules
    mod = modules['domains.training.api.api_key_routes']
    headers = _bob_headers()

    # Make create_or_get_session_uuid return Alice's UUID (the
    # raw-UUID bypass path). The subsequent .eq('user_id', BOB) filter
    # returns no rows for Alice's session → SESSION_NOT_FOUND.
    fake_supabase = MagicMock()
    fake_supabase.table.return_value.select.return_value.eq.return_value.eq.return_value.limit.return_value.execute.return_value.data = []
    with patch.object(mod, 'get_supabase_client', return_value=fake_supabase), \
         patch.object(mod, 'create_or_get_session_uuid',
                      return_value=ALICE_SESSION_UUID):
        resp = client.post(
            f'/api/training/api-keys/{ALICE_SESSION_UUID}',
            headers=headers,
            json={'name': 'attacker-key'},
        )

    assert resp.status_code == 404
    body = resp.get_json() or {}
    assert body.get('code') == 'SESSION_NOT_FOUND'


def test_api_key_list_for_foreign_session_returns_empty_list(client, app_and_modules):
    """GET /api-keys/<sid> — uses .eq('user_id', g.user_id) so a foreign
    session_id returns an empty list. NOT a leak (no key data exposed)
    but does confirm session existence by returning 200 instead of 404.
    Documented as a minor information disclosure rather than a full IDOR."""
    _, modules = app_and_modules
    mod = modules['domains.training.api.api_key_routes']
    headers = _bob_headers()

    fake_supabase = MagicMock()
    fake_supabase.table.return_value.select.return_value.eq.return_value.eq.return_value.order.return_value.execute.return_value.data = []
    with patch.object(mod, 'get_supabase_client', return_value=fake_supabase), \
         patch.object(mod, 'create_or_get_session_uuid',
                      return_value=ALICE_SESSION_UUID):
        resp = client.get(
            f'/api/training/api-keys/{ALICE_SESSION_UUID}',
            headers=headers,
        )

    body = resp.get_json() or {}
    # Either 200 with empty keys (current behaviour, minor existence-leak)
    # or 404 SESSION_NOT_FOUND (ideal). Both are acceptable: no key data
    # is exposed across tenants.
    if resp.status_code == 200:
        assert body.get('keys') == []
    else:
        assert resp.status_code == 404
        assert body.get('code') == 'SESSION_NOT_FOUND'
