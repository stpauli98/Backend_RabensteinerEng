"""C-2 (CRITICAL IDOR) regression test for POST /generate-datasets/<session_id>.

The route resolves the path session_id to a UUID via
``create_or_get_session_uuid``, which returns raw UUIDs UNCHANGED without any
ownership validation. Without an explicit ``assert_session_ownership`` guard,
an authenticated attacker (Bob) can submit Alice's raw session UUID and have
the backend load and process Alice's CSV data — billing his own quota against
her session.

The sibling handler ``train_models`` already guards this correctly. This test
pins the same semantics on ``generate_datasets``: a foreign / non-owned session
must yield 404 SESSION_NOT_FOUND BEFORE any data access (the dataset generator
is patched to fail loudly if it is ever reached).

Harness follows the established pattern in ``test_idor_multi_user.py``: stub the
three auth/subscription decorators in ``shared.auth`` before (re)importing the
route module so the blueprint can be exercised without Supabase/JWT, then patch
the module-level ``assert_session_ownership`` to raise.
"""
import os

# Force testing mode (relaxed rate limit) BEFORE any app import.
os.environ.setdefault('FLASK_ENV', 'testing')

import importlib
from functools import wraps
from unittest.mock import patch, MagicMock

import pytest
from flask import Flask


ALICE_SESSION_UUID = 'aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa'
BOB_ID = 'user-bob-bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb'


def _stub_require_auth(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        from flask import g
        g.user_id = BOB_ID
        g.user_email = f'{BOB_ID}@example.com'
        g.access_token = 'test-token'
        return f(*args, **kwargs)
    return wrapper


def _stub_require_subscription(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        from flask import g
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


def _stub_passthrough(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        return f(*args, **kwargs)
    return wrapper


@pytest.fixture
def app_and_module():
    """Build a Flask app with the training_routes blueprint, auth decorators
    stubbed out. Yields (app, route_module)."""
    from core.rate_limits import limiter

    with patch('shared.auth.jwt.require_auth', side_effect=_stub_require_auth), \
         patch('shared.auth.subscription.require_subscription', side_effect=_stub_require_subscription), \
         patch('shared.auth.subscription.check_processing_limit', side_effect=_stub_passthrough), \
         patch('shared.auth.subscription.check_training_limit', side_effect=_stub_passthrough):
        import domains.training.api.common as common_module
        importlib.reload(common_module)
        mod = importlib.import_module('domains.training.api.training_routes')
        importlib.reload(mod)

        app = Flask(__name__)
        app.config['TESTING'] = True
        limiter.init_app(app)
        app.register_blueprint(mod.bp, url_prefix='/api/training')

    yield app, mod

    # Restore real decorators for subsequent test files.
    import domains.training.api.common as common_module
    importlib.reload(common_module)
    importlib.reload(importlib.import_module('domains.training.api.training_routes'))


@pytest.fixture
def client(app_and_module):
    app, _ = app_and_module
    with app.test_client() as c:
        yield c


def test_generate_datasets_rejects_cross_tenant_request(client, app_and_module):
    """POST /generate-datasets/<sid> MUST return 404 SESSION_NOT_FOUND when the
    caller does not own the session, BEFORE any dataset generation occurs.

    create_or_get_session_uuid returns the raw UUID unchanged (the production
    bypass path); assert_session_ownership raises to simulate Bob hitting
    Alice's session. The data layer (generate_violin_plots_for_session) is
    patched to explode if it is ever reached — proving the guard fires first.
    """
    from shared.auth.ownership import SessionOwnershipError

    _, mod = app_and_module

    data_layer_reached = MagicMock(
        side_effect=AssertionError(
            'generate_violin_plots_for_session was called — ownership guard '
            'did NOT fire before data access (IDOR).'
        )
    )

    with patch.object(mod, 'assert_session_ownership',
                      side_effect=SessionOwnershipError('not yours')), \
         patch.object(mod, 'create_or_get_session_uuid',
                      return_value=ALICE_SESSION_UUID), \
         patch.object(mod, 'generate_violin_plots_for_session',
                      data_layer_reached):
        resp = client.post(
            f'/api/training/generate-datasets/{ALICE_SESSION_UUID}',
            headers={'Authorization': 'Bearer bob-token'},
            json={'model_parameters': {}, 'training_split': {}},
        )

    assert resp.status_code == 404, (
        f"generate-datasets accepted cross-tenant request. "
        f"status={resp.status_code} body={resp.get_data(as_text=True)}"
    )
    body = resp.get_json() or {}
    assert body.get('code') == 'SESSION_NOT_FOUND', (
        f"expected SESSION_NOT_FOUND, got body={body}"
    )
    assert not data_layer_reached.called, (
        'IDOR: data layer was reached despite ownership failure.'
    )
