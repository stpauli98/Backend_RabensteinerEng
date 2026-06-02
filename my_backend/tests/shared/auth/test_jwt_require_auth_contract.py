"""W11-ADV-3: require_auth 401 responses must follow {success, code, error} contract.

Pre-fix, the decorator returned bare `{error: "..."}` 401 bodies, which broke the
FE error mapper (it keys off `code`, not the free-form `error` string).
"""
import os

# Force testing mode (1000/min rate limit) BEFORE any app import.
os.environ.setdefault('FLASK_ENV', 'testing')

import importlib
from functools import wraps

import pytest
from flask import Flask, g


def _stub_require_subscription(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        g.subscription = {'id': 'sub-1'}
        g.plan = {'name': 'pro', 'max_processing_jobs_per_month': 100, 'total_compute_hours': 0}
        g.usage = {'processing_jobs_count': 0, 'processing_count': 0}
        return f(*args, **kwargs)
    return wrapper


def _stub_check_processing_limit(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        return f(*args, **kwargs)
    return wrapper


@pytest.fixture
def client():
    """Build a minimal app with the REAL require_auth decorator wired in.

    We stub only the subscription/usage decorators so the auth contract can be
    exercised in isolation without hitting Stripe or Supabase usage tables.
    """
    from unittest.mock import patch
    from core.rate_limits import limiter

    with patch(
        'shared.auth.subscription.require_subscription',
        side_effect=_stub_require_subscription,
    ), patch(
        'shared.auth.subscription.check_processing_limit',
        side_effect=_stub_check_processing_limit,
    ):
        import domains.training.api.common as common_module
        importlib.reload(common_module)
        import domains.training.api.upload_routes as upload_routes
        importlib.reload(upload_routes)

        app = Flask(__name__)
        app.config['TESTING'] = True
        limiter.init_app(app)
        app.register_blueprint(upload_routes.bp, url_prefix='/api/training')

        with app.test_client() as c:
            yield c

    # Restore real decorators for downstream tests.
    import domains.training.api.common as common_module
    importlib.reload(common_module)
    import domains.training.api.upload_routes as upload_routes
    importlib.reload(upload_routes)


def test_missing_auth_header_returns_structured_401(client):
    """No Authorization header → 401 with code=MISSING_AUTHORIZATION."""
    r = client.get("/api/training/csv-files/00000000-0000-0000-0000-000000000000")
    assert r.status_code == 401
    body = r.get_json()
    assert body is not None, f"Response must be JSON, got: {r.get_data(as_text=True)}"
    assert body.get('success') is False, f"Expected success=False, got: {body}"
    assert body.get('code') == 'MISSING_AUTHORIZATION', (
        f"Expected code=MISSING_AUTHORIZATION, got: {body}"
    )
    assert 'error' in body


def test_malformed_token_returns_structured_401(client):
    """Bearer with garbage payload → 401 with code=INVALID_TOKEN."""
    r = client.get(
        "/api/training/csv-files/00000000-0000-0000-0000-000000000000",
        headers={"Authorization": "Bearer not.a.jwt"},
    )
    assert r.status_code == 401
    body = r.get_json()
    assert body is not None
    assert body.get('success') is False
    assert body.get('code') == 'INVALID_TOKEN', (
        f"Expected code=INVALID_TOKEN, got: {body}"
    )
    assert 'error' in body


def test_wrong_token_type_returns_structured_401(client):
    """Non-Bearer token type → 401 with code=INVALID_TOKEN."""
    r = client.get(
        "/api/training/csv-files/00000000-0000-0000-0000-000000000000",
        headers={"Authorization": "Basic dXNlcjpwYXNz"},
    )
    assert r.status_code == 401
    body = r.get_json()
    assert body is not None
    assert body.get('success') is False
    assert body.get('code') == 'INVALID_TOKEN'
