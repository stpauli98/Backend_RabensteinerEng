"""
Coverage for POST /api/stripe/verify-session (FIX-6).

Pins the contract of the verify-session endpoint that closes the
/payment/success spoof:
  1) Missing/malformed session_id → 400 BAD_REQUEST
  2) Stripe session not found → 404 SESSION_NOT_FOUND
  3) Session belongs to a different user → 403 SESSION_FORBIDDEN
  4) payment_status != 'paid' → 402 PAYMENT_NOT_COMPLETED
  5) Happy path → 200 {verified, plan_name, amount, customer_id}

Tests stub `require_auth` so they exercise the route logic, not JWT
verification. Mirror of the test-app pattern used in
tests/domains/training/api/test_session_routes.py.
"""
import os

# Force testing mode (1000/min rate limit) BEFORE any app import.
os.environ.setdefault('FLASK_ENV', 'testing')

import importlib
import json
from functools import wraps
from unittest.mock import patch, MagicMock

import pytest
import stripe
from flask import Flask, g


def _stub_require_auth(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        g.user_id = 'user-id-123'
        g.user_email = 'test@example.com'
        g.access_token = 'test-token'
        return f(*args, **kwargs)
    return wrapper


def _build_app_with_auth_stub():
    """Build a Flask test app with require_auth stubbed and the stripe
    blueprint freshly registered so the stub is the decorator captured at
    import time."""
    from core.rate_limits import limiter

    with patch('shared.auth.jwt.require_auth', side_effect=_stub_require_auth):
        import domains.payments.api.stripe as stripe_module
        importlib.reload(stripe_module)

        app = Flask(__name__)
        app.config['TESTING'] = True
        limiter.init_app(app)
        app.register_blueprint(stripe_module.bp, url_prefix='/api/stripe')

    # Restore the real decorator on the module so subsequent imports get the
    # production behaviour.
    import domains.payments.api.stripe as stripe_module
    importlib.reload(stripe_module)
    return app


@pytest.fixture
def client():
    app = _build_app_with_auth_stub()
    with app.test_client() as c:
        yield c


def _auth_headers():
    return {'Authorization': 'Bearer test-token'}


def _make_session_obj(
    *,
    metadata_user_id='user-id-123',
    payment_status='paid',
    customer='cus_test_owner',
    line_items_data=None,
):
    """Build a dict that mimics a Stripe checkout.Session with .get() access."""
    if line_items_data is None:
        line_items_data = [{
            'amount_total': 4900,
            'currency': 'eur',
            'price': {'id': 'price_basic_monthly', 'product': 'prod_basic'},
        }]
    return {
        'id': 'cs_test_abc',
        'payment_status': payment_status,
        'customer': customer,
        'metadata': {'user_id': metadata_user_id} if metadata_user_id else {},
        'line_items': {'data': line_items_data},
    }


# ---------------------------------------------------------------------------
# Request validation
# ---------------------------------------------------------------------------

def test_missing_session_id_returns_400(client):
    resp = client.post(
        '/api/stripe/verify-session',
        data=json.dumps({}),
        content_type='application/json',
        headers=_auth_headers(),
    )
    assert resp.status_code == 400
    body = json.loads(resp.data)
    assert body['code'] == 'BAD_REQUEST'


def test_malformed_session_id_returns_400(client):
    """Reject anything that doesn't look like a Stripe checkout session ID."""
    resp = client.post(
        '/api/stripe/verify-session',
        data=json.dumps({'session_id': 'not_a_stripe_session_id'}),
        content_type='application/json',
        headers=_auth_headers(),
    )
    assert resp.status_code == 400
    body = json.loads(resp.data)
    assert body['code'] == 'BAD_REQUEST'


# ---------------------------------------------------------------------------
# Stripe lookup failures
# ---------------------------------------------------------------------------

def test_stripe_session_not_found_returns_404(client):
    import domains.payments.api.stripe as stripe_module

    with patch.object(
        stripe_module.stripe.checkout.Session,
        'retrieve',
        side_effect=stripe.error.InvalidRequestError('No such session', 'session_id'),
    ):
        resp = client.post(
            '/api/stripe/verify-session',
            data=json.dumps({'session_id': 'cs_test_DOES_NOT_EXIST'}),
            content_type='application/json',
            headers=_auth_headers(),
        )

    assert resp.status_code == 404
    body = json.loads(resp.data)
    assert body['code'] == 'SESSION_NOT_FOUND'


# ---------------------------------------------------------------------------
# Ownership check
# ---------------------------------------------------------------------------

def test_session_belonging_to_other_user_returns_403(client):
    """Session metadata.user_id matches a different user and the caller's
    customer mapping doesn't match either — must 403."""
    import domains.payments.api.stripe as stripe_module

    other_session = _make_session_obj(metadata_user_id='different-user-456')

    # Empty mapping lookup → caller is not the customer either.
    fake_supabase = MagicMock()
    fake_supabase.table.return_value.select.return_value.eq.return_value \
        .eq.return_value.limit.return_value.execute.return_value = MagicMock(data=[])

    with patch.object(stripe_module.stripe.checkout.Session, 'retrieve', return_value=other_session), \
         patch.object(stripe_module, 'get_supabase_admin_client', return_value=fake_supabase):
        resp = client.post(
            '/api/stripe/verify-session',
            data=json.dumps({'session_id': 'cs_test_belongs_to_other'}),
            content_type='application/json',
            headers=_auth_headers(),
        )

    assert resp.status_code == 403
    body = json.loads(resp.data)
    assert body['code'] == 'SESSION_FORBIDDEN'


# ---------------------------------------------------------------------------
# Payment status check
# ---------------------------------------------------------------------------

def test_unpaid_session_returns_402(client):
    import domains.payments.api.stripe as stripe_module

    unpaid = _make_session_obj(payment_status='unpaid')

    with patch.object(stripe_module.stripe.checkout.Session, 'retrieve', return_value=unpaid):
        resp = client.post(
            '/api/stripe/verify-session',
            data=json.dumps({'session_id': 'cs_test_unpaid'}),
            content_type='application/json',
            headers=_auth_headers(),
        )

    assert resp.status_code == 402
    body = json.loads(resp.data)
    assert body['code'] == 'PAYMENT_NOT_COMPLETED'


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_valid_session_returns_200_with_plan_info(client):
    import domains.payments.api.stripe as stripe_module

    session = _make_session_obj()

    # Subscription_plans lookup returns the plan name; user_subscriptions
    # mapping lookup is not used because metadata.user_id matches.
    fake_supabase = MagicMock()
    plan_chain = fake_supabase.table.return_value.select.return_value.or_ \
        .return_value.limit.return_value.execute
    plan_chain.return_value = MagicMock(data=[{'name': 'BASIC'}])

    with patch.object(stripe_module.stripe.checkout.Session, 'retrieve', return_value=session), \
         patch.object(stripe_module, 'get_supabase_admin_client', return_value=fake_supabase):
        resp = client.post(
            '/api/stripe/verify-session',
            data=json.dumps({'session_id': 'cs_test_paid_for_caller'}),
            content_type='application/json',
            headers=_auth_headers(),
        )

    assert resp.status_code == 200
    body = json.loads(resp.data)
    assert body['verified'] is True
    assert body['plan_name'] == 'BASIC'
    assert body['amount'] == 49.0  # 4900 cents → 49.00
    assert body['currency'] == 'eur'
    assert body['customer_id'] == 'cus_test_owner'
