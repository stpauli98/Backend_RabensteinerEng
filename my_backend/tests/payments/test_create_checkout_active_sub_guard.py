"""
Coverage for the FIX-6 active-subscription guard on
POST /api/stripe/create-checkout-session.

Pins:
  1) User with an active subscription → 409 SUBSCRIPTION_ALREADY_ACTIVE
     and Stripe Session.create is NEVER called.
  2) User with a trialing subscription → same 409 + no Stripe call.
  3) User with no active/trialing row → falls through to Stripe Session.create
     and we DO call it.
  4) User with only cancelled rows → falls through (treated as no active sub).

This is a security boundary — clients can spoof a "no active sub" claim,
so the BE check is the source of truth.
"""
import os

# Force testing mode (1000/min rate limit) BEFORE any app import.
os.environ.setdefault('FLASK_ENV', 'testing')

import importlib
import json
from functools import wraps
from unittest.mock import patch, MagicMock

import pytest
from flask import Flask, g


def _stub_require_auth(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        g.user_id = 'user-id-active-sub'
        g.user_email = 'active@example.com'
        g.access_token = 'test-token'
        return f(*args, **kwargs)
    return wrapper


def _build_app_with_auth_stub():
    from core.rate_limits import limiter

    with patch('shared.auth.jwt.require_auth', side_effect=_stub_require_auth):
        import domains.payments.api.stripe as stripe_module
        importlib.reload(stripe_module)

        app = Flask(__name__)
        app.config['TESTING'] = True
        limiter.init_app(app)
        app.register_blueprint(stripe_module.bp, url_prefix='/api/stripe')

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


def _supabase_with_subscription_rows(rows):
    """Build a MagicMock supabase whose
    table('user_subscriptions').select(...).eq('user_id', ...).in_('status', ...).execute()
    returns `rows`.
    """
    fake = MagicMock()
    fake.table.return_value.select.return_value.eq.return_value.in_.return_value \
        .execute.return_value = MagicMock(data=rows)
    return fake


def test_active_subscription_rejects_with_409(client):
    """User already on an active plan must NOT be able to create a new
    checkout session — that would result in double-billing."""
    import domains.payments.api.stripe as stripe_module

    fake_supabase = _supabase_with_subscription_rows([{
        'id': 'sub-row-1',
        'plan_id': 'plan-premium-uuid',
        'status': 'active',
        'expires_at': '2030-01-01T00:00:00+00:00',
    }])

    with patch.object(stripe_module, 'get_supabase_admin_client', return_value=fake_supabase), \
         patch.object(stripe_module, 'get_or_create_stripe_customer') as mock_customer, \
         patch.object(stripe_module.stripe.checkout.Session, 'create') as mock_create:
        resp = client.post(
            '/api/stripe/create-checkout-session',
            data=json.dumps({'plan_id': 'plan-basic-uuid', 'billing_cycle': 'monthly'}),
            content_type='application/json',
            headers=_auth_headers(),
        )

    assert resp.status_code == 409
    body = json.loads(resp.data)
    assert body['code'] == 'SUBSCRIPTION_ALREADY_ACTIVE'
    assert 'suggestion' in body
    # CRITICAL: Stripe must NOT have been called — that's the whole point.
    mock_create.assert_not_called()
    mock_customer.assert_not_called()


def test_trialing_subscription_rejects_with_409(client):
    """Trialing counts as active for billing safety — same guard applies."""
    import domains.payments.api.stripe as stripe_module

    fake_supabase = _supabase_with_subscription_rows([{
        'id': 'sub-row-2',
        'plan_id': 'plan-premium-uuid',
        'status': 'trialing',
        'expires_at': '2030-01-01T00:00:00+00:00',
    }])

    with patch.object(stripe_module, 'get_supabase_admin_client', return_value=fake_supabase), \
         patch.object(stripe_module, 'get_or_create_stripe_customer') as mock_customer, \
         patch.object(stripe_module.stripe.checkout.Session, 'create') as mock_create:
        resp = client.post(
            '/api/stripe/create-checkout-session',
            data=json.dumps({'plan_id': 'plan-basic-uuid', 'billing_cycle': 'monthly'}),
            content_type='application/json',
            headers=_auth_headers(),
        )

    assert resp.status_code == 409
    body = json.loads(resp.data)
    assert body['code'] == 'SUBSCRIPTION_ALREADY_ACTIVE'
    mock_create.assert_not_called()
    mock_customer.assert_not_called()


def test_no_existing_subscription_falls_through_to_stripe(client):
    """If user has no active/trialing row, checkout should proceed normally.
    This is the regression guard — the new check must not block legitimate
    first-time signups."""
    import domains.payments.api.stripe as stripe_module

    fake_supabase = _supabase_with_subscription_rows([])

    mock_session = MagicMock(
        id='cs_test_new_signup',
        url='https://checkout.stripe.com/c/pay/cs_test_new_signup',
    )

    with patch.object(stripe_module, 'get_supabase_admin_client', return_value=fake_supabase), \
         patch.object(stripe_module, 'get_or_create_stripe_customer', return_value='cus_test_new'), \
         patch.object(stripe_module, 'get_stripe_price_id', return_value='price_basic_monthly'), \
         patch.object(stripe_module.stripe.checkout.Session, 'create', return_value=mock_session) as mock_create:
        resp = client.post(
            '/api/stripe/create-checkout-session',
            data=json.dumps({'plan_id': 'plan-basic-uuid', 'billing_cycle': 'monthly'}),
            content_type='application/json',
            headers=_auth_headers(),
        )

    assert resp.status_code == 200
    body = json.loads(resp.data)
    assert body['session_id'] == 'cs_test_new_signup'
    mock_create.assert_called_once()


def test_cancelled_subscription_does_not_block(client):
    """A row with status='cancelled' must not block a new checkout — only
    'active' and 'trialing' should. Without this, every user who has ever
    cancelled would be permanently locked out of buying again."""
    import domains.payments.api.stripe as stripe_module

    # The .in_('status', ['active', 'trialing']) filter is applied in the
    # query, so a cancelled-only user's query returns []. We simulate that.
    fake_supabase = _supabase_with_subscription_rows([])

    mock_session = MagicMock(
        id='cs_test_returning_user',
        url='https://checkout.stripe.com/c/pay/cs_test_returning_user',
    )

    with patch.object(stripe_module, 'get_supabase_admin_client', return_value=fake_supabase), \
         patch.object(stripe_module, 'get_or_create_stripe_customer', return_value='cus_test_returning'), \
         patch.object(stripe_module, 'get_stripe_price_id', return_value='price_basic_monthly'), \
         patch.object(stripe_module.stripe.checkout.Session, 'create', return_value=mock_session) as mock_create:
        resp = client.post(
            '/api/stripe/create-checkout-session',
            data=json.dumps({'plan_id': 'plan-basic-uuid', 'billing_cycle': 'monthly'}),
            content_type='application/json',
            headers=_auth_headers(),
        )

    assert resp.status_code == 200
    mock_create.assert_called_once()
