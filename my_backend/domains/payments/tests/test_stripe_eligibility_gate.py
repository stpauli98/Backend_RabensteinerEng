"""Checkout gate + eligibility endpoint tests (#131)."""
from types import SimpleNamespace

import domains.payments.api.stripe as stripe_routes


def _patch_sb(monkeypatch, fake_supabase, *, active_subs=None, plans=None):
    sb = fake_supabase({
        'user_subscriptions': active_subs or [],
        'subscription_plans': plans or [],
    })
    monkeypatch.setattr(stripe_routes, 'get_supabase_admin_client', lambda: sb)
    return sb


def _block_stripe(monkeypatch):
    # Fail loudly if Stripe is reached on a path that must short-circuit first.
    def _boom(*a, **k):
        raise AssertionError("Stripe should not be called on this path")
    monkeypatch.setattr(stripe_routes, 'get_or_create_stripe_customer', _boom)


def _allow_stripe(monkeypatch):
    monkeypatch.setattr(stripe_routes, 'get_or_create_stripe_customer',
                        lambda uid, email: 'cus_test')
    monkeypatch.setattr(stripe_routes, 'get_stripe_price_id',
                        lambda pid, cycle: 'price_test')
    monkeypatch.setattr(stripe_routes.stripe.checkout.Session, 'create',
                        staticmethod(lambda **kw: SimpleNamespace(
                            id='cs_test', url='https://checkout.stripe.com/c/pay/cs_test')))


def test_checkout_blocks_api_only_when_no_model(client, auth_header, monkeypatch, fake_supabase):
    _patch_sb(monkeypatch, fake_supabase,
              plans=[{'id': 'p_api', 'slug': 'api_only', 'is_upgrade_only': True}])
    monkeypatch.setattr(stripe_routes, 'has_trained_model', lambda sb, uid: False)
    _block_stripe(monkeypatch)
    resp = client.post('/api/stripe/create-checkout-session',
                       json={'plan_id': 'p_api', 'billing_cycle': 'monthly'},
                       headers=auth_header)
    assert resp.status_code == 403
    assert resp.get_json()['code'] == 'ELIGIBILITY_NOT_MET'


def test_checkout_allows_api_only_when_model_exists(client, auth_header, monkeypatch, fake_supabase):
    _patch_sb(monkeypatch, fake_supabase,
              plans=[{'id': 'p_api', 'slug': 'api_only', 'is_upgrade_only': True}])
    monkeypatch.setattr(stripe_routes, 'has_trained_model', lambda sb, uid: True)
    _allow_stripe(monkeypatch)
    resp = client.post('/api/stripe/create-checkout-session',
                       json={'plan_id': 'p_api', 'billing_cycle': 'monthly'},
                       headers=auth_header)
    assert resp.status_code == 200
    assert resp.get_json()['url'].startswith('https://checkout.stripe.com/')


def test_checkout_non_upgrade_only_skips_eligibility(client, auth_header, monkeypatch, fake_supabase):
    _patch_sb(monkeypatch, fake_supabase,
              plans=[{'id': 'p_std', 'slug': 'standard', 'is_upgrade_only': False}])

    def _must_not_call(sb, uid):
        raise AssertionError("has_trained_model must not run for non-upgrade-only plans")
    monkeypatch.setattr(stripe_routes, 'has_trained_model', _must_not_call)
    _allow_stripe(monkeypatch)
    resp = client.post('/api/stripe/create-checkout-session',
                       json={'plan_id': 'p_std', 'billing_cycle': 'monthly'},
                       headers=auth_header)
    assert resp.status_code == 200


def test_eligibility_endpoint_returns_true(client, auth_header, monkeypatch, fake_supabase):
    monkeypatch.setattr(stripe_routes, 'get_supabase_admin_client',
                        lambda: fake_supabase({}))
    monkeypatch.setattr(stripe_routes, 'has_trained_model', lambda sb, uid: True)
    resp = client.get('/api/stripe/api-only-eligibility', headers=auth_header)
    assert resp.status_code == 200
    assert resp.get_json() == {'eligible': True}


def test_eligibility_endpoint_returns_false(client, auth_header, monkeypatch, fake_supabase):
    monkeypatch.setattr(stripe_routes, 'get_supabase_admin_client',
                        lambda: fake_supabase({}))
    monkeypatch.setattr(stripe_routes, 'has_trained_model', lambda sb, uid: False)
    resp = client.get('/api/stripe/api-only-eligibility', headers=auth_header)
    assert resp.status_code == 200
    assert resp.get_json() == {'eligible': False}


def test_eligibility_endpoint_requires_auth(client):
    resp = client.get('/api/stripe/api-only-eligibility')  # no auth header
    assert resp.status_code == 401
