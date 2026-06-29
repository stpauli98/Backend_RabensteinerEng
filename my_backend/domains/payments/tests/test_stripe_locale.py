"""#127 — Stripe Checkout / Billing Portal locale is the app's language.

Without an explicit `locale`, Stripe auto-detects (often German). These tests
assert the routes forward the client-supplied language, normalized + defaulted
to 'en'.
"""
from types import SimpleNamespace

import domains.payments.api.stripe as stripe_routes


def _capture_checkout(monkeypatch, captured):
    monkeypatch.setattr(stripe_routes, 'get_or_create_stripe_customer',
                        lambda uid, email: 'cus_x')
    monkeypatch.setattr(stripe_routes, 'get_stripe_price_id',
                        lambda pid, cyc: 'price_x')

    def _create(**kw):
        captured.update(kw)
        return SimpleNamespace(id='cs_x', url='https://checkout.stripe.com/c/pay/cs_x')

    monkeypatch.setattr(stripe_routes.stripe.checkout.Session, 'create',
                        staticmethod(_create))


def _checkout(client, auth_header, monkeypatch, fake_supabase, body):
    sb = fake_supabase({
        'user_subscriptions': [],
        'subscription_plans': [{'id': 'p_std', 'slug': 'standard', 'is_upgrade_only': False}],
    })
    monkeypatch.setattr(stripe_routes, 'get_supabase_admin_client', lambda: sb)
    cap = {}
    _capture_checkout(monkeypatch, cap)
    resp = client.post('/api/stripe/create-checkout-session', json=body, headers=auth_header)
    return resp, cap


def test_checkout_forwards_de_locale(client, auth_header, monkeypatch, fake_supabase):
    resp, cap = _checkout(client, auth_header, monkeypatch, fake_supabase,
                          {'plan_id': 'p_std', 'billing_cycle': 'monthly', 'locale': 'de'})
    assert resp.status_code == 200
    assert cap.get('locale') == 'de'


def test_checkout_defaults_to_en_when_locale_missing(client, auth_header, monkeypatch, fake_supabase):
    resp, cap = _checkout(client, auth_header, monkeypatch, fake_supabase,
                          {'plan_id': 'p_std', 'billing_cycle': 'monthly'})
    assert resp.status_code == 200
    assert cap.get('locale') == 'en'


def test_checkout_normalizes_unknown_locale_to_en(client, auth_header, monkeypatch, fake_supabase):
    resp, cap = _checkout(client, auth_header, monkeypatch, fake_supabase,
                          {'plan_id': 'p_std', 'billing_cycle': 'monthly', 'locale': 'fr'})
    assert resp.status_code == 200
    assert cap.get('locale') == 'en'


def _capture_portal(monkeypatch, captured):
    monkeypatch.setattr(stripe_routes, 'get_or_create_stripe_customer',
                        lambda uid, email: 'cus_x')

    def _create(**kw):
        captured.update(kw)
        return SimpleNamespace(url='https://billing.stripe.com/p/session/x')

    monkeypatch.setattr(stripe_routes.stripe.billing_portal.Session, 'create',
                        staticmethod(_create))


def test_portal_forwards_de_locale(client, auth_header, monkeypatch):
    cap = {}
    _capture_portal(monkeypatch, cap)
    resp = client.post('/api/stripe/customer-portal', json={'locale': 'de'}, headers=auth_header)
    assert resp.status_code == 200
    assert cap.get('locale') == 'de'


def test_portal_defaults_to_en_when_no_body(client, auth_header, monkeypatch):
    cap = {}
    _capture_portal(monkeypatch, cap)
    resp = client.post('/api/stripe/customer-portal', headers=auth_header)
    assert resp.status_code == 200
    assert cap.get('locale') == 'en'
