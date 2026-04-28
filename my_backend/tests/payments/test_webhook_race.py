"""
Webhook race-condition coverage.

These tests pin three properties of the webhook endpoint:
  1) When the event was previously processed, the handler is not called.
  2) When the handler raises, mark_webhook_processed is NOT called and we
     return 500 so Stripe retries.
  3) When the handler succeeds, mark_webhook_processed IS called and we
     return 200.
"""
from unittest.mock import patch, MagicMock
import json

import pytest
import stripe


@pytest.fixture
def app():
    from core.app_factory import create_app
    flask_app, _ = create_app()
    flask_app.config['TESTING'] = True
    return flask_app


@pytest.fixture
def client(app):
    return app.test_client()


def _construct_event_payload(event_id='evt_test_123', event_type='checkout.session.completed'):
    """Return a dict that mimics a Stripe Event for our handler dispatcher."""
    return {
        'id': event_id,
        'type': event_type,
        'data': {'object': {'id': 'cs_test_xyz', 'subscription': 'sub_test', 'customer': 'cus_test', 'metadata': {}}}
    }


def test_already_processed_event_skips_handler_and_returns_200(client):
    event = _construct_event_payload()
    with patch('domains.payments.api.stripe.stripe.Webhook.construct_event', return_value=event), \
         patch('domains.payments.api.stripe.is_webhook_processed', return_value=True) as mock_is_processed, \
         patch('domains.payments.api.stripe.handle_successful_payment') as mock_handler, \
         patch('domains.payments.api.stripe.mark_webhook_processed') as mock_mark:
        resp = client.post(
            '/api/stripe/webhook',
            data=b'{}',
            headers={'Stripe-Signature': 't=1,v1=fake'}
        )

    assert resp.status_code == 200
    body = json.loads(resp.data)
    assert body.get('message') == 'already processed'
    mock_is_processed.assert_called_once_with('evt_test_123')
    mock_handler.assert_not_called()
    mock_mark.assert_not_called()


def test_handler_exception_returns_500_and_does_not_mark(client):
    event = _construct_event_payload()
    with patch('domains.payments.api.stripe.stripe.Webhook.construct_event', return_value=event), \
         patch('domains.payments.api.stripe.is_webhook_processed', return_value=False), \
         patch(
             'domains.payments.api.stripe.handle_successful_payment',
             side_effect=RuntimeError('simulated DB failure'),
         ) as mock_handler, \
         patch('domains.payments.api.stripe.mark_webhook_processed') as mock_mark:
        resp = client.post(
            '/api/stripe/webhook',
            data=b'{}',
            headers={'Stripe-Signature': 't=1,v1=fake'}
        )

    assert resp.status_code == 500
    mock_handler.assert_called_once()
    mock_mark.assert_not_called()


def test_handler_success_marks_processed_and_returns_200(client):
    event = _construct_event_payload()
    with patch('domains.payments.api.stripe.stripe.Webhook.construct_event', return_value=event), \
         patch('domains.payments.api.stripe.is_webhook_processed', return_value=False), \
         patch('domains.payments.api.stripe.handle_successful_payment') as mock_handler, \
         patch('domains.payments.api.stripe.mark_webhook_processed') as mock_mark:
        resp = client.post(
            '/api/stripe/webhook',
            data=b'{}',
            headers={'Stripe-Signature': 't=1,v1=fake'}
        )

    assert resp.status_code == 200
    mock_handler.assert_called_once()
    mock_mark.assert_called_once_with('evt_test_123', 'checkout.session.completed')


def test_invalid_signature_returns_400(client):
    with patch(
        'domains.payments.api.stripe.stripe.Webhook.construct_event',
        side_effect=stripe.error.SignatureVerificationError('bad sig', 'sig'),
    ):
        resp = client.post(
            '/api/stripe/webhook',
            data=b'{}',
            headers={'Stripe-Signature': 't=1,v1=fake'}
        )

    assert resp.status_code == 400


def test_mark_failure_after_handler_success_still_returns_200(client, caplog):
    """If the handler succeeded but mark_webhook_processed raises (e.g.
    transient DB error after the user-visible work landed), the endpoint
    must still return 200 and emit a WARNING log so the failure is
    observable in production logs.
    """
    import logging
    event = _construct_event_payload()
    with patch('domains.payments.api.stripe.stripe.Webhook.construct_event', return_value=event), \
         patch('domains.payments.api.stripe.is_webhook_processed', return_value=False), \
         patch('domains.payments.api.stripe.handle_successful_payment') as mock_handler, \
         patch(
             'domains.payments.api.stripe.mark_webhook_processed',
             side_effect=RuntimeError('simulated DB write fail'),
         ) as mock_mark:
        with caplog.at_level(logging.WARNING, logger='domains.payments.api.stripe'):
            resp = client.post(
                '/api/stripe/webhook',
                data=b'{}',
                headers={'Stripe-Signature': 't=1,v1=fake'}
            )

    assert resp.status_code == 200
    mock_handler.assert_called_once()
    mock_mark.assert_called_once()
    assert any('failed to mark processed' in rec.message for rec in caplog.records), (
        f"expected a WARNING about mark failure; saw {[r.message for r in caplog.records]}"
    )
