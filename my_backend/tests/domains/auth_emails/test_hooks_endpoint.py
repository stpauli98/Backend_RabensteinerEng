import base64
import hashlib
import hmac
import json
import time
from unittest.mock import patch

import pytest
from flask import Flask

from domains.auth_emails import auth_emails_bp

SECRET = "v1,whsec_" + base64.b64encode(b"super-secret-key-32-bytes-long!!").decode()
SECRET_BYTES = base64.b64decode(SECRET.split(",")[1].replace("whsec_", ""))


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("AUTH_HOOK_SECRET", SECRET)
    monkeypatch.setenv("RESEND_API_KEY", "re_test")
    monkeypatch.setenv("EMAIL_FROM_ADDRESS", "noreply@forecast-engine.com")
    monkeypatch.setenv("EMAIL_FROM_NAME", "Forecast Engine")

    app = Flask(__name__)
    app.register_blueprint(auth_emails_bp, url_prefix="/api/auth")
    return app.test_client()


def _signed_headers(body: bytes):
    ts = str(int(time.time()))
    msg_id = "msg_test"
    signed = f"{msg_id}.{ts}.".encode() + body
    sig = base64.b64encode(hmac.new(SECRET_BYTES, signed, hashlib.sha256).digest()).decode()
    return {
        "webhook-id": msg_id,
        "webhook-timestamp": ts,
        "webhook-signature": f"v1,{sig}",
        "Content-Type": "application/json",
    }


def _payload(lang="en", action="signup"):
    user_metadata = {"lang": lang} if lang else {}
    return {
        "user": {
            "id": "u1",
            "email": "alice@example.com",
            "user_metadata": user_metadata,
        },
        "email_data": {
            "token_hash": "abc123",
            "email_action_type": action,
            "redirect_to": "https://forecast-engine.com/dashboard",
            "site_url": "https://luvjebsltuttakatnzaa.supabase.co",
        },
    }


@patch("domains.auth_emails.api.hooks.send_email")
def test_signup_en_dispatches_to_resend(send_mock, client):
    send_mock.return_value = "msg_123"
    body = json.dumps(_payload(lang="en")).encode()
    resp = client.post("/api/auth/send-email", data=body, headers=_signed_headers(body))

    assert resp.status_code == 200
    assert resp.get_json() == {"status": "ok"}
    send_mock.assert_called_once()
    kwargs = send_mock.call_args.kwargs
    assert kwargs["to"] == "alice@example.com"
    assert kwargs["subject"] == "Confirm your email – Forecast Engine"
    assert kwargs["from_addr"] == "Forecast Engine <noreply@forecast-engine.com>"
    assert "Confirm your registration" in kwargs["html"]
    assert "token_hash=abc123" in kwargs["html"]


@patch("domains.auth_emails.api.hooks.send_email")
def test_missing_lang_defaults_to_de(send_mock, client):
    send_mock.return_value = "msg_124"
    body = json.dumps(_payload(lang=None)).encode()
    resp = client.post("/api/auth/send-email", data=body, headers=_signed_headers(body))
    assert resp.status_code == 200
    assert "Bestätigen Sie Ihre Registrierung" in send_mock.call_args.kwargs["html"]


@patch("domains.auth_emails.api.hooks.send_email")
def test_invalid_signature_returns_401(send_mock, client):
    body = json.dumps(_payload()).encode()
    bad_headers = _signed_headers(body)
    bad_headers["webhook-signature"] = "v1,AAAA"
    resp = client.post("/api/auth/send-email", data=body, headers=bad_headers)
    assert resp.status_code == 401
    send_mock.assert_not_called()


@patch("domains.auth_emails.api.hooks.send_email")
def test_stale_timestamp_returns_401(send_mock, client):
    body = json.dumps(_payload()).encode()
    ts = str(int(time.time()) - 600)
    msg_id = "msg_stale"
    signed = f"{msg_id}.{ts}.".encode() + body
    sig = base64.b64encode(hmac.new(SECRET_BYTES, signed, hashlib.sha256).digest()).decode()
    headers = {
        "webhook-id": msg_id,
        "webhook-timestamp": ts,
        "webhook-signature": f"v1,{sig}",
        "Content-Type": "application/json",
    }
    resp = client.post("/api/auth/send-email", data=body, headers=headers)
    assert resp.status_code == 401


@patch("domains.auth_emails.api.hooks.send_email")
def test_unknown_action_returns_400(send_mock, client):
    body = json.dumps(_payload(action="invite")).encode()
    resp = client.post("/api/auth/send-email", data=body, headers=_signed_headers(body))
    assert resp.status_code == 400
    send_mock.assert_not_called()


@patch("domains.auth_emails.api.hooks.send_email")
def test_resend_failure_returns_502(send_mock, client):
    from domains.auth_emails.services.resend_client import ResendError
    send_mock.side_effect = ResendError("boom")
    body = json.dumps(_payload()).encode()
    resp = client.post("/api/auth/send-email", data=body, headers=_signed_headers(body))
    assert resp.status_code == 502


@patch("domains.auth_emails.api.hooks.send_email")
def test_missing_email_returns_400(send_mock, client):
    payload = {
        "user": {"id": "u1", "user_metadata": {"lang": "en"}},
        # `email` deliberately omitted
        "email_data": {
            "token_hash": "abc",
            "email_action_type": "signup",
            "redirect_to": "https://forecast-engine.com/dashboard",
            "site_url": "https://luvjebsltuttakatnzaa.supabase.co",
        },
    }
    body = json.dumps(payload).encode()
    resp = client.post("/api/auth/send-email", data=body, headers=_signed_headers(body))
    assert resp.status_code == 400
    send_mock.assert_not_called()
