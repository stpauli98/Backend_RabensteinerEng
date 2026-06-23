"""Integration tests for the retention HTTP routes (webhook + admin).

Uses a minimal Flask app with only retention_bp registered, so it exercises the
real routes + real svix signature verification without create_app's scheduler
and DB-cleanup side effects. The DB and email sender are mocked; the signature
verification is REAL.
"""
import base64
import hashlib
import hmac
import json

import pytest
from flask import Flask

from domains.retention.api import retention_bp
import domains.retention.api.routes as routes
import domains.retention.sweep as sweep_mod


_SECRET_B64 = base64.b64encode(b"testsecretkey16!").decode()
_WEBHOOK_SECRET = f"whsec_{_SECRET_B64}"


@pytest.fixture
def client():
    app = Flask(__name__)
    app.register_blueprint(retention_bp, url_prefix="/api/retention")
    return app.test_client()


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("RESEND_WEBHOOK_SECRET", _WEBHOOK_SECRET)
    monkeypatch.setenv("RETENTION_ADMIN_SECRET", "admin-top-secret")
    monkeypatch.setenv("RETENTION_ADMIN_EMAIL", "admin@example.com")
    monkeypatch.setenv("RESEND_API_KEY", "re_test")
    monkeypatch.setenv("EMAIL_FROM_ADDRESS", "noreply@info.forecast-engine.com")
    monkeypatch.setenv("EMAIL_FROM_NAME", "Forecast Engine")


def _sign(body: str, msg_id="msg_evt_1", ts="1782200000"):
    key = base64.b64decode(_SECRET_B64)
    signed = f"{msg_id}.{ts}.{body}".encode()
    sig = base64.b64encode(hmac.new(key, signed, hashlib.sha256).digest()).decode()
    return {"svix-id": msg_id, "svix-timestamp": ts, "svix-signature": f"v1,{sig}"}


# --- webhook ---------------------------------------------------------------

def test_webhook_rejects_bad_signature(client, monkeypatch):
    db_called = {"n": 0}
    monkeypatch.setattr(routes, "get_supabase_admin_client",
                        lambda: db_called.__setitem__("n", db_called["n"] + 1))
    body = json.dumps({"type": "email.bounced", "data": {"email_id": "m1"}})
    resp = client.post("/api/retention/resend-webhook", data=body,
                       headers={"svix-id": "x", "svix-timestamp": "1",
                                "svix-signature": "v1,deadbeef"})
    assert resp.status_code == 401
    assert db_called["n"] == 0  # no DB work on the 401 path


def test_webhook_valid_bounce_marks_and_alerts(client, monkeypatch):
    calls = {}
    monkeypatch.setattr(routes, "get_supabase_admin_client", lambda: "SB")
    monkeypatch.setattr(routes, "mark_status_by_message_id",
                        lambda sb, mid, status, now: calls.update(mark=(sb, mid, status)) or True)
    sent = {"n": 0}
    monkeypatch.setattr(routes, "send_email",
                        lambda **kw: sent.__setitem__("n", sent["n"] + 1) or "id")
    body = json.dumps({"type": "email.bounced", "data": {"email_id": "msg_bounced_9"}})
    resp = client.post("/api/retention/resend-webhook", data=body, headers=_sign(body))
    assert resp.status_code == 200
    assert resp.get_json() == {"status": "ok", "matched": True}
    assert calls["mark"] == ("SB", "msg_bounced_9", "bounced")
    assert sent["n"] == 1  # admin alerted on bounce


def test_webhook_unknown_event_is_ignored(client, monkeypatch):
    marked = {"n": 0}
    monkeypatch.setattr(routes, "get_supabase_admin_client", lambda: "SB")
    monkeypatch.setattr(routes, "mark_status_by_message_id",
                        lambda *a, **k: marked.__setitem__("n", marked["n"] + 1) or True)
    body = json.dumps({"type": "email.opened", "data": {"email_id": "m"}})
    resp = client.post("/api/retention/resend-webhook", data=body, headers=_sign(body))
    assert resp.status_code == 200
    assert resp.get_json() == {"status": "ignored"}
    assert marked["n"] == 0  # unknown event never touches the DB


def test_webhook_delivered_marks_but_no_alert(client, monkeypatch):
    monkeypatch.setattr(routes, "get_supabase_admin_client", lambda: "SB")
    monkeypatch.setattr(routes, "mark_status_by_message_id", lambda *a, **k: True)
    sent = {"n": 0}
    monkeypatch.setattr(routes, "send_email",
                        lambda **kw: sent.__setitem__("n", sent["n"] + 1))
    body = json.dumps({"type": "email.delivered", "data": {"email_id": "m"}})
    resp = client.post("/api/retention/resend-webhook", data=body, headers=_sign(body))
    assert resp.status_code == 200
    assert sent["n"] == 0  # delivered is not an alert-worthy status


# --- admin run-sweep -------------------------------------------------------

def test_run_sweep_requires_secret(client, monkeypatch):
    ran = {"n": 0}
    monkeypatch.setattr(sweep_mod, "run_sweep",
                        lambda *a, **k: ran.__setitem__("n", ran["n"] + 1))
    resp = client.post("/api/retention/run-sweep")  # no header
    assert resp.status_code == 401
    assert ran["n"] == 0


def test_run_sweep_with_secret_runs(client, monkeypatch):
    monkeypatch.setattr(routes, "get_supabase_admin_client", lambda: "SB")
    monkeypatch.setattr(sweep_mod, "run_sweep",
                        lambda sb, *, dry_run: {"ran": True, "planned": 0, "dry": dry_run})
    resp = client.post("/api/retention/run-sweep",
                       headers={"X-Retention-Secret": "admin-top-secret"})
    assert resp.status_code == 200
    assert resp.get_json()["ran"] is True


def test_run_sweep_wrong_secret_rejected(client, monkeypatch):
    ran = {"n": 0}
    monkeypatch.setattr(sweep_mod, "run_sweep",
                        lambda *a, **k: ran.__setitem__("n", ran["n"] + 1))
    resp = client.post("/api/retention/run-sweep",
                       headers={"X-Retention-Secret": "wrong"})
    assert resp.status_code == 401
    assert ran["n"] == 0
