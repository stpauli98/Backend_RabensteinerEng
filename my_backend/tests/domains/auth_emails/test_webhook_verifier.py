import base64
import hashlib
import hmac
import time

import pytest

from domains.auth_emails.services.webhook_verifier import (
    InvalidSignature,
    StaleWebhook,
    verify_webhook,
)

SECRET = "v1,whsec_" + base64.b64encode(b"super-secret-key-32-bytes-long!!").decode()
SECRET_BYTES = base64.b64decode(SECRET.split(",")[1].replace("whsec_", ""))


def _sign(webhook_id: str, timestamp: str, body: bytes) -> str:
    signed = f"{webhook_id}.{timestamp}.".encode() + body
    sig = base64.b64encode(hmac.new(SECRET_BYTES, signed, hashlib.sha256).digest()).decode()
    return f"v1,{sig}"


def test_accepts_valid_signature():
    body = b'{"hello":"world"}'
    ts = str(int(time.time()))
    headers = {
        "webhook-id": "msg_1",
        "webhook-timestamp": ts,
        "webhook-signature": _sign("msg_1", ts, body),
    }
    verify_webhook(headers, body, SECRET)  # does not raise


def test_rejects_tampered_body():
    body = b'{"hello":"world"}'
    ts = str(int(time.time()))
    headers = {
        "webhook-id": "msg_1",
        "webhook-timestamp": ts,
        "webhook-signature": _sign("msg_1", ts, body),
    }
    with pytest.raises(InvalidSignature):
        verify_webhook(headers, b'{"hello":"evil"}', SECRET)


def test_rejects_wrong_secret():
    body = b"{}"
    ts = str(int(time.time()))
    headers = {
        "webhook-id": "msg_1",
        "webhook-timestamp": ts,
        "webhook-signature": _sign("msg_1", ts, body),
    }
    other = "v1,whsec_" + base64.b64encode(b"different-key-bytes-32-bytes-lng").decode()
    with pytest.raises(InvalidSignature):
        verify_webhook(headers, body, other)


def test_rejects_stale_timestamp():
    body = b"{}"
    old_ts = str(int(time.time()) - 600)  # 10 min ago
    headers = {
        "webhook-id": "msg_1",
        "webhook-timestamp": old_ts,
        "webhook-signature": _sign("msg_1", old_ts, body),
    }
    with pytest.raises(StaleWebhook):
        verify_webhook(headers, body, SECRET)


def test_rejects_future_timestamp():
    body = b"{}"
    future_ts = str(int(time.time()) + 600)
    headers = {
        "webhook-id": "msg_1",
        "webhook-timestamp": future_ts,
        "webhook-signature": _sign("msg_1", future_ts, body),
    }
    with pytest.raises(StaleWebhook):
        verify_webhook(headers, body, SECRET)


def test_accepts_when_one_of_multiple_signatures_matches():
    """During key rotation, header may carry several `v1,...` sigs space-separated."""
    body = b"{}"
    ts = str(int(time.time()))
    real_sig = _sign("msg_1", ts, body)
    fake_sig = "v1,AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA="
    headers = {
        "webhook-id": "msg_1",
        "webhook-timestamp": ts,
        "webhook-signature": f"{fake_sig} {real_sig}",
    }
    verify_webhook(headers, body, SECRET)


def test_missing_header_raises():
    with pytest.raises(InvalidSignature):
        verify_webhook({}, b"{}", SECRET)


def test_accepts_at_300s_boundary():
    """Exact 300s age is the boundary; condition is `> _MAX_AGE_SECONDS`, so 300 must pass.

    Using 299s instead of 300s avoids a sub-second timing race: int(time.time()) truncates
    down, so a -300 offset can already be >300.0s old by the time verify_webhook runs.
    299s is definitively within the acceptance window regardless of wall-clock jitter.
    """
    body = b"{}"
    ts = str(int(time.time()) - 299)
    headers = {
        "webhook-id": "msg_1",
        "webhook-timestamp": ts,
        "webhook-signature": _sign("msg_1", ts, body),
    }
    verify_webhook(headers, body, SECRET)  # does not raise


def test_accepts_non_utf8_body():
    """Spec signs raw bytes; body may legitimately not be valid UTF-8 text."""
    body = b"\xff\xfe\xfd"
    ts = str(int(time.time()))
    headers = {
        "webhook-id": "msg_1",
        "webhook-timestamp": ts,
        "webhook-signature": _sign("msg_1", ts, body),
    }
    verify_webhook(headers, body, SECRET)  # does not raise


def test_rejects_unknown_secret_prefix():
    """Future-proofing: if Supabase rotates to v2 or whpk_, we should fail loudly, not silently."""
    body = b"{}"
    ts = str(int(time.time()))
    headers = {
        "webhook-id": "msg_1",
        "webhook-timestamp": ts,
        "webhook-signature": _sign("msg_1", ts, body),
    }
    bad_secret = "v2,whsec_" + base64.b64encode(b"another-32-byte-key-yyyyyyyyyyyy").decode()
    with pytest.raises(InvalidSignature):
        verify_webhook(headers, body, bad_secret)
