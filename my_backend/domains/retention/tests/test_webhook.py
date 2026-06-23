import base64, hashlib, hmac, json, time
from domains.retention.webhook import verify_svix

def _sign(secret_b64, msg_id, ts, body):
    key = base64.b64decode(secret_b64)
    signed = f"{msg_id}.{ts}.{body}".encode()
    sig = base64.b64encode(hmac.new(key, signed, hashlib.sha256).digest()).decode()
    return f"v1,{sig}"

def test_verify_svix_accepts_valid_signature():
    secret = base64.b64encode(b"0123456789abcdef").decode()
    body = json.dumps({"type": "email.bounced"})
    ts = str(int(time.time()))
    headers = {"svix-id": "msg_1", "svix-timestamp": ts,
               "svix-signature": _sign(secret, "msg_1", ts, body)}
    assert verify_svix(f"whsec_{secret}", headers, body) is True

def test_verify_svix_rejects_bad_signature():
    secret = base64.b64encode(b"0123456789abcdef").decode()
    body = json.dumps({"type": "email.bounced"})
    ts = str(int(time.time()))
    headers = {"svix-id": "msg_1", "svix-timestamp": ts,
               "svix-signature": "v1,AAAA"}
    assert verify_svix(f"whsec_{secret}", headers, body) is False

def test_verify_svix_missing_headers_returns_false():
    assert verify_svix("whsec_x", {}, "{}") is False
