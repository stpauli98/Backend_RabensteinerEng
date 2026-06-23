"""Svix webhook signature verification (Resend uses Svix). Stdlib only."""
import base64
import hashlib
import hmac


def verify_svix(secret: str, headers: dict, body: str) -> bool:
    msg_id = headers.get("svix-id")
    timestamp = headers.get("svix-timestamp")
    sig_header = headers.get("svix-signature")
    if not (msg_id and timestamp and sig_header and secret):
        return False
    raw = secret.split("_", 1)[1] if secret.startswith("whsec_") else secret
    try:
        key = base64.b64decode(raw)
    except Exception:
        return False
    signed = f"{msg_id}.{timestamp}.{body}".encode()
    expected = base64.b64encode(hmac.new(key, signed, hashlib.sha256).digest()).decode()
    # svix-signature can carry multiple space-separated "vN,sig" entries.
    for part in sig_header.split(" "):
        _, _, sig = part.partition(",")
        if sig and hmac.compare_digest(sig, expected):
            return True
    return False
