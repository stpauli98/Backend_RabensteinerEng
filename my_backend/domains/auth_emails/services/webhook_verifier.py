"""Standard Webhooks signature verification for Supabase Send Email Hook.

Spec: https://www.standardwebhooks.com/
"""

import base64
import hashlib
import hmac
import time
from typing import Mapping


class InvalidSignature(Exception):
    """Raised when the webhook signature does not match the expected HMAC."""


class StaleWebhook(Exception):
    """Raised when the webhook timestamp is more than 5 minutes off."""


_MAX_AGE_SECONDS = 300
_SECRET_PREFIX = "v1,whsec_"


def _decode_secret(secret: str) -> bytes:
    """Decode a Supabase-style secret `v1,whsec_<base64>` into raw bytes."""
    if not secret.startswith(_SECRET_PREFIX):
        raise InvalidSignature(
            f"Unrecognised secret format (expected '{_SECRET_PREFIX}<base64>')"
        )
    encoded = secret[len(_SECRET_PREFIX):]
    return base64.b64decode(encoded)


def verify_webhook(headers: Mapping[str, str], body: bytes, secret: str) -> None:
    """Verify a Standard Webhooks delivery. Raises on failure, returns None on success.

    Expected headers (lowercase keys):
      - webhook-id
      - webhook-timestamp (unix seconds, as a string)
      - webhook-signature ("v1,<base64sig>" possibly space-separated for rotation)

    Pass Flask's request.headers directly (case-insensitive lookup works). If
    passing a plain dict, keys must be lowercase; 'Webhook-Id' != 'webhook-id'.
    """
    try:
        webhook_id = headers["webhook-id"]
        timestamp = headers["webhook-timestamp"]
        signature_header = headers["webhook-signature"]
    except KeyError as exc:
        raise InvalidSignature(f"Missing header: {exc.args[0]}") from exc

    try:
        ts_int = int(timestamp)
    except ValueError as exc:
        raise InvalidSignature("Non-integer timestamp") from exc

    age = abs(time.time() - ts_int)
    if age > _MAX_AGE_SECONDS:
        raise StaleWebhook(f"Webhook age {age:.0f}s exceeds {_MAX_AGE_SECONDS}s")

    secret_bytes = _decode_secret(secret)
    prefix = f"{webhook_id}.{timestamp}.".encode("utf-8")
    signed_content = prefix + body
    expected = base64.b64encode(
        hmac.new(secret_bytes, signed_content, hashlib.sha256).digest()
    ).decode()

    provided = [
        token.split(",", 1)[1]
        for token in signature_header.split(" ")
        if token.startswith("v1,")
    ]
    if not provided:
        raise InvalidSignature("No v1 signature in header")

    match = False
    for sig in provided:
        match |= hmac.compare_digest(expected, sig)
    if not match:
        raise InvalidSignature("Signature mismatch")
