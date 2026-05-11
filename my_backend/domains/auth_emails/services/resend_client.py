"""Thin wrapper around the Resend HTTP API."""

import requests


_RESEND_URL = "https://api.resend.com/emails"
_TIMEOUT_SECONDS = 10


class ResendError(Exception):
    """Raised when Resend returns a non-2xx response or the request fails."""


def send_email(
    *,
    api_key: str,
    from_addr: str,
    to: str,
    subject: str,
    html: str,
) -> str:
    """POST one email to Resend. Returns the Resend message id on success."""
    payload = {
        "from": from_addr,
        "to": [to],
        "subject": subject,
        "html": html,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    try:
        resp = requests.post(
            _RESEND_URL, json=payload, headers=headers, timeout=_TIMEOUT_SECONDS
        )
    except requests.RequestException as exc:
        raise ResendError(f"Network error contacting Resend: {exc}") from exc

    if resp.status_code >= 400:
        raise ResendError(f"Resend HTTP {resp.status_code}: {resp.text}")

    return resp.json().get("id", "")
