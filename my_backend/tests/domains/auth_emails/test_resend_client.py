import json
from unittest.mock import patch, MagicMock

import pytest
import requests

from domains.auth_emails.services.resend_client import ResendError, send_email


def _mock_response(*, status_code: int = 200, body: dict = None) -> MagicMock:
    resp = MagicMock(spec=requests.Response)
    resp.status_code = status_code
    resp.text = json.dumps(body or {})
    resp.json.return_value = body or {}
    return resp


@patch("domains.auth_emails.services.resend_client.requests.post")
def test_sends_email_with_expected_payload(mock_post):
    mock_post.return_value = _mock_response(body={"id": "msg_123"})

    msg_id = send_email(
        api_key="re_test",
        from_addr="Forecast Engine <noreply@forecast-engine.com>",
        to="alice@example.com",
        subject="Hello",
        html="<p>Hi</p>",
    )

    assert msg_id == "msg_123"
    assert mock_post.call_count == 1
    args, kwargs = mock_post.call_args
    assert args[0] == "https://api.resend.com/emails"
    assert kwargs["headers"]["Authorization"] == "Bearer re_test"
    assert kwargs["headers"]["Content-Type"] == "application/json"
    assert kwargs["json"] == {
        "from": "Forecast Engine <noreply@forecast-engine.com>",
        "to": ["alice@example.com"],
        "subject": "Hello",
        "html": "<p>Hi</p>",
    }
    assert kwargs["timeout"] == 10


@patch("domains.auth_emails.services.resend_client.requests.post")
def test_raises_on_4xx(mock_post):
    mock_post.return_value = _mock_response(
        status_code=401, body={"error": "Invalid API key"}
    )
    with pytest.raises(ResendError) as exc:
        send_email(
            api_key="bad", from_addr="x@y", to="a@b",
            subject="s", html="h",
        )
    assert "401" in str(exc.value)


@patch("domains.auth_emails.services.resend_client.requests.post")
def test_raises_on_5xx(mock_post):
    mock_post.return_value = _mock_response(
        status_code=503, body={"error": "Service Unavailable"}
    )
    with pytest.raises(ResendError):
        send_email(api_key="k", from_addr="f@x", to="t@x", subject="s", html="h")


@patch("domains.auth_emails.services.resend_client.requests.post")
def test_raises_on_network_error(mock_post):
    mock_post.side_effect = requests.exceptions.ConnectionError("boom")
    with pytest.raises(ResendError) as exc:
        send_email(api_key="k", from_addr="f@x", to="t@x", subject="s", html="h")
    assert "boom" in str(exc.value)


@patch("domains.auth_emails.services.resend_client.requests.post")
def test_raises_on_timeout(mock_post):
    mock_post.side_effect = requests.exceptions.Timeout("timed out")
    with pytest.raises(ResendError):
        send_email(api_key="k", from_addr="f@x", to="t@x", subject="s", html="h")


@patch("domains.auth_emails.services.resend_client.requests.post")
def test_missing_id_in_response_returns_empty_string(mock_post):
    """Resend should always return an id, but defensively handle the case where it doesn't."""
    mock_post.return_value = _mock_response(body={})
    msg_id = send_email(
        api_key="k", from_addr="f@x", to="t@x", subject="s", html="h",
    )
    assert msg_id == ""
