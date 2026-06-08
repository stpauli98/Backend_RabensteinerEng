"""M-3 info-leak regression: upload_chunk must NOT return a Python traceback.

The top-level ``except`` in domains/processing/api/first_processing.upload_chunk
previously returned ``{"error": str(e), "traceback": traceback.format_exc()}``,
leaking file paths and library internals to the client. This test forces an
internal error to flow into that generic catch-all and asserts the response
body is sanitized: a 4xx/5xx status with NO ``traceback`` key and no raw
"Traceback (most recent call last)" string.

Unit-style: bypasses @require_auth by calling the unwrapped handler with a
manually-set g.user_id, mirroring tests/security/test_upload_validation.py.
No Docker, Supabase, or storage backends required.
"""
from __future__ import annotations

import io
from unittest.mock import patch

import pytest
from flask import Flask, g

from domains.processing.api.first_processing import bp as first_bp, upload_chunk


USER_A = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"


@pytest.fixture
def app() -> Flask:
    app = Flask(__name__)
    app.register_blueprint(first_bp, url_prefix="/api/firstProcessing")
    app.config["TESTING"] = True
    return app


def _unwrap(view):
    """Strip @require_auth / @require_subscription / @check_processing_limit layers."""
    inner = view
    while hasattr(inner, "__wrapped__"):
        inner = inner.__wrapped__
    return inner


def _status(response):
    return response[1] if isinstance(response, tuple) else response.status_code


def _body(response):
    resp = response[0] if isinstance(response, tuple) else response
    return resp.get_json()


def _multipart_with_chunk():
    """Valid form data that passes the handler's early validation gates."""
    return {
        "uploadId": "test-upload-123",
        "chunkIndex": "0",
        "totalChunks": "1",
        "tss": "15",
        "offset": "0",
        "mode": "mean",
        "intrplMax": "60",
        "fileChunk": (io.BytesIO(b"UTC;Value\n2020-01-01 00:00:00;1.0\n"), "chunk.csv"),
    }


def test_upload_chunk_generic_error_does_not_leak_traceback(app):
    """An internal Exception must surface as a sanitized 4xx/5xx, not a traceback."""
    inner = _unwrap(upload_chunk)

    # Force a generic, unexpected error from an internal service call so it
    # bubbles into the top-level catch-all (not a deliberate validation branch).
    with patch(
        "domains.processing.api.first_processing.local_chunk_service.upload_chunk",
        side_effect=Exception("boom /usr/lib/python3.9/site-packages/secret.py internal"),
    ):
        with app.test_request_context(
            "/api/firstProcessing/upload_chunk",
            method="POST",
            data=_multipart_with_chunk(),
            content_type="multipart/form-data",
        ):
            g.user_id = USER_A
            response = inner()

    status = _status(response)
    assert 400 <= status < 600, f"expected a 4xx/5xx error status, got {status}"

    body = _body(response)
    assert body is not None, "response body must be JSON"

    # The core assertions: no traceback key, no raw traceback string anywhere.
    assert "traceback" not in body, f"response leaked a 'traceback' key: {body}"

    serialized = repr(body)
    assert "Traceback (most recent call last)" not in serialized, (
        f"response leaked a Python traceback: {serialized}"
    )
    # The raw internal exception text (with library file paths) must not leak.
    assert "site-packages" not in serialized, (
        f"response leaked internal file paths: {serialized}"
    )


def test_upload_chunk_missing_file_chunk_still_returns_safe_400(app):
    """The deliberate, user-safe validation branch (no fileChunk) is preserved."""
    inner = _unwrap(upload_chunk)
    with app.test_request_context(
        "/api/firstProcessing/upload_chunk",
        method="POST",
        data={"uploadId": "x", "tss": "15"},
        content_type="multipart/form-data",
    ):
        g.user_id = USER_A
        response = inner()

    assert _status(response) == 400
    body = _body(response)
    assert "traceback" not in body
    assert body.get("error") == "No file chunk found"
