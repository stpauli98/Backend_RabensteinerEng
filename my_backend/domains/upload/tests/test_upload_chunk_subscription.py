"""Verify /upload-chunk requires active subscription (FB1 — BE BUG #3).

Authenticated users without an active subscription must be rejected at the
chunk-receive layer, not only at /finalize-upload. This prevents disk-filling
DoS where an unsubscribed user could upload unlimited chunks before hitting
the subscription check on finalize.

Test strategy:
  - Full decorator stack (require_auth + require_subscription) exercised via
    the Flask test client — no _unwrap bypass.
  - require_auth is bypassed by mocking _verify_jwt_local so a fake-but-valid
    token is accepted without a real JWT secret.
  - require_subscription is tested by mocking get_user_subscription to return
    None (no subscription) or a stub subscription dict (active subscription).
"""
from __future__ import annotations

import io
from unittest.mock import patch, MagicMock

import pytest
from flask import Flask

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FAKE_USER_ID = "cccccccc-cccc-cccc-cccc-cccccccccccc"
FAKE_EMAIL = "test@example.com"
FAKE_TOKEN = "fake-bearer-token"

# Minimal JWT claims that require_auth would set on g after _verify_jwt_local
FAKE_CLAIMS = {
    "sub": FAKE_USER_ID,
    "email": FAKE_EMAIL,
    "role": "authenticated",
    "aud": "authenticated",
}

# Minimal subscription dict that require_subscription expects
ACTIVE_SUBSCRIPTION = {
    "user_id": FAKE_USER_ID,
    "status": "active",
    "subscription_plans": {
        "name": "STANDARD",
        "max_uploads_per_month": 100,
        "max_processing_jobs_per_month": 50,
        "max_storage_gb": 10,
    },
}

_MULTIPART_FIELDS = {
    "uploadId": "fb1-test-upload-id",
    "chunkIndex": "0",
    "totalChunks": "1",
    "delimiter": ";",
    "selected_columns": "{}",
    "timezone": "UTC",
    "dropdown_count": "2",
    "hasHeader": "ja",
}


def _chunk_data():
    """Return a multipart payload for a minimal CSV chunk."""
    return {
        **_MULTIPART_FIELDS,
        "fileChunk": (io.BytesIO(b"col1;col2\n1;2\n"), "test.csv"),
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def app():
    """Minimal Flask app with the upload blueprint registered."""
    from domains.upload.api.load_data import bp as upload_bp

    flask_app = Flask(__name__)
    flask_app.register_blueprint(upload_bp, url_prefix="/api/loadRowData")
    flask_app.config["TESTING"] = True
    return flask_app


@pytest.fixture
def client(app):
    return app.test_client()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_upload_chunk_rejects_user_without_active_subscription(client):
    """Authenticated user with NO subscription must receive 403 from /upload-chunk.

    Before FB1 fix: only @require_auth was on the endpoint, so this would have
    reached the handler body and returned 200/400.
    After FB1 fix: @require_subscription fires first and returns 403.
    """
    with patch(
        "shared.auth.jwt._verify_jwt_local", return_value=FAKE_CLAIMS
    ), patch(
        "shared.auth.subscription.get_user_subscription", return_value=None
    ):
        resp = client.post(
            "/api/loadRowData/upload-chunk",
            data=_chunk_data(),
            headers={"Authorization": f"Bearer {FAKE_TOKEN}"},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 403, (
            f"Expected 403 for no-subscription user, got {resp.status_code}: "
            f"{resp.data}"
        )
        body = resp.get_json()
        assert body is not None
        assert "error" in body


def test_upload_chunk_allows_user_with_active_subscription(client):
    """Authenticated user WITH an active subscription must pass the subscription gate.

    We don't care about the rest of the upload logic — we only need to confirm
    that the decorator stack doesn't reject the request. We patch the chunk
    storage service to return success so the handler completes cleanly.
    """
    with patch(
        "shared.auth.jwt._verify_jwt_local", return_value=FAKE_CLAIMS
    ), patch(
        "shared.auth.subscription.get_user_subscription",
        return_value=ACTIVE_SUBSCRIPTION,
    ), patch(
        "domains.upload.api.load_data.local_chunk_service.upload_chunk",
        return_value=True,
    ), patch(
        "domains.upload.api.load_data.local_chunk_service.save_upload_metadata",
        return_value=None,
    ), patch(
        "domains.upload.api.load_data.local_chunk_service.get_upload_metadata",
        return_value=None,  # first chunk, no prior metadata
    ):
        resp = client.post(
            "/api/loadRowData/upload-chunk",
            data=_chunk_data(),
            headers={"Authorization": f"Bearer {FAKE_TOKEN}"},
            content_type="multipart/form-data",
        )
        # Must NOT be a 403 (subscription rejection) or 401 (auth rejection)
        assert resp.status_code not in (401, 403), (
            f"Expected subscription to be accepted, got {resp.status_code}: "
            f"{resp.data}"
        )


def test_upload_chunk_rejects_unauthenticated_request(client):
    """Request without Authorization header must be rejected with 401."""
    resp = client.post(
        "/api/loadRowData/upload-chunk",
        data=_chunk_data(),
        content_type="multipart/form-data",
    )
    assert resp.status_code == 401, (
        f"Expected 401 for missing auth header, got {resp.status_code}: {resp.data}"
    )
