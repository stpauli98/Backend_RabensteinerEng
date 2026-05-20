"""Verify /upload-chunk returns 409 when chunk-0 metadata never materialises (FB3).

Race condition fix: when a non-zero chunk arrives but chunk-0 metadata is still
absent after the full wait window, the endpoint must return 409 Conflict so the
client can retry — rather than silently proceeding and failing later at
/finalize-upload.

Test strategy:
  - Auth/subscription decorators are bypassed the same way as FB1 tests.
  - local_chunk_service.get_upload_metadata is patched to ALWAYS return None,
    simulating the race where chunk-0 metadata never arrives.
  - time.sleep is patched to a no-op so the 5-second loop completes instantly.
  - Verifies status 409 and error='chunk_zero_pending'.
"""
from __future__ import annotations

import io
from unittest.mock import patch

import pytest
from flask import Flask

# ---------------------------------------------------------------------------
# Helpers (mirrors FB1 constants so tests are self-contained)
# ---------------------------------------------------------------------------

FAKE_USER_ID = "cccccccc-cccc-cccc-cccc-cccccccccccc"
FAKE_EMAIL = "test@example.com"
FAKE_TOKEN = "fake-bearer-token"

FAKE_CLAIMS = {
    "sub": FAKE_USER_ID,
    "email": FAKE_EMAIL,
    "role": "authenticated",
    "aud": "authenticated",
}

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


def _non_zero_chunk_data():
    """Multipart payload for chunk index 1 (non-zero — triggers wait loop)."""
    return {
        "fileChunk": (io.BytesIO(b"data"), "test.csv"),
        "uploadId": "fb3-race-test",
        "chunkIndex": "1",       # NON-ZERO — exercises the metadata wait path
        "totalChunks": "3",
        "delimiter": ";",
        "selected_columns": "{}",
        "timezone": "UTC",
        "dropdown_count": "2",
        "hasHeader": "ja",
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


def test_upload_chunk_returns_409_when_chunk0_metadata_missing(client):
    """Non-zero chunk with chunk-0 metadata permanently absent must get 409.

    Before FB3 fix: the loop runs 10 iterations (1 s) then falls through and
    calls upload_chunk anyway — returning 200 and corrupting the upload.
    After FB3 fix: the loop runs 50 iterations (5 s) then returns 409 with
    error='chunk_zero_pending'.

    time.sleep is patched to a no-op so the test completes in milliseconds.
    """
    with patch(
        "shared.auth.jwt._verify_jwt_local", return_value=FAKE_CLAIMS
    ), patch(
        "shared.auth.subscription.get_user_subscription",
        return_value=ACTIVE_SUBSCRIPTION,
    ), patch(
        "domains.upload.api.load_data.local_chunk_service.get_upload_metadata",
        return_value=None,   # metadata NEVER arrives
    ), patch(
        "time.sleep",        # skip the actual sleeps so test is fast
        return_value=None,
    ):
        resp = client.post(
            "/api/loadRowData/upload-chunk",
            data=_non_zero_chunk_data(),
            headers={"Authorization": f"Bearer {FAKE_TOKEN}"},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 409, (
            f"Expected 409 Conflict when chunk-0 metadata missing, "
            f"got {resp.status_code}: {resp.data}"
        )
        body = resp.get_json()
        assert body is not None, "Response body must be JSON"
        assert body.get("error") == "chunk_zero_pending", (
            f"Expected error='chunk_zero_pending', got {body}"
        )
