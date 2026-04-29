"""Regression tests for the IDOR fix on /api/loadRowData/download/<file_id>.

The download handler accepts a user-controlled file_id from the URL path. The
file_id is server-constructed as either:

  - "{user_id}_{upload_id}_{uuid_hex8}"   (local-chunk format)
  - "{user_id}/{...}"                      (Supabase Storage format)

A logged-in user must not be able to download another user's file by passing
that user's file_id. The handler enforces this with a prefix check that runs
before any storage call. These tests exercise that early-return guard.

The tests are unit-style: they call the underlying view function with
g.user_id pre-populated, so they do not require Docker, Supabase, or the
JWT middleware to be reachable.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest
from flask import Flask, g

# Importing the blueprint also imports the route function. We use
# `__wrapped__` (set by functools.wraps inside @require_auth) to bypass the
# auth decorator and call the handler directly with a manually-set g.user_id.
from domains.upload.api.load_data import bp as upload_bp, download_file


USER_A = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
USER_B = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"


@pytest.fixture
def app() -> Flask:
    """Minimal Flask app with the upload blueprint registered."""
    app = Flask(__name__)
    app.register_blueprint(upload_bp, url_prefix="/api/loadRowData")
    app.config["TESTING"] = True
    return app


def _call_download(app: Flask, caller_user_id: str, file_id: str):
    """Invoke the download_file view directly inside a request context.

    Bypasses @require_auth by calling the unwrapped function and pre-setting
    g.user_id, mirroring what require_auth would have done after a successful
    JWT validation.
    """
    inner = getattr(download_file, "__wrapped__", download_file)
    with app.test_request_context(f"/api/loadRowData/download/{file_id}"):
        g.user_id = caller_user_id
        return inner(file_id=file_id)


def test_download_rejects_other_users_local_file_id(app):
    """User A asking for User B's local-format file_id must get 403.

    The 403 must short-circuit before any local-chunk read happens.
    """
    victim_file_id = f"{USER_B}_upload123_deadbeef"

    with patch(
        "domains.upload.api.load_data.local_chunk_service.get_processed_result"
    ) as mock_local, patch(
        "domains.upload.api.load_data.storage_service.get_download_url"
    ) as mock_storage_url, patch(
        "domains.upload.api.load_data.storage_service.download_csv"
    ) as mock_storage_dl:
        response = _call_download(app, caller_user_id=USER_A, file_id=victim_file_id)

        # Flask views can return (Response, status) tuples
        if isinstance(response, tuple):
            body, status = response[0], response[1]
        else:
            body, status = response, response.status_code

        assert status == 403, f"expected 403, got {status}"
        # The guard must fire before any storage backend is touched.
        mock_local.assert_not_called()
        mock_storage_url.assert_not_called()
        mock_storage_dl.assert_not_called()


def test_download_rejects_other_users_storage_file_id(app):
    """Storage-format file_id (with a slash) belonging to User B is rejected."""
    victim_file_id = f"{USER_B}/20251015_120000_abcd1234"

    with patch(
        "domains.upload.api.load_data.local_chunk_service.get_processed_result"
    ) as mock_local, patch(
        "domains.upload.api.load_data.storage_service.get_download_url"
    ) as mock_storage_url, patch(
        "domains.upload.api.load_data.storage_service.download_csv"
    ) as mock_storage_dl:
        response = _call_download(app, caller_user_id=USER_A, file_id=victim_file_id)

        if isinstance(response, tuple):
            status = response[1]
        else:
            status = response.status_code

        assert status == 403
        mock_local.assert_not_called()
        mock_storage_url.assert_not_called()
        mock_storage_dl.assert_not_called()


def test_download_allows_own_local_file_id(app):
    """User A asking for their OWN local-format file_id passes the prefix check.

    We don't care about the rest of the handler for this test; we only verify
    that the prefix check did not short-circuit. We assert this by checking
    that the local-chunk service was actually queried.
    """
    own_file_id = f"{USER_A}_upload999_cafef00d"

    with patch(
        "domains.upload.api.load_data.local_chunk_service.get_processed_result",
        return_value="col1,col2\n1,2\n",
    ) as mock_local:
        response = _call_download(app, caller_user_id=USER_A, file_id=own_file_id)

        # Local-chunk lookup must have been reached.
        mock_local.assert_called_once_with(own_file_id)
        # And the response must NOT be 403.
        if isinstance(response, tuple):
            status = response[1]
        else:
            status = response.status_code
        assert status != 403


def test_download_allows_own_storage_file_id(app):
    """User A asking for their OWN storage-format file_id passes the prefix check."""
    own_file_id = f"{USER_A}/20251015_120000_cafef00d"

    with patch(
        "domains.upload.api.load_data.local_chunk_service.get_processed_result",
        return_value=None,
    ) as mock_local, patch(
        "domains.upload.api.load_data.storage_service.get_download_url",
        return_value="https://signed.example/url",
    ) as mock_storage_url:
        response = _call_download(app, caller_user_id=USER_A, file_id=own_file_id)

        # Both backends should have been consulted in order.
        mock_local.assert_called_once_with(own_file_id)
        mock_storage_url.assert_called_once()
        # Owner request must not be 403.
        if isinstance(response, tuple):
            status = response[1]
        else:
            status = response.status_code
        assert status != 403


def test_download_rejects_prefix_collision_attempts(app):
    """Defence against prefix-collision tricks like 'aaaa...' targeting 'aaaa...bbbb'.

    A file_id that merely *contains* the caller's user_id but does not start
    with "{user_id}_" or "{user_id}/" must still be rejected. This guards
    against e.g. someone crafting a path like "evil_{victim}_..." or
    "{victim_prefix}{caller_id}...".
    """
    # file_id starts with USER_B (the victim) and happens to contain USER_A
    # in the upload segment. Must be rejected because it does not start with
    # USER_A's prefix.
    sneaky_file_id = f"{USER_B}_upload_{USER_A}_deadbeef"

    with patch(
        "domains.upload.api.load_data.local_chunk_service.get_processed_result"
    ) as mock_local:
        response = _call_download(app, caller_user_id=USER_A, file_id=sneaky_file_id)

        if isinstance(response, tuple):
            status = response[1]
        else:
            status = response.status_code

        assert status == 403
        mock_local.assert_not_called()
