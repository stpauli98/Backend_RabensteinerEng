"""Ownership / IDOR tests for file_id-scoped upload endpoints.

file_ids are minted server-side as ``f"{user_id}_{random}"`` so the owning
user is encoded as the prefix. A logged-in user must not be able to act on
another user's file by supplying that user's file_id to ``download``,
``cleanup-files`` or ``merge-and-prepare``.

The first block unit-tests the ``_file_id_owned_by_user`` helper directly
(hermetic, no request context). The second block exercises the endpoint
guards by calling the unwrapped view functions with ``g.user_id`` pre-set,
mirroring what ``@require_auth`` would do after JWT validation.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest
from flask import Flask, g

import domains.upload.api.load_data as ld
from domains.upload.api.load_data import bp as upload_bp, cleanup_files, merge_and_prepare


USER_A = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
USER_B = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"


# --------------------------------------------------------------------------- #
# Unit tests: the helper                                                       #
# --------------------------------------------------------------------------- #

def test_owner_prefix_required():
    assert ld._file_id_owned_by_user("alice_123", "alice") is True
    assert ld._file_id_owned_by_user("bob_123", "alice") is False
    # No traversal: an id that smuggles a slash/.. must never pass.
    assert ld._file_id_owned_by_user("alice_/../bob_123", "alice") is False
    assert ld._file_id_owned_by_user(5, "alice") is False
    assert ld._file_id_owned_by_user("alice_..", "alice") is False


def test_helper_rejects_backslash_and_parent_segments():
    assert ld._file_id_owned_by_user("alice_\\..\\bob", "alice") is False
    assert ld._file_id_owned_by_user("alice_..bob", "alice") is False
    # A different user's clean id is still rejected.
    assert ld._file_id_owned_by_user("alice2_123", "alice") is False


# --------------------------------------------------------------------------- #
# Endpoint tests: cleanup-files + merge-and-prepare                            #
# --------------------------------------------------------------------------- #

@pytest.fixture
def app() -> Flask:
    app = Flask(__name__)
    app.register_blueprint(upload_bp, url_prefix="/api/loadRowData")
    app.config["TESTING"] = True
    return app


def _status(response):
    if isinstance(response, tuple):
        return response[1]
    return response.status_code


def _unwrap(view):
    """Drill through stacked @wraps decorators to the raw view function."""
    inner = view
    while hasattr(inner, "__wrapped__"):
        inner = inner.__wrapped__
    return inner


def _call(app, view, caller_user_id, json_body):
    inner = _unwrap(view)
    with app.test_request_context(
        "/api/loadRowData/", method="POST", json=json_body
    ):
        g.user_id = caller_user_id
        return inner()


def test_cleanup_rejects_foreign_file_id_before_delete(app):
    """cleanup-files must 403 before deleting if ANY id is not owned."""
    body = {"fileIds": [f"{USER_A}_mine", f"{USER_B}_victim"]}

    with patch(
        "domains.upload.api.load_data.local_chunk_service.delete_processed_result"
    ) as mock_local_del, patch(
        "domains.upload.api.load_data.storage_service.delete_file"
    ) as mock_storage_del:
        response = _call(app, cleanup_files, USER_A, body)

        assert _status(response) == 403, f"expected 403, got {_status(response)}"
        # No deletion may happen for any id once one is foreign.
        mock_local_del.assert_not_called()
        mock_storage_del.assert_not_called()


def test_cleanup_allows_all_owned_file_ids(app):
    body = {"fileIds": [f"{USER_A}_one", f"{USER_A}_two"]}

    with patch(
        "domains.upload.api.load_data.local_chunk_service.delete_processed_result",
        return_value=True,
    ) as mock_local_del, patch(
        "domains.upload.api.load_data.storage_service.delete_file"
    ):
        response = _call(app, cleanup_files, USER_A, body)

        assert _status(response) == 200
        assert mock_local_del.call_count == 2


def test_merge_rejects_foreign_file_id_before_read(app):
    """merge-and-prepare must 403 before reading if ANY id is not owned."""
    body = {"fileIds": [f"{USER_A}_mine", f"{USER_B}_victim"]}

    with patch(
        "domains.upload.api.load_data.local_chunk_service.get_processed_result"
    ) as mock_local_get, patch(
        "domains.upload.api.load_data.storage_service.download_csv"
    ) as mock_storage_get:
        response = _call(app, merge_and_prepare, USER_A, body)

        assert _status(response) == 403, f"expected 403, got {_status(response)}"
        mock_local_get.assert_not_called()
        mock_storage_get.assert_not_called()


def test_merge_single_element_foreign_id_rejected(app):
    """The single-element short-circuit path must also be ownership-guarded."""
    body = {"fileIds": [f"{USER_B}_victim"]}

    response = _call(app, merge_and_prepare, USER_A, body)
    assert _status(response) == 403
