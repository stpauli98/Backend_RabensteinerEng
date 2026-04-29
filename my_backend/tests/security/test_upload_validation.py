"""Schema-validation regression tests for upload JSON endpoints.

Each handler in domains/upload/api/load_data.py that consumes
``request.json`` is now wrapped with a marshmallow schema. These tests
exercise the rejection path: posting a malformed body must produce a
400 with a ``fields`` error map, instead of crashing downstream code
as 500.

The tests are unit-style and bypass @require_auth by calling the
unwrapped handler with a manually-set g.user_id, mirroring
test_download_idor.py. No Docker, Supabase, or storage backends are
required.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest
from flask import Flask, g

from domains.upload.api.load_data import (
    bp as upload_bp,
    finalize_upload,
    cancel_upload,
    prepare_save,
    merge_and_prepare,
    cleanup_files,
)


USER_A = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"


@pytest.fixture
def app() -> Flask:
    app = Flask(__name__)
    app.register_blueprint(upload_bp, url_prefix="/api/loadRowData")
    app.config["TESTING"] = True
    return app


def _unwrap(view):
    """Strip @require_auth (and any other functools.wraps decorator) layers."""
    inner = view
    while hasattr(inner, "__wrapped__"):
        inner = inner.__wrapped__
    return inner


def _post_json(app: Flask, view, path: str, body):
    """Invoke a handler under a request context with a JSON body."""
    inner = _unwrap(view)
    with app.test_request_context(path, method="POST", json=body):
        g.user_id = USER_A
        return inner()


def _status(response):
    if isinstance(response, tuple):
        return response[1]
    return response.status_code


def _body(response):
    """Extract the JSON dict from a Flask Response (or (Response, status) tuple)."""
    resp = response[0] if isinstance(response, tuple) else response
    return resp.get_json()


def test_finalize_upload_rejects_missing_uploadId(app):
    """Empty body -> 400 with fields error, not 500."""
    response = _post_json(app, finalize_upload, "/api/loadRowData/finalize-upload", {})
    assert _status(response) == 400
    body = _body(response)
    assert body["error"] == "invalid request"
    assert "uploadId" in body["fields"]


def test_finalize_upload_rejects_wrong_type(app):
    """uploadId of wrong type -> 400, not 500."""
    response = _post_json(
        app, finalize_upload, "/api/loadRowData/finalize-upload",
        {"uploadId": 12345},
    )
    assert _status(response) == 400
    assert "uploadId" in _body(response)["fields"]


def test_cancel_upload_rejects_missing_uploadId(app):
    response = _post_json(app, cancel_upload, "/api/loadRowData/cancel-upload", {})
    assert _status(response) == 400
    assert "uploadId" in _body(response)["fields"]


def test_prepare_save_rejects_missing_data_wrapper(app):
    """Missing top-level 'data' wrapper -> 400."""
    response = _post_json(app, prepare_save, "/api/loadRowData/prepare-save", {})
    assert _status(response) == 400
    assert "data" in _body(response)["fields"]


def test_prepare_save_rejects_missing_inner_data(app):
    """Wrapper present but inner 'data' rows missing -> 400."""
    response = _post_json(
        app, prepare_save, "/api/loadRowData/prepare-save",
        {"data": {"fileName": "x.csv"}},
    )
    assert _status(response) == 400
    body = _body(response)
    assert "data" in body["fields"]


def test_merge_and_prepare_rejects_missing_fileIds(app):
    response = _post_json(
        app, merge_and_prepare, "/api/loadRowData/merge-and-prepare",
        {"fileName": "out.csv"},
    )
    assert _status(response) == 400
    assert "fileIds" in _body(response)["fields"]


def test_merge_and_prepare_rejects_non_string_fileIds(app):
    response = _post_json(
        app, merge_and_prepare, "/api/loadRowData/merge-and-prepare",
        {"fileIds": [123, 456]},
    )
    assert _status(response) == 400
    assert "fileIds" in _body(response)["fields"]


def test_cleanup_files_accepts_empty_list(app):
    """Empty list is a no-op success per existing handler intent."""
    response = _post_json(
        app, cleanup_files, "/api/loadRowData/cleanup-files",
        {"fileIds": []},
    )
    assert _status(response) == 200


def test_cleanup_files_rejects_non_list_fileIds(app):
    response = _post_json(
        app, cleanup_files, "/api/loadRowData/cleanup-files",
        {"fileIds": "not-a-list"},
    )
    assert _status(response) == 400
    assert "fileIds" in _body(response)["fields"]


def test_finalize_upload_valid_shape_passes_schema(app):
    """A well-formed body must reach the downstream metadata lookup.

    We patch the local-chunk service to return None so the handler
    returns 404 ('Upload not found'), proving the schema accepted the
    payload and execution moved past validation.
    """
    with patch(
        "domains.upload.api.load_data.local_chunk_service.get_upload_metadata",
        return_value=None,
    ) as mock_meta:
        response = _post_json(
            app, finalize_upload, "/api/loadRowData/finalize-upload",
            {"uploadId": "valid-upload-id-123"},
        )
        # Schema accepted -> handler reached metadata lookup -> 404 path.
        mock_meta.assert_called_once_with("valid-upload-id-123")
        assert _status(response) == 404
