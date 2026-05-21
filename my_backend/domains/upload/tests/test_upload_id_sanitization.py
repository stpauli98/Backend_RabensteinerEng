"""Verify upload_id is sanitized against path traversal (GB5)."""
import os
import tempfile
import pytest
from unittest.mock import patch


# Path-traversal payloads to test
TRAVERSAL_PAYLOADS = [
    "../evil",
    "../../evil",
    "/etc/passwd",
    "/tmp/evil",
    "subdir/../../etc",
    "..\\evil",
    "evil/with/slashes",
    "evil\\with\\backslashes",
    "..",
    ".",
]


@pytest.mark.parametrize("payload", TRAVERSAL_PAYLOADS)
def test_get_chunk_dir_rejects_path_traversal(payload):
    """get_chunk_dir must raise ValueError for any upload_id containing path separators or '..'"""
    from domains.processing.services.local_chunk_service import LocalChunkService

    service = LocalChunkService()
    with pytest.raises(ValueError, match=r"(invalid|unsafe|sanitiz)"):
        service.get_chunk_dir(payload)


def test_get_chunk_dir_accepts_safe_uuid_format():
    """get_chunk_dir must accept normal UUID-style upload IDs."""
    from domains.processing.services.local_chunk_service import LocalChunkService

    service = LocalChunkService()
    safe_ids = [
        "4633c88e-36fb-446d-a17e-90374359875c_1779273211549-7kttd4wa_c4896d60",
        "abc123",
        "test-upload-id_456",
        "ABCdef-123_456",
    ]
    for upload_id in safe_ids:
        result = service.get_chunk_dir(upload_id)
        # Should not raise; should produce a path UNDER the CHUNK_DIR
        from domains.processing.services.local_chunk_service import CHUNK_DIR
        normalized = os.path.normpath(result)
        normalized_chunk_dir = os.path.normpath(CHUNK_DIR)
        assert normalized.startswith(normalized_chunk_dir + os.sep), \
            f"upload_id {upload_id!r} produced path outside CHUNK_DIR: {result}"


def test_save_upload_metadata_rejects_path_traversal():
    """High-level entrypoint should also fail when given a path-traversal upload_id."""
    from domains.processing.services.local_chunk_service import LocalChunkService

    service = LocalChunkService()
    with pytest.raises(ValueError):
        service.save_upload_metadata(
            upload_id="../malicious",
            total_chunks=1,
            parameters={'delimiter': ';'},
            user_id="any-user",
        )


def test_upload_chunk_method_rejects_path_traversal():
    from domains.processing.services.local_chunk_service import LocalChunkService

    service = LocalChunkService()
    with pytest.raises(ValueError):
        service.upload_chunk(upload_id="../evil", chunk_index=0, data=b"x")


def test_delete_upload_chunks_rejects_path_traversal():
    from domains.processing.services.local_chunk_service import LocalChunkService

    service = LocalChunkService()
    with pytest.raises(ValueError):
        service.delete_upload_chunks(upload_id="../evil")


def test_get_upload_metadata_returns_none_for_traversal(tmp_path):
    """get_upload_metadata is read-only; instead of raising, returning None is acceptable
    (defense-in-depth: caller still cannot read arbitrary files since the check fails)."""
    from domains.processing.services.local_chunk_service import LocalChunkService

    service = LocalChunkService()
    # Either raise OR return None — both prevent the traversal. We accept both:
    try:
        result = service.get_upload_metadata("../evil")
        assert result is None, f"Expected None or ValueError, got {result!r}"
    except ValueError:
        pass  # acceptable


def test_upload_chunk_endpoint_returns_400_for_path_traversal_upload_id(client):
    """The /upload-chunk endpoint should reject path-traversal upload_id with 400, not 500."""
    import io
    from unittest.mock import patch

    fake_sub = {
        'id': 'sub-1', 'user_id': 'u1', 'plan_name': 'STANDARD', 'status': 'active',
        'expires_at': '2099-01-01T00:00:00+00:00',
        'subscription_plans': {
            'name': 'STANDARD',
            'max_uploads_per_month': 100,
            'max_processing_jobs_per_month': 50,
            'max_storage_gb': 10,
            'total_compute_hours': 0,
        },
    }
    with patch('shared.auth.jwt._verify_jwt_local', return_value={'sub': 'u1', 'email': 'a@b.com'}):
        with patch('shared.auth.subscription.get_user_subscription', return_value=fake_sub):
            data = {
                'fileChunk': (io.BytesIO(b'x'), 'test.csv'),
                'uploadId': '../traversal-attempt',
                'chunkIndex': '0',
                'totalChunks': '1',
                'delimiter': ';',
                'selected_columns': '{}',
                'timezone': 'UTC',
                'dropdown_count': '2',
                'hasHeader': 'ja',
            }
            resp = client.post(
                '/api/loadRowData/upload-chunk',
                data=data,
                headers={'Authorization': 'Bearer fake'},
                content_type='multipart/form-data',
            )
            # Should be 400 (bad request) — NOT 500 (uncaught exception)
            assert resp.status_code in (400, 422), \
                f"Expected 400/422 for path-traversal upload_id, got {resp.status_code}: {resp.data}"


# ---- fixtures ----

from flask import Flask, g

FAKE_USAGE = {'processing_count': 0, 'processing_jobs_count': 0}


@pytest.fixture
def app():
    from domains.upload.api.load_data import bp
    flask_app = Flask(__name__)
    flask_app.config['TESTING'] = True
    flask_app.register_blueprint(bp, url_prefix='/api/loadRowData')

    @flask_app.before_request
    def _seed_g_usage():
        g.usage = FAKE_USAGE

    return flask_app


@pytest.fixture
def client(app):
    return app.test_client()
