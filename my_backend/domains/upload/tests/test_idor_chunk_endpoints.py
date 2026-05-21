"""Verify IDOR protection on chunk endpoints (GB2)."""
import io
import json
import os
import tempfile
import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime, timezone, timedelta
from flask import Flask, g


FAKE_SUB = {
    'id': 'sub-1',
    'user_id': 'owner-user-uuid',
    'status': 'active',
    'expires_at': '2026-12-31T00:00:00+00:00',
    'subscription_plans': {
        'name': 'STANDARD',
        'max_uploads_per_month': 100,
        'max_processing_jobs_per_month': 50,
        'max_storage_gb': 10,
        'total_compute_hours': 0,
    },
}

FAKE_USAGE = {'processing_count': 0, 'processing_jobs_count': 0}


@pytest.fixture
def app():
    from domains.upload.api.load_data import bp
    flask_app = Flask(__name__)
    flask_app.config['TESTING'] = True
    flask_app.register_blueprint(bp, url_prefix='/api/loadRowData')

    # Pre-set g.usage so check_processing_limit's legacy branch does not raise
    # AttributeError when accessing g.usage (g attributes raise AttributeError
    # rather than returning None when unset).
    @flask_app.before_request
    def _seed_g_usage():
        g.usage = FAKE_USAGE

    return flask_app


@pytest.fixture
def client(app):
    return app.test_client()


@pytest.fixture
def metadata_with_owner(tmp_path):
    """Create a temp chunk dir with metadata owned by 'owner-user-uuid'."""
    upload_id = 'idor-test-upload-1'
    chunk_dir = tmp_path / upload_id
    chunk_dir.mkdir()
    metadata = {
        'total_chunks': 3,
        'parameters': {'delimiter': ';'},
        'user_id': 'owner-user-uuid',  # legit owner
        'created_at': datetime.now().isoformat(),
    }
    (chunk_dir / '_metadata.json').write_text(json.dumps(metadata))
    return upload_id, str(tmp_path)


def test_upload_chunk_rejects_when_attacker_uses_owner_upload_id(client, metadata_with_owner):
    """Attacker with valid JWT cannot upload chunks against another user's upload_id."""
    upload_id, chunk_root = metadata_with_owner

    with patch('shared.auth.jwt._verify_jwt_local', return_value={'sub': 'attacker-uuid', 'email': 'evil@example.com'}), \
         patch('shared.auth.subscription.get_user_subscription', return_value=FAKE_SUB), \
         patch('domains.processing.services.local_chunk_service.CHUNK_DIR', chunk_root), \
         patch('time.sleep', return_value=None):
        data = {
            'fileChunk': (io.BytesIO(b'malicious data'), 'evil.csv'),
            'uploadId': upload_id,
            'chunkIndex': '1',  # non-zero — triggers existing-metadata path
            'totalChunks': '3',
            'delimiter': ';',
            'selected_columns': '{}',
            'timezone': 'UTC',
            'dropdown_count': '2',
            'hasHeader': 'ja',
        }
        resp = client.post(
            '/api/loadRowData/upload-chunk',
            data=data,
            headers={'Authorization': 'Bearer attacker-token'},
            content_type='multipart/form-data',
        )
        assert resp.status_code == 403, \
            f"Expected 403 Forbidden when attacker uses owner's upload_id, got {resp.status_code}: {resp.data}"


def test_finalize_upload_rejects_when_attacker_uses_owner_upload_id(client, metadata_with_owner):
    """Attacker cannot finalize another user's upload."""
    upload_id, chunk_root = metadata_with_owner

    with patch('shared.auth.jwt._verify_jwt_local', return_value={'sub': 'attacker-uuid', 'email': 'evil@example.com'}), \
         patch('shared.auth.subscription.get_user_subscription', return_value=FAKE_SUB), \
         patch('shared.auth.subscription.get_user_usage', return_value=FAKE_USAGE), \
         patch('domains.processing.services.local_chunk_service.CHUNK_DIR', chunk_root):
        resp = client.post(
            '/api/loadRowData/finalize-upload',
            json={'uploadId': upload_id},
            headers={'Authorization': 'Bearer attacker-token'},
        )
        assert resp.status_code == 403, \
            f"Expected 403 Forbidden, got {resp.status_code}: {resp.data}"


def test_cancel_upload_rejects_when_attacker_uses_owner_upload_id(client, metadata_with_owner):
    """Attacker cannot cancel another user's upload."""
    upload_id, chunk_root = metadata_with_owner

    with patch('shared.auth.jwt._verify_jwt_local', return_value={'sub': 'attacker-uuid', 'email': 'evil@example.com'}), \
         patch('shared.auth.subscription.get_user_subscription', return_value=FAKE_SUB), \
         patch('domains.processing.services.local_chunk_service.CHUNK_DIR', chunk_root):
        resp = client.post(
            '/api/loadRowData/cancel-upload',
            json={'uploadId': upload_id},
            headers={'Authorization': 'Bearer attacker-token'},
        )
        assert resp.status_code == 403, \
            f"Expected 403 Forbidden, got {resp.status_code}: {resp.data}"
