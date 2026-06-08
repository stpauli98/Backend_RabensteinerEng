"""IDOR hardening tests (Finding C-6) — cloud upload sessions bound to user_id.

Two layers:
1. Unit tests for UploadManager.set_owner / is_owner ownership tracking.
2. Endpoint test: /interpolate-chunked returns 403 when the upload session
   belongs to a DIFFERENT user than g.user_id.

The endpoint test mirrors test_cloud_routes.py's stub/reload pattern so it runs
without Docker network, Supabase, or Stripe — but is executed in Docker per
project policy.
"""
import os

# Force testing mode (1000/min rate limit) BEFORE any app import.
os.environ.setdefault('FLASK_ENV', 'testing')

from unittest.mock import patch
from functools import wraps

import pytest
from flask import Flask, g


# ---------------------------------------------------------------------------
# Unit tests: UploadManager ownership tracking
# ---------------------------------------------------------------------------

@pytest.fixture
def manager():
    from domains.cloud.services.upload_manager import UploadManager
    m = UploadManager()
    yield m
    m.clear()


def _seed_session(m, upload_id, owner):
    """Create a session the way /upload-chunk does, then bind the owner."""
    m[upload_id] = {
        'temp_file': {'total_chunks': 0, 'received_chunks': set(), 'filename': None},
        'load_file': {'total_chunks': 0, 'received_chunks': set(), 'filename': None},
        'interpolate_file': {'total_chunks': 0, 'received_chunks': set(), 'filename': None},
    }
    m.set_owner(upload_id, owner)


def test_is_owner_true_for_creating_user(manager):
    _seed_session(manager, 'upload-alice-1', 'alice')
    assert manager.is_owner('upload-alice-1', 'alice') is True


def test_is_owner_false_for_other_user(manager):
    _seed_session(manager, 'upload-alice-1', 'alice')
    assert manager.is_owner('upload-alice-1', 'bob') is False


def test_is_owner_false_for_nonexistent_session(manager):
    assert manager.is_owner('nonexistent', 'alice') is False


def test_set_owner_idempotent_for_same_user(manager):
    _seed_session(manager, 'upload-alice-1', 'alice')
    # Re-binding the same owner (e.g. subsequent chunk) must keep ownership.
    manager.set_owner('upload-alice-1', 'alice')
    assert manager.is_owner('upload-alice-1', 'alice') is True


def test_owner_survives_session_data_access(manager):
    _seed_session(manager, 'upload-alice-1', 'alice')
    # Reading session data must not clear ownership.
    _ = manager['upload-alice-1']
    assert manager.is_owner('upload-alice-1', 'alice') is True


def test_owner_removed_with_session(manager):
    _seed_session(manager, 'upload-alice-1', 'alice')
    manager.remove('upload-alice-1')
    assert manager.is_owner('upload-alice-1', 'alice') is False


# ---------------------------------------------------------------------------
# Endpoint test: cross-user access on /interpolate-chunked must be rejected
# ---------------------------------------------------------------------------

def _stub_require_auth_as(user_id):
    def deco(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            g.user_id = user_id
            g.user_email = f'{user_id}@example.com'
            g.access_token = 'test-token'
            return f(*args, **kwargs)
        return wrapper
    return deco


def _stub_require_subscription(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        g.subscription = {'id': 'sub-1'}
        g.plan = {'name': 'pro', 'max_processing_jobs_per_month': 100, 'total_compute_hours': 0}
        g.usage = {'processing_jobs_count': 0, 'processing_count': 0}
        return f(*args, **kwargs)
    return wrapper


def _stub_check_processing_limit(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        return f(*args, **kwargs)
    return wrapper


def _build_app_as_user(user_id):
    from core.rate_limits import limiter
    import importlib

    with patch('shared.auth.jwt.require_auth', side_effect=_stub_require_auth_as(user_id)), \
         patch('shared.auth.subscription.require_subscription', side_effect=_stub_require_subscription), \
         patch('shared.auth.subscription.check_processing_limit', side_effect=_stub_check_processing_limit):
        import domains.cloud.api.cloud as cloud_module
        importlib.reload(cloud_module)
        _app = Flask(__name__)
        _app.config['TESTING'] = True
        limiter.init_app(_app)
        _app.register_blueprint(cloud_module.bp, url_prefix='/api/cloud')

    importlib.reload(cloud_module)
    return _app


def test_interpolate_chunked_rejects_foreign_owner_with_403():
    """C-6: a session owned by alice must not be processable by bob."""
    from domains.cloud.services.upload_manager import chunk_uploads
    chunk_uploads.clear()

    upload_id = 'foreign-upload-123'
    # alice creates and owns the session
    chunk_uploads[upload_id] = {
        'temp_file': {'total_chunks': 0, 'received_chunks': set(), 'filename': None},
        'load_file': {'total_chunks': 0, 'received_chunks': set(), 'filename': None},
        'interpolate_file': {'total_chunks': 1, 'received_chunks': {0}, 'filename': 'x.csv'},
    }
    chunk_uploads.set_owner(upload_id, 'alice')

    # bob attempts to drive interpolation on alice's upload
    app = _build_app_as_user('bob')
    with app.test_client() as c:
        resp = c.post('/api/cloud/interpolate-chunked', json={'uploadId': upload_id})

    chunk_uploads.clear()

    assert resp.status_code == 403, (
        f"Expected 403 for cross-user upload access, got {resp.status_code}. "
        f"Body: {resp.get_json()}"
    )
    body = resp.get_json()
    assert body.get('success') is False, f"Expected success=False, got: {body}"
