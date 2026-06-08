"""Input type-confusion hardening tests (Finding M-1).

Wrong-typed JSON inputs to cloud endpoints must produce a structured HTTP 400
(error-contract), NOT a 500 leaking a Python exception via str(e).

Specific paths covered:
1. /interpolate-chunked with uploadId of wrong type (dict) -> 400, not 500.
   (re.match on a dict raises TypeError inside sanitize_upload_id.)
2. /interpolate-chunked with a VALID owned upload + max_time_span of wrong type
   (dict) -> 400, not 500. (float({}) raises TypeError.)
3. /upload-chunk with totalChunks=0 -> 400, not 500. (division by zero.)

The harness mirrors test_upload_ownership.py's stub/reload pattern so it runs
without Docker network, Supabase, or Stripe — but is executed in Docker per
project policy.
"""
import os
import io

# Force testing mode (1000/min rate limit) BEFORE any app import.
os.environ.setdefault('FLASK_ENV', 'testing')

from unittest.mock import patch
from functools import wraps

from flask import Flask, g


# ---------------------------------------------------------------------------
# Harness (mirrors test_upload_ownership.py)
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


def _assert_no_exception_leak(body):
    """Structured contract must not leak raw Python exception text."""
    text = str(body)
    assert 'Traceback' not in text, f"Response leaked a traceback: {body}"
    # The bare str(e) for these inputs would contain typical exception phrasing.
    assert "'NoneType'" not in text and 'argument of type' not in text, (
        f"Response leaked raw exception text: {body}"
    )


# ---------------------------------------------------------------------------
# 1. /interpolate-chunked: uploadId wrong type -> 400 (sanitize_upload_id TypeError)
# ---------------------------------------------------------------------------

def test_interpolate_chunked_uploadid_dict_returns_400():
    from domains.cloud.services.upload_manager import chunk_uploads
    chunk_uploads.clear()

    app = _build_app_as_user('alice')
    with app.test_client() as c:
        resp = c.post('/api/cloud/interpolate-chunked', json={'uploadId': {'x': 1}})

    chunk_uploads.clear()

    assert resp.status_code == 400, (
        f"Expected 400 for dict uploadId (type-confusion), got {resp.status_code}. "
        f"Body: {resp.get_json()}"
    )
    body = resp.get_json()
    assert body.get('success') is False, f"Expected success=False, got: {body}"
    _assert_no_exception_leak(body)


# ---------------------------------------------------------------------------
# 2. /interpolate-chunked: max_time_span wrong type on owned upload -> 400
#    (float({}) raises TypeError; reached before any chunk file I/O.)
# ---------------------------------------------------------------------------

def test_interpolate_chunked_max_time_span_dict_returns_400():
    from domains.cloud.services.upload_manager import chunk_uploads
    chunk_uploads.clear()

    upload_id = 'owned-upload-mts-1'
    chunk_uploads[upload_id] = {
        'temp_file': {'total_chunks': 0, 'received_chunks': set(), 'filename': None},
        'load_file': {'total_chunks': 0, 'received_chunks': set(), 'filename': None},
        'interpolate_file': {'total_chunks': 1, 'received_chunks': {0}, 'filename': 'x.csv'},
    }
    chunk_uploads.set_owner(upload_id, 'alice')

    app = _build_app_as_user('alice')
    with app.test_client() as c:
        resp = c.post(
            '/api/cloud/interpolate-chunked',
            json={'uploadId': upload_id, 'max_time_span': {}},
        )

    chunk_uploads.clear()

    assert resp.status_code == 400, (
        f"Expected 400 for dict max_time_span (type-confusion), got {resp.status_code}. "
        f"Body: {resp.get_json()}"
    )
    body = resp.get_json()
    assert body.get('success') is False, f"Expected success=False, got: {body}"
    _assert_no_exception_leak(body)


# ---------------------------------------------------------------------------
# 3. /upload-chunk: totalChunks=0 -> 400 (avoid ZeroDivisionError -> 500)
# ---------------------------------------------------------------------------

def test_upload_chunk_total_chunks_zero_returns_400():
    from domains.cloud.services.upload_manager import chunk_uploads
    chunk_uploads.clear()

    app = _build_app_as_user('alice')
    with app.test_client() as c:
        data = {
            'uploadId': 'zero-chunks-upload-1',
            'fileType': 'interpolate_file',
            'chunkIndex': '0',
            'totalChunks': '0',
            'file': (io.BytesIO(b'a,b\n1,2\n'), 'x.csv'),
        }
        resp = c.post(
            '/api/cloud/upload-chunk',
            data=data,
            content_type='multipart/form-data',
        )

    chunk_uploads.clear()

    assert resp.status_code == 400, (
        f"Expected 400 for totalChunks=0, got {resp.status_code}. "
        f"Body: {resp.get_json()}"
    )
    body = resp.get_json()
    assert body.get('success') is False, f"Expected success=False, got: {body}"
    _assert_no_exception_leak(body)
