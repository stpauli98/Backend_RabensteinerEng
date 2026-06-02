"""W11-ADV-4: global 404/405/500 errorhandlers return JSON, not HTML.

Flask defaults to HTML for these status codes; the FE error mapper keys off
`code` in the JSON body. Without these handlers, FE shows the raw HTML or
falls back to a generic message.
"""
import os

# Force testing mode BEFORE any app import.
os.environ.setdefault('FLASK_ENV', 'testing')

import pytest
from flask import Flask, jsonify, request


def _build_minimal_app():
    """Build a small Flask app and register the same handlers app_factory installs.

    We don't call create_app() because it boots SocketIO, the APScheduler,
    Supabase clients, etc. — far too heavy for an error-handler test. Instead
    we re-register the handlers under test on a clean Flask app.
    """
    app = Flask(__name__)
    app.config['TESTING'] = True

    # A single GET-only route so we can probe 404 (unknown path) and 405
    # (wrong method on a known path).
    @app.route('/api/training/probe', methods=['GET'])
    def _probe_get():
        return jsonify({'ok': True})

    # Mirror the handlers added in core/app_factory.create_app().
    @app.errorhandler(404)
    def _not_found(error):
        if hasattr(error, 'response') and error.response:
            return error.response
        return jsonify({
            'success': False,
            'code': 'NOT_FOUND',
            'error': 'Resource not found',
        }), 404

    @app.errorhandler(405)
    def _method_not_allowed(error):
        if hasattr(error, 'response') and error.response:
            return error.response
        return jsonify({
            'success': False,
            'code': 'METHOD_NOT_ALLOWED',
            'error': f'Method {request.method} not allowed for this endpoint',
        }), 405

    return app


@pytest.fixture
def client():
    app = _build_minimal_app()
    with app.test_client() as c:
        yield c


def test_404_returns_json_with_code(client):
    """Unknown route → 404 with JSON {code: NOT_FOUND}."""
    r = client.get("/api/training/nonexistent-endpoint-xyz")
    assert r.status_code == 404
    assert r.content_type.startswith('application/json'), (
        f"404 must be JSON, got content_type={r.content_type}, body={r.get_data(as_text=True)[:200]}"
    )
    body = r.get_json()
    assert body is not None
    assert body.get('success') is False
    assert body.get('code') == 'NOT_FOUND', f"Expected code=NOT_FOUND, got: {body}"


def test_405_returns_json_with_code(client):
    """Wrong method on a known route → 405 with JSON {code: METHOD_NOT_ALLOWED}."""
    r = client.patch("/api/training/probe")
    assert r.status_code == 405
    assert r.content_type.startswith('application/json'), (
        f"405 must be JSON, got content_type={r.content_type}, body={r.get_data(as_text=True)[:200]}"
    )
    body = r.get_json()
    assert body is not None
    assert body.get('success') is False
    assert body.get('code') == 'METHOD_NOT_ALLOWED', (
        f"Expected code=METHOD_NOT_ALLOWED, got: {body}"
    )


# ---------------------------------------------------------------------------
# Integration check: the handlers above must also be registered on the real
# app produced by core.app_factory.create_app(). We probe a known unknown
# route to verify the global 404 handler from app_factory wires up correctly.
# ---------------------------------------------------------------------------

def test_404_on_real_app_factory_returns_json():
    """create_app() must register a JSON 404 handler."""
    from core.app_factory import create_app
    real_app, _ = create_app()
    real_app.config['TESTING'] = True
    with real_app.test_client() as c:
        r = c.get("/api/training/nonexistent-endpoint-xyz")
        assert r.status_code == 404
        assert r.content_type.startswith('application/json'), (
            f"Real app 404 must be JSON, got: {r.content_type}"
        )
        body = r.get_json()
        assert body is not None
        assert body.get('code') == 'NOT_FOUND'


def test_405_on_real_app_factory_returns_json():
    """create_app() must register a JSON 405 handler.

    PATCH on /health (which is GET-only) should trigger the 405 handler.
    """
    from core.app_factory import create_app
    real_app, _ = create_app()
    real_app.config['TESTING'] = True
    with real_app.test_client() as c:
        r = c.patch("/health")
        assert r.status_code == 405
        assert r.content_type.startswith('application/json'), (
            f"Real app 405 must be JSON, got: {r.content_type}"
        )
        body = r.get_json()
        assert body is not None
        assert body.get('code') == 'METHOD_NOT_ALLOWED'
