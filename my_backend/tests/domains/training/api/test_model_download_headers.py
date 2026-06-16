"""Verify download-model-h5 attaches X-Model-Env-* headers."""
from unittest.mock import patch, MagicMock
from flask import Flask


def _make_client():
    # Avoid registering the whole training blueprint (other sub-blueprints have heavy imports).
    # Register only model_routes' bp at /api/training.
    from domains.training.api.model_routes import bp
    app = Flask(__name__)
    app.register_blueprint(bp, url_prefix='/api/training')
    return app.test_client()


def _auth_patch():
    # require_auth now verifies the JWT locally via _verify_jwt_local (no Supabase
    # network call). Patch it to return a valid claims dict so the decorator passes.
    fake_claims = {
        'sub': 'test-user',
        'email': 't@e.com',
        'user_metadata': {},
        'role': 'authenticated',
    }
    return patch('shared.auth.jwt._verify_jwt_local', return_value=fake_claims)


def test_download_response_has_env_headers():
    client = _make_client()
    # Use a valid UUID as session_id so validate_training_session_format passes.
    # Also patch _resolve_and_assert_ownership to bypass DB lookup and ownership check.
    valid_session_id = 'b2be65df-ce96-4305-b4c7-6530c7bc7096'
    with _auth_patch(), \
         patch('domains.training.api.model_routes._resolve_and_assert_ownership',
               return_value=(valid_session_id, None)), \
         patch('domains.training.api.model_routes.download_model_file',
               return_value=(b'fake-bytes', 'fake.keras')):
        resp = client.get(
            f'/api/training/download-model-h5/{valid_session_id}?filename=fake.keras',
            headers={'Authorization': 'Bearer fake'}
        )
    assert resp.status_code == 200, resp.get_data(as_text=True)
    for header in ('X-Model-Env-Python', 'X-Model-Env-TensorFlow',
                   'X-Model-Env-Keras', 'X-Model-Env-Numpy'):
        assert header in resp.headers, f'missing: {header}'
        assert resp.headers[header], f'empty: {header}'
