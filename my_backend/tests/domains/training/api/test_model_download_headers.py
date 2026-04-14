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
    mock_supabase = MagicMock()
    mock_supabase.auth.get_user.return_value = MagicMock(user=MagicMock(
        id='test-user', email='t@e.com', user_metadata={}
    ))
    return patch('shared.auth.jwt.get_supabase_client', return_value=mock_supabase)


def test_download_response_has_env_headers():
    client = _make_client()
    with _auth_patch(), \
         patch('domains.training.api.model_routes.download_model_file',
               return_value=(b'fake-bytes', 'fake.keras')):
        resp = client.get(
            '/api/training/download-model-h5/test-session?filename=fake.keras',
            headers={'Authorization': 'Bearer fake'}
        )
    assert resp.status_code == 200, resp.get_data(as_text=True)
    for header in ('X-Model-Env-Python', 'X-Model-Env-TensorFlow',
                   'X-Model-Env-Keras', 'X-Model-Env-Numpy'):
        assert header in resp.headers, f'missing: {header}'
        assert resp.headers[header], f'empty: {header}'
