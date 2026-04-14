"""Tests for /api/training/environment-info endpoint."""
from unittest.mock import patch, MagicMock
from flask import Flask


def test_environment_info_returns_versions():
    from domains.training.api.environment_routes import bp
    app = Flask(__name__)
    app.register_blueprint(bp, url_prefix='/api/training')
    client = app.test_client()

    # require_auth reads request header then calls supabase.auth.get_user
    # Patch get_supabase_client to return a mock where get_user returns a valid user
    mock_supabase = MagicMock()
    mock_supabase.auth.get_user.return_value = MagicMock(user=MagicMock(
        id='test-user', email='t@e.com', user_metadata={}
    ))
    with patch('shared.auth.jwt.get_supabase_client', return_value=mock_supabase):
        resp = client.get('/api/training/environment-info',
                          headers={'Authorization': 'Bearer fake'})

    assert resp.status_code == 200
    data = resp.get_json()
    for field in ('python', 'tensorflow', 'keras', 'numpy', 'pip_install'):
        assert field in data and data[field]
    assert 'tensorflow==' in data['pip_install']
    assert 'keras==' in data['pip_install']


def test_environment_info_without_auth_returns_401():
    from domains.training.api.environment_routes import bp
    app = Flask(__name__)
    app.register_blueprint(bp, url_prefix='/api/training')
    client = app.test_client()
    resp = client.get('/api/training/environment-info')
    assert resp.status_code == 401
