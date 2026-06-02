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


# ---------------------------------------------------------------------------
# W11-A T7: rate-limit decorator presence
# ---------------------------------------------------------------------------

def _has_rate_limit(fn) -> bool:
    """Walk decorator chain to find Flask-Limiter's marker."""
    current = fn
    while current is not None:
        if hasattr(current, '_rate_limits'):
            return True
        if hasattr(current, '__dict__') and '__wrapper-limiter-instance' in current.__dict__:
            return True
        current = getattr(current, '__wrapped__', None)
    return False


def test_environment_info_handler_has_rate_limit():
    """W11-BE1: environment_info must declare @limiter.limit(training_limit_string)."""
    import importlib
    import domains.training.api.environment_routes as environment_routes
    importlib.reload(environment_routes)

    handler = getattr(environment_routes, 'environment_info', None)
    assert handler is not None, "environment_info not exported from environment_routes"
    assert _has_rate_limit(handler), (
        "environment_info missing @limiter.limit decorator. "
        "Apply @limiter.limit(training_limit_string) between @bp.route and @require_auth."
    )
