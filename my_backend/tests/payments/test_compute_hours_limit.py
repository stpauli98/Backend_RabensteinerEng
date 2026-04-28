"""
Compute-hours limit gating coverage.

Pins three properties of the hybrid check_processing_limit decorator:
  1) For a plan with total_compute_hours > 0, gating uses the RPC and
     returns 403 with error_code='compute_hours_exhausted' when exceeded.
  2) For a plan without compute hours, the legacy job-count gate fires.
  3) When neither limit is reached, the wrapped function runs.
"""
from unittest.mock import patch, MagicMock
import json
from functools import wraps

import pytest
from flask import Flask, g, jsonify


@pytest.fixture
def app():
    """Build a minimal Flask app that wires the decorator to a test endpoint."""
    flask_app = Flask(__name__)
    flask_app.config['TESTING'] = True

    # Import after env vars from conftest are set.
    from shared.auth.subscription import check_processing_limit

    # Stand-in for @require_auth + @require_subscription so the decorator
    # under test sees the g attributes it expects.
    def fake_auth(plan, usage, user_id='4633c88e-36fb-446d-a17e-90374359875c'):
        def deco(f):
            @wraps(f)
            def wrapper(*args, **kwargs):
                g.user_id = user_id
                g.user_email = 'test@example.com'
                g.access_token = 'test_token'
                g.plan = plan
                g.usage = usage
                return f(*args, **kwargs)
            return wrapper
        return deco

    @flask_app.route('/test-compute-plan', methods=['POST'])
    @fake_auth(
        plan={'name': 'STANDARD', 'total_compute_hours': 50, 'max_processing_jobs_per_month': 999},
        usage={'processing_jobs_count': 0, 'processing_count': 0},
    )
    @check_processing_limit
    def test_compute_plan():
        return jsonify({'ok': True}), 200

    @flask_app.route('/test-count-plan', methods=['POST'])
    @fake_auth(
        plan={'name': 'Free', 'total_compute_hours': 0, 'max_processing_jobs_per_month': 5},
        usage={'processing_jobs_count': 5, 'processing_count': 5},
    )
    @check_processing_limit
    def test_count_plan():
        return jsonify({'ok': True}), 200

    @flask_app.route('/test-allowed', methods=['POST'])
    @fake_auth(
        plan={'name': 'STANDARD', 'total_compute_hours': 50, 'max_processing_jobs_per_month': 999},
        usage={'processing_jobs_count': 0, 'processing_count': 0},
    )
    @check_processing_limit
    def test_allowed():
        return jsonify({'ok': True}), 200

    return flask_app


@pytest.fixture
def client(app):
    return app.test_client()


def test_compute_hours_exhausted_returns_403_with_error_code(client):
    """User has used 50.5h of a 50h plan → 403 with compute_hours_exhausted code."""
    seconds_used = int(50.5 * 3600)
    fake_supabase = MagicMock()
    fake_supabase.rpc.return_value.execute.return_value.data = seconds_used

    with patch('shared.auth.subscription.get_supabase_user_client', return_value=fake_supabase):
        resp = client.post('/test-compute-plan')

    assert resp.status_code == 403
    body = json.loads(resp.data)
    assert body['error_code'] == 'compute_hours_exhausted'
    assert body['redirect_to'] == '/pricing'
    assert body['limit_hours'] == 50
    assert body['used_hours'] == pytest.approx(50.5, abs=0.01)
    fake_supabase.rpc.assert_called_with(
        'get_total_compute_seconds',
        {'p_user_id': '4633c88e-36fb-446d-a17e-90374359875c'},
    )


def test_compute_hours_under_limit_allows_operation(client):
    """User has used 10h of a 50h plan → operation runs, 200."""
    seconds_used = int(10 * 3600)
    fake_supabase = MagicMock()
    fake_supabase.rpc.return_value.execute.return_value.data = seconds_used

    with patch('shared.auth.subscription.get_supabase_user_client', return_value=fake_supabase):
        resp = client.post('/test-allowed')

    assert resp.status_code == 200
    assert json.loads(resp.data) == {'ok': True}


def test_count_plan_at_limit_returns_403_legacy_message(client):
    """Free plan with no compute hours and count >= max → legacy 403 path."""
    # No supabase patch needed — the legacy branch uses g.usage which the
    # fake_auth fixture already set to processing_jobs_count=5 against
    # max_processing_jobs_per_month=5.
    resp = client.post('/test-count-plan')
    assert resp.status_code == 403
    body = json.loads(resp.data)
    # Legacy branch does not set error_code = 'compute_hours_exhausted'.
    assert body.get('error_code') != 'compute_hours_exhausted'
