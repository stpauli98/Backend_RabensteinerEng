"""Verify @require_subscription is applied to all chunk lifecycle endpoints (GB3)."""
import pytest
from unittest.mock import patch
from flask import Flask, g


@pytest.fixture
def app():
    from domains.upload.api.load_data import bp
    flask_app = Flask(__name__)
    flask_app.config['TESTING'] = True
    flask_app.register_blueprint(bp, url_prefix='/api/loadRowData')

    # Seed g.usage for legacy middleware quirk (pre-existing — see GB2 fixture)
    @flask_app.before_request
    def _seed_g_usage():
        if not hasattr(g, 'usage'):
            g.usage = None

    return flask_app


@pytest.fixture
def client(app):
    return app.test_client()


def test_cancel_upload_rejects_user_without_subscription(client):
    with patch('shared.auth.jwt._verify_jwt_local', return_value={'sub': 'no-sub-user', 'email': 'a@b.com'}):
        with patch('shared.auth.subscription.get_user_subscription', return_value=None):
            resp = client.post(
                '/api/loadRowData/cancel-upload',
                json={'uploadId': 'any-id'},
                headers={'Authorization': 'Bearer fake'},
            )
            assert resp.status_code == 403, \
                f"Expected 403 for no-subscription cancel-upload, got {resp.status_code}: {resp.data}"


def test_prepare_save_rejects_user_without_subscription(client):
    with patch('shared.auth.jwt._verify_jwt_local', return_value={'sub': 'no-sub-user', 'email': 'a@b.com'}):
        with patch('shared.auth.subscription.get_user_subscription', return_value=None):
            resp = client.post(
                '/api/loadRowData/prepare-save',
                json={'data': {'data': [['header1', 'header2'], ['val1', 'val2']], 'fileName': 'test.csv'}},
                headers={'Authorization': 'Bearer fake'},
            )
            assert resp.status_code == 403, \
                f"Expected 403 for no-subscription prepare-save, got {resp.status_code}: {resp.data}"


def test_merge_and_prepare_rejects_user_without_subscription(client):
    with patch('shared.auth.jwt._verify_jwt_local', return_value={'sub': 'no-sub-user', 'email': 'a@b.com'}):
        with patch('shared.auth.subscription.get_user_subscription', return_value=None):
            resp = client.post(
                '/api/loadRowData/merge-and-prepare',
                json={'fileIds': ['file-a', 'file-b'], 'fileName': 'merged.csv'},
                headers={'Authorization': 'Bearer fake'},
            )
            assert resp.status_code == 403, \
                f"Expected 403 for no-subscription merge-and-prepare, got {resp.status_code}: {resp.data}"
