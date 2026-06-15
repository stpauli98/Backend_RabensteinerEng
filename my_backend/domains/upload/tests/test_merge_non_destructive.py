"""merge-and-prepare must NOT delete source files (re-download / re-merge support)."""
import json
from unittest.mock import patch
import pytest
from flask import Flask, g

FAKE_SUB = {
    'id': 'sub-1', 'user_id': 'owner-user-uuid', 'status': 'active',
    'expires_at': '2026-12-31T00:00:00+00:00',
    'subscription_plans': {
        'name': 'STANDARD', 'max_uploads_per_month': 100,
        'max_processing_jobs_per_month': 50, 'max_storage_gb': 10,
        'total_compute_hours': 0,
    },
}
FAKE_USAGE = {'processing_count': 0, 'processing_jobs_count': 0}


@pytest.fixture
def client():
    from domains.upload.api.load_data import bp
    app = Flask(__name__)
    app.config['TESTING'] = True
    app.register_blueprint(bp, url_prefix='/api/loadRowData')

    @app.before_request
    def _seed():
        g.usage = FAKE_USAGE

    return app.test_client()


def test_merge_and_prepare_does_not_delete_sources(client):
    csv = "UTC;val\n2026-01-01 00:00:00;1\n"
    with patch('shared.auth.jwt._verify_jwt_local',
               return_value={'sub': 'owner-user-uuid', 'email': 'o@e.com'}), \
         patch('shared.auth.subscription.get_user_subscription', return_value=FAKE_SUB), \
         patch('domains.upload.api.load_data.local_chunk_service.get_processed_result',
               return_value=csv), \
         patch('domains.upload.api.load_data.local_chunk_service.save_processed_result',
               return_value=True), \
         patch('domains.upload.api.load_data.local_chunk_service.delete_processed_result') as del_local, \
         patch('domains.upload.api.load_data.storage_service.delete_file') as del_remote:
        resp = client.post(
            '/api/loadRowData/merge-and-prepare',
            json={'fileIds': ['owner-user-uuid_a', 'owner-user-uuid_b'],
                  'fileName': 'out.csv'},
            headers={'Authorization': 'Bearer owner-token'},
        )
    assert resp.status_code == 200, resp.data
    body = json.loads(resp.data)
    assert body.get('downloadFileId'), body
    del_local.assert_not_called()
    del_remote.assert_not_called()
