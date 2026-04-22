"""End-to-end test: upload_files() with duplicate-named columns."""
import pytest
from unittest.mock import patch, MagicMock
from flask import Flask

from domains.upload.api.load_data import upload_files


DUP_CSV = (
    "Vrijeme,Temp,Temp\n"
    "2024-01-15 10:00:00,21.5,22.1\n"
    "2024-01-15 11:00:00,22.3,23.0\n"
    "2024-01-15 12:00:00,23.1,24.2\n"
)


@pytest.fixture
def app():
    app = Flask(__name__)
    app.extensions['socketio'] = MagicMock()
    return app


def test_duplicate_columns_use_selected_position(app):
    """Selecting index 2 should yield the SECOND Temp column's values."""
    params = {
        'uploadId': 'test-dup-upload',
        'delimiter': ',',
        'timezone': 'UTC',
        'selected_columns': {'column1': '0', 'column2': '2'},
        'custom_date_format': None,
        'value_column_name': 'Temp_B',
        'dropdown_count': 2,
        'has_header': 'ja',
    }

    with app.test_request_context():
        with patch('domains.upload.api.load_data.local_chunk_service') as mock_svc, \
             patch('domains.upload.api.load_data.increment_processing_count'), \
             patch('domains.upload.api.load_data.update_storage_usage'), \
             patch('domains.upload.api.load_data.log_compute_duration'), \
             patch('domains.upload.api.load_data.g') as mock_g:
            mock_g.user_id = 'test-user'
            mock_svc.save_processed_result.return_value = True

            response, status = upload_files(DUP_CSV, params)
            assert status == 200
            body = response.get_json()
            assert body['success'] is True
            assert 'Temp_B' in body['headers']
            # Preview includes header row + data rows
            data_rows = body['preview'][1:]
            values = [row[1] for row in data_rows]
            # Second Temp column values (22.1, 23.0, 24.2) — NOT first (21.5, 22.3, 23.1)
            assert '22.1' in values
            assert '23.0' in values
            assert '24.2' in values
            assert '21.5' not in values


def test_first_temp_when_index_is_1(app):
    """Selecting index 1 should yield the FIRST Temp column's values."""
    params = {
        'uploadId': 'test-dup-upload-2',
        'delimiter': ',',
        'timezone': 'UTC',
        'selected_columns': {'column1': '0', 'column2': '1'},
        'custom_date_format': None,
        'value_column_name': 'Temp_A',
        'dropdown_count': 2,
        'has_header': 'ja',
    }

    with app.test_request_context():
        with patch('domains.upload.api.load_data.local_chunk_service') as mock_svc, \
             patch('domains.upload.api.load_data.increment_processing_count'), \
             patch('domains.upload.api.load_data.update_storage_usage'), \
             patch('domains.upload.api.load_data.log_compute_duration'), \
             patch('domains.upload.api.load_data.g') as mock_g:
            mock_g.user_id = 'test-user'
            mock_svc.save_processed_result.return_value = True

            response, status = upload_files(DUP_CSV, params)
            assert status == 200
            body = response.get_json()
            values = [row[1] for row in body['preview'][1:]]
            assert '21.5' in values
            assert '22.3' in values
            assert '23.1' in values
            assert '22.1' not in values
