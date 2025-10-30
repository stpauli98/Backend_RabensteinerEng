"""
Integration Tests for load_data.py

Tests cover:
- /prepare-save: Prepare processed data for download
- /download/<file_id>: Download processed CSV files
- End-to-end upload flows with real data processing
"""

import pytest
import json
import time
import os
import tempfile
from unittest.mock import patch, MagicMock
from io import BytesIO, StringIO
import pandas as pd

# Import Flask app and routes
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

# Middleware mocking is now handled by conftest.py
from api.routes.load_data import bp, temp_files, chunk_storage
from flask import g


@pytest.fixture
def app():
    """Create Flask test app."""
    from flask import Flask
    app = Flask(__name__)
    app.config['TESTING'] = True
    app.config['SECRET_KEY'] = 'test-secret-key'
    
    # Mock socketio extension
    app.extensions = {
        'socketio': MagicMock()
    }
    
    # Register blueprint
    app.register_blueprint(bp, url_prefix='/api')
    
    # Mock g.user_id for usage tracking
    @app.before_request
    def set_user_id():
        g.user_id = 'test-user-123'
    
    return app


@pytest.fixture
def client(app):
    """Create Flask test client."""
    return app.test_client()


@pytest.fixture(autouse=True)
def cleanup_storage():
    """Clean up storage after each test."""
    yield
    temp_files.clear()
    chunk_storage.clear()


class TestPrepareSaveEndpoint:
    """Test /prepare-save endpoint."""
    
    def test_prepare_save_missing_data(self, client):
        """Should return error when data is missing."""
        response = client.post('/api/prepare-save',
                              data=json.dumps({}),
                              content_type='application/json')
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data
        assert 'No data received' in data['error']
    
    def test_prepare_save_empty_data(self, client):
        """Should return error when data array is empty."""
        response = client.post('/api/prepare-save',
                              data=json.dumps({'data': {'data': []}}),
                              content_type='application/json')
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data
        assert 'Empty data' in data['error']
    
    def test_prepare_save_success(self, client):
        """Should create temp file and return file ID."""
        test_data = {
            'data': {
                'data': [
                    ['UTC', 'value'],
                    ['2024-01-01 10:00:00', '100'],
                    ['2024-01-01 11:00:00', '200']
                ],
                'fileName': 'test.csv'
            }
        }
        
        response = client.post('/api/prepare-save',
                              data=json.dumps(test_data),
                              content_type='application/json')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'fileId' in data
        assert data['fileId'] in temp_files
        assert os.path.exists(temp_files[data['fileId']]['path'])
    
    def test_prepare_save_creates_valid_csv(self, client):
        """Should create valid CSV file with correct data."""
        test_data = {
            'data': {
                'data': [
                    ['UTC', 'value'],
                    ['2024-01-01 10:00:00', '100'],
                    ['2024-01-01 11:00:00', '200']
                ],
                'fileName': 'test.csv'
            }
        }
        
        response = client.post('/api/prepare-save',
                              data=json.dumps(test_data),
                              content_type='application/json')
        
        data = json.loads(response.data)
        file_id = data['fileId']
        file_path = temp_files[file_id]['path']
        
        # Read and verify CSV content
        with open(file_path, 'r') as f:
            content = f.read()
            assert 'UTC;value' in content
            assert '2024-01-01 10:00:00;100' in content
            assert '2024-01-01 11:00:00;200' in content


class TestDownloadEndpoint:
    """Test /download/<file_id> endpoint."""
    
    def test_download_file_not_found(self, client):
        """Should return 404 when file ID doesn't exist."""
        response = client.get('/api/download/non-existent-id')
        
        assert response.status_code == 404
        data = json.loads(response.data)
        assert 'error' in data
        assert 'File not found' in data['error']
    
    def test_download_file_success(self, client):
        """Should download file with correct content."""
        # First prepare a file
        test_data = {
            'data': {
                'data': [
                    ['UTC', 'value'],
                    ['2024-01-01 10:00:00', '100']
                ],
                'fileName': 'download_test.csv'
            }
        }
        
        prep_response = client.post('/api/prepare-save',
                                   data=json.dumps(test_data),
                                   content_type='application/json')
        
        prep_data = json.loads(prep_response.data)
        file_id = prep_data['fileId']
        
        # Download the file
        download_response = client.get(f'/api/download/{file_id}')
        
        assert download_response.status_code == 200
        assert download_response.mimetype == 'text/csv'
        
        # Verify content
        content = download_response.data.decode('utf-8')
        assert 'UTC;value' in content
        assert '2024-01-01 10:00:00;100' in content
    
    def test_download_cleans_up_temp_file(self, client):
        """Should delete temp file after download."""
        # Prepare file
        test_data = {
            'data': {
                'data': [['UTC', 'value'], ['2024-01-01 10:00:00', '100']],
                'fileName': 'cleanup_test.csv'
            }
        }
        
        prep_response = client.post('/api/prepare-save',
                                   data=json.dumps(test_data),
                                   content_type='application/json')
        
        file_id = json.loads(prep_response.data)['fileId']
        file_path = temp_files[file_id]['path']
        
        # Download
        client.get(f'/api/download/{file_id}')
        
        # Verify cleanup
        assert file_id not in temp_files
        assert not os.path.exists(file_path)


class TestEndToEndUploadFlow:
    """Test complete upload workflow."""
    
    @patch('api.routes.load_data.get_socketio')
    def test_full_upload_workflow(self, mock_socketio, client):
        """Test complete upload: chunk → finalize → prepare → download."""
        mock_socketio.return_value = MagicMock()
        
        # Sample CSV data
        csv_content = """datetime;value
01.01.2024 10:00:00;100.5
01.01.2024 11:00:00;101.2
02.01.2024 12:00:00;102.8"""
        
        selected_columns = json.dumps({'column1': 'datetime', 'column2': 'value'})
        
        # Step 1: Upload chunks
        chunks = [csv_content[i:i+50].encode('utf-8') for i in range(0, len(csv_content), 50)]
        total_chunks = len(chunks)
        
        for idx, chunk in enumerate(chunks):
            response = client.post('/api/upload-chunk', data={
                'fileChunk': (BytesIO(chunk), 'test.csv'),
                'uploadId': 'e2e-test-1',
                'chunkIndex': str(idx),
                'totalChunks': str(total_chunks),
                'delimiter': ';',
                'selected_columns': selected_columns,
                'timezone': 'UTC',
                'dropdown_count': '2',
                'hasHeader': 'ja'
            })
            assert response.status_code == 200
        
        # Step 2: Finalize upload
        finalize_response = client.post('/api/finalize-upload',
                                        data=json.dumps({'uploadId': 'e2e-test-1'}),
                                        content_type='application/json')
        
        assert finalize_response.status_code == 200
        finalize_data = json.loads(finalize_response.data)
        assert 'data' in finalize_data
        assert len(finalize_data['data']) > 0
        
        # Step 3: Prepare for download
        prepare_response = client.post('/api/prepare-save',
                                      data=json.dumps({'data': finalize_data}),
                                      content_type='application/json')
        
        assert prepare_response.status_code == 200
        prepare_data = json.loads(prepare_response.data)
        assert 'fileId' in prepare_data
        
        # Step 4: Download
        download_response = client.get(f'/api/download/{prepare_data["fileId"]}')
        
        assert download_response.status_code == 200
        content = download_response.data.decode('utf-8')
        assert 'UTC' in content
        assert '100.5' in content or '100,5' in content  # Handle locale differences
    
    @patch('api.routes.load_data.get_socketio')
    def test_upload_with_separate_date_time_columns(self, mock_socketio, client):
        """Test upload with separate date and time columns."""
        mock_socketio.return_value = MagicMock()
        
        csv_content = """date;time;value
01.01.2024;10:00:00;100.5
02.01.2024;11:00:00;101.2"""
        
        selected_columns = json.dumps({
            'column1': 'date',
            'column2': 'time',
            'column3': 'value'
        })
        
        # Upload chunk
        response = client.post('/api/upload-chunk', data={
            'fileChunk': (BytesIO(csv_content.encode('utf-8')), 'test.csv'),
            'uploadId': 'e2e-test-2',
            'chunkIndex': '0',
            'totalChunks': '1',
            'delimiter': ';',
            'selected_columns': selected_columns,
            'timezone': 'Europe/Vienna',
            'dropdown_count': '3',
            'hasHeader': 'ja'
        })
        assert response.status_code == 200
        
        # Finalize
        finalize_response = client.post('/api/finalize-upload',
                                        data=json.dumps({'uploadId': 'e2e-test-2'}),
                                        content_type='application/json')
        
        assert finalize_response.status_code == 200
        data = json.loads(finalize_response.data)
        assert 'data' in data
        assert len(data['data']) > 0
    
    @patch('api.routes.load_data.get_socketio')
    def test_upload_cancel_workflow(self, mock_socketio, client):
        """Test upload cancellation flow."""
        mock_socketio.return_value = MagicMock()
        
        csv_content = "datetime;value\n01.01.2024 10:00:00;100"
        selected_columns = json.dumps({'column1': 'datetime', 'column2': 'value'})
        
        # Upload first chunk
        response = client.post('/api/upload-chunk', data={
            'fileChunk': (BytesIO(csv_content.encode('utf-8')), 'test.csv'),
            'uploadId': 'e2e-test-3',
            'chunkIndex': '0',
            'totalChunks': '2',
            'delimiter': ';',
            'selected_columns': selected_columns,
            'timezone': 'UTC',
            'dropdown_count': '2',
            'hasHeader': 'ja'
        })
        assert response.status_code == 200
        assert 'e2e-test-3' in chunk_storage
        
        # Cancel upload
        cancel_response = client.post('/api/cancel-upload',
                                     data=json.dumps({'uploadId': 'e2e-test-3'}),
                                     content_type='application/json')
        
        assert cancel_response.status_code == 200
        assert 'e2e-test-3' not in chunk_storage
