"""
Tests for route handler endpoints in load_data.py

Tests cover:
- /upload-chunk: Chunked file upload handling
- /finalize-upload: Upload completion and processing
- /cancel-upload: Upload cancellation
- process_chunks(): Chunk processing pipeline
"""

import pytest
import json
import time
from unittest.mock import patch, MagicMock, mock_open
from io import BytesIO, StringIO
import pandas as pd

# Import Flask app and routes
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

# Middleware mocking is now handled by conftest.py
from api.routes.load_data import bp, chunk_storage, temp_files


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
    
    return app


@pytest.fixture
def client(app):
    """Create Flask test client."""
    return app.test_client()


@pytest.fixture
def mock_user():
    """Mock authenticated user."""
    return {
        'user_id': 'test-user-123',
        'email': 'test@example.com'
    }


@pytest.fixture
def auth_headers():
    """Mock auth headers for requests."""
    return {
        'Authorization': 'Bearer test-token-123',
        'Content-Type': 'application/json'
    }


@pytest.fixture(autouse=True)
def cleanup_storage():
    """Clean up chunk_storage and temp_files after each test."""
    yield
    chunk_storage.clear()
    temp_files.clear()


class TestUploadChunkEndpoint:
    """Test /upload-chunk endpoint."""
    
    def test_upload_chunk_missing_file(self, client):
        """Should return error when file chunk is missing."""
        response = client.post('/api/upload-chunk', data={
            'uploadId': 'test-upload-1',
            'chunkIndex': '0',
            'totalChunks': '3',
            'delimiter': ',',
            'selected_columns': '{}',
            'timezone': 'UTC',
            'dropdown_count': '2',
            'hasHeader': 'ja'
        })
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data
        assert 'Chunk file not found' in data['error']
    
    def test_upload_chunk_missing_required_params(self, client):
        """Should return error when required parameters are missing."""
        response = client.post('/api/upload-chunk', data={
            'fileChunk': (BytesIO(b'test data'), 'test.csv'),
            'uploadId': 'test-upload-1'
            # Missing: chunkIndex, totalChunks, delimiter, etc.
        })
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data
        assert 'Missing required parameters' in data['error']
    
    @patch('api.routes.load_data.get_socketio')
    def test_upload_chunk_first_chunk_creates_storage(self, mock_socketio, client):
        """Should create storage entry for first chunk."""
        mock_socketio.return_value = MagicMock()
        
        selected_columns = json.dumps({'column1': 'datetime', 'column2': 'value'})
        
        response = client.post('/api/upload-chunk', data={
            'fileChunk': (BytesIO(b'chunk data 1'), 'test.csv'),
            'uploadId': 'test-upload-1',
            'chunkIndex': '0',
            'totalChunks': '3',
            'delimiter': ',',
            'selected_columns': selected_columns,
            'timezone': 'UTC',
            'dropdown_count': '2',
            'hasHeader': 'ja'
        })
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'uploadId' in data
        assert data['uploadId'] == 'test-upload-1'
        assert 'test-upload-1' in chunk_storage
        assert chunk_storage['test-upload-1']['total_chunks'] == 3
        assert chunk_storage['test-upload-1']['received_chunks'] == 1
    
    @patch('api.routes.load_data.get_socketio')
    def test_upload_chunk_subsequent_chunks_update_storage(self, mock_socketio, client):
        """Should update storage for subsequent chunks."""
        mock_socketio.return_value = MagicMock()
        
        selected_columns = json.dumps({'column1': 'datetime', 'column2': 'value'})
        
        # Upload first chunk
        client.post('/api/upload-chunk', data={
            'fileChunk': (BytesIO(b'chunk 1'), 'test.csv'),
            'uploadId': 'test-upload-2',
            'chunkIndex': '0',
            'totalChunks': '3',
            'delimiter': ',',
            'selected_columns': selected_columns,
            'timezone': 'UTC',
            'dropdown_count': '2',
            'hasHeader': 'ja'
        })
        
        # Upload second chunk
        response = client.post('/api/upload-chunk', data={
            'fileChunk': (BytesIO(b'chunk 2'), 'test.csv'),
            'uploadId': 'test-upload-2',
            'chunkIndex': '1',
            'totalChunks': '3',
            'delimiter': ',',
            'selected_columns': selected_columns,
            'timezone': 'UTC',
            'dropdown_count': '2',
            'hasHeader': 'ja'
        })
        
        assert response.status_code == 200
        assert chunk_storage['test-upload-2']['received_chunks'] == 2
        assert 0 in chunk_storage['test-upload-2']['chunks']
        assert 1 in chunk_storage['test-upload-2']['chunks']
    
    @patch('api.routes.load_data.get_socketio')
    def test_upload_chunk_progress_tracking(self, mock_socketio, client):
        """Should emit progress updates via socketio."""
        mock_socket = MagicMock()
        mock_socketio.return_value = mock_socket
        
        selected_columns = json.dumps({'column1': 'datetime', 'column2': 'value'})
        
        client.post('/api/upload-chunk', data={
            'fileChunk': (BytesIO(b'chunk 1'), 'test.csv'),
            'uploadId': 'test-upload-3',
            'chunkIndex': '0',
            'totalChunks': '2',
            'delimiter': ',',
            'selected_columns': selected_columns,
            'timezone': 'UTC',
            'dropdown_count': '2',
            'hasHeader': 'ja'
        })
        
        # Verify socketio.emit was called
        assert mock_socket.emit.called
        call_args = mock_socket.emit.call_args
        assert call_args[0][0] == 'upload_progress'
        assert 'progress' in call_args[0][1]
    
    @patch('api.routes.load_data.get_socketio')
    def test_upload_chunk_all_chunks_received_message(self, mock_socketio, client):
        """Should emit completion message when all chunks received."""
        mock_socket = MagicMock()
        mock_socketio.return_value = mock_socket
        
        selected_columns = json.dumps({'column1': 'datetime', 'column2': 'value'})
        
        # Upload first chunk
        client.post('/api/upload-chunk', data={
            'fileChunk': (BytesIO(b'chunk 1'), 'test.csv'),
            'uploadId': 'test-upload-4',
            'chunkIndex': '0',
            'totalChunks': '2',
            'delimiter': ',',
            'selected_columns': selected_columns,
            'timezone': 'UTC',
            'dropdown_count': '2',
            'hasHeader': 'ja'
        })
        
        # Upload second chunk (final)
        client.post('/api/upload-chunk', data={
            'fileChunk': (BytesIO(b'chunk 2'), 'test.csv'),
            'uploadId': 'test-upload-4',
            'chunkIndex': '1',
            'totalChunks': '2',
            'delimiter': ',',
            'selected_columns': selected_columns,
            'timezone': 'UTC',
            'dropdown_count': '2',
            'hasHeader': 'ja'
        })
        
        # Find the emit call with progress=100 and waiting message
        emit_calls = mock_socket.emit.call_args_list
        completion_calls = [
            call for call in emit_calls 
            if call[0][1].get('progress') == 100 and 'waiting' in call[0][1].get('message', '').lower()
        ]
        assert len(completion_calls) > 0, f"Expected completion call not found. Calls: {emit_calls}"


class TestFinalizeUploadEndpoint:
    """Test /finalize-upload endpoint."""
    
    def test_finalize_upload_missing_upload_id(self, client):
        """Should return error when uploadId is missing."""
        response = client.post('/api/finalize-upload',
                              data=json.dumps({}),
                              content_type='application/json')
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data
        assert 'uploadId is required' in data['error']
    
    def test_finalize_upload_not_found(self, client):
        """Should return 404 when upload not found."""
        response = client.post('/api/finalize-upload',
                              data=json.dumps({'uploadId': 'non-existent'}),
                              content_type='application/json')
        
        assert response.status_code == 404
        data = json.loads(response.data)
        assert 'error' in data
        assert 'Upload not found' in data['error']
    
    def test_finalize_upload_missing_chunks(self, client):
        """Should return error when not all chunks received."""
        # Create incomplete upload
        chunk_storage['test-upload-5'] = {
            'chunks': {0: b'chunk1'},
            'total_chunks': 3,
            'received_chunks': 1,
            'last_activity': time.time(),
            'parameters': {}
        }
        
        response = client.post('/api/finalize-upload',
                              data=json.dumps({'uploadId': 'test-upload-5'}),
                              content_type='application/json')
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data
        assert 'Not all chunks received' in data['error']
    
    @patch('api.routes.load_data.process_chunks')
    def test_finalize_upload_success(self, mock_process_chunks, client):
        """Should call process_chunks when all chunks received."""
        mock_process_chunks.return_value = (json.dumps({'data': []}), 200)
        
        # Create complete upload
        chunk_storage['test-upload-6'] = {
            'chunks': {0: b'chunk1', 1: b'chunk2'},
            'total_chunks': 2,
            'received_chunks': 2,
            'last_activity': time.time(),
            'parameters': {}
        }
        
        response = client.post('/api/finalize-upload',
                              data=json.dumps({'uploadId': 'test-upload-6'}),
                              content_type='application/json')
        
        assert mock_process_chunks.called
        assert mock_process_chunks.call_args[0][0] == 'test-upload-6'


class TestCancelUploadEndpoint:
    """Test /cancel-upload endpoint."""
    
    def test_cancel_upload_missing_upload_id(self, client):
        """Should return error when uploadId is missing."""
        response = client.post('/api/cancel-upload',
                              data=json.dumps({}),
                              content_type='application/json')
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data
        assert 'uploadId is required' in data['error']
    
    @patch('api.routes.load_data.get_socketio')
    def test_cancel_upload_removes_storage(self, mock_socketio, client):
        """Should remove upload from storage."""
        mock_socketio.return_value = MagicMock()
        
        # Create upload
        chunk_storage['test-upload-7'] = {
            'chunks': {0: b'chunk1'},
            'total_chunks': 2,
            'received_chunks': 1,
            'last_activity': time.time(),
            'parameters': {}
        }
        
        response = client.post('/api/cancel-upload',
                              data=json.dumps({'uploadId': 'test-upload-7'}),
                              content_type='application/json')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['success'] is True
        assert 'test-upload-7' not in chunk_storage
    
    @patch('api.routes.load_data.get_socketio')
    def test_cancel_upload_emits_socketio_event(self, mock_socketio, client):
        """Should emit cancellation event via socketio."""
        mock_socket = MagicMock()
        mock_socketio.return_value = mock_socket
        
        chunk_storage['test-upload-8'] = {
            'chunks': {},
            'total_chunks': 2,
            'received_chunks': 0,
            'last_activity': time.time(),
            'parameters': {}
        }
        
        response = client.post('/api/cancel-upload',
                              data=json.dumps({'uploadId': 'test-upload-8'}),
                              content_type='application/json')
        
        assert response.status_code == 200
        assert mock_socket.emit.called
        call_args = mock_socket.emit.call_args
        assert call_args[0][0] == 'upload_progress'
        assert 'canceled' in call_args[0][1]['message'].lower()
    
    @patch('api.routes.load_data.get_socketio')
    def test_cancel_upload_non_existent(self, mock_socketio, client):
        """Should succeed even when upload doesn't exist."""
        mock_socketio.return_value = MagicMock()
        
        response = client.post('/api/cancel-upload',
                              data=json.dumps({'uploadId': 'non-existent'}),
                              content_type='application/json')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['success'] is True


class TestProcessChunksFunction:
    """Test process_chunks() helper function."""
    
    @patch('api.routes.load_data.get_socketio')
    @patch('api.routes.load_data.upload_files')
    def test_process_chunks_combines_chunks(self, mock_upload_files, mock_socketio):
        """Should combine chunks in correct order."""
        from api.routes.load_data import process_chunks
        
        mock_socketio.return_value = MagicMock()
        mock_upload_files.return_value = (json.dumps({'data': []}), 200)
        
        # Create upload with 3 chunks
        chunk_storage['test-process-1'] = {
            'chunks': {
                0: b'datetime,value\n',
                1: b'2024-01-01,100\n',
                2: b'2024-01-02,200\n'
            },
            'total_chunks': 3,
            'received_chunks': 3,
            'last_activity': time.time(),
            'parameters': {
                'delimiter': ',',
                'timezone': 'UTC',
                'has_header': 'ja',
                'selected_columns': {'column1': 'datetime', 'column2': 'value'},
                'dropdown_count': 2
            }
        }
        
        process_chunks('test-process-1')
        
        # Verify upload_files was called
        assert mock_upload_files.called
        
        # Verify chunks were combined
        call_args = mock_upload_files.call_args[0]
        combined_content = call_args[0]
        assert 'datetime,value' in combined_content
        assert '2024-01-01,100' in combined_content
        assert '2024-01-02,200' in combined_content
    
    @patch('api.routes.load_data.get_socketio')
    def test_process_chunks_encoding_fallback(self, mock_socketio):
        """Should try multiple encodings if UTF-8 fails."""
        from api.routes.load_data import process_chunks
        
        mock_socketio.return_value = MagicMock()
        
        # Create upload with latin1 encoded data
        latin1_data = 'Ä Ö Ü ß'.encode('latin1')
        
        chunk_storage['test-process-2'] = {
            'chunks': {0: latin1_data},
            'total_chunks': 1,
            'received_chunks': 1,
            'last_activity': time.time(),
            'parameters': {
                'delimiter': ',',
                'timezone': 'UTC',
                'has_header': 'ja',
                'selected_columns': {'column1': 'datetime', 'column2': 'value'},
                'dropdown_count': 2
            }
        }
        
        with patch('api.routes.load_data.upload_files') as mock_upload:
            mock_upload.return_value = (json.dumps({'data': []}), 200)
            process_chunks('test-process-2')
            
            # Verify that upload_files was called (encoding succeeded)
            assert mock_upload.called
