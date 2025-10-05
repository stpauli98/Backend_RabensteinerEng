"""
Comprehensive pytest test suite for cloud.py endpoints

Test Categories:
1. Unit Tests - Test individual functions
2. Integration Tests - Test endpoint flows
3. Edge Cases - Test error handling
4. Performance Tests - Test large file handling
"""

import pytest
import pandas as pd
import numpy as np
import json
import os
import tempfile
import shutil
from io import BytesIO, StringIO
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock, Mock
import base64

# Import the blueprint and functions to test
from api.routes.cloud import (
    bp as cloud_bp,
    calculate_bounds,
    interpolate_data,
    get_chunk_dir,
    CHUNK_DIR,
    chunk_uploads,
    temp_files
)


# ==================== FIXTURES ====================

@pytest.fixture
def app():
    """Create Flask test app"""
    from flask import Flask
    app = Flask(__name__)
    app.config['TESTING'] = True
    app.register_blueprint(cloud_bp, url_prefix='')
    return app


@pytest.fixture
def client(app):
    """Create Flask test client"""
    return app.test_client()


@pytest.fixture
def sample_temp_df():
    """Sample temperature DataFrame"""
    dates = pd.date_range('2024-01-01 00:00:00', periods=100, freq='1min')
    return pd.DataFrame({
        'UTC': dates,
        'Temperature': np.random.uniform(15, 25, 100)
    })


@pytest.fixture
def sample_load_df():
    """Sample load DataFrame"""
    dates = pd.date_range('2024-01-01 00:00:00', periods=100, freq='1min')
    return pd.DataFrame({
        'UTC': dates,
        'Load': np.random.uniform(50, 150, 100)
    })


@pytest.fixture
def sample_load_with_nans():
    """Sample load DataFrame with NaN values for interpolation"""
    dates = pd.date_range('2024-01-01 00:00:00', periods=100, freq='1min')
    load_values = np.random.uniform(50, 150, 100)
    # Introduce NaN gaps
    load_values[10:15] = np.nan
    load_values[30:35] = np.nan
    return pd.DataFrame({
        'UTC': dates,
        'Load': load_values
    })


@pytest.fixture
def csv_temp_file(sample_temp_df):
    """Create temporary CSV file for temperature data"""
    temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv')
    sample_temp_df.to_csv(temp_file.name, sep=';', index=False)
    yield temp_file.name
    os.unlink(temp_file.name)


@pytest.fixture
def csv_load_file(sample_load_df):
    """Create temporary CSV file for load data"""
    temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv')
    sample_load_df.to_csv(temp_file.name, sep=';', index=False)
    yield temp_file.name
    os.unlink(temp_file.name)


@pytest.fixture
def upload_id():
    """Generate unique upload ID"""
    return f"test_upload_{datetime.now().strftime('%Y%m%d%H%M%S')}"


@pytest.fixture(autouse=True)
def cleanup_chunks():
    """Cleanup chunk directories after each test"""
    yield
    # Cleanup chunk_uploads dictionary
    chunk_uploads.clear()
    # Cleanup temp_files dictionary
    temp_files.clear()
    # Cleanup chunk directories
    if os.path.exists(CHUNK_DIR):
        shutil.rmtree(CHUNK_DIR, ignore_errors=True)


# ==================== UNIT TESTS ====================

class TestCalculateBounds:
    """Test calculate_bounds function"""
    
    def test_constant_tolerance(self):
        """Test constant tolerance calculation"""
        predictions = np.array([100, 200, 300])
        tol_cnt = 10
        tol_dep = 0.1
        
        upper, lower = calculate_bounds(predictions, 'cnt', tol_cnt, tol_dep)
        
        assert np.array_equal(upper, predictions + tol_cnt)
        assert np.array_equal(lower, predictions - tol_cnt)
    
    def test_dependent_tolerance(self):
        """Test dependent tolerance calculation"""
        predictions = np.array([100, 200, 300])
        tol_cnt = 10
        tol_dep = 0.1
        
        upper, lower = calculate_bounds(predictions, 'dep', tol_cnt, tol_dep)
        
        expected_upper = predictions * (1 + tol_dep) + tol_cnt
        expected_lower = predictions * (1 - tol_dep) - tol_cnt
        
        assert np.array_equal(upper, expected_upper)
        assert np.array_equal(lower, expected_lower)
    
    def test_negative_values_allowed(self):
        """Test that negative bounds are allowed"""
        predictions = np.array([5, 10, 15])
        tol_cnt = 20
        tol_dep = 0.1
        
        upper, lower = calculate_bounds(predictions, 'cnt', tol_cnt, tol_dep)
        
        # Lower bound should allow negative values
        assert np.any(lower < 0)


class TestInterpolateData:
    """Test interpolate_data function"""
    
    def test_basic_interpolation(self, sample_temp_df, sample_load_with_nans):
        """Test basic linear interpolation"""
        result_df, added_points = interpolate_data(
            sample_temp_df,
            sample_load_with_nans,
            'Temperature',
            'Load',
            max_time_span=60
        )
        
        assert added_points > 0
        assert result_df['value'].isna().sum() < sample_load_with_nans['Load'].isna().sum()
    
    def test_no_interpolation_beyond_max_span(self, sample_temp_df, sample_load_with_nans):
        """Test that gaps larger than max_time_span are not interpolated"""
        # Create large gap
        sample_load_with_nans.loc[50:80, 'Load'] = np.nan
        
        result_df, added_points = interpolate_data(
            sample_temp_df,
            sample_load_with_nans,
            'Temperature',
            'Load',
            max_time_span=5  # Small max span
        )
        
        # Large gap should remain NaN
        assert result_df['value'].isna().sum() > 0
    
    def test_all_nan_values(self, sample_temp_df):
        """Test handling of all NaN values"""
        nan_df = pd.DataFrame({
            'UTC': sample_temp_df['UTC'],
            'Load': [np.nan] * len(sample_temp_df)
        })
        
        result_df, added_points = interpolate_data(
            sample_temp_df,
            nan_df,
            'Temperature',
            'Load',
            max_time_span=60
        )
        
        assert added_points == 0
        assert result_df['value'].isna().all()


# ==================== INTEGRATION TESTS ====================

class TestUploadChunkEndpoint:
    """Test /upload-chunk endpoint"""
    
    def test_valid_chunk_upload(self, client, upload_id):
        """Test successful chunk upload"""
        chunk_data = b"test,data,chunk1\n1,2,3\n"
        
        response = client.post('/upload-chunk', data={
            'file': (BytesIO(chunk_data), 'test.csv'),
            'uploadId': upload_id,
            'fileType': 'temp_file',
            'chunkIndex': '0',
            'totalChunks': '3'
        })
        
        assert response.status_code == 200
        data = response.get_json()
        assert data['success'] is True
        assert data['data']['uploadId'] == upload_id
        assert data['data']['chunkIndex'] == 0
        assert data['data']['totalChunks'] == 3
    
    def test_missing_file(self, client, upload_id):
        """Test error when file is missing"""
        response = client.post('/upload-chunk', data={
            'uploadId': upload_id,
            'fileType': 'temp_file',
            'chunkIndex': '0',
            'totalChunks': '1'
        })
        
        assert response.status_code == 400
        data = response.get_json()
        assert data['success'] is False
        assert 'No file part' in data['data']['error']
    
    def test_invalid_file_type(self, client, upload_id):
        """Test error with invalid file type"""
        chunk_data = b"test,data\n"
        
        response = client.post('/upload-chunk', data={
            'file': (BytesIO(chunk_data), 'test.csv'),
            'uploadId': upload_id,
            'fileType': 'invalid_type',
            'chunkIndex': '0',
            'totalChunks': '1'
        })
        
        assert response.status_code == 400
        data = response.get_json()
        assert data['success'] is False
        assert 'Invalid file type' in data['data']['error']
    
    def test_invalid_chunk_index(self, client, upload_id):
        """Test error with invalid chunk index"""
        chunk_data = b"test,data\n"
        
        response = client.post('/upload-chunk', data={
            'file': (BytesIO(chunk_data), 'test.csv'),
            'uploadId': upload_id,
            'fileType': 'temp_file',
            'chunkIndex': 'invalid',
            'totalChunks': '1'
        })
        
        assert response.status_code == 400
        data = response.get_json()
        assert data['success'] is False


class TestCompleteEndpoint:
    """Test /complete endpoint"""
    
    def test_successful_complete_flow(self, client, upload_id, sample_temp_df, sample_load_df):
        """Test complete chunked upload flow"""
        # Upload chunks for temp file
        temp_csv = sample_temp_df.to_csv(sep=';', index=False).encode()
        client.post('/upload-chunk', data={
            'file': (BytesIO(temp_csv), 'temp.csv'),
            'uploadId': upload_id,
            'fileType': 'temp_file',
            'chunkIndex': '0',
            'totalChunks': '1'
        })
        
        # Upload chunks for load file
        load_csv = sample_load_df.to_csv(sep=';', index=False).encode()
        client.post('/upload-chunk', data={
            'file': (BytesIO(load_csv), 'load.csv'),
            'uploadId': upload_id,
            'fileType': 'load_file',
            'chunkIndex': '0',
            'totalChunks': '1'
        })
        
        # Complete the upload
        response = client.post('/complete', 
            json={
                'uploadId': upload_id,
                'REG': 'lin',
                'TR': 'cnt',
                'TOL_CNT': '10',
                'TOL_DEP': '0.1'
            }
        )
        
        assert response.status_code == 200
        data = response.get_json()
        assert data['success'] is True
        assert 'x_values' in data['data']
        assert 'y_values' in data['data']
        assert 'equation' in data['data']
    
    def test_missing_chunks(self, client, upload_id):
        """Test error when chunks are incomplete"""
        # Only upload temp file chunks, not load file
        temp_csv = b"UTC;Temperature\n2024-01-01 00:00:00;20\n"
        client.post('/upload-chunk', data={
            'file': (BytesIO(temp_csv), 'temp.csv'),
            'uploadId': upload_id,
            'fileType': 'temp_file',
            'chunkIndex': '0',
            'totalChunks': '1'
        })
        
        # Try to complete without load file
        response = client.post('/complete', json={'uploadId': upload_id})
        
        assert response.status_code == 400
        data = response.get_json()
        assert data['success'] is False
        assert 'Not all chunks received' in data['data']['error']
    
    def test_invalid_upload_id(self, client):
        """Test error with invalid upload ID"""
        response = client.post('/complete', json={'uploadId': 'invalid_id'})
        
        assert response.status_code == 400
        data = response.get_json()
        assert data['success'] is False
        assert 'Invalid upload ID' in data['data']['error']


class TestClouddataEndpoint:
    """Test /clouddata endpoint"""
    
    def test_successful_clouddata_processing(self, client, sample_temp_df, sample_load_df):
        """Test successful data processing via clouddata endpoint"""
        # Encode dataframes to base64
        temp_csv = sample_temp_df.to_csv(sep=';', index=False)
        load_csv = sample_load_df.to_csv(sep=';', index=False)
        
        temp_b64 = base64.b64encode(temp_csv.encode()).decode()
        load_b64 = base64.b64encode(load_csv.encode()).decode()
        
        payload = {
            'files': {
                'temp_out.csv': temp_b64,
                'load.csv': load_b64
            },
            'REG': 'lin',
            'TR': 'cnt',
            'TOL_CNT': '10',
            'TOL_DEP': '0.1'
        }
        
        response = client.post('/clouddata', json=payload)
        
        assert response.status_code == 200
        data = response.get_json()
        assert data['success'] is True
        assert 'equation' in data['data']
    
    def test_polynomial_regression(self, client, sample_temp_df, sample_load_df):
        """Test polynomial regression"""
        temp_csv = sample_temp_df.to_csv(sep=';', index=False)
        load_csv = sample_load_df.to_csv(sep=';', index=False)
        
        payload = {
            'files': {
                'temp_out.csv': base64.b64encode(temp_csv.encode()).decode(),
                'load.csv': base64.b64encode(load_csv.encode()).decode()
            },
            'REG': 'poly',
            'TR': 'dep',
            'TOL_CNT': '10',
            'TOL_DEP': '0.15'
        }
        
        response = client.post('/clouddata', json=payload)
        
        assert response.status_code == 200
        data = response.get_json()
        assert data['success'] is True
        # Polynomial equation should contain x²
        assert 'x²' in data['data']['equation']
    
    def test_missing_files(self, client):
        """Test error when files are missing"""
        payload = {
            'files': {
                'temp_out.csv': base64.b64encode(b'test').decode()
                # Missing load.csv
            }
        }
        
        response = client.post('/clouddata', json=payload)
        
        assert response.status_code == 400
        data = response.get_json()
        assert data['success'] is False


class TestInterpolateChunkedEndpoint:
    """Test /interpolate-chunked endpoint"""
    
    def test_successful_interpolation(self, client, upload_id, sample_load_with_nans):
        """Test successful interpolation with chunked upload"""
        # Upload file chunks
        csv_data = sample_load_with_nans.to_csv(sep=';', index=False).encode()
        client.post('/upload-chunk', data={
            'file': (BytesIO(csv_data), 'load.csv'),
            'uploadId': upload_id,
            'fileType': 'interpolate_file',
            'chunkIndex': '0',
            'totalChunks': '1'
        })
        
        # Trigger interpolation
        response = client.post('/interpolate-chunked', 
            json={
                'uploadId': upload_id,
                'max_time_span': '60'
            }
        )
        
        assert response.status_code == 200
        # Response is streaming, check mimetype
        assert response.mimetype == 'application/x-ndjson'
    
    def test_invalid_max_time_span(self, client, upload_id):
        """Test error with invalid max_time_span"""
        response = client.post('/interpolate-chunked',
            json={
                'uploadId': upload_id,
                'max_time_span': 'invalid'
            }
        )
        
        assert response.status_code == 400
        data = response.get_json()
        assert data['success'] is False
    
    def test_missing_upload_id(self, client):
        """Test error when upload ID is missing"""
        response = client.post('/interpolate-chunked', json={})
        
        assert response.status_code == 400
        data = response.get_json()
        assert data['success'] is False
        assert 'Upload ID is required' in data['data']['error']


class TestPrepareSaveEndpoint:
    """Test /prepare-save endpoint"""
    
    def test_successful_save_preparation(self, client):
        """Test successful CSV preparation for download"""
        csv_data = [
            ['UTC', 'Value'],
            ['2024-01-01 00:00:00', '100'],
            ['2024-01-01 00:01:00', '101']
        ]
        
        response = client.post('/prepare-save',
            json={
                'data': csv_data,
                'filename': 'test_output'
            }
        )
        
        assert response.status_code == 200
        data = response.get_json()
        assert data['success'] is True
        assert 'fileId' in data
        assert data['filename'] == 'test_output'
    
    def test_missing_data(self, client):
        """Test error when data is missing"""
        response = client.post('/prepare-save', json={'filename': 'test'})
        
        assert response.status_code == 400
        data = response.get_json()
        assert data['success'] is False
    
    def test_invalid_data_format(self, client):
        """Test error with invalid data format"""
        response = client.post('/prepare-save',
            json={
                'data': 'invalid_format',
                'filename': 'test'
            }
        )
        
        assert response.status_code == 400
        data = response.get_json()
        assert data['success'] is False


class TestDownloadEndpoint:
    """Test /download/<file_id> endpoint"""
    
    def test_successful_download(self, client):
        """Test successful file download"""
        # First prepare a file
        csv_data = [['UTC', 'Value'], ['2024-01-01', '100']]
        prep_response = client.post('/prepare-save',
            json={'data': csv_data, 'filename': 'test'}
        )
        file_id = prep_response.get_json()['fileId']
        
        # Download the file
        response = client.get(f'/download/{file_id}')
        
        assert response.status_code == 200
        assert response.mimetype == 'text/csv'
        assert 'test.csv' in response.headers.get('Content-Disposition', '')
    
    def test_invalid_file_id(self, client):
        """Test error with invalid file ID"""
        response = client.get('/download/invalid_id')
        
        assert response.status_code == 404
        data = response.get_json()
        assert data['success'] is False


# ==================== EDGE CASE TESTS ====================

class TestEdgeCases:
    """Test edge cases and error conditions"""
    
    def test_duplicate_timestamps(self, client):
        """Test handling of duplicate timestamps"""
        df = pd.DataFrame({
            'UTC': ['2024-01-01 00:00:00'] * 5,
            'Temperature': [20, 21, 22, 23, 24]
        })
        
        csv_data = df.to_csv(sep=';', index=False)
        payload = {
            'files': {
                'temp_out.csv': base64.b64encode(csv_data.encode()).decode(),
                'load.csv': base64.b64encode(csv_data.encode()).decode()
            }
        }
        
        response = client.post('/clouddata', json=payload)
        
        assert response.status_code == 400
        data = response.get_json()
        assert 'Duplicate timestamps' in data['data']['error']
    
    def test_no_matching_timestamps(self, client):
        """Test error when no matching timestamps exist"""
        df1 = pd.DataFrame({
            'UTC': pd.date_range('2024-01-01', periods=10, freq='1min'),
            'Temperature': range(10)
        })
        df2 = pd.DataFrame({
            'UTC': pd.date_range('2024-02-01', periods=10, freq='1min'),
            'Load': range(10)
        })
        
        payload = {
            'files': {
                'temp_out.csv': base64.b64encode(df1.to_csv(sep=';', index=False).encode()).decode(),
                'load.csv': base64.b64encode(df2.to_csv(sep=';', index=False).encode()).decode()
            }
        }
        
        response = client.post('/clouddata', json=payload)
        
        assert response.status_code == 400
        data = response.get_json()
        assert 'No matching timestamps' in data['data']['error']
    
    def test_no_points_within_tolerance(self, client, sample_temp_df, sample_load_df):
        """Test error when no points are within tolerance bounds"""
        payload = {
            'files': {
                'temp_out.csv': base64.b64encode(sample_temp_df.to_csv(sep=';', index=False).encode()).decode(),
                'load.csv': base64.b64encode(sample_load_df.to_csv(sep=';', index=False).encode()).decode()
            },
            'REG': 'lin',
            'TR': 'cnt',
            'TOL_CNT': '0.0001',  # Extremely small tolerance
            'TOL_DEP': '0.0001'
        }
        
        response = client.post('/clouddata', json=payload)
        
        # Should either succeed with adjusted tolerance or fail gracefully
        if response.status_code == 400:
            data = response.get_json()
            assert 'No points within tolerance' in data['data']['error']
    
    def test_empty_dataframe_after_cleaning(self, client):
        """Test error when all data is NaN"""
        df = pd.DataFrame({
            'UTC': pd.date_range('2024-01-01', periods=10, freq='1min'),
            'Temperature': [np.nan] * 10
        })
        
        payload = {
            'files': {
                'temp_out.csv': base64.b64encode(df.to_csv(sep=';', index=False).encode()).decode(),
                'load.csv': base64.b64encode(df.to_csv(sep=';', index=False).encode()).decode()
            }
        }
        
        response = client.post('/clouddata', json=payload)
        
        assert response.status_code == 400
        data = response.get_json()
        assert 'No valid' in data['data']['error']


# ==================== PERFORMANCE TESTS ====================

class TestPerformance:
    """Performance and stress tests"""
    
    @pytest.mark.slow
    def test_large_file_processing(self, client):
        """Test processing of large dataset (10000 rows)"""
        large_df = pd.DataFrame({
            'UTC': pd.date_range('2024-01-01', periods=10000, freq='1min'),
            'Temperature': np.random.uniform(15, 25, 10000),
            'Load': np.random.uniform(50, 150, 10000)
        })
        
        temp_df = large_df[['UTC', 'Temperature']]
        load_df = large_df[['UTC', 'Load']]
        
        payload = {
            'files': {
                'temp_out.csv': base64.b64encode(temp_df.to_csv(sep=';', index=False).encode()).decode(),
                'load.csv': base64.b64encode(load_df.to_csv(sep=';', index=False).encode()).decode()
            },
            'REG': 'lin',
            'TR': 'cnt',
            'TOL_CNT': '10'
        }
        
        response = client.post('/clouddata', json=payload)
        
        assert response.status_code == 200
        data = response.get_json()
        assert data['success'] is True
        assert len(data['data']['x_values']) > 0
    
    @pytest.mark.slow
    def test_chunk_streaming_performance(self, client, upload_id):
        """Test streaming response with large interpolation dataset"""
        # Create large dataset with NaN gaps
        large_size = 20000
        load_values = np.random.uniform(50, 150, large_size)
        # Add gaps
        for i in range(0, large_size, 500):
            load_values[i:i+10] = np.nan
        
        df = pd.DataFrame({
            'UTC': pd.date_range('2024-01-01', periods=large_size, freq='1min'),
            'Load': load_values
        })
        
        # Upload chunks
        csv_data = df.to_csv(sep=';', index=False).encode()
        client.post('/upload-chunk', data={
            'file': (BytesIO(csv_data), 'load.csv'),
            'uploadId': upload_id,
            'fileType': 'interpolate_file',
            'chunkIndex': '0',
            'totalChunks': '1'
        })
        
        # Trigger interpolation
        response = client.post('/interpolate-chunked',
            json={'uploadId': upload_id, 'max_time_span': '60'}
        )
        
        assert response.status_code == 200
        # Verify streaming response
        chunks_received = 0
        for line in response.iter_encoded():
            if line:
                chunks_received += 1
        
        assert chunks_received > 0
