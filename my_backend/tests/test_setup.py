"""
Test setup validation.

This module contains basic tests to verify that the test framework
is properly configured and working.
"""

import pytest
import os
import pandas as pd
from pathlib import Path


class TestFrameworkSetup:
    """Test that pytest framework is properly configured."""
    
    def test_pytest_working(self):
        """Verify pytest is working."""
        assert True
    
    def test_imports(self):
        """Verify required packages can be imported."""
        import pandas
        import flask
        assert pandas is not None
        assert flask is not None
    
    def test_fixtures_directory_exists(self):
        """Verify fixtures directory exists."""
        fixtures_dir = Path(__file__).parent / 'fixtures'
        assert fixtures_dir.exists()
        assert fixtures_dir.is_dir()
    
    def test_sample_data_files_exist(self):
        """Verify sample data files exist."""
        fixtures_dir = Path(__file__).parent / 'fixtures'
        
        required_files = [
            'sample_data.csv',
            'sample_data_german.csv',
            'sample_data_semicolon.csv',
            'sample_data_separate_datetime.csv'
        ]
        
        for filename in required_files:
            file_path = fixtures_dir / filename
            assert file_path.exists(), f"Missing file: {filename}"


class TestFixtures:
    """Test that fixtures are working correctly."""
    
    def test_app_fixture(self, app):
        """Test Flask app fixture."""
        assert app is not None
        assert app.config['TESTING'] is True
    
    def test_client_fixture(self, client):
        """Test Flask test client fixture."""
        assert client is not None
    
    def test_mock_socketio_fixture(self, mock_socketio):
        """Test SocketIO mock fixture."""
        assert mock_socketio is not None
        assert hasattr(mock_socketio, 'emit')
    
    def test_sample_csv_content_fixture(self, sample_csv_content):
        """Test sample CSV content fixture."""
        assert sample_csv_content is not None
        assert 'datetime,value' in sample_csv_content
        assert '2024-01-01' in sample_csv_content
    
    def test_sample_dataframe_fixture(self, sample_dataframe):
        """Test sample DataFrame fixture."""
        assert sample_dataframe is not None
        assert isinstance(sample_dataframe, pd.DataFrame)
        assert 'datetime' in sample_dataframe.columns
        assert 'value' in sample_dataframe.columns
        assert len(sample_dataframe) > 0
    
    def test_upload_chunk_data_fixture(self, upload_chunk_data):
        """Test upload chunk data fixture."""
        assert upload_chunk_data is not None
        assert 'uploadId' in upload_chunk_data
        assert 'chunkIndex' in upload_chunk_data
        assert 'totalChunks' in upload_chunk_data
    
    def test_supported_date_formats_fixture(self, supported_date_formats):
        """Test supported date formats fixture."""
        assert supported_date_formats is not None
        assert isinstance(supported_date_formats, list)
        assert len(supported_date_formats) == 19
        assert '%Y-%m-%d %H:%M:%S' in supported_date_formats
        assert '%d.%m.%Y %H:%M' in supported_date_formats


class TestSampleData:
    """Test that sample data files are valid."""
    
    def test_read_sample_csv(self):
        """Test reading sample CSV file."""
        fixtures_dir = Path(__file__).parent / 'fixtures'
        csv_file = fixtures_dir / 'sample_data.csv'
        
        df = pd.read_csv(csv_file)
        assert df is not None
        assert len(df) > 0
        assert 'datetime' in df.columns
        assert 'value' in df.columns
    
    def test_read_german_format_csv(self):
        """Test reading German format CSV."""
        fixtures_dir = Path(__file__).parent / 'fixtures'
        csv_file = fixtures_dir / 'sample_data_german.csv'
        
        df = pd.read_csv(csv_file)
        assert df is not None
        assert 'datum' in df.columns
        assert 'wert' in df.columns
    
    def test_read_semicolon_csv(self):
        """Test reading semicolon-delimited CSV."""
        fixtures_dir = Path(__file__).parent / 'fixtures'
        csv_file = fixtures_dir / 'sample_data_semicolon.csv'
        
        df = pd.read_csv(csv_file, delimiter=';')
        assert df is not None
        assert 'datetime' in df.columns
        assert 'value' in df.columns
    
    def test_read_separate_datetime_csv(self):
        """Test reading CSV with separate date/time columns."""
        fixtures_dir = Path(__file__).parent / 'fixtures'
        csv_file = fixtures_dir / 'sample_data_separate_datetime.csv'
        
        df = pd.read_csv(csv_file)
        assert df is not None
        assert 'date' in df.columns
        assert 'time' in df.columns
        assert 'value' in df.columns


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
