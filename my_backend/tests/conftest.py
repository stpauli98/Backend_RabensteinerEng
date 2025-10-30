"""
Pytest configuration and fixtures for backend tests.

This module provides common fixtures used across all test modules.
"""

import pytest
import os
import sys
from io import BytesIO, StringIO
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


@pytest.fixture
def app():
    """
    Create Flask app for testing.
    
    Returns:
        Flask application instance configured for testing
    """
    from flask import Flask
    
    app = Flask(__name__)
    app.config['TESTING'] = True
    app.config['SECRET_KEY'] = 'test-secret-key'
    app.config['UPLOAD_EXPIRY_SECONDS'] = 1800
    
    # Mock extensions
    app.extensions = {'socketio': MagicMock()}
    
    return app


@pytest.fixture
def client(app):
    """
    Create test client.
    
    Args:
        app: Flask application fixture
        
    Returns:
        Flask test client
    """
    return app.test_client()


@pytest.fixture
def mock_socketio():
    """
    Mock SocketIO instance.
    
    Returns:
        Mocked SocketIO object
    """
    mock = MagicMock()
    mock.emit = MagicMock()
    return mock


@pytest.fixture
def mock_auth(monkeypatch):
    """
    Mock authentication middleware.
    
    Bypasses authentication for testing.
    """
    def mock_require_auth(f):
        """Passthrough decorator."""
        return f
    
    # This would need to be adjusted based on actual middleware location
    # monkeypatch.setattr('middleware.auth.require_auth', mock_require_auth)
    return mock_require_auth


@pytest.fixture
def mock_subscription(monkeypatch):
    """
    Mock subscription middleware.
    
    Bypasses subscription checks for testing.
    """
    def mock_require_subscription(f):
        return f
    
    def mock_check_processing_limit(f):
        return f
    
    return mock_require_subscription, mock_check_processing_limit


@pytest.fixture
def sample_csv_content():
    """
    Sample CSV data for testing.
    
    Returns:
        CSV string with datetime and value columns
    """
    return """datetime,value
2024-01-01 10:00:00,100.5
2024-01-01 11:00:00,101.2
2024-01-01 12:00:00,102.0
2024-01-01 13:00:00,103.5
2024-01-01 14:00:00,104.1"""


@pytest.fixture
def sample_csv_semicolon():
    """
    Sample CSV with semicolon delimiter.
    
    Returns:
        CSV string with semicolon delimiter
    """
    return """datetime;value
2024-01-01 10:00:00;100.5
2024-01-01 11:00:00;101.2"""


@pytest.fixture
def sample_csv_german_format():
    """
    Sample CSV with German date format.
    
    Returns:
        CSV string with dd.mm.YYYY HH:MM format
    """
    return """datum,wert
01.01.2024 10:00,100.5
01.01.2024 11:00,101.2
01.01.2024 12:00,102.0"""


@pytest.fixture
def sample_csv_separate_datetime():
    """
    Sample CSV with separate date and time columns.
    
    Returns:
        CSV string with date, time, and value columns
    """
    return """date,time,value
2024-01-01,10:00:00,100.5
2024-01-01,11:00:00,101.2
2024-01-01,12:00:00,102.0"""


@pytest.fixture
def sample_dataframe():
    """
    Sample pandas DataFrame for testing.
    
    Returns:
        DataFrame with datetime and value columns
    """
    return pd.DataFrame({
        'datetime': pd.to_datetime([
            '2024-01-01 10:00:00',
            '2024-01-01 11:00:00',
            '2024-01-01 12:00:00'
        ]),
        'value': [100.5, 101.2, 102.0]
    })


@pytest.fixture
def upload_chunk_data():
    """
    Sample data for chunk upload testing.
    
    Returns:
        Dictionary with upload chunk parameters
    """
    return {
        'uploadId': 'test-upload-123',
        'chunkIndex': '0',
        'totalChunks': '3',
        'delimiter': ',',
        'selected_columns': '{"column1": "datetime", "column2": "value"}',
        'timezone': 'UTC',
        'dropdown_count': '2',
        'hasHeader': 'ja',
        'fileChunk': (BytesIO(b'datetime,value\n2024-01-01 10:00:00,100.5'), 'test.csv')
    }


@pytest.fixture
def mock_flask_request():
    """
    Mock Flask request object.
    
    Returns:
        Mocked request object
    """
    mock = MagicMock()
    mock.form = {}
    mock.files = {}
    mock.json = {}
    return mock


@pytest.fixture
def mock_flask_g():
    """
    Mock Flask g object.
    
    Returns:
        Mocked g object with user_id
    """
    mock = MagicMock()
    mock.user_id = 'test-user-123'
    return mock


@pytest.fixture(autouse=True)
def reset_state():
    """
    Reset any global state before each test.
    
    This ensures tests are isolated and don't affect each other.
    """
    # This will be important when we have state managers
    yield
    # Cleanup after test


@pytest.fixture
def temp_csv_file(tmp_path):
    """
    Create temporary CSV file for testing.
    
    Args:
        tmp_path: pytest's temporary directory fixture
        
    Returns:
        Path to temporary CSV file
    """
    csv_file = tmp_path / "test.csv"
    csv_file.write_text("""datetime,value
2024-01-01 10:00:00,100.5
2024-01-01 11:00:00,101.2""")
    return csv_file


@pytest.fixture
def supported_date_formats():
    """
    List of supported date formats for testing.
    
    Returns:
        List of datetime format strings
    """
    return [
        '%Y-%m-%dT%H:%M:%S%z',
        '%Y-%m-%dT%H:%M%z',
        '%Y-%m-%dT%H:%M:%S',
        '%Y-%m-%d %H:%M:%S',
        '%d.%m.%Y %H:%M',
        '%Y-%m-%d %H:%M',
        '%d.%m.%Y %H:%M:%S',
        '%d.%m.%Y %H:%M:%S.%f',
        '%Y/%m/%d %H:%M:%S',
        '%d/%m/%Y %H:%M:%S',
        '%Y/%m/%d',
        '%d/%m/%Y',
        '%d-%m-%Y %H:%M:%S',
        '%d-%m-%Y %H:%M',
        '%Y/%m/%d %H:%M',
        '%d/%m/%Y %H:%M',
        '%d-%m-%Y',
        '%H:%M:%S',
        '%H:%M'
    ]
