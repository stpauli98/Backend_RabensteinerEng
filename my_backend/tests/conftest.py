"""
Pytest configuration and shared fixtures for cloud.py tests

This file provides:
- Shared fixtures for all test files
- Mock configurations
- Test data generators
- Cleanup utilities
"""

import pytest
import os
import sys
import tempfile
import shutil
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# ==================== SESSION FIXTURES ====================

@pytest.fixture(scope='session')
def test_data_dir():
    """Create temporary directory for test data files"""
    temp_dir = tempfile.mkdtemp(prefix='cloud_test_data_')
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture(scope='session')
def chunk_storage_dir():
    """Create temporary directory for chunk storage during tests"""
    temp_dir = tempfile.mkdtemp(prefix='cloud_chunks_')
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


# ==================== MODULE FIXTURES ====================

@pytest.fixture(scope='module')
def mock_chunk_dir(chunk_storage_dir):
    """Mock CHUNK_DIR to use test directory"""
    from api.routes import cloud
    original_chunk_dir = cloud.CHUNK_DIR
    cloud.CHUNK_DIR = chunk_storage_dir
    yield chunk_storage_dir
    cloud.CHUNK_DIR = original_chunk_dir


# ==================== FUNCTION FIXTURES ====================

@pytest.fixture
def clean_environment():
    """Ensure clean test environment before and after each test"""
    # Clean before test
    from api.routes.cloud import chunk_uploads, temp_files
    chunk_uploads.clear()
    temp_files.clear()
    
    yield
    
    # Clean after test
    chunk_uploads.clear()
    temp_files.clear()


@pytest.fixture
def mock_logger():
    """Mock logger to suppress log output during tests"""
    import logging
    from unittest.mock import MagicMock
    
    logger = logging.getLogger('api.routes.cloud')
    original_handlers = logger.handlers[:]
    logger.handlers = [MagicMock()]
    
    yield logger
    
    logger.handlers = original_handlers


# ==================== HELPER FUNCTIONS ====================

def create_csv_file(data_dict, filename, sep=';'):
    """Helper function to create CSV file from dictionary
    
    Args:
        data_dict: Dictionary with column names as keys and lists as values
        filename: Path to save the CSV file
        sep: CSV separator (default ';')
    
    Returns:
        Path to created file
    """
    import pandas as pd
    df = pd.DataFrame(data_dict)
    df.to_csv(filename, sep=sep, index=False)
    return filename


def create_chunked_file(file_path, chunk_size=5*1024*1024):
    """Helper function to split file into chunks
    
    Args:
        file_path: Path to file to chunk
        chunk_size: Size of each chunk in bytes (default 5MB)
    
    Returns:
        List of chunk file paths
    """
    chunks = []
    chunk_index = 0
    
    with open(file_path, 'rb') as f:
        while True:
            chunk_data = f.read(chunk_size)
            if not chunk_data:
                break
            
            chunk_path = f"{file_path}.chunk{chunk_index}"
            with open(chunk_path, 'wb') as chunk_file:
                chunk_file.write(chunk_data)
            
            chunks.append(chunk_path)
            chunk_index += 1
    
    return chunks


# ==================== PYTEST CONFIGURATION ====================

def pytest_configure(config):
    """Configure pytest markers"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "performance: marks tests as performance tests"
    )


def pytest_collection_modifyitems(config, items):
    """Automatically mark tests based on their location/name"""
    for item in items:
        # Mark slow tests
        if 'slow' in item.nodeid or 'large_file' in item.nodeid:
            item.add_marker(pytest.mark.slow)
        
        # Mark integration tests
        if 'Endpoint' in item.nodeid or 'integration' in item.nodeid:
            item.add_marker(pytest.mark.integration)
        
        # Mark unit tests
        if 'Test' in item.nodeid and 'Endpoint' not in item.nodeid:
            item.add_marker(pytest.mark.unit)
        
        # Mark performance tests
        if 'performance' in item.nodeid.lower():
            item.add_marker(pytest.mark.performance)


# ==================== DOCKER SUPPORT ====================

@pytest.fixture(scope='session')
def docker_available():
    """Check if Docker is available for integration tests"""
    import subprocess
    try:
        result = subprocess.run(
            ['docker', '--version'],
            capture_output=True,
            timeout=5
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


@pytest.fixture(scope='session')
def docker_app_url(docker_available):
    """Start Flask app in Docker container for integration tests
    
    Only runs if Docker is available and --docker flag is passed
    """
    if not docker_available:
        pytest.skip("Docker not available")
    
    # This would start the Docker container
    # Implementation depends on your Docker setup
    yield "http://localhost:8080"
    
    # Cleanup: stop Docker container
    pass


# ==================== ENV CONFIGURATION ====================

@pytest.fixture(autouse=True)
def setup_test_environment(monkeypatch):
    """Setup test environment variables"""
    monkeypatch.setenv('FLASK_ENV', 'testing')
    monkeypatch.setenv('TESTING', 'True')
    
    # Mock Supabase credentials if needed
    monkeypatch.setenv('SUPABASE_URL', 'https://test.supabase.co')
    monkeypatch.setenv('SUPABASE_KEY', 'test-key')


# ==================== DATA VALIDATORS ====================

def validate_regression_response(data):
    """Validate structure of regression response data
    
    Args:
        data: Response data dictionary
    
    Returns:
        bool: True if valid
    
    Raises:
        AssertionError: If structure is invalid
    """
    required_keys = [
        'x_values', 'y_values', 'predicted_y',
        'upper_bound', 'lower_bound',
        'filtered_x', 'filtered_y',
        'equation', 'removed_points'
    ]
    
    for key in required_keys:
        assert key in data, f"Missing required key: {key}"
    
    # Validate types
    assert isinstance(data['x_values'], list)
    assert isinstance(data['y_values'], list)
    assert isinstance(data['equation'], str)
    assert isinstance(data['removed_points'], int)
    
    # Validate lengths match
    assert len(data['x_values']) == len(data['y_values'])
    assert len(data['filtered_x']) == len(data['filtered_y'])
    
    return True


def validate_interpolation_response(response_data):
    """Validate structure of interpolation streaming response
    
    Args:
        response_data: List of parsed NDJSON lines
    
    Returns:
        bool: True if valid
    """
    assert len(response_data) > 0, "No data in response"
    
    # First message should be metadata
    meta = response_data[0]
    assert meta['type'] == 'meta'
    assert 'total_rows' in meta
    assert 'total_chunks' in meta
    
    # Last message should be completion
    last = response_data[-1]
    assert last['type'] == 'complete'
    assert last['success'] is True
    
    return True
