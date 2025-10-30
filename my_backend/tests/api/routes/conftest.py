"""
Shared fixtures and configuration for route tests.

This file ensures consistent middleware mocking across all test files.
"""

import pytest
from unittest.mock import MagicMock
import sys

# Mock middleware modules BEFORE any other imports
# This ensures the mocks are in place when load_data.py is imported
def mock_decorator(f):
    """Mock decorator that does nothing."""
    return f

# Create mock modules
sys.modules['middleware.auth'] = MagicMock()
sys.modules['middleware.subscription'] = MagicMock()
sys.modules['utils.usage_tracking'] = MagicMock()

# Set up mock decorators
sys.modules['middleware.auth'].require_auth = mock_decorator
sys.modules['middleware.subscription'].require_subscription = mock_decorator
sys.modules['middleware.subscription'].check_processing_limit = mock_decorator

# Set up mock functions
sys.modules['utils.usage_tracking'].increment_processing_count = MagicMock()
sys.modules['utils.usage_tracking'].update_storage_usage = MagicMock()


@pytest.fixture
def mock_socketio():
    """Mock SocketIO instance."""
    return MagicMock()
