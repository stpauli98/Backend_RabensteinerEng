"""Unit tests for core/rate_limits.py."""
import os
from unittest.mock import patch

import pytest

from core.rate_limits import (
    cloud_limit_string,
    forecast_limit_string,
    _is_testing,
)


class TestCloudLimitString:
    def test_returns_60_per_minute_in_production(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop('FLASK_ENV', None)
            os.environ.pop('PYTEST_CURRENT_TEST', None)
            assert cloud_limit_string() == "60 per minute"

    def test_returns_1000_per_minute_in_testing(self):
        with patch.dict(os.environ, {'FLASK_ENV': 'testing'}, clear=False):
            assert cloud_limit_string() == "1000 per minute"

    def test_returns_1000_per_minute_under_pytest(self):
        # PYTEST_CURRENT_TEST is auto-set by pytest while running
        assert cloud_limit_string() == "1000 per minute"
