"""Unit tests for core/rate_limits.py."""
import pytest

from core.rate_limits import cloud_limit_string


class TestCloudLimitString:
    def test_returns_60_per_minute_in_production(self, monkeypatch):
        monkeypatch.delenv('FLASK_ENV', raising=False)
        monkeypatch.delenv('PYTEST_CURRENT_TEST', raising=False)
        assert cloud_limit_string() == "60 per minute"

    def test_returns_1000_per_minute_in_testing(self, monkeypatch):
        monkeypatch.setenv('FLASK_ENV', 'testing')
        assert cloud_limit_string() == "1000 per minute"

    def test_returns_1000_per_minute_under_pytest(self):
        # PYTEST_CURRENT_TEST is auto-set by pytest while running
        assert cloud_limit_string() == "1000 per minute"
