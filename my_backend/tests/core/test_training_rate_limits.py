"""Tests for training_limit_string rate-limit helper."""

from core.rate_limits import training_limit_string


class TestTrainingLimitString:
    def test_returns_30_per_minute_in_production(self, monkeypatch):
        """Without testing env markers, helper returns the production limit."""
        monkeypatch.delenv("FLASK_ENV", raising=False)
        monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
        assert training_limit_string() == "30 per minute"

    def test_returns_1000_per_minute_in_testing(self, monkeypatch):
        """With FLASK_ENV=testing set, helper returns the testing limit."""
        monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
        monkeypatch.setenv("FLASK_ENV", "testing")
        assert training_limit_string() == "1000 per minute"

    def test_returns_1000_per_minute_under_pytest(self):
        """During pytest run PYTEST_CURRENT_TEST is set → helper returns testing limit."""
        assert training_limit_string() == "1000 per minute"
