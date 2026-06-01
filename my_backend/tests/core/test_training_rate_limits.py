"""Tests for training_limit_string rate-limit helper."""

from core.rate_limits import training_limit_string


def test_training_limit_returns_test_value_when_pytest_active():
    """During pytest run PYTEST_CURRENT_TEST is set → helper returns testing limit."""
    assert training_limit_string() == "1000 per minute"


def test_training_limit_returns_production_value(monkeypatch):
    """Without testing env markers, helper returns the production limit."""
    monkeypatch.delenv("FLASK_ENV", raising=False)
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    assert training_limit_string() == "30 per minute"
