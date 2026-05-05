import logging
import pytest
from unittest.mock import patch

@pytest.fixture
def disable_debug(monkeypatch):
    monkeypatch.setenv("DEBUG_ANOMALY", "false")
    # Force re-import so DEBUG constant is reset
    import importlib
    import domains.adjustments.debug_log as dl
    importlib.reload(dl)
    yield dl
    importlib.reload(dl)  # clean up

@pytest.fixture
def enable_debug(monkeypatch):
    monkeypatch.setenv("DEBUG_ANOMALY", "true")
    import importlib
    import domains.adjustments.debug_log as dl
    importlib.reload(dl)
    yield dl
    importlib.reload(dl)


def test_dlog_noop_when_debug_disabled(disable_debug, caplog):
    caplog.set_level(logging.DEBUG)
    disable_debug.dlog("TEST", key="value")
    assert len([r for r in caplog.records if r.name == "domains.adjustments.debug"]) == 0


def test_dlog_emits_when_debug_enabled(enable_debug, caplog):
    caplog.set_level(logging.DEBUG, logger="domains.adjustments.debug")
    enable_debug.dlog("TEST", key="value")
    matching = [r for r in caplog.records if r.name == "domains.adjustments.debug" and "[TEST]" in r.message]
    assert len(matching) == 1
    assert "key='value'" in matching[0].message


def test_log_phase_decorator_times_and_logs(enable_debug, caplog):
    caplog.set_level(logging.DEBUG, logger="domains.adjustments.debug")

    @enable_debug.log_phase("test_phase")
    def doit():
        return 42

    result = doit()
    assert result == 42
    msgs = [r.message for r in caplog.records if r.name == "domains.adjustments.debug"]
    assert any("[PHASE_START] phase='test_phase'" in m for m in msgs)
    assert any("[PHASE_END] phase='test_phase'" in m for m in msgs)


def test_log_phase_logs_failure(enable_debug, caplog):
    caplog.set_level(logging.DEBUG, logger="domains.adjustments.debug")

    @enable_debug.log_phase("failing")
    def bad():
        raise ValueError("nope")

    with pytest.raises(ValueError):
        bad()

    msgs = [r.message for r in caplog.records if r.name == "domains.adjustments.debug"]
    assert any("[PHASE_FAIL]" in m and "ValueError" in m for m in msgs)
