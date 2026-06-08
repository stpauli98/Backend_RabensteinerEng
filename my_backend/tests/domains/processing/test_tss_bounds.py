"""
Tests for H-2 DoS fix: bounding the time step size (tss) and the generated
time-grid size in csv_processor.process_csv.

A tiny tss (e.g. 0.0001) over a non-trivial time span builds billions of
datetime objects -> memory exhaustion -> gunicorn worker SIGKILL. These tests
verify the validator rejects out-of-range / non-numeric tss and that the
span-based grid cap raises ValueError BEFORE any large allocation occurs.
"""
import datetime

import pytest

import domains.processing.services.csv_processor as cp


@pytest.mark.parametrize("bad", [0, -1, 0.0001, 100000, "abc", {"x": 1}])
def test_validate_tss_rejects(bad):
    with pytest.raises(ValueError):
        cp.validate_tss(bad)


def test_validate_tss_ok():
    assert cp.validate_tss("15") == 15.0


def test_validate_tss_accepts_bounds():
    assert cp.validate_tss(cp._TSS_MIN) == cp._TSS_MIN
    assert cp.validate_tss(cp._TSS_MAX) == cp._TSS_MAX


def test_process_csv_rejects_tiny_tss_without_allocating():
    """
    A multi-day span with a tiny tss would, without the fix, allocate billions
    of datetimes. validate_tss must reject the out-of-range tss first, so the
    grid loop is never entered and no large allocation happens.

    We build a tiny 2-row CSV spanning 3 days; the span cap / validator must
    fire and raise ValueError cheaply.
    """
    csv_content = (
        "UTC;Value\n"
        "2020-01-01 00:00:00;1.0\n"
        "2020-01-04 00:00:00;2.0\n"
    )
    with pytest.raises(ValueError):
        cp.process_csv(
            csv_content,
            tss=0.0001,
            offset=0,
            mode_input="mean",
            intrpl_max=60,
        )
