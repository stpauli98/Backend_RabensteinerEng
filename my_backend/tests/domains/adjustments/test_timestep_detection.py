"""Detected timestep + offset shown after a file loads in anomaly detection.

Mirrors the canonical definition in services.utils:
  timestep [min] = round(median(Δt) / 60)
  offset   [min] = first_timestamp.minute % timestep
"""
import pandas as pd

from domains.adjustments.services.timestep_detection import (
    detect_timestep_offset_minutes,
)


def _df(start, step_min, n=6):
    idx = pd.date_range(start=start, periods=n, freq=f"{step_min}min")
    return pd.DataFrame({"UTC": idx, "value": range(n)})


def test_15min_grid_aligned_to_quarter_hour_offset_zero():
    # 14:45, 15:00, ... → 45 % 15 == 0
    ts, off = detect_timestep_offset_minutes(_df("2026-02-09 14:45:00", 15))
    assert ts == 15.0
    assert off == 0.0


def test_15min_grid_with_five_minute_offset():
    # first minute 05 → 5 % 15 == 5
    ts, off = detect_timestep_offset_minutes(_df("2026-02-09 14:05:00", 15))
    assert ts == 15.0
    assert off == 5.0


def test_five_minute_grid_offset():
    # 5-min step starting 10:02 → 2 % 5 == 2
    ts, off = detect_timestep_offset_minutes(_df("2026-02-09 10:02:00", 5))
    assert ts == 5.0
    assert off == 2.0


def test_single_row_is_safe():
    ts, off = detect_timestep_offset_minutes(_df("2026-02-09 10:00:00", 15, n=1))
    assert ts == 0.0
    assert off == 0.0
