"""
Time Grid Setting (#60): the processed grid must START at the correct timestamp.

Owner report: raw start 2025-04-03 22:01:00, anchor "must-appear" point
2025-04-03 22:02:00, tss=7 → grid should start at 22:02:00 (matches reference
Python). The app previously started at 22:09:00 (one tss too late) due to an
off-by-one (`i - 2`) in the anchor branch.
"""
import pandas as pd

import domains.processing.services.csv_processor as cp


def _ts(s):
    return pd.to_datetime(s)


def test_owner_case_grid_starts_at_anchor():
    # anchor 22:02 >= raw 22:01 → lowest aligned point >= raw is the anchor itself
    assert cp._anchor_grid_start(_ts("2025-04-03 22:02:00"), _ts("2025-04-03 22:01:00"), 7) \
        == _ts("2025-04-03 22:02:00")


def test_lowest_aligned_point_ge_raw():
    # anchor 22:02, raw 21:50, tss 7 → aligned: 22:02, 21:55, 21:48; lowest >= 21:50 is 21:55
    assert cp._anchor_grid_start(_ts("2025-04-03 22:02:00"), _ts("2025-04-03 21:50:00"), 7) \
        == _ts("2025-04-03 21:55:00")


def test_anchor_before_raw_uses_last_aligned_le_raw():
    # anchor 22:00 < raw 22:05, tss 7 → last aligned <= raw is 22:00 (next is 22:07)
    assert cp._anchor_grid_start(_ts("2025-04-03 22:00:00"), _ts("2025-04-03 22:05:00"), 7) \
        == _ts("2025-04-03 22:00:00")


def test_anchor_equals_raw():
    assert cp._anchor_grid_start(_ts("2025-04-03 22:01:00"), _ts("2025-04-03 22:01:00"), 7) \
        == _ts("2025-04-03 22:01:00")


def test_const_branch_grid_does_not_start_before_raw():
    """Owner report: raw start 2026-02-09 14:45:00, tss=15, offset=0, intrpl.

    Because 60/15 is integer the 'const' branch runs (no anchor_time). The grid
    must NOT emit points before the first raw sample: those have no left
    neighbour and come out as 'nan'. result.csv showed 3 leading 'nan'
    (14:00/14:15/14:30) because the grid floored to the top of the hour.
    """
    import json
    csv_content = (
        "UTC;load_grid [kW]\n"
        "2026-02-09 14:45:00;0\n"
        "2026-02-09 15:00:00;0\n"
        "2026-02-09 15:15:00;5\n"
    )
    resp = cp.process_csv(
        csv_content,
        tss=15,
        offset=0,
        mode_input="intrpl",
        intrpl_max=60,
    )
    body = "".join(s.decode() if isinstance(s, (bytes, bytearray)) else s for s in resp.response)
    records = [json.loads(line) for line in body.splitlines() if line.strip()]
    rows = [r for r in records if "UTC" in r]
    # First output timestamp must be the first aligned point >= raw start (14:45),
    # not 14:00.
    assert rows[0]["UTC"] == "2026-02-09 14:45:00", f"grid started at {rows[0]['UTC']}"
    # No leading nan: the value column must be present (not None) for the first row.
    value_key = next(k for k in rows[0] if k != "UTC")
    assert rows[0][value_key] is not None, "first row is nan -> grid started before raw data"


def test_process_csv_first_output_timestamp_matches_anchor():
    """End-to-end: process_csv must emit its FIRST grid timestamp at the anchor."""
    import json
    csv_content = (
        "UTC;Value\n"
        "2025-04-03 22:01:00;1.0\n"
        "2025-04-03 23:00:00;2.0\n"
    )
    resp = cp.process_csv(
        csv_content,
        tss=7,
        offset=0,
        mode_input="intrpl",
        intrpl_max=60,
        anchor_time="2025-04-03 22:02:00",
    )
    body = "".join(s.decode() if isinstance(s, (bytes, bytearray)) else s for s in resp.response)
    records = [json.loads(line) for line in body.splitlines() if line.strip()]
    first = next(r for r in records if "UTC" in r)
    assert first["UTC"] == "2025-04-03 22:02:00", f"grid started at {first['UTC']}, expected 22:02:00"
