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
