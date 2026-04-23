"""Tests for the lgMin parameter in the data-cleaning pipeline.

Covers:
- validate_processing_params() accepts/rejects lgMin correctly
- clean_data() runs STEP 7 (short valid-range removal) only when lgMin is set
- STEP 7 runs AFTER STEP 6 GAP_MAX so gap-filling's output is subject to it
"""
import numpy as np
import pandas as pd
import pytest

from domains.processing.services.data_cleaner import (
    clean_data,
    validate_processing_params,
)


# ---------------------------------------------------------------------------
# validate_processing_params — parameter acceptance/rejection
# ---------------------------------------------------------------------------

def test_lg_min_valid_value_accepted():
    out = validate_processing_params({"lgMin": "720"})
    assert out["lgMin"] == 720.0


def test_lg_min_zero_accepted():
    out = validate_processing_params({"lgMin": "0"})
    assert out["lgMin"] == 0.0


def test_lg_min_negative_rejected():
    with pytest.raises(ValueError, match="Minimum valid range duration"):
        validate_processing_params({"lgMin": "-1"})


def test_lg_min_above_max_rejected():
    with pytest.raises(ValueError, match="Minimum valid range duration"):
        validate_processing_params({"lgMin": "1000001"})


def test_lg_min_non_numeric_rejected():
    with pytest.raises(ValueError, match="must be a valid number"):
        validate_processing_params({"lgMin": "abc"})


def test_lg_min_empty_string_ignored():
    out = validate_processing_params({"lgMin": ""})
    assert "lgMin" not in out


def test_lg_min_missing_ignored():
    out = validate_processing_params({})
    assert "lgMin" not in out


# ---------------------------------------------------------------------------
# clean_data — STEP 7 behavior
# ---------------------------------------------------------------------------

def _ts(values, minutes_between=1):
    """Build a DataFrame with UTC timestamps `minutes_between` min apart."""
    start = pd.Timestamp("2026-01-01 00:00:00")
    utc = [start + pd.Timedelta(minutes=i * minutes_between) for i in range(len(values))]
    df = pd.DataFrame(
        {
            "UTC": [t.strftime("%Y-%m-%d %H:%M:%S") for t in utc],
            "val": values,
        }
    )
    return df


def test_short_valid_range_is_removed():
    # 30 valid samples (30 min) flanked by NaN on both sides; lgMin=60 means
    # "keep only intervals >= 60 min". The middle island must become NaN.
    values = ([np.nan] * 10) + list(range(30)) + ([np.nan] * 10)
    df = _ts(values)
    cleaned = clean_data(df, "val", {"lgMin": 60}, tracker=None)
    assert cleaned.iloc[10:40]["val"].isna().all()


def test_long_valid_range_kept():
    # 120 valid samples — longer than lgMin=60, kept as-is.
    values = ([np.nan] * 5) + list(range(120)) + ([np.nan] * 5)
    df = _ts(values)
    cleaned = clean_data(df, "val", {"lgMin": 60}, tracker=None)
    assert cleaned.iloc[5:125]["val"].notna().all()


def test_without_lg_min_param_data_untouched():
    # Without lgMin, the 30-sample island survives.
    values = ([np.nan] * 10) + list(range(30)) + ([np.nan] * 10)
    df = _ts(values)
    cleaned = clean_data(df, "val", {}, tracker=None)
    assert cleaned.iloc[10:40]["val"].notna().all()


def test_lg_min_runs_after_gap_filling():
    # Build: 5 valid, 3 NaN (gap), 5 valid, 20 NaN (long gap), 5 valid.
    values = [
        1.0, 2.0, 3.0, 4.0, 5.0,
        np.nan, np.nan, np.nan,
        6.0, 7.0, 8.0, 9.0, 10.0,
        *([np.nan] * 20),
        11.0, 12.0, 13.0, 14.0, 15.0,
    ]
    df = _ts(values)
    # gapMax=5 min: first gap (3 min) gets filled → island becomes 13 min long,
    # second gap (20 min) stays NaN.
    # lgMin=30 min: the 13-min island is still shorter than 30 so it must be
    # discarded AFTER gap-filling ran. The last 5-min island too.
    cleaned = clean_data(df, "val", {"gapMax": 5, "lgMin": 30}, tracker=None)
    assert cleaned.iloc[0:13]["val"].isna().all()   # first island removed
    assert cleaned.iloc[-5:]["val"].isna().all()    # last island removed
