"""Unit tests for anomaly detection helper functions."""
import numpy as np
import pandas as pd
import pytest

from domains.adjustments.services.anomaly_helpers import (
    create_sequences,
    hms,
    intrpl,
    rd,
    slope_calc,
    tr,
)


def make_df(timestamps, values):
    return pd.DataFrame({"UTC": pd.to_datetime(timestamps), "v": values})


def test_tr_returns_de_for_de():
    assert tr("hi", "hallo", "de") == "hallo"


def test_tr_returns_en_for_en_or_unknown():
    assert tr("hi", "hallo", "en") == "hi"
    assert tr("hi", "hallo", "fr") == "hi"


def test_rd_handles_none():
    assert rd(1.234, None) == 1.234


def test_rd_rounds():
    assert rd(1.234, 1) == 1.2


def test_rd_passes_nan_through():
    assert pd.isna(rd(np.nan, 1))


def test_hms_timedelta():
    assert hms(pd.Timedelta(minutes=3), "timedelta") == "00:03:00"


def test_hms_hours():
    assert hms(1.5, "hours") == "01:30:00"


def test_slope_calc_basic():
    times = ["2025-01-01 00:00", "2025-01-01 00:03", "2025-01-01 00:06"]
    df = make_df(times, [10.0, 16.0, 22.0])  # +6 every 3 minutes → slope = 2/min
    res = slope_calc(df)
    assert pd.isna(res.iloc[0, 1])
    assert res.iloc[1, 1] == pytest.approx(2.0)
    assert res.iloc[2, 1] == pytest.approx(2.0)


def test_intrpl_fills_short_gap():
    times = pd.date_range("2025-01-01", periods=5, freq="3min")
    df = pd.DataFrame({"UTC": times, "v": [10.0, np.nan, np.nan, np.nan, 22.0]})
    out = intrpl(df.copy(), gap_max=60.0, dec=1, lang="en")
    # Linear from 10 → 22 over 12 minutes: 10, 13, 16, 19, 22
    assert list(out["v"]) == [10.0, 13.0, 16.0, 19.0, 22.0]


def test_intrpl_skips_too_long_gap():
    times = pd.date_range("2025-01-01", periods=5, freq="60min")
    df = pd.DataFrame({"UTC": times, "v": [10.0, np.nan, np.nan, np.nan, 22.0]})
    out = intrpl(df.copy(), gap_max=30.0, dec=1, lang="en")
    # All NaN preserved
    assert pd.isna(out["v"]).sum() == 3


def test_create_sequences_shape():
    X, y = create_sequences(np.arange(10), sequence_length=3)
    assert X.shape == (7, 3)
    assert y.shape == (7,)
    assert list(X[0]) == [0, 1, 2]
    assert y[0] == 3
