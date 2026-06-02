"""avg=True parity: the optimized transformer's averaging branch must match the
original (create_training_arrays_original lines 149-158 / 240-249).

The original computes the mean of the RAW data points whose timestamp falls in the
window [utc_ref+th_strt, utc_ref+th_end], via:
    idx1 = utc_idx_post(raw, th_strt)   # first index >= th_strt (or None)
    idx2 = utc_idx_pre(raw, th_end)     # last index <= th_end  (or None)
    val  = raw.iloc[idx1:idx2, 1].mean()   # half-open, idx2 EXCLUDED; skipna
and fills the window with [val] * N.

The optimized branch previously interpolated N evenly-spaced points and took
np.nanmean of those — a different statistic for non-linear data. These tests pin
the new helper to the original's exact slice-mean semantics (including the
None-slice edge behavior).
"""
import numpy as np
import pandas as pd

from domains.training.data.loader import utc_idx_pre, utc_idx_post
from domains.training.data.transformer import extract_avg_windows_matching_original


def _raw(n_hours, value_fn):
    """Raw frame as i_dat[key]: 'UTC' column + value column, default RangeIndex
    (so iloc positions == index labels, matching the original)."""
    base = pd.Timestamp("2024-01-01 00:00:00")
    times = [base + pd.Timedelta(hours=h) for h in range(n_hours)]
    values = [float(value_fn(h)) for h in range(n_hours)]
    return base, pd.DataFrame({"UTC": times, "val": values})


def _expected_original(raw, ref, th_strt, th_end):
    """Exactly what create_training_arrays_original computes for one avg window."""
    idx1 = utc_idx_post(raw, ref + pd.Timedelta(hours=th_strt))
    idx2 = utc_idx_pre(raw, ref + pd.Timedelta(hours=th_end))
    return raw.iloc[idx1:idx2, 1].mean()


def test_avg_matches_original_for_nonlinear_data():
    # Non-linear values so raw-mean != mean-of-interpolated-points (the old bug).
    base, raw = _raw(24, lambda h: h * h)
    interp_df = raw.set_index("UTC")  # what preprocess_and_interpolate_file produces

    refs = [base + pd.Timedelta(hours=h) for h in (3, 10, 18)]
    th_strt, th_end, n_points = 0.0, 5.0, 4

    got = extract_avg_windows_matching_original(interp_df, refs, th_strt, th_end, n_points)

    assert got.shape == (len(refs), n_points)
    for i, ref in enumerate(refs):
        expected = _expected_original(raw, ref, th_strt, th_end)
        # window filled with the single mean value (matches [val] * N)
        assert np.allclose(got[i, :], expected), f"row {i}: {got[i]} != {expected}"


def test_avg_replicates_original_none_slice_edge():
    # Window entirely AFTER the data -> utc_idx_post returns None -> original
    # slices iloc[None:idx2] (from start). The helper must reproduce that quirk.
    base, raw = _raw(10, lambda h: h)
    interp_df = raw.set_index("UTC")
    ref = base + pd.Timedelta(hours=100)
    th_strt, th_end, n_points = 0.0, 5.0, 3

    got = extract_avg_windows_matching_original(interp_df, [ref], th_strt, th_end, n_points)
    expected = _expected_original(raw, ref, th_strt, th_end)

    assert np.allclose(got[0, :], expected)


def test_avg_drops_sample_when_window_empty():
    # Sub-hour window sitting strictly between two hourly points -> no raw points
    # inside -> empty slice -> mean is NaN -> all-NaN row, which the downstream NaN
    # mask drops (matching the original's error/break that drops the whole sample).
    base, raw = _raw(10, lambda h: h)
    interp_df = raw.set_index("UTC")
    ref = base + pd.Timedelta(hours=3)        # on a data point
    th_strt, th_end, n_points = 0.1, 0.4, 3   # window (3.1h, 3.4h): no points inside

    got = extract_avg_windows_matching_original(interp_df, [ref], th_strt, th_end, n_points)
    expected = _expected_original(raw, ref, th_strt, th_end)

    assert np.isnan(float(expected))       # original: empty slice -> NaN
    assert np.all(np.isnan(got[0, :]))     # helper: all-NaN row -> sample dropped
