"""
Regression test for C3: storage tracking in /adjustdata/complete must measure
result dataframes BEFORE they are deleted from state, not after.

The handler used to compute `total_size_bytes` from
`adjustment_chunks[upload_id]['dataframes']` *after* the per-file loop had
already `del`-ed every entry out of that same dict, so the measured size was
always 0 regardless of how much data was actually processed.

This test exercises the extracted `_result_size_mb` helper directly, and
proves the "measure before delete" ordering is what makes the value correct.
"""
import pandas as pd
import pytest

from domains.adjustments.api.adjustments import _result_size_mb


def make_df(n_rows: int) -> pd.DataFrame:
    return pd.DataFrame({
        "UTC": pd.date_range("2024-01-01", periods=n_rows, freq="min"),
        "value": [float(i) for i in range(n_rows)],
    })


def test_result_size_mb_is_positive_for_nonempty_frames():
    frames = [make_df(1000), make_df(2000)]
    size_mb = _result_size_mb(frames)
    assert size_mb > 0


def test_result_size_mb_matches_expected_magnitude():
    df1 = make_df(1000)
    df2 = make_df(2000)
    expected_bytes = int(df1.memory_usage(deep=True).sum()) + int(df2.memory_usage(deep=True).sum())
    expected_mb = expected_bytes / (1024 * 1024)

    size_mb = _result_size_mb([df1, df2])

    assert size_mb == pytest.approx(expected_mb)


def test_result_size_mb_ignores_none_entries():
    df = make_df(500)
    size_mb = _result_size_mb([df, None])
    assert size_mb == pytest.approx(int(df.memory_usage(deep=True).sum()) / (1024 * 1024))


def test_result_size_mb_empty_iterable_is_zero():
    assert _result_size_mb([]) == 0


def test_measured_value_survives_deletion_of_source_dict():
    """
    Simulates the bug scenario: a dict of dataframes (like
    adjustment_chunks[upload_id]['dataframes']) gets its entries deleted
    during per-file processing. If size is computed BEFORE deletion (the
    fix), the captured value stays correct even after `del`. If it were
    computed AFTER (the bug), it would read 0 from the now-empty dict.
    """
    state_dataframes = {
        "file_a.csv": make_df(1000),
        "file_b.csv": make_df(1500),
    }

    # Correct ordering: measure BEFORE cleanup.
    captured_size_mb = _result_size_mb(list(state_dataframes.values()))
    assert captured_size_mb > 0

    # Simulate the per-file cleanup that happens later in the handler.
    for filename in list(state_dataframes.keys()):
        del state_dataframes[filename]

    assert state_dataframes == {}
    # The captured value must be unaffected by the later deletion.
    assert captured_size_mb > 0

    # Demonstrate what the BUG looked like: measuring after deletion reads 0.
    buggy_size_mb = _result_size_mb(list(state_dataframes.values()))
    assert buggy_size_mb == 0
    assert buggy_size_mb != captured_size_mb
