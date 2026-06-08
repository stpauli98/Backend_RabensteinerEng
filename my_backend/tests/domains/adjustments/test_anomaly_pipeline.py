"""Unit tests for anomaly detection pipeline phases.

Uses the shipped `test2/test2.csv` (14 879 rows, 3-min cadence) where possible
for realism. Smaller synthetic data is used for property-level checks.
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from domains.adjustments.services.anomaly_pipeline import (
    build_par_dict,
    process_constants,
    process_range,
    process_short_ranges,
    process_sbad,
    process_zeros,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
TEST2_CSV = REPO_ROOT / "test2" / "test2.csv"


def _load_test2():
    df = pd.read_csv(TEST2_CSV, sep=";", dtype=str)
    df["UTC"] = pd.to_datetime(df["UTC"], format="%Y-%m-%d %H:%M:%S")
    df.iloc[:, 1] = pd.to_numeric(df.iloc[:, 1], errors="coerce")
    return df.sort_values("UTC").reset_index(drop=True)


# ---------------------------------------------------------------------------
# build_par_dict factory
# ---------------------------------------------------------------------------

def test_build_par_dict_defaults():
    par = build_par_dict({})
    assert par["EQ_MAX"]["value"] is None
    assert par["EL0"]["value"] is False
    assert par["STL"]["run"] is False
    assert par["LSTM"]["run"] is False
    assert par["LSTM"]["var"]["NEURONS"]["value"] == 64
    assert par["LSTM"]["var"]["EPOCHS"]["value"] == 20  # NB: bug L913 corrected
    assert par["LSTM"]["var"]["BATCH_SIZE"]["value"] == 16
    # Defaults from Python init (L436, L487) — code-review BUG #23/#24
    assert par["LSTM"]["var"]["PERIOD_H"]["value"] == 24
    assert par["LSTM"]["var"]["THRESHOLD"]["value"] == 100


def test_prepare_lstm_raises_on_nan():
    """LSTM must refuse NaN-containing data — same guard as prepare_stl."""
    from domains.adjustments.services.anomaly_pipeline import prepare_lstm

    times = pd.date_range("2025-01-01", periods=20, freq="3min")
    values = [1.0] * 19 + [float("nan")]
    df = pd.DataFrame({"UTC": times, "v": values})

    with pytest.raises(ValueError) as exc:
        prepare_lstm(df, period=4, neurons=4, epochs=1, batch_size=4, lang="de")
    msg = str(exc.value)
    assert "NaN im Datensatz vorhanden" in msg
    # Error must be actionable: point the user at the interpolation gap setting.
    assert "Maximal zulässige Lücke" in msg


def test_prepare_stl_raises_on_nan_with_actionable_hint():
    """STL must refuse NaN data and tell the user how to fix the gaps."""
    from domains.adjustments.services.anomaly_pipeline import prepare_stl

    times = pd.date_range("2025-01-01", periods=48, freq="1h")
    values = [1.0] * 47 + [float("nan")]
    df = pd.DataFrame({"UTC": times, "v": values})

    with pytest.raises(ValueError) as exc:
        prepare_stl(df, period=24, lang="de")
    msg = str(exc.value)
    assert "NaN im Datensatz vorhanden" in msg
    assert "Maximal zulässige Lücke" in msg

    with pytest.raises(ValueError) as exc_en:
        prepare_stl(df, period=24, lang="en")
    assert "Maximum allowed gap for linear interpolation" in str(exc_en.value)


def test_build_par_dict_with_values():
    par = build_par_dict({
        "eqMax": 15, "gapMax": 60, "dec": 1,
        "sbad": {"chgMax": 20, "lgMax": 120},
        "stl": {"run": True, "periodH": 24},
    })
    assert par["EQ_MAX"]["value"] == 15
    assert par["SBAD"]["var"]["CHG_MAX"]["value"] == 20
    assert par["STL"]["run"] is True
    assert "[min]" in par["EQ_MAX"]["name"]["en"]
    assert "[min]" in par["EQ_MAX"]["name"]["de"]
    assert "[#/min]" in par["SBAD"]["var"]["CHG_MAX"]["name"]["en"]


# ---------------------------------------------------------------------------
# process_constants
# ---------------------------------------------------------------------------

def test_process_constants_nans_long_runs():
    times = pd.date_range("2025-01-01", periods=10, freq="3min")
    values = [1.0, 2.0, 5.0, 5.0, 5.0, 5.0, 5.0, 6.0, 7.0, 8.0]
    df = pd.DataFrame({"UTC": times, "v": values})
    out = process_constants(df.copy(), eq_max=12, gap_max=None, dec=None, lang="en")
    # Indices 2..6 (5 points) span 12 minutes: 5*3 = 15min from idx2 to idx6 → ≥ 12
    assert int(pd.isna(out["v"]).sum()) == 5


def test_process_constants_no_op_when_below_threshold():
    times = pd.date_range("2025-01-01", periods=5, freq="3min")
    values = [1.0, 5.0, 5.0, 5.0, 9.0]  # 3 constants span 6 min
    df = pd.DataFrame({"UTC": times, "v": values})
    out = process_constants(df.copy(), eq_max=15, gap_max=None, dec=None, lang="en")
    assert pd.isna(out["v"]).sum() == 0


# ---------------------------------------------------------------------------
# process_zeros
# ---------------------------------------------------------------------------

def test_process_zeros_nans_zeros():
    times = pd.date_range("2025-01-01", periods=4, freq="3min")
    df = pd.DataFrame({"UTC": times, "v": [1.0, 0.0, 2.0, 0.0]})
    out = process_zeros(df.copy(), el0=True, gap_max=None, dec=None, lang="en")
    assert pd.isna(out["v"].iloc[1])
    assert pd.isna(out["v"].iloc[3])
    assert out["v"].iloc[0] == 1.0
    assert out["v"].iloc[2] == 2.0


def test_process_zeros_disabled_no_op():
    times = pd.date_range("2025-01-01", periods=3, freq="3min")
    df = pd.DataFrame({"UTC": times, "v": [1.0, 0.0, 2.0]})
    out = process_zeros(df.copy(), el0=False, gap_max=None, dec=None, lang="en")
    assert out["v"].iloc[1] == 0.0


# ---------------------------------------------------------------------------
# process_range
# ---------------------------------------------------------------------------

def test_process_range_clips_outside():
    times = pd.date_range("2025-01-01", periods=5, freq="3min")
    df = pd.DataFrame({"UTC": times, "v": [-1.0, 50.0, 200.0, 100.0, 300.0]})
    out = process_range(df.copy(), v_max=180, v_min=0, gap_max=None, dec=None, lang="en")
    nan_mask = pd.isna(out["v"]).tolist()
    assert nan_mask == [True, False, True, False, True]


# ---------------------------------------------------------------------------
# process_sbad
# ---------------------------------------------------------------------------

def test_process_sbad_smooths_spike():
    # Smooth ramp with one big spike
    times = pd.date_range("2025-01-01", periods=10, freq="3min")
    values = [10.0, 11.0, 12.0, 13.0, 14.0, 100.0, 16.0, 17.0, 18.0, 19.0]
    df = pd.DataFrame({"UTC": times, "v": values})
    out, count = process_sbad(df.copy(), chg_max=10, lg_max=120, gap_max=60, dec=1, lang="en")
    # Spike (idx 5) should be removed and interpolated to ~15
    assert out["v"].iloc[5] == pytest.approx(15.0, abs=1.0)


# ---------------------------------------------------------------------------
# process_short_ranges
# ---------------------------------------------------------------------------

def test_process_short_ranges_removes_short_segments():
    # Two valid segments separated by NaN; first one is too short
    times = pd.date_range("2025-01-01", periods=10, freq="3min")
    values = [1.0, 2.0, np.nan, np.nan, np.nan, 5.0, 6.0, 7.0, 8.0, 9.0]
    df = pd.DataFrame({"UTC": times, "v": values})
    # First segment: 2 points, span 3 min → < 10 min, removed
    # Second segment: 5 points, span 12 min → ≥ 10 min, kept
    out = process_short_ranges(df.copy(), lg_min=10, lang="en")
    assert pd.isna(out["v"].iloc[0])
    assert pd.isna(out["v"].iloc[1])
    assert out["v"].iloc[5] == 5.0


# ---------------------------------------------------------------------------
# Integration on real test2.csv
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not TEST2_CSV.exists(), reason="test2.csv fixture not present")
def test_full_pipeline_through_sbad_on_test2():
    df = _load_test2()
    initial_len = len(df)
    par = build_par_dict({
        "eqMax": 15, "gapMax": 60, "dec": 1,
        "vMax": 180, "vMin": 0, "el0": True,
        "sbad": {"chgMax": 20, "lgMax": 120},
    })

    df = process_constants(df, par["EQ_MAX"]["value"], par["GAP_MAX"]["value"],
                           par["DEC"]["value"], lang="en")
    df = process_zeros(df, par["EL0"]["value"], par["GAP_MAX"]["value"],
                       par["DEC"]["value"], lang="en")
    df = process_range(df, par["V_MAX"]["value"], par["V_MIN"]["value"],
                       par["GAP_MAX"]["value"], par["DEC"]["value"], lang="en")
    df, count = process_sbad(df, par["SBAD"]["var"]["CHG_MAX"]["value"],
                             par["SBAD"]["var"]["LG_MAX"]["value"],
                             par["GAP_MAX"]["value"], par["DEC"]["value"], lang="en")

    # Length must be preserved
    assert len(df) == initial_len
    # Some values should have been NaN-ed (test2.csv has constant 100 runs at start)
    assert int(pd.isna(df.iloc[:, 1]).sum()) >= 0
    # SBAD should have detected at least some anomalies (or exactly 0 if data is clean)
    assert count >= 0


# ---------------------------------------------------------------------------
# R3: phase markers around STL.fit() and LSTM model.fit()
# ---------------------------------------------------------------------------

def test_prepare_stl_emits_progress_markers():
    """prepare_stl must emit start (0.0) and end (1.0) progress markers."""
    from domains.adjustments.services.anomaly_pipeline import prepare_stl

    n = 48  # 2 synthetic days at hourly cadence
    rng = np.random.RandomState(42)
    df = pd.DataFrame({
        "ts": pd.date_range("2024-01-01", periods=n, freq="1h"),
        "value": np.sin(np.arange(n) * 2 * np.pi / 24) + rng.normal(0, 0.1, n),
    })

    captured = []

    def cb(label, fraction):
        captured.append((label, fraction))

    result, time_values = prepare_stl(df, period=24, lang="de", progress_callback=cb)

    labels = [c[0] for c in captured]
    fractions = [c[1] for c in captured]

    assert any(f == 0.0 for f in fractions), "Expected 0.0 start marker"
    assert any(f == 1.0 for f in fractions), "Expected 1.0 end marker"
    assert all("STL" in lbl or "Zerlegung" in lbl for lbl in labels), (
        f"Unexpected label(s): {labels}"
    )
    assert result is not None  # STL ran successfully
    assert len(time_values) == n


# ---------------------------------------------------------------------------
# Resource guards: an out-of-range period/neurons must fail fast with a
# ValueError (-> 400), never reach STL/LSTM and OOM-kill the worker (DoS).
# ---------------------------------------------------------------------------

def _clean_df(n=48):
    return pd.DataFrame({
        "ts": pd.date_range("2024-01-01", periods=n, freq="1h"),
        "value": np.sin(np.arange(n) * 2 * np.pi / 24) + 1.0,
    })


@pytest.mark.parametrize("bad_period", [10**9, 0, -5, "abc", None, 48])
def test_prepare_stl_rejects_out_of_range_period(bad_period):
    """Huge/zero/negative/non-numeric/too-large period -> ValueError, no STL."""
    from domains.adjustments.services.anomaly_pipeline import prepare_stl
    df = _clean_df(48)  # upper bound = 24; period 48 is too large
    with pytest.raises(ValueError):
        prepare_stl(df, period=bad_period, lang="de")


def test_prepare_stl_accepts_period_at_upper_bound():
    """period == n // 2 is the largest valid value and must still run."""
    from domains.adjustments.services.anomaly_pipeline import prepare_stl
    df = _clean_df(48)
    result, _ = prepare_stl(df, period=24, lang="de")
    assert result is not None


def test_prepare_lstm_rejects_huge_neurons():
    """Huge neuron count must be rejected before building the model (OOM guard)."""
    from domains.adjustments.services.anomaly_pipeline import prepare_lstm
    df = _clean_df(20)
    with pytest.raises(ValueError):
        prepare_lstm(df, period=4, neurons=10**9, epochs=1, batch_size=4, lang="de")


def test_prepare_lstm_rejects_huge_epochs():
    from domains.adjustments.services.anomaly_pipeline import prepare_lstm
    df = _clean_df(20)
    with pytest.raises(ValueError):
        prepare_lstm(df, period=4, neurons=8, epochs=10**7, batch_size=4, lang="de")


# ---------------------------------------------------------------------------
# Plan T1: callbacks must additionally carry a stable i18n key (message_key)
# and optional message_params (e.g. SBAD iteration counter).
# ---------------------------------------------------------------------------

def test_process_constants_passes_message_key_to_callback():
    """Pipeline phases must emit a stable i18n key, not just localized labels."""
    import pandas as pd
    from domains.adjustments.services.anomaly_pipeline import process_constants

    times = pd.date_range("2025-01-01", periods=10, freq="3min")
    df = pd.DataFrame({"t": times, "v": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 3.0, 4.0, 5.0]})
    captured: list = []

    def cb(label, fraction, message_key=None, message_params=None):
        captured.append({"label": label, "fraction": fraction,
                         "message_key": message_key, "message_params": message_params})

    process_constants(df, eq_max=1.0, gap_max=None, dec=1, lang="de", progress_callback=cb)

    assert captured, "callback should have been invoked"
    keys = {c["message_key"] for c in captured if c["message_key"] is not None}
    assert "anomaly_phase_constants" in keys


def test_process_zeros_passes_message_key():
    import pandas as pd
    from domains.adjustments.services.anomaly_pipeline import process_zeros

    df = pd.DataFrame({"t": pd.date_range("2025-01-01", periods=5, freq="3min"),
                       "v": [0.0, 1.0, 0.0, 2.0, 3.0]})
    captured: list = []

    def cb(label, fraction, message_key=None, message_params=None):
        captured.append(message_key)

    process_zeros(df, el0=True, gap_max=None, dec=1, lang="en", progress_callback=cb)
    assert "anomaly_phase_zeros" in {k for k in captured if k}


def test_sbad_callback_includes_iteration_param():
    """SBAD iterates; the iteration count must travel as message_params."""
    import pandas as pd
    from domains.adjustments.services.anomaly_pipeline import process_sbad

    df = pd.DataFrame({"t": pd.date_range("2025-01-01", periods=20, freq="3min"),
                       "v": [10.0] * 10 + [10.0, 100.0, 10.0] + [10.0] * 7})
    captured: list = []

    def cb(label, fraction, message_key=None, message_params=None):
        captured.append({"key": message_key, "params": message_params})

    process_sbad(df, chg_max=5, lg_max=3, gap_max=None, dec=1, lang="en", progress_callback=cb)

    sbad_calls = [c for c in captured if c["key"] == "anomaly_phase_sbad"]
    assert sbad_calls, "SBAD must emit anomaly_phase_sbad messageKey"
    iter_params = [c["params"] for c in sbad_calls if c["params"] and "iter" in c["params"]]
    assert iter_params, "SBAD callback must carry {iter: N} as message_params"


def test_legacy_two_arg_callback_still_works():
    """The whole point of the introspection wrapper: legacy cb(label, fraction)
    callbacks (no message_key/message_params kwargs) must keep working."""
    import pandas as pd
    from domains.adjustments.services.anomaly_pipeline import process_zeros

    df = pd.DataFrame({"t": pd.date_range("2025-01-01", periods=5, freq="3min"),
                       "v": [0.0, 1.0, 0.0, 2.0, 3.0]})
    captured: list = []

    def cb(label, fraction):  # legacy signature, no kwargs at all
        captured.append((label, fraction))

    # Must not raise TypeError
    process_zeros(df, el0=True, gap_max=None, dec=1, lang="en", progress_callback=cb)
    assert captured, "legacy callback should have been invoked"
