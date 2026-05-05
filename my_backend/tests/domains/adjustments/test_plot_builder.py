"""Unit tests for plot_builder JSON serialization."""
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from domains.adjustments.services.plot_builder import (
    build_anomaly_overlay,
    build_line_plot,
    build_lstm_error,
    build_original_plot,
    build_processed_plot,
    build_slope_plot,
    build_stl_decomposition,
)


def make_df():
    times = pd.date_range("2025-01-01", periods=5, freq="3min")
    return pd.DataFrame({"UTC": times, "Q_RGK [kW]": [10.0, np.nan, 12.0, float("inf"), 14.0]})


def test_build_line_plot_shape():
    df = make_df()
    plot = build_line_plot(df, "UTC", "Q_RGK [kW]", "title", "UTC", "kW")
    assert plot["type"] == "line"
    assert "traces" in plot
    assert "xaxis" in plot
    assert "yaxis" in plot
    assert "title" in plot
    assert plot["xaxis"]["type"] == "datetime"
    assert plot["title"] == "title"


def test_build_line_plot_serializes_nan_inf_as_null():
    df = make_df()
    plot = build_line_plot(df, "UTC", "Q_RGK [kW]", "t", "x", "y")
    y = plot["traces"][0]["y"]
    assert y[0] == 10.0
    assert y[1] is None  # NaN
    assert y[2] == 12.0
    assert y[3] is None  # Inf
    assert y[4] == 14.0


def test_build_line_plot_x_iso_format():
    df = make_df()
    plot = build_line_plot(df, "UTC", "Q_RGK [kW]", "t", "x", "y")
    x = plot["traces"][0]["x"]
    assert x[0] == "2025-01-01T00:00:00"
    assert x[1] == "2025-01-01T00:03:00"


def test_build_anomaly_overlay_two_traces():
    df = make_df()
    mask = np.array([False, False, True, False, True])
    plot = build_anomaly_overlay(df, "UTC", "Q_RGK [kW]", mask, "t", "kW", lang="en")
    assert plot["type"] == "multi"
    assert len(plot["traces"]) == 2
    assert plot["traces"][1]["mode"] == "markers"
    assert len(plot["traces"][1]["x"]) == 2  # 2 anomalies


def test_build_anomaly_overlay_label_localized_de():
    df = make_df()
    mask = np.array([False, False, True, False, True])
    plot = build_anomaly_overlay(df, "UTC", "Q_RGK [kW]", mask, "t", "kW", lang="de")
    assert plot["traces"][1]["label"] == "Anomalien"
    assert plot["traces"][0]["label"] == "Originaldaten"


def test_build_anomaly_overlay_mask_length_mismatch_raises():
    df = make_df()
    mask = np.array([False, True])  # too short
    with pytest.raises(ValueError):
        build_anomaly_overlay(df, "UTC", "Q_RGK [kW]", mask, "t", "kW", lang="en")


def test_build_stl_decomposition_returns_4_subplots():
    class FakeSTL:
        observed = pd.Series([1.0, 2.0, 3.0])
        trend = pd.Series([1.5, 2.5, 3.5])
        seasonal = pd.Series([0.1, -0.1, 0.0])
        resid = pd.Series([0.0, 0.0, 0.0])
    times = pd.date_range("2025-01-01", periods=3, freq="3min")
    plots = build_stl_decomposition(FakeSTL(), pd.Series(times), lang="de")
    assert len(plots) == 4
    components = [p["component"] for p in plots]
    assert components == ["observed", "trend", "seasonal", "resid"]
    assert "STL-Zerlegung" in plots[0]["title"]


def test_build_lstm_error_shape():
    results = pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=3, freq="3min"),
        "absolute_error": [0.1, 0.5, 0.2],
    })
    plot = build_lstm_error(results, lang="en")
    assert plot["type"] == "line"
    assert plot["traces"][0]["y"] == [0.1, 0.5, 0.2]
    assert "Absolute error" in plot["title"]


def test_build_slope_plot_de_label():
    df = pd.DataFrame({
        "UTC": pd.date_range("2025-01-01", periods=3, freq="3min"),
        "Q_RGK": [1.0, 2.0, 3.0],
    })
    plot = build_slope_plot(df, lang="de")
    assert plot["title"] == "Steigung der Originaldaten"
    assert plot["yaxis"]["label"] == "Steigung [#/min]"


def test_build_original_plot_en_title():
    df = make_df()
    plot = build_original_plot(df, lang="en")
    assert plot["title"] == "Original data"


def test_build_processed_plot_de_title():
    df = make_df()
    plot = build_processed_plot(df, lang="de")
    assert plot["title"] == "Verarbeitete Daten"


def test_serialize_x_handles_nat():
    times = pd.to_datetime(["2025-01-01", pd.NaT, "2025-01-02"])
    df = pd.DataFrame({"UTC": times, "v": [1.0, 2.0, 3.0]})
    plot = build_line_plot(df, "UTC", "v", "t", "x", "y")
    x = plot["traces"][0]["x"]
    assert x[1] is None


def test_anomaly_overlay_lstm_uses_original_value_label():
    """LSTM overlay should use 'Original value' label per Python L1428."""
    times = pd.date_range("2025-01-01", periods=3, freq="3min")
    df = pd.DataFrame({"UTC": times, "v": [1.0, 2.0, 3.0]})
    mask = np.array([False, True, False])
    plot = build_anomaly_overlay(
        df, "UTC", "v", mask, "title", "y", lang="en",
        base_label="Original value",
    )
    assert plot["traces"][0]["label"] == "Original value"
