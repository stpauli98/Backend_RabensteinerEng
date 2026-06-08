"""
Plot data builders for the anomaly detection pipeline.

Each builder returns a JSON-serializable dict (`PlotData`) that the frontend
WebGLChart can render directly. Format:

    {
        "type": "line" | "scatter" | "multi",
        "traces": [
            {
                "x": [...],          # ISO datetime strings
                "y": [...],          # floats (with `null` for NaN)
                "color": "blue",     # optional
                "label": "...",      # optional, localized
                "mode": "lines"      # "lines" | "markers"
            },
            ...
        ],
        "xaxis": {"label": "UTC", "type": "datetime"},
        "yaxis": {"label": "..."},
        "title": "..."   # localized
    }

NO matplotlib output — those plot blocks (L685-724, L1238-1305, L1378-1453,
L1547-1565) are reproduced as JSON; the frontend renders them with
zoom/pan/hover via WebGLChart.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from domains.adjustments.services.anomaly_helpers import tr


def _serialize_x(series: pd.Series) -> List:
    """Convert a datetime-like series to ISO 8601 strings; NaT → None."""
    dt_series = pd.to_datetime(series, errors="coerce")
    formatted = dt_series.dt.strftime("%Y-%m-%dT%H:%M:%S")
    return [None if pd.isna(v) else v for v in formatted.tolist()]


def _serialize_y(series: pd.Series) -> List:
    """Convert numeric series to list, with `None` replacing NaN/Inf."""
    out = []
    for v in series.values:
        try:
            f = float(v)
        except (TypeError, ValueError):
            out.append(None)
            continue
        if math.isnan(f) or math.isinf(f):
            out.append(None)
        else:
            out.append(f)
    return out


# Cloud Run rejects responses larger than 32 MiB with a platform-level 500
# (no CORS headers, so the browser reports a misleading CORS error). A full-
# resolution CSV serialized every row as x (ISO string) + y (float), which
# blew past that cap on large files. Plot traces are decimated to this many
# points for TRANSPORT only — the analysis still runs on the full DataFrame.
# A line chart can't render more than a few thousand points on screen anyway.
MAX_PLOT_POINTS = 10_000


def _minmax_keep_indices(y: np.ndarray, max_points: int) -> np.ndarray:
    """
    Indices of a min/max bucket decimation of `y` down to ~`max_points`.

    The series is split into equal buckets; the min and max of each bucket are
    kept (plus the first and last points). Unlike uniform striding this
    preserves vertical extremes, so spikes/anomalies stay visible.
    """
    n = int(y.shape[0])
    if n <= max_points:
        return np.arange(n)

    n_buckets = max(1, max_points // 2)
    bounds = np.linspace(0, n, n_buckets + 1, dtype=int)
    keep = {0, n - 1}
    for b in range(n_buckets):
        lo, hi = int(bounds[b]), int(bounds[b + 1])
        if hi <= lo:
            continue
        seg = y[lo:hi]
        finite = np.nonzero(np.isfinite(seg))[0]
        if finite.size == 0:
            keep.add(lo)  # all-NaN bucket: keep one point to preserve the gap
            continue
        keep.add(lo + int(finite[np.argmin(seg[finite])]))
        keep.add(lo + int(finite[np.argmax(seg[finite])]))
    return np.array(sorted(keep), dtype=int)


def _downsample_xy(x_series: pd.Series, y_series: pd.Series) -> tuple[List, List]:
    """
    Serialize an (x, y) pair for a trace, decimated to <= MAX_PLOT_POINTS so the
    JSON response stays under Cloud Run's 32 MiB cap. Returns JSON-ready lists.
    """
    x_series = pd.Series(x_series).reset_index(drop=True)
    y_series = pd.Series(y_series).reset_index(drop=True)
    y_num = pd.to_numeric(y_series, errors="coerce").to_numpy(dtype=float)
    keep = _minmax_keep_indices(y_num, MAX_PLOT_POINTS)
    if keep.shape[0] == len(y_series):
        return _serialize_x(x_series), _serialize_y(y_series)
    return _serialize_x(x_series.iloc[keep]), _serialize_y(y_series.iloc[keep])


def build_line_plot(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    xaxis_label: str,
    yaxis_label: str,
    color: str = "blue",
    label: Optional[str] = None,
) -> Dict[str, Any]:
    """Single-line plot — original or processed values over time."""
    x_ser, y_ser = _downsample_xy(df[x_col], df[y_col])
    return {
        "type": "line",
        "traces": [
            {
                "x": x_ser,
                "y": y_ser,
                "color": color,
                "label": label,
                "mode": "lines",
            }
        ],
        "xaxis": {"label": xaxis_label, "type": "datetime"},
        "yaxis": {"label": yaxis_label},
        "title": title,
    }


def build_anomaly_overlay(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    anomaly_mask: np.ndarray,
    title: str,
    yaxis_label: str,
    lang: str = "en",
    base_label: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Two-trace plot: original line + anomaly markers.

    Replicates L1277-1305 (STL) and L1423-1453 (LSTM) anomaly overlays.
    `base_label` overrides the default series label — STL uses "Original data"
    (Python L1287) while LSTM uses "Original value" (Python L1428). Pass the
    pre-localized string to avoid drift.
    """
    if len(anomaly_mask) != len(df):
        raise ValueError(
            tr(
                "anomaly_mask length must match dataframe length.",
                "Länge der Anomalie-Maske muss mit DataFrame übereinstimmen.",
                lang,
            )
        )

    # Markers are sparse — index them from the FULL-resolution series so every
    # anomaly is kept; only the dense base line is decimated for transport.
    full_x = _serialize_x(df[x_col])
    full_y = _serialize_y(df[y_col])
    anomaly_indices = np.where(anomaly_mask)[0]
    anomaly_x = [full_x[i] for i in anomaly_indices]
    anomaly_y = [full_y[i] for i in anomaly_indices]
    base_x, base_y = _downsample_xy(df[x_col], df[y_col])

    series_label = base_label if base_label is not None else tr(
        "Original data", "Originaldaten", lang
    )
    return {
        "type": "multi",
        "traces": [
            {
                "x": base_x,
                "y": base_y,
                "color": "blue",
                "label": series_label,
                "mode": "lines",
            },
            {
                "x": anomaly_x,
                "y": anomaly_y,
                "color": "red",
                "label": tr("Anomalies", "Anomalien", lang),
                "mode": "markers",
            },
        ],
        "xaxis": {"label": "UTC", "type": "datetime"},
        "yaxis": {"label": yaxis_label},
        "title": title,
    }


def build_stl_decomposition(stl_result, time_values: pd.Series, lang: str = "en") -> List[Dict[str, Any]]:
    """
    Four sub-plots for the STL decomposition (L1238-1253):
      observed, trend, seasonal, residual.

    Returns a list — frontend stacks them vertically.
    """
    components = [
        ("observed", stl_result.observed, tr("Observed", "Beobachtet", lang)),
        ("trend", stl_result.trend, tr("Trend", "Trend", lang)),
        ("seasonal", stl_result.seasonal, tr("Seasonal", "Saisonal", lang)),
        ("resid", stl_result.resid, tr("Residual", "Residuum", lang)),
    ]

    plots: List[Dict[str, Any]] = []
    base_title = tr("STL decomposition", "STL-Zerlegung", lang)

    for component_key, series, component_label in components:
        x_ser, y_ser = _downsample_xy(time_values, pd.Series(series))
        plots.append({
            "type": "line",
            "component": component_key,
            "traces": [
                {
                    "x": x_ser,
                    "y": y_ser,
                    "color": "blue",
                    "label": component_label,
                    "mode": "lines",
                }
            ],
            "xaxis": {"label": "UTC", "type": "datetime"},
            "yaxis": {"label": component_label},
            "title": f"{base_title} — {component_label}",
        })

    return plots


def build_lstm_error(results_df: pd.DataFrame, lang: str = "en") -> Dict[str, Any]:
    """
    LSTM absolute-error plot (L1378-1392).

    `results_df` must have columns: timestamp, absolute_error.
    """
    x_ser, y_ser = _downsample_xy(results_df["timestamp"], results_df["absolute_error"])
    return {
        "type": "line",
        "traces": [
            {
                "x": x_ser,
                "y": y_ser,
                "color": "blue",
                "label": tr("Absolute error", "Absoluter Fehler", lang),
                "mode": "lines",
            }
        ],
        "xaxis": {"label": "UTC", "type": "datetime"},
        "yaxis": {"label": tr("Absolute error", "Absoluter Fehler", lang)},
        "title": tr(
            "Absolute error by LSTM-based anomaly detection",
            "Absoluter Fehler durch LSTM-basierte Anomalieerkennung",
            lang,
        ),
    }


def build_slope_plot(df_slope: pd.DataFrame, lang: str = "en") -> Dict[str, Any]:
    """Slope plot (L708-724) — green line of slope values over time."""
    return build_line_plot(
        df_slope,
        x_col=df_slope.columns[0],
        y_col=df_slope.columns[1],
        title=tr(
            "Slope of the original data",
            "Steigung der Originaldaten",
            lang,
        ),
        xaxis_label=df_slope.columns[0],
        yaxis_label=tr("Slope [#/min]", "Steigung [#/min]", lang),
        color="green",
    )


def build_original_plot(df: pd.DataFrame, lang: str = "en") -> Dict[str, Any]:
    """Original data plot (L685-701)."""
    return build_line_plot(
        df,
        x_col=df.columns[0],
        y_col=df.columns[1],
        title=tr("Original data", "Originaldaten", lang),
        xaxis_label=df.columns[0],
        yaxis_label=str(df.columns[1]),
        color="blue",
    )


def build_processed_plot(df: pd.DataFrame, lang: str = "en") -> Dict[str, Any]:
    """Processed data plot (L1547-1565)."""
    return build_line_plot(
        df,
        x_col=df.columns[0],
        y_col=df.columns[1],
        title=tr("Processed data", "Verarbeitete Daten", lang),
        xaxis_label=df.columns[0],
        yaxis_label=str(df.columns[1]),
        color="blue",
    )
