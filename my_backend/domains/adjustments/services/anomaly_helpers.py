"""
Anomaly detection helper functions.

Pure utility functions ported from `anomaly_detection_1.py` (L23-225).
NO matplotlib, NO global LANG state — all language-dependent strings go
through `tr(en, de, lang)`.
"""
from __future__ import annotations

import logging
from typing import Callable, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ProgressCallback = Optional[Callable[[str, float], None]]


def tr(en: str, de: str, lang: str) -> str:
    """Return localized string (en/de). Default fallback to en for unknown lang."""
    return de if lang == "de" else en


def rd(value, decimals):
    """Round helper that respects None decimals (no rounding)."""
    if decimals is None:
        return value
    if pd.isna(value):
        return value
    return round(value, decimals)


def hms(dt, fmt: str) -> str:
    """Format duration as HH:MM:SS. fmt='timedelta' for pd.Timedelta, fmt='hours' for float hours."""
    if fmt == "timedelta":
        total_seconds = int(dt.total_seconds())
    elif fmt == "hours":
        total_seconds = int(float(dt) * 3600)
    else:
        raise ValueError(f"Unknown hms fmt: {fmt}")
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def slope_calc(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute slope (dv/dt in #/min) between consecutive rows.

    Vectorized port of L33-54.
    Column 0 must be datetime, column 1 numeric.
    Returns a copy with column 1 replaced by slope; first row is NaN.
    """
    if len(df) == 0:
        return df.copy()

    df_slope = df.copy()
    times = df.iloc[:, 0]
    values = df.iloc[:, 1].astype(float)

    dv = values.diff()
    dt_min = times.diff().dt.total_seconds() / 60.0

    with np.errstate(divide="ignore", invalid="ignore"):
        slope = dv / dt_min

    slope.iloc[0] = np.nan
    df_slope.iloc[:, 1] = slope.values
    return df_slope


def intrpl(
    df: pd.DataFrame,
    gap_max: float,
    dec,
    lang: str = "en",
    progress_callback: ProgressCallback = None,
) -> pd.DataFrame:
    """
    Linear interpolation of NaN gaps shorter than gap_max minutes.

    Port of L56-103. Modifies df in place AND returns it.
    progress_callback(step_label, fraction_0_to_1) is invoked sparingly.
    """
    if gap_max is None:
        return df

    label = tr("Interpolation", "Lineare Interpolation", lang)
    n = len(df)
    if n < 2:
        return df

    frm = 0
    idx_strt: Optional[int] = None

    # Cache columns for speed
    times = df.iloc[:, 0].values
    values = df.iloc[:, 1].values

    last_progress = 0.0
    for i in range(1, n):
        v_prev = values[i - 1]
        v_curr = values[i]

        # NaN detected → open analyzing time window (ATW)
        if pd.isna(v_curr) and not pd.isna(v_prev) and frm == 0:
            idx_strt = i - 1
            frm = 1
        # Closing ATW
        elif not pd.isna(v_curr) and frm == 1:
            idx_end = i
            t_end = pd.Timestamp(times[idx_end])
            t_strt = pd.Timestamp(times[idx_strt])
            gap = (t_end - t_strt).total_seconds() / 60.0

            if gap <= gap_max:
                slope = (values[idx_end] - values[idx_strt]) / gap
                for i_frm in range(idx_strt + 1, idx_end):
                    t_frm = pd.Timestamp(times[i_frm])
                    dt = (t_frm - t_strt).total_seconds() / 60.0
                    new_val = values[idx_strt] + dt * slope
                    values[i_frm] = rd(new_val, dec)

            frm = 0

        if progress_callback is not None:
            progress = i / max(n - 1, 1)
            if progress - last_progress >= 0.05 or i == n - 1:
                progress_callback(label, progress)
                last_progress = progress

    # Write back
    df.iloc[:, 1] = values
    return df


def create_sequences(values: np.ndarray, sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
    """Sliding-window sequence builder for LSTM. Port of L218-225."""
    X, y = [], []
    for i in range(len(values) - sequence_length):
        X.append(values[i : i + sequence_length])
        y.append(values[i + sequence_length])
    return np.array(X), np.array(y)
