"""Detect a loaded file's time step and offset, in minutes.

Single source of truth for the values shown after a file is loaded (e.g. the
anomaly-detection `/load` response). Mirrors the canonical detection in
`services.utils` so the displayed numbers match the file-info elsewhere:

    timestep [min] = round(median(Δt) / 60)
    offset   [min] = first_timestamp.minute % timestep

The input DataFrame's first column must be a parsed UTC datetime column.
"""
from typing import Tuple

import numpy as np
import pandas as pd


def detect_timestep_offset_minutes(df: pd.DataFrame) -> Tuple[float, float]:
    """Return (timestep_min, offset_min) for `df`.

    Returns (0.0, 0.0) when the timestep cannot be determined (e.g. a single
    row), so callers always get plain floats to display.
    """
    utc_col = df.columns[0]
    ts = pd.to_datetime(df[utc_col])

    diffs_sec = np.diff(ts.values.astype("datetime64[s]").astype("int64"))
    if diffs_sec.size == 0:
        return 0.0, 0.0

    timestep = round(float(np.median(diffs_sec)) / 60.0)
    if timestep <= 0:
        return float(timestep), 0.0

    offset = float(ts.iloc[0].minute % timestep)
    return float(timestep), offset
