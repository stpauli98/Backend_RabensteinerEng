"""
CSV file loader for the anomaly detection pipeline.

Validates that the uploaded CSV matches the strict format expected by the
Python reference script (`anomaly_detection_1.py`, L494-622):
  - Extension `.csv`
  - Delimiter `;`
  - Exactly 2 columns
  - First column named `UTC` with format `%Y-%m-%d %H:%M:%S`
  - At least 2 data rows
  - No invalid timestamps, no duplicates
  - Time-step deviation ≤ 0.1 % from mean

Returns the parsed DataFrame plus its mean time step (`pd.Timedelta`) so the
calling layer can pass `dt_avg` to `validate_par_dict` for STL period checks.
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

from domains.adjustments.services.anomaly_helpers import tr
from domains.adjustments.debug_log import dlog
from shared.exceptions.errors import AnomalyException

logger = logging.getLogger(__name__)


_REQ = {
    "FILE_EXT": ".csv",
    "DEL": ";",
    "COLUMNS": 2,
    "ROWS_MIN": 2,
    "UTC_NAME": "UTC",
    "UTC_FMT": "%Y-%m-%d %H:%M:%S",
}


def detect_delimiter(file_path: Path) -> str:
    """Return inferred CSV delimiter (port of L25-28)."""
    with open(file_path, "r", encoding="utf-8") as f:
        sample = f.read(5000)
    return csv.Sniffer().sniff(sample).delimiter


def load_and_validate_csv(
    file_path: Path | str,
    lang: str = "en",
    *,
    max_size_bytes: int = 200 * 1024 * 1024,  # 200 MB hard cap to prevent OOM
    allowed_root: Optional[Path] = None,
) -> Tuple[pd.DataFrame, pd.Timedelta]:
    """
    Load and validate the CSV. Raises ValueError with a localized message on
    any of the 8 failure modes from the reference script.

    Returns (df_sorted, dt_avg) where df has columns [UTC: datetime, value: float].

    Security guards:
      - `max_size_bytes`: rejects files larger than 200 MB (configurable). Prevents
        an attacker from triggering an OOM kill via crafted upload.
      - `allowed_root`: if provided, path must resolve under this directory; blocks
        path traversal (`../etc/passwd`).
    """
    path = Path(file_path).resolve()

    if allowed_root is not None:
        try:
            path.relative_to(Path(allowed_root).resolve())
        except ValueError:
            raise ValueError(
                tr(
                    f"File path '{file_path}' is outside the allowed upload directory.",
                    f"Pfad '{file_path}' liegt außerhalb des erlaubten Upload-Verzeichnisses.",
                    lang,
                )
            )

    if path.stat().st_size > max_size_bytes:
        raise ValueError(
            tr(
                f"File too large ({path.stat().st_size} bytes). Limit: {max_size_bytes} bytes.",
                f"Datei zu groß ({path.stat().st_size} Bytes). Limit: {max_size_bytes} Bytes.",
                lang,
            )
        )

    # 1. file extension
    actual_ext = path.suffix.lower()
    if actual_ext != _REQ["FILE_EXT"]:
        raise ValueError(
            tr(
                f"Incorrect file extension detected: '{actual_ext}'."
                f" Expected: '{_REQ['FILE_EXT']}'. "
                "Please standardize the file before processing!",
                f"Falsche Dateiendung erkannt: '{actual_ext}'. "
                f"Erwartet: '{_REQ['FILE_EXT']}'. "
                "Bitte standardisieren Sie zuerst die Datei!",
                lang,
            )
        )

    # 2. delimiter
    actual_del = detect_delimiter(path)
    dlog("DELIMITER_DETECTED", delimiter=actual_del)
    if actual_del != _REQ["DEL"]:
        raise ValueError(
            tr(
                f"Incorrect delimiter detected: '{actual_del}', "
                f"expected: '{_REQ['DEL']}'. "
                "Please standardize the file before processing!",
                f"Falsches Trennzeichen erkannt: '{actual_del}', "
                f"erwartet: '{_REQ['DEL']}'. "
                "Bitte standardisieren Sie zuerst die Datei!",
                lang,
            )
        )

    df = pd.read_csv(path, sep=_REQ["DEL"], dtype=str)

    # 3. number of columns
    if df.shape[1] != _REQ["COLUMNS"]:
        raise ValueError(
            tr(
                f"Incorrect number of columns detected: {df.shape[1]}, "
                f"expected: {_REQ['COLUMNS']}. "
                "Please standardize the file before processing!",
                f"Falsche Anzahl von Spalten erkannt: {df.shape[1]}, "
                f"erwartet: {_REQ['COLUMNS']}. "
                "Bitte standardisieren Sie zuerst die Datei!",
                lang,
            )
        )

    # 4. row count
    if df.shape[0] < _REQ["ROWS_MIN"]:
        raise ValueError(
            tr(
                f"Incorrect number of rows detected: {df.shape[0]}, "
                f"expected: minimal {_REQ['ROWS_MIN']}",
                f"Falsche Anzahl von Zeilen erkannt: {df.shape[0]}, "
                f"erwartet: mindestens {_REQ['ROWS_MIN']}",
                lang,
            )
        )

    # 5. first column name
    actual_utc = df.columns[0]
    if actual_utc != _REQ["UTC_NAME"]:
        raise ValueError(
            tr(
                "Incorrect name of the first column detected: "
                f"'{actual_utc}', "
                f"expected: '{_REQ['UTC_NAME']}'. "
                "Please standardize the file before processing!",
                "Ungültiger Spaltenname in der ersten Spalte erkannt: "
                f"'{actual_utc}', "
                f"erwarteter Name: '{_REQ['UTC_NAME']}'. "
                "Bitte standardisieren Sie zuerst die Datei!",
                lang,
            )
        )

    # 6. parse UTC timestamps — assign by column name to preserve datetime dtype
    utc_col = df.columns[0]
    df[utc_col] = pd.to_datetime(df[utc_col], format=_REQ["UTC_FMT"], errors="coerce")
    if df[utc_col].isna().any():
        raise ValueError(
            tr(
                f"Incorrect time format detected in the first column, expected: "
                f"'{_REQ['UTC_FMT']}'. "
                "Please standardize the file before processing!",
                "Ungültiger Zeitformat in der ersten Spalte erkannt, erwartet: "
                f"'{_REQ['UTC_FMT']}'. "
                "Bitte standardisieren Sie zuerst die Datei!",
                lang,
            )
        )

    # 7. duplicate timestamps
    if df.iloc[:, 0].duplicated().any():
        raise ValueError(
            tr(
                "Duplicate datetime values found. "
                "Please standardize the file before processing!",
                "Doppelte Datums- und Zeitwerte gefunden. "
                "Bitte standardisieren Sie zuerst die Datei!",
                lang,
            )
        )

    # Sort + numeric coerce
    df = df.sort_values(by=utc_col).reset_index(drop=True)
    value_col = df.columns[1]
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")

    dlog("CSV_PARSED", rows=len(df), cols=list(df.columns))

    # 8. time-step deviation ≤ 0.1 %
    dt_series = df[utc_col].diff().dropna()
    dt_avg = dt_series.mean()
    dt_dev = (dt_series - dt_avg).abs() / dt_avg
    dlog("TIME_GRID_CHECK", dt_avg_s=dt_avg.total_seconds(), n_samples=len(dt_series))
    if (dt_dev > 0.001).any():
        dlog("TIME_GRID_FAIL", dt_dev_max=float(dt_dev.max()), threshold=0.001)
        # Raise as an AnomalyException with a code so the route returns
        # `error_code: TIME_GRID_REQUIRED`. The frontend keys off that code to
        # (a) keep the error persistent instead of auto-dismissing, and
        # (b) show the "configure the time grid first / open Data Adjustments"
        # hint. A plain ValueError would be returned without a code and the UI
        # would silently reset to the upload step.
        raise AnomalyException(
            tr(
                "Time step deviates by more than 0.1% from the mean. "
                "Please configure the time grid first!",
                "Die Zeitschrittweite weicht um mehr als 0,1 % vom Mittelwert "
                "ab. Bitte definieren Sie zuerst das Zeitraster!",
                lang,
            ),
            error_code="TIME_GRID_REQUIRED",
            suggestions=[
                tr(
                    "Open the Data Adjustments page, create a uniform time grid, "
                    "then return here.",
                    "Öffnen Sie die Seite „Datenanpassung“, erstellen Sie ein "
                    "einheitliches Zeitraster und kehren Sie dann hierher zurück.",
                    lang,
                )
            ],
        )
    dlog("TIME_GRID_OK")

    return df, dt_avg
