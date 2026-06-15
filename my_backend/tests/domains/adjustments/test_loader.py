"""Unit tests for CSV loader. Verifies all 8 file-validation errors byte-for-byte."""
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from domains.adjustments.data.loader import load_and_validate_csv
from shared.exceptions.errors import AnomalyException

REPO_ROOT = Path(__file__).resolve().parents[3]
TEST2_CSV = REPO_ROOT / "test2" / "test2.csv"


def write_csv(content: str, suffix: str = ".csv") -> Path:
    f = tempfile.NamedTemporaryFile("w", suffix=suffix, delete=False, encoding="utf-8")
    f.write(content)
    f.close()
    return Path(f.name)


# ----------------------------------------------------------------------------
# Happy path
# ----------------------------------------------------------------------------

@pytest.mark.skipif(not TEST2_CSV.exists(), reason="test2.csv fixture missing")
def test_load_test2_returns_dataframe_and_dt_avg():
    df, dt_avg = load_and_validate_csv(TEST2_CSV, lang="en")
    assert len(df) == 14880  # 14881 raw lines − 1 header
    assert df.columns[0] == "UTC"
    assert pd.api.types.is_datetime64_any_dtype(df.iloc[:, 0])
    assert dt_avg.total_seconds() == pytest.approx(180.0)  # 3-min cadence


# ----------------------------------------------------------------------------
# 8 validation errors — exact wording in DE/EN
# ----------------------------------------------------------------------------

def test_wrong_extension_de():
    p = write_csv("UTC;v\n2025-01-01 00:00:00;1.0\n2025-01-01 00:03:00;2.0", suffix=".txt")
    with pytest.raises(ValueError) as exc:
        load_and_validate_csv(p, lang="de")
    assert "Falsche Dateiendung erkannt" in str(exc.value)


def test_wrong_delimiter_en():
    p = write_csv("UTC,v\n2025-01-01 00:00:00,1.0\n2025-01-01 00:03:00,2.0")
    with pytest.raises(ValueError) as exc:
        load_and_validate_csv(p, lang="en")
    assert "Incorrect delimiter detected" in str(exc.value)


def test_wrong_delimiter_de():
    p = write_csv("UTC,v\n2025-01-01 00:00:00,1.0\n2025-01-01 00:03:00,2.0")
    with pytest.raises(ValueError) as exc:
        load_and_validate_csv(p, lang="de")
    assert "Falsches Trennzeichen erkannt" in str(exc.value)


def test_wrong_column_count_en():
    p = write_csv(
        "UTC;v;extra\n"
        "2025-01-01 00:00:00;1.0;x\n"
        "2025-01-01 00:03:00;2.0;y"
    )
    with pytest.raises(ValueError) as exc:
        load_and_validate_csv(p, lang="en")
    assert "Incorrect number of columns detected" in str(exc.value)


def test_wrong_first_column_name_de():
    p = write_csv(
        "Time;v\n"
        "2025-01-01 00:00:00;1.0\n"
        "2025-01-01 00:03:00;2.0"
    )
    with pytest.raises(ValueError) as exc:
        load_and_validate_csv(p, lang="de")
    assert "Ungültiger Spaltenname in der ersten Spalte erkannt" in str(exc.value)


def test_invalid_time_format_en():
    p = write_csv(
        "UTC;v\n"
        "01/01/2025 00:00;1.0\n"
        "01/01/2025 00:03;2.0"
    )
    with pytest.raises(ValueError) as exc:
        load_and_validate_csv(p, lang="en")
    assert "Incorrect time format detected" in str(exc.value)


def test_duplicate_timestamps_en():
    p = write_csv(
        "UTC;v\n"
        "2025-01-01 00:00:00;1.0\n"
        "2025-01-01 00:00:00;2.0"
    )
    with pytest.raises(ValueError) as exc:
        load_and_validate_csv(p, lang="en")
    assert "Duplicate datetime values found" in str(exc.value)


def test_too_few_rows_en():
    p = write_csv("UTC;v\n2025-01-01 00:00:00;1.0")
    with pytest.raises(ValueError) as exc:
        load_and_validate_csv(p, lang="en")
    assert "Incorrect number of rows detected" in str(exc.value)


def test_time_step_drift_en():
    # 3-min steps, then sudden 60-min jump
    rows = ["UTC;v"]
    rows.append("2025-01-01 00:00:00;1.0")
    rows.append("2025-01-01 00:03:00;2.0")
    rows.append("2025-01-01 00:06:00;3.0")
    rows.append("2025-01-01 01:06:00;4.0")  # 60 min skip
    p = write_csv("\n".join(rows))
    with pytest.raises(AnomalyException) as exc:
        load_and_validate_csv(p, lang="en")
    assert "Time step deviates by more than 0.1%" in str(exc.value)
    assert exc.value.error_code == "TIME_GRID_REQUIRED"


def test_path_traversal_rejected():
    p = write_csv("UTC;v\n2025-01-01 00:00:00;1.0\n2025-01-01 00:03:00;2.0")
    other_root = Path(tempfile.mkdtemp())
    with pytest.raises(ValueError) as exc:
        load_and_validate_csv(p, lang="en", allowed_root=other_root)
    assert "outside the allowed upload directory" in str(exc.value)


def test_file_size_cap_rejects_large():
    # Build a tiny file but cap to 1 byte
    p = write_csv("UTC;v\n2025-01-01 00:00:00;1.0\n2025-01-01 00:03:00;2.0")
    with pytest.raises(ValueError) as exc:
        load_and_validate_csv(p, lang="de", max_size_bytes=1)
    assert "Datei zu groß" in str(exc.value)
