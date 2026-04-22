"""Tests for index-based column resolution in _parse_csv_to_dataframe."""
import pytest
import pandas as pd

from domains.upload.api.load_data import _parse_csv_to_dataframe
from shared.exceptions import InvalidColumnIndexError


DUP_CSV = "Vrijeme,Temp,Temp\n2024-01-15 10:00:00,21.5,22.1\n2024-01-15 11:00:00,22.3,23.0\n"
NO_HEADER_CSV = "2024-01-15 10:00:00,21.5\n2024-01-15 11:00:00,22.3\n"


def _params(*, idx1, idx2, has_header='ja', idx3=None):
    if idx3 is not None:
        # 3-column mode
        return {
            'delimiter': ',',
            'has_header': has_header,
            'date_column_idx': idx1,
            'time_column_idx': idx2,
            'value_column_idx': idx3,
            'has_separate_date_time': True,
        }
    return {
        'delimiter': ',',
        'has_header': has_header,
        'date_column_idx': idx1,
        'time_column_idx': None,
        'value_column_idx': idx2,
        'has_separate_date_time': False,
    }


def test_resolves_duplicate_column_by_index():
    """Selecting index 2 on Vrijeme,Temp,Temp returns the second Temp (22.1, 23.0)."""
    validated = _params(idx1=0, idx2=2)
    df, resolved = _parse_csv_to_dataframe(DUP_CSV, validated)
    # pandas auto-renames duplicates
    assert resolved['date_column'] == 'Vrijeme'
    assert resolved['value_column'] == 'Temp.1'
    # Confirm the actual values are from the SECOND Temp column
    assert list(df[resolved['value_column']]) == [22.1, 23.0]


def test_first_temp_when_index_is_1():
    validated = _params(idx1=0, idx2=1)
    df, resolved = _parse_csv_to_dataframe(DUP_CSV, validated)
    assert resolved['value_column'] == 'Temp'
    assert list(df[resolved['value_column']]) == [21.5, 22.3]


def test_out_of_range_index_raises():
    validated = _params(idx1=0, idx2=99)
    with pytest.raises(InvalidColumnIndexError):
        _parse_csv_to_dataframe(DUP_CSV, validated)


def test_no_header_uses_string_index_names():
    validated = _params(idx1=0, idx2=1, has_header='nein')
    df, resolved = _parse_csv_to_dataframe(NO_HEADER_CSV, validated)
    # Without header, pandas columns are '0', '1'
    assert resolved['date_column'] == '0'
    assert resolved['value_column'] == '1'
