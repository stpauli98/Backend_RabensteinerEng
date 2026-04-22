"""Tests for _validate_and_extract_params in load_data."""
import pytest

from domains.upload.api.load_data import _validate_and_extract_params
from shared.exceptions import MissingParameterError


SAMPLE_CSV = "Vrijeme,Temp,Temp\n2024-01-15 10:00:00,21.5,22.1\n"


def _base_params(**overrides):
    p = {
        'delimiter': ',',
        'timezone': 'UTC',
        'selected_columns': {'column1': '0', 'column2': '1'},
        'custom_date_format': None,
        'value_column_name': '',
        'dropdown_count': 2,
        'has_header': 'ja',
        'uploadId': 'test-id',
    }
    p.update(overrides)
    return p


def test_accepts_integer_string_indices():
    result = _validate_and_extract_params(_base_params(), SAMPLE_CSV)
    assert result['date_column_idx'] == 0
    assert result['value_column_idx'] == 1
    assert result['time_column_idx'] is None
    assert result['has_separate_date_time'] is False


def test_three_column_mode_parses_all_indices():
    params = _base_params(
        dropdown_count=3,
        selected_columns={'column1': '0', 'column2': '1', 'column3': '2'}
    )
    result = _validate_and_extract_params(params, SAMPLE_CSV)
    assert result['date_column_idx'] == 0
    assert result['time_column_idx'] == 1
    assert result['value_column_idx'] == 2
    assert result['has_separate_date_time'] is True


def test_rejects_non_numeric_index():
    params = _base_params(selected_columns={'column1': 'Vrijeme', 'column2': '1'})
    with pytest.raises(MissingParameterError):
        _validate_and_extract_params(params, SAMPLE_CSV)


def test_rejects_null_index_for_required_column():
    params = _base_params(selected_columns={'column1': None, 'column2': '1'})
    with pytest.raises(MissingParameterError):
        _validate_and_extract_params(params, SAMPLE_CSV)


def test_column3_null_is_allowed_in_two_column_mode():
    params = _base_params(
        selected_columns={'column1': '0', 'column2': '1', 'column3': None}
    )
    # Should not raise (column3 is ignored in dropdown_count=2 mode)
    result = _validate_and_extract_params(params, SAMPLE_CSV)
    assert result['time_column_idx'] is None
