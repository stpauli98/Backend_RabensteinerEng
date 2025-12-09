"""
Tests for refactored helper functions in load_data.py

This module tests the 4 new helper functions extracted during FAZA 2:
- _validate_and_extract_params: Parameter validation and extraction
- _parse_csv_to_dataframe: CSV parsing and DataFrame creation
- _process_datetime_columns: Datetime processing and parsing
- _build_result_dataframe: Final result structure building
"""

import pytest
import pandas as pd
from datetime import datetime
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from api.routes.load_data import (
    _validate_and_extract_params,
    _parse_csv_to_dataframe,
    _process_datetime_columns,
    _build_result_dataframe
)
from shared.exceptions import (
    MissingParameterError,
    DelimiterMismatchError,
    CSVParsingError,
    DateTimeParsingError
)


class TestValidateAndExtractParams:
    """Test _validate_and_extract_params() function for parameter validation."""
    
    def test_valid_params_all_fields(self):
        """Test successful validation with all parameters provided."""
        params = {
            'delimiter': ',',
            'timezone': 'Europe/Berlin',
            'selected_columns': {
                'column1': 'date',
                'column2': 'time',
                'column3': 'value'
            },
            'custom_date_format': '%d.%m.%Y %H:%M:%S',
            'value_column_name': 'Temperature',
            'dropdown_count': '3',
            'has_header': 'ja',
            'uploadId': 'test-upload-123'
        }
        file_content = "date,time,value\n2024-01-01,10:00:00,100"
        
        result = _validate_and_extract_params(params, file_content)
        
        assert result['upload_id'] == 'test-upload-123'
        assert result['delimiter'] == ','
        assert result['timezone'] == 'Europe/Berlin'
        assert result['custom_date_format'] == '%d.%m.%Y %H:%M:%S'
        assert result['value_column_name'] == 'Temperature'
        assert result['has_separate_date_time'] is True
        assert result['has_header'] == 'ja'
        assert result['date_column'] == 'date'
        assert result['time_column'] == 'time'
        assert result['value_column'] == 'value'
    
    def test_valid_params_two_column_mode(self):
        """Test validation with 2-column mode (combined datetime)."""
        params = {
            'delimiter': ',',
            'selected_columns': {
                'column1': 'datetime',
                'column2': 'value'
            },
            'dropdown_count': '2',
            'has_header': 'ja'
        }
        file_content = "datetime,value\n2024-01-01 10:00:00,100"
        
        result = _validate_and_extract_params(params, file_content)
        
        assert result['has_separate_date_time'] is False
        assert result['date_column'] == 'datetime'
        assert result['time_column'] is None
        assert result['value_column'] == 'value'
    
    def test_missing_delimiter(self):
        """Test that missing delimiter raises ValueError."""
        params = {
            'selected_columns': {},
            'has_header': 'ja'
        }
        file_content = "a,b,c\n1,2,3"
        
        with pytest.raises(MissingParameterError):
            _validate_and_extract_params(params, file_content)
    
    def test_delimiter_mismatch(self):
        """Test that delimiter mismatch raises ValueError."""
        params = {
            'delimiter': ';',  # Claimed semicolon
            'selected_columns': {},
            'has_header': 'ja'
        }
        file_content = "a,b,c\n1,2,3"  # Actually comma
        
        with pytest.raises(DelimiterMismatchError):
            _validate_and_extract_params(params, file_content)
    
    def test_default_values(self):
        """Test that default values are applied correctly."""
        params = {
            'delimiter': ',',
            'selected_columns': {}
        }
        file_content = "a,b,c\n1,2,3"
        
        result = _validate_and_extract_params(params, file_content)
        
        assert result['timezone'] == 'UTC'  # Default timezone
        assert result['custom_date_format'] is None
        assert result['value_column_name'] == ''
        assert result['has_separate_date_time'] is False  # Default dropdown_count=2
        assert result['has_header'] is False
    
    def test_value_column_name_whitespace_stripping(self):
        """Test that value_column_name whitespace is stripped."""
        params = {
            'delimiter': ',',
            'value_column_name': '  Temperature  ',
            'selected_columns': {}
        }
        file_content = "a,b,c\n1,2,3"
        
        result = _validate_and_extract_params(params, file_content)
        
        assert result['value_column_name'] == 'Temperature'


class TestParseCSVToDataFrame:
    """Test _parse_csv_to_dataframe() function for CSV parsing."""
    
    def test_parse_csv_with_header(self):
        """Test parsing CSV with header row."""
        file_content = "datetime,value\n2024-01-01,100\n2024-01-02,200"
        validated_params = {
            'delimiter': ',',
            'has_header': 'ja',
            'value_column': 'value'
        }
        
        df = _parse_csv_to_dataframe(file_content, validated_params)
        
        assert not df.empty
        assert len(df) == 2
        assert 'datetime' in df.columns
        assert 'value' in df.columns
        assert df['value'].dtype in ['float64', 'int64']  # Numeric conversion
    
    def test_parse_csv_without_header(self):
        """Test parsing CSV without header (column indices)."""
        file_content = "2024-01-01,100\n2024-01-02,200"
        validated_params = {
            'delimiter': ',',
            'has_header': 'nein',
            'value_column': '1'
        }
        
        df = _parse_csv_to_dataframe(file_content, validated_params)
        
        assert not df.empty
        assert len(df) == 2
        assert '0' in df.columns  # Numeric column names
        assert '1' in df.columns
        assert df['1'].dtype in ['float64', 'int64']
    
    def test_empty_content(self):
        """Test that empty content raises ValueError."""
        file_content = ""
        validated_params = {
            'delimiter': ',',
            'has_header': 'ja',
            'value_column': 'value'
        }
        
        with pytest.raises(CSVParsingError):
            _parse_csv_to_dataframe(file_content, validated_params)
    
    def test_invalid_csv(self):
        """Test that malformed CSV is handled gracefully (pandas fills NaN)."""
        file_content = "a,b\n1,2,3,4,5"  # Inconsistent columns
        validated_params = {
            'delimiter': ',',
            'has_header': 'ja',
            'value_column': 'b'
        }
        
        # Pandas handles inconsistent columns by filling with NaN
        df = _parse_csv_to_dataframe(file_content, validated_params)
        assert not df.empty
        assert 'a' in df.columns
        assert 'b' in df.columns
    
    def test_column_cleaning(self):
        """Test that column names are stripped of whitespace."""
        file_content = " datetime , value \n2024-01-01,100"
        validated_params = {
            'delimiter': ',',
            'has_header': 'ja',
            'value_column': 'value'
        }
        
        df = _parse_csv_to_dataframe(file_content, validated_params)
        
        assert 'datetime' in df.columns  # Whitespace stripped
        assert 'value' in df.columns
        assert ' datetime ' not in df.columns
    
    def test_numeric_conversion_of_value_column(self):
        """Test that value column is converted to numeric."""
        file_content = "datetime,value\n2024-01-01,100\n2024-01-02,invalid"
        validated_params = {
            'delimiter': ',',
            'has_header': 'ja',
            'value_column': 'value'
        }
        
        df = _parse_csv_to_dataframe(file_content, validated_params)
        
        assert df['value'].dtype == 'float64'
        assert pd.notna(df.loc[0, 'value'])
        assert pd.isna(df.loc[1, 'value'])  # Invalid converted to NaN
    
    def test_drop_empty_columns(self):
        """Test that completely empty columns are dropped."""
        file_content = "a,b,c,d\n1,2,,\n3,4,,"
        validated_params = {
            'delimiter': ',',
            'has_header': 'ja',
            'value_column': 'b'
        }
        
        df = _parse_csv_to_dataframe(file_content, validated_params)
        
        # Columns c and d should be dropped (all NaN)
        assert 'a' in df.columns
        assert 'b' in df.columns


class TestProcessDatetimeColumns:
    """Test _process_datetime_columns() function for datetime processing."""
    
    def test_combined_datetime_column(self):
        """Test processing single combined datetime column."""
        df = pd.DataFrame({
            'datetime': ['2024-01-01 10:00:00', '2024-01-02 11:00:00'],
            'value': [100, 200]
        })
        validated_params = {
            'has_separate_date_time': False,
            'date_column': 'datetime',
            'time_column': None,
            'custom_date_format': None
        }
        
        result_df = _process_datetime_columns(df, validated_params)
        
        assert 'datetime' in result_df.columns
        assert pd.api.types.is_datetime64_any_dtype(result_df['datetime'])
        assert len(result_df) == 2
    
    def test_separate_date_time_columns(self):
        """Test processing separate date and time columns."""
        df = pd.DataFrame({
            'date': ['01.01.2024', '02.01.2024'],
            'time': ['10:00:00', '11:00:00'],
            'value': [100, 200]
        })
        validated_params = {
            'has_separate_date_time': True,
            'date_column': 'date',
            'time_column': 'time',
            'custom_date_format': None
        }
        
        result_df = _process_datetime_columns(df, validated_params)
        
        assert 'datetime' in result_df.columns
        assert pd.api.types.is_datetime64_any_dtype(result_df['datetime'])
        assert len(result_df) == 2
    
    def test_custom_date_format(self):
        """Test datetime parsing with custom format."""
        df = pd.DataFrame({
            'datetime': ['01.01.2024 10:00', '02.01.2024 11:00'],
            'value': [100, 200]
        })
        validated_params = {
            'has_separate_date_time': False,
            'date_column': 'datetime',
            'time_column': None,
            'custom_date_format': '%d.%m.%Y %H:%M'
        }
        
        result_df = _process_datetime_columns(df, validated_params)
        
        assert 'datetime' in result_df.columns
        assert pd.api.types.is_datetime64_any_dtype(result_df['datetime'])
    
    def test_unsupported_format_raises_error(self):
        """Test that unsupported format raises ValueError."""
        df = pd.DataFrame({
            'datetime': ['invalid-date-format', 'another-invalid'],
            'value': [100, 200]
        })
        validated_params = {
            'has_separate_date_time': False,
            'date_column': 'datetime',
            'time_column': None,
            'custom_date_format': None
        }
        
        with pytest.raises(DateTimeParsingError):
            _process_datetime_columns(df, validated_params)
    
    def test_separate_datetime_with_invalid_format(self):
        """Test separate date/time with invalid format raises error."""
        df = pd.DataFrame({
            'date': ['completely-invalid-format-xyz', 'another-bad-one'],
            'time': ['also-bad', 'not-a-time'],
            'value': [100, 200]
        })
        validated_params = {
            'has_separate_date_time': True,
            'date_column': 'date',
            'time_column': 'time',
            'custom_date_format': None
        }
        
        with pytest.raises(DateTimeParsingError):
            _process_datetime_columns(df, validated_params)
    
    def test_fallback_to_first_column(self):
        """Test that first column is used if date_column is None."""
        df = pd.DataFrame({
            '0': ['2024-01-01 10:00:00', '2024-01-02 11:00:00'],
            '1': [100, 200]
        })
        validated_params = {
            'has_separate_date_time': False,
            'date_column': None,  # Should use first column
            'time_column': None,
            'custom_date_format': None
        }
        
        result_df = _process_datetime_columns(df, validated_params)
        
        assert 'datetime' in result_df.columns
        assert pd.api.types.is_datetime64_any_dtype(result_df['datetime'])


class TestBuildResultDataFrame:
    """Test _build_result_dataframe() function for result building."""
    
    def test_build_result_with_default_column_name(self):
        """Test building result with default value column name."""
        df = pd.DataFrame({
            'datetime': pd.to_datetime(['2024-01-01 10:00:00', '2024-01-02 11:00:00']),
            'value': [100.5, 200.3]
        })
        validated_params = {
            'value_column': 'value',
            'value_column_name': ''  # Empty = use original
        }
        
        result = _build_result_dataframe(df, validated_params)
        
        assert isinstance(result, list)
        assert len(result) == 3  # Header + 2 data rows
        assert result[0] == ['UTC', 'value']  # Headers
        assert result[1][0] == '2024-01-01 10:00:00'  # UTC timestamp
        assert result[1][1] == '100.5'
    
    def test_build_result_with_custom_column_name(self):
        """Test building result with custom value column name."""
        df = pd.DataFrame({
            'datetime': pd.to_datetime(['2024-01-01 10:00:00']),
            'value': [100.5]
        })
        validated_params = {
            'value_column': 'value',
            'value_column_name': 'Temperature'
        }
        
        result = _build_result_dataframe(df, validated_params)
        
        assert result[0] == ['UTC', 'Temperature']  # Custom name in header
        assert result[1][1] == '100.5'
    
    def test_missing_value_column(self):
        """Test that missing value column raises ValueError."""
        df = pd.DataFrame({
            'datetime': pd.to_datetime(['2024-01-01 10:00:00']),
            'other': [100]
        })
        validated_params = {
            'value_column': 'value',  # Column doesn't exist
            'value_column_name': ''
        }
        
        with pytest.raises(ValueError, match="Datum, Wert 1 oder Wert 2 nicht ausgewählt"):
            _build_result_dataframe(df, validated_params)
    
    def test_null_value_column(self):
        """Test that null value_column raises ValueError."""
        df = pd.DataFrame({
            'datetime': pd.to_datetime(['2024-01-01 10:00:00']),
            'value': [100]
        })
        validated_params = {
            'value_column': None,
            'value_column_name': ''
        }
        
        with pytest.raises(ValueError, match="Datum, Wert 1 oder Wert 2 nicht ausgewählt"):
            _build_result_dataframe(df, validated_params)
    
    def test_sorting_by_utc(self):
        """Test that results are sorted by UTC timestamp."""
        df = pd.DataFrame({
            'datetime': pd.to_datetime([
                '2024-01-03 10:00:00',
                '2024-01-01 10:00:00',
                '2024-01-02 10:00:00'
            ]),
            'value': [300, 100, 200]
        })
        validated_params = {
            'value_column': 'value',
            'value_column_name': ''
        }
        
        result = _build_result_dataframe(df, validated_params)
        
        # Should be sorted: 2024-01-01, 2024-01-02, 2024-01-03
        assert result[1][0] == '2024-01-01 10:00:00'
        assert result[2][0] == '2024-01-02 10:00:00'
        assert result[3][0] == '2024-01-03 10:00:00'
    
    def test_null_value_handling(self):
        """Test that null values are converted to empty strings."""
        df = pd.DataFrame({
            'datetime': pd.to_datetime(['2024-01-01 10:00:00', '2024-01-02 10:00:00']),
            'value': [100.5, None]
        })
        validated_params = {
            'value_column': 'value',
            'value_column_name': ''
        }
        
        result = _build_result_dataframe(df, validated_params)
        
        assert result[1][1] == '100.5'
        assert result[2][1] == ''  # None converted to empty string
    
    def test_drops_rows_with_null_utc(self):
        """Test that rows with null UTC timestamps are dropped."""
        df = pd.DataFrame({
            'datetime': pd.to_datetime(['2024-01-01 10:00:00', None, '2024-01-02 10:00:00']),
            'value': [100, 200, 300]
        })
        validated_params = {
            'value_column': 'value',
            'value_column_name': ''
        }
        
        result = _build_result_dataframe(df, validated_params)
        
        # Should drop the row with None datetime
        assert len(result) == 3  # Header + 2 valid rows


class TestRefactoredHelpersIntegration:
    """Integration tests for refactored helper functions working together."""
    
    def test_full_pipeline_with_header(self):
        """Test complete processing pipeline with header."""
        # Step 1: Validate params
        params = {
            'delimiter': ',',
            'timezone': 'UTC',
            'selected_columns': {
                'column1': 'datetime',
                'column2': 'value'
            },
            'dropdown_count': '2',
            'has_header': 'ja',
            'uploadId': 'test-123'
        }
        file_content = "datetime,value\n2024-01-01 10:00:00,100\n2024-01-02 11:00:00,200"
        
        validated_params = _validate_and_extract_params(params, file_content)
        assert validated_params['delimiter'] == ','
        
        # Step 2: Parse CSV
        df = _parse_csv_to_dataframe(file_content, validated_params)
        assert not df.empty
        assert len(df) == 2
        
        # Step 3: Process datetime
        df = _process_datetime_columns(df, validated_params)
        assert 'datetime' in df.columns
        assert pd.api.types.is_datetime64_any_dtype(df['datetime'])
        
        # Step 4: Build result
        result = _build_result_dataframe(df, validated_params)
        assert len(result) == 3  # Header + 2 rows
        assert result[0] == ['UTC', 'value']
    
    def test_full_pipeline_separate_datetime(self):
        """Test complete pipeline with separate date/time columns."""
        params = {
            'delimiter': ',',
            'selected_columns': {
                'column1': 'date',
                'column2': 'time',
                'column3': 'value'
            },
            'dropdown_count': '3',
            'has_header': 'ja',
            'value_column_name': 'Temperature'
        }
        file_content = "date,time,value\n01.01.2024,10:00:00,25.5\n02.01.2024,11:00:00,26.3"
        
        validated_params = _validate_and_extract_params(params, file_content)
        df = _parse_csv_to_dataframe(file_content, validated_params)
        df = _process_datetime_columns(df, validated_params)
        result = _build_result_dataframe(df, validated_params)
        
        assert result[0] == ['UTC', 'Temperature']
        assert len(result) == 3


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
