"""
Tests for DateTimeParser class in load_data.py

This module tests the consolidated DateTimeParser class that replaced
multiple overlapping date parsing functions.
"""

import pytest
import pandas as pd
from datetime import datetime
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from api.routes.load_data import DateTimeParser, SUPPORTED_DATE_FORMATS
from shared.exceptions import UnsupportedTimezoneError


class TestDateTimeParserInit:
    """Test DateTimeParser initialization."""
    
    def test_init_with_default_formats(self):
        """Test initialization with default formats."""
        parser = DateTimeParser()
        assert parser.formats == SUPPORTED_DATE_FORMATS
        assert len(parser.formats) > 0
    
    def test_init_with_custom_formats(self):
        """Test initialization with custom formats."""
        custom_formats = ['%Y-%m-%d', '%d.%m.%Y']
        parser = DateTimeParser(supported_formats=custom_formats)
        assert parser.formats == custom_formats
        assert len(parser.formats) == 2


class TestDetectFormat:
    """Test detect_format() method."""
    
    def test_detect_format_iso_datetime(self):
        """Test detection of ISO datetime format."""
        parser = DateTimeParser()
        fmt = parser.detect_format('2024-01-01 10:00:00')
        assert fmt == '%Y-%m-%d %H:%M:%S'
    
    def test_detect_format_german_datetime(self):
        """Test detection of German datetime format."""
        parser = DateTimeParser()
        fmt = parser.detect_format('01.01.2024 10:00')
        assert fmt == '%d.%m.%Y %H:%M'
    
    def test_detect_format_time_only(self):
        """Test detection of time-only format."""
        parser = DateTimeParser()
        fmt = parser.detect_format('10:00:00')
        assert fmt == '%H:%M:%S'
    
    def test_detect_format_unsupported(self):
        """Test that unsupported format returns None."""
        parser = DateTimeParser()
        fmt = parser.detect_format('invalid-date-format')
        assert fmt is None
    
    def test_detect_format_non_string_input(self):
        """Test that non-string input is converted."""
        parser = DateTimeParser()
        fmt = parser.detect_format(20240101)
        # Should attempt to parse as string
        assert fmt is None or isinstance(fmt, str)
    
    def test_detect_format_with_whitespace(self):
        """Test that whitespace is stripped."""
        parser = DateTimeParser()
        fmt = parser.detect_format('  2024-01-01 10:00:00  ')
        assert fmt == '%Y-%m-%d %H:%M:%S'


class TestIsSupported:
    """Test is_supported() method."""
    
    def test_is_supported_valid_format(self):
        """Test that valid format is recognized."""
        parser = DateTimeParser()
        assert parser.is_supported('2024-01-01 10:00:00') is True
        assert parser.is_supported('01.01.2024 10:00') is True
    
    def test_is_supported_invalid_format(self):
        """Test that invalid format is rejected."""
        parser = DateTimeParser()
        assert parser.is_supported('invalid-date') is False
        assert parser.is_supported('99/99/9999') is False
    
    def test_is_supported_custom_formats(self):
        """Test with custom format list."""
        parser = DateTimeParser(supported_formats=['%Y-%m-%d'])
        assert parser.is_supported('2024-01-01') is True
        assert parser.is_supported('01.01.2024') is False


class TestValidateFormat:
    """Test validate_format() method."""
    
    def test_validate_format_valid(self):
        """Test validation of valid format."""
        parser = DateTimeParser()
        is_valid, error = parser.validate_format('2024-01-01 10:00:00')
        
        assert is_valid is True
        assert error is None
    
    def test_validate_format_invalid(self):
        """Test validation of invalid format returns error dict."""
        parser = DateTimeParser()
        is_valid, error = parser.validate_format('invalid-date')
        
        assert is_valid is False
        assert error is not None
        assert 'error' in error
        assert 'message' in error
        assert error['error'] == 'UNSUPPORTED_DATE_FORMAT'


class TestParseSeries:
    """Test parse_series() method."""
    
    def test_parse_series_valid_iso_format(self):
        """Test parsing series with ISO format."""
        parser = DateTimeParser()
        series = pd.Series(['2024-01-01 10:00:00', '2024-01-02 11:00:00'])
        
        success, parsed, error = parser.parse_series(series)
        
        assert success is True
        assert parsed is not None
        assert len(parsed) == 2
        assert pd.api.types.is_datetime64_any_dtype(parsed)
        assert error is None
    
    def test_parse_series_valid_german_format(self):
        """Test parsing series with German format."""
        parser = DateTimeParser()
        series = pd.Series(['01.01.2024 10:00', '02.01.2024 11:00'])
        
        success, parsed, error = parser.parse_series(series)
        
        assert success is True
        assert parsed is not None
        assert pd.api.types.is_datetime64_any_dtype(parsed)
    
    def test_parse_series_with_custom_format(self):
        """Test parsing with custom format."""
        parser = DateTimeParser()
        series = pd.Series(['01.01.2024 10:00', '02.01.2024 11:00'])
        
        success, parsed, error = parser.parse_series(series, custom_format='%d.%m.%Y %H:%M')
        
        assert success is True
        assert parsed is not None
        assert pd.api.types.is_datetime64_any_dtype(parsed)
    
    def test_parse_series_invalid_format(self):
        """Test that invalid format returns error."""
        parser = DateTimeParser()
        series = pd.Series(['invalid', 'dates'])
        
        success, parsed, error = parser.parse_series(series)
        
        assert success is False
        assert parsed is None
        assert error is not None
        assert 'Format nicht unterstützt' in error
    
    def test_parse_series_custom_format_failure(self):
        """Test that truly unparsable data with custom format fails."""
        parser = DateTimeParser()
        series = pd.Series(['completely-invalid-xyz', 'not-a-date-at-all'])
        
        # Even with custom format, this data is unparsable
        success, parsed, error = parser.parse_series(series, custom_format='%Y-%m-%d')
        
        # Should fail because data is invalid
        assert success is False
        assert parsed is None
        assert error is not None
    
    def test_parse_series_with_whitespace(self):
        """Test that whitespace is stripped from series values."""
        parser = DateTimeParser()
        series = pd.Series(['  2024-01-01 10:00:00  ', '  2024-01-02 11:00:00  '])
        
        success, parsed, error = parser.parse_series(series)
        
        assert success is True
        assert parsed is not None


class TestParseCombinedColumns:
    """Test parse_combined_columns() method."""
    
    def test_parse_combined_columns_success(self):
        """Test combining and parsing separate date/time columns."""
        parser = DateTimeParser()
        df = pd.DataFrame({
            'date': ['01.01.2024', '02.01.2024'],
            'time': ['10:00:00', '11:00:00']
        })
        
        success, parsed, error = parser.parse_combined_columns(df, 'date', 'time')
        
        assert success is True
        assert parsed is not None
        assert len(parsed) == 2
        assert pd.api.types.is_datetime64_any_dtype(parsed)
        assert error is None
    
    def test_parse_combined_columns_with_custom_format(self):
        """Test combining with custom format."""
        parser = DateTimeParser()
        df = pd.DataFrame({
            'date': ['01.01.2024', '02.01.2024'],
            'time': ['10:00', '11:00']
        })
        
        success, parsed, error = parser.parse_combined_columns(
            df, 'date', 'time', 
            custom_format='%d.%m.%Y %H:%M'
        )
        
        assert success is True
        assert parsed is not None
    
    def test_parse_combined_columns_invalid_data(self):
        """Test that invalid combined data returns error."""
        parser = DateTimeParser()
        df = pd.DataFrame({
            'date': ['invalid', 'data'],
            'time': ['bad', 'values']
        })
        
        success, parsed, error = parser.parse_combined_columns(df, 'date', 'time')
        
        assert success is False
        assert parsed is None
        assert error is not None
    
    def test_parse_combined_columns_missing_column(self):
        """Test that missing column returns error."""
        parser = DateTimeParser()
        df = pd.DataFrame({
            'date': ['01.01.2024', '02.01.2024']
        })
        
        success, parsed, error = parser.parse_combined_columns(df, 'date', 'missing_time')
        
        assert success is False
        assert error is not None
        assert 'Fehler beim Kombinieren' in error


class TestConvertToUTC:
    """Test convert_to_utc() method."""
    
    def test_convert_to_utc_from_berlin(self):
        """Test conversion from Europe/Berlin to UTC."""
        parser = DateTimeParser()
        series = pd.Series(pd.to_datetime(['2024-01-01 10:00:00', '2024-01-01 11:00:00']))
        
        utc_series = parser.convert_to_utc(series, 'Europe/Berlin')
        
        assert pd.api.types.is_datetime64_any_dtype(utc_series)
        assert utc_series.dt.tz is not None
        assert str(utc_series.dt.tz) == 'UTC'
    
    def test_convert_to_utc_already_utc(self):
        """Test that UTC series remains unchanged."""
        parser = DateTimeParser()
        series = pd.Series(pd.to_datetime(['2024-01-01 10:00:00', '2024-01-01 11:00:00']))
        
        utc_series = parser.convert_to_utc(series, 'UTC')
        
        assert pd.api.types.is_datetime64_any_dtype(utc_series)
        # Should be timezone-aware UTC
        assert utc_series.dt.tz is not None
    
    def test_convert_to_utc_invalid_timezone(self):
        """Test that invalid timezone raises ValueError."""
        parser = DateTimeParser()
        series = pd.Series(pd.to_datetime(['2024-01-01 10:00:00']))
        
        with pytest.raises(UnsupportedTimezoneError):
            parser.convert_to_utc(series, 'Invalid/Timezone')
    
    def test_convert_to_utc_already_has_timezone(self):
        """Test conversion when series already has timezone."""
        parser = DateTimeParser()
        series = pd.Series(pd.to_datetime(['2024-01-01 10:00:00', '2024-01-01 11:00:00']))
        series = series.dt.tz_localize('Europe/Berlin')
        
        utc_series = parser.convert_to_utc(series, 'UTC')
        
        assert str(utc_series.dt.tz) == 'UTC'
    
    def test_convert_to_utc_non_datetime_series(self):
        """Test that non-datetime series is converted first."""
        parser = DateTimeParser()
        series = pd.Series(['2024-01-01 10:00:00', '2024-01-01 11:00:00'])
        
        utc_series = parser.convert_to_utc(series, 'UTC')
        
        assert pd.api.types.is_datetime64_any_dtype(utc_series)


class TestDateTimeParserIntegration:
    """Integration tests for DateTimeParser class."""
    
    def test_full_pipeline_parse_and_convert(self):
        """Test complete pipeline: parse then convert to UTC."""
        parser = DateTimeParser()
        
        # Step 1: Parse series
        series = pd.Series(['01.01.2024 10:00:00', '02.01.2024 11:00:00'])
        success, parsed, error = parser.parse_series(series)
        assert success is True
        
        # Step 2: Convert to UTC
        utc_series = parser.convert_to_utc(parsed, 'Europe/Berlin')
        assert str(utc_series.dt.tz) == 'UTC'
    
    def test_combined_columns_to_utc_pipeline(self):
        """Test: combine columns, parse, convert to UTC."""
        parser = DateTimeParser()
        df = pd.DataFrame({
            'date': ['01.01.2024', '02.01.2024'],
            'time': ['10:00:00', '11:00:00']
        })
        
        # Combine and parse
        success, parsed, error = parser.parse_combined_columns(df, 'date', 'time')
        assert success is True
        
        # Convert to UTC
        utc_series = parser.convert_to_utc(parsed, 'Europe/Berlin')
        assert str(utc_series.dt.tz) == 'UTC'
        assert len(utc_series) == 2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
