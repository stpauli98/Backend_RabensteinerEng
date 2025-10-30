"""
Tests for helper functions in load_data.py

This module tests utility functions used for CSV parsing and datetime handling:
- clean_time: Time string cleaning
- detect_delimiter: CSV delimiter detection
- clean_file_content: Content cleaning
- check_date_format: Date format validation
- is_format_supported: Format matching
- validate_datetime_format: DateTime validation
"""

import pytest
import pandas as pd
from datetime import datetime
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from api.routes.load_data import (
    clean_time,
    detect_delimiter,
    clean_file_content,
    check_date_format,
    is_format_supported,
    validate_datetime_format,
    SUPPORTED_DATE_FORMATS,
    SUPPORTED_DELIMITERS,
    DEFAULT_DELIMITER
)


class TestCleanTime:
    """Test clean_time() function for removing invalid characters from time strings."""
    
    def test_clean_time_valid_time_string(self):
        """Test that valid time strings pass through unchanged."""
        assert clean_time("10:30:45") == "10:30:45"
        assert clean_time("23:59:59") == "23:59:59"
        assert clean_time("00:00:00") == "00:00:00"
    
    def test_clean_time_with_invalid_characters(self):
        """Test removal of invalid characters from time strings."""
        assert clean_time("00:00:00.000Kdd") == "00:00:00.000"
        assert clean_time("10:30ABC:45XYZ") == "10:30:45"
        assert clean_time("12:34:56.789extra") == "12:34:56.789"
    
    def test_clean_time_non_string_input(self):
        """Test that non-string inputs are returned unchanged."""
        assert clean_time(123) == 123
        assert clean_time(None) is None
        assert clean_time(12.34) == 12.34
    
    def test_clean_time_empty_string(self):
        """Test handling of empty string."""
        assert clean_time("") == ""
    
    def test_clean_time_with_date_separators(self):
        """Test that date/time separators are preserved."""
        assert clean_time("2024-01-01T10:30:45") == "2024-01-01T10:30:45"
        assert clean_time("01.01.2024 10:30") == "01.01.2024 10:30"


class TestDetectDelimiter:
    """Test detect_delimiter() function for CSV delimiter detection."""
    
    def test_detect_delimiter_comma(self):
        """Test detection of comma delimiter."""
        content = "name,age,city\nJohn,25,NYC\nJane,30,LA"
        assert detect_delimiter(content) == ','
    
    def test_detect_delimiter_semicolon(self):
        """Test detection of semicolon delimiter."""
        content = "name;age;city\nJohn;25;NYC\nJane;30;LA"
        assert detect_delimiter(content) == ';'
    
    def test_detect_delimiter_tab(self):
        """Test detection of tab delimiter."""
        content = "name\tage\tcity\nJohn\t25\tNYC\nJane\t30\tLA"
        assert detect_delimiter(content) == '\t'
    
    def test_detect_delimiter_mixed(self):
        """Test delimiter detection with mixed delimiters (majority wins)."""
        # More commas than semicolons
        content = "a,b,c;d\n1,2,3;4\n5,6,7;8"
        assert detect_delimiter(content) == ','
    
    def test_detect_delimiter_no_delimiter(self):
        """Test default delimiter when no delimiter detected."""
        content = "just some text\nwith no delimiters"
        assert detect_delimiter(content) == DEFAULT_DELIMITER
    
    def test_detect_delimiter_sample_lines(self):
        """Test that only specified number of lines are sampled."""
        # First 3 lines have commas, rest have semicolons
        content = "a,b,c\n1,2,3\n4,5,6\na;b;c\n7;8;9"
        assert detect_delimiter(content, sample_lines=3) == ','


class TestCleanFileContent:
    """Test clean_file_content() function for cleaning CSV content."""
    
    def test_clean_file_content_removes_trailing_delimiter(self):
        """Test removal of trailing delimiters."""
        content = "a,b,c,\n1,2,3,"
        cleaned = clean_file_content(content, ',')
        assert cleaned == "a,b,c\n1,2,3"
    
    def test_clean_file_content_removes_multiple_trailing(self):
        """Test removal of multiple trailing delimiters and whitespace."""
        content = "a,b,c,,,\n1,2,3;,,"
        cleaned = clean_file_content(content, ',')
        assert not cleaned.endswith(',')
        assert not cleaned.endswith(';')
    
    def test_clean_file_content_preserves_valid_content(self):
        """Test that valid content is preserved."""
        content = "datetime,value\n2024-01-01,100"
        cleaned = clean_file_content(content, ',')
        assert cleaned == content
    
    def test_clean_file_content_empty_string(self):
        """Test handling of empty string."""
        content = ""
        cleaned = clean_file_content(content, ',')
        assert cleaned == ""


class TestCheckDateFormat:
    """Test check_date_format() function for date format validation."""
    
    def test_check_date_format_supported_formats(self):
        """Test recognition of supported date formats."""
        # Test various supported formats
        valid_dates = [
            "2024-01-01 10:00:00",
            "01.01.2024 10:00",
            "01/01/2024 10:00:00",
            "2024-01-01T10:00:00",
            "10:00:00"
        ]
        
        for date_str in valid_dates:
            is_supported, error = check_date_format(date_str)
            assert is_supported is True, f"Failed for: {date_str}"
            assert error is None
    
    def test_check_date_format_unsupported_format(self):
        """Test rejection of unsupported date formats."""
        invalid_dates = [
            "invalid-date",
            "2024/13/45",
            "not a date at all",
            "32.13.2024"
        ]
        
        for date_str in invalid_dates:
            is_supported, error = check_date_format(date_str)
            assert is_supported is False, f"Should reject: {date_str}"
            assert error is not None
            assert "error" in error
            assert "message" in error
    
    def test_check_date_format_non_string_input(self):
        """Test handling of non-string input (converts to string)."""
        # Should convert to string and check
        is_supported, error = check_date_format(20240101)
        assert isinstance(is_supported, bool)


class TestIsFormatSupported:
    """Test is_format_supported() function for format matching."""
    
    def test_is_format_supported_matching_format(self):
        """Test detection of matching format."""
        value = "2024-01-01 10:00:00"
        is_supported, fmt = is_format_supported(value, SUPPORTED_DATE_FORMATS)
        
        assert is_supported is True
        assert fmt is not None
        assert fmt in SUPPORTED_DATE_FORMATS
    
    def test_is_format_supported_no_match(self):
        """Test when no format matches."""
        value = "invalid-date-format"
        is_supported, fmt = is_format_supported(value, SUPPORTED_DATE_FORMATS)
        
        assert is_supported is False
        assert fmt is None
    
    def test_is_format_supported_custom_formats(self):
        """Test with custom format list."""
        custom_formats = ['%Y-%m-%d', '%d.%m.%Y']
        
        # Should match first format
        is_supported, fmt = is_format_supported("2024-01-01", custom_formats)
        assert is_supported is True
        assert fmt == '%Y-%m-%d'
        
        # Should match second format
        is_supported, fmt = is_format_supported("01.01.2024", custom_formats)
        assert is_supported is True
        assert fmt == '%d.%m.%Y'
        
        # Should not match
        is_supported, fmt = is_format_supported("01/01/2024", custom_formats)
        assert is_supported is False
    
    def test_is_format_supported_non_string_input(self):
        """Test conversion of non-string input."""
        value = 20240101
        is_supported, fmt = is_format_supported(value, SUPPORTED_DATE_FORMATS)
        assert isinstance(is_supported, bool)


class TestValidateDatetimeFormat:
    """Test validate_datetime_format() function."""
    
    def test_validate_datetime_format_valid(self):
        """Test validation of valid datetime strings."""
        valid_datetimes = [
            "2024-01-01 10:00:00",
            "01.01.2024 10:00",
            "2024-01-01T10:00:00",
            "10:00:00"
        ]
        
        for dt_str in valid_datetimes:
            assert validate_datetime_format(dt_str) is True, f"Failed for: {dt_str}"
    
    def test_validate_datetime_format_invalid(self):
        """Test validation rejects invalid formats."""
        invalid_datetimes = [
            "not a datetime",
            "2024/13/45 25:99:99",
            "invalid"
            # Note: Empty string is not tested as pandas may accept it in some contexts
        ]
        
        for dt_str in invalid_datetimes:
            assert validate_datetime_format(dt_str) is False, f"Should reject: {dt_str}"
    
    def test_validate_datetime_format_non_string(self):
        """Test handling of non-string input."""
        # Should convert and validate
        result = validate_datetime_format(20240101100000)
        assert isinstance(result, bool)


class TestHelperFunctionsIntegration:
    """Integration tests for helper functions working together."""
    
    def test_delimiter_detection_and_cleaning(self):
        """Test delimiter detection followed by content cleaning."""
        content = "a,b,c,\n1,2,3,\n4,5,6,"
        
        # Detect delimiter
        delimiter = detect_delimiter(content)
        assert delimiter == ','
        
        # Clean content
        cleaned = clean_file_content(content, delimiter)
        assert not cleaned.endswith(',')
    
    def test_datetime_format_validation_chain(self):
        """Test complete datetime validation chain."""
        test_datetime = "01.01.2024 10:00"
        
        # Check format support
        is_valid = validate_datetime_format(test_datetime)
        assert is_valid is True
        
        # Check with check_date_format
        is_supported, error = check_date_format(test_datetime)
        assert is_supported is True
        assert error is None
        
        # Find matching format
        is_match, fmt = is_format_supported(test_datetime, SUPPORTED_DATE_FORMATS)
        assert is_match is True
        assert fmt == '%d.%m.%Y %H:%M'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
