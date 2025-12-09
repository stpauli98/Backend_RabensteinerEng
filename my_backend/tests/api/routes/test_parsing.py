"""
Tests for parsing functions in load_data.py

Tests cover:
- cleanup_old_uploads(): Expired upload cleanup
- parse_datetime_column(): DateTime column parsing
- parse_datetime(): Separate date/time column parsing
- convert_to_utc(): Timezone conversion
"""

import pytest
import pandas as pd
import time
from unittest.mock import patch, MagicMock
from datetime import datetime
from io import StringIO

# Import functions to test
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from api.routes.load_data import (
    cleanup_old_uploads,
    parse_datetime_column,
    parse_datetime,
    convert_to_utc,
    chunk_storage,
    UPLOAD_EXPIRY_SECONDS
)
from shared.exceptions import UnsupportedTimezoneError


class TestCleanupOldUploads:
    """Test cleanup_old_uploads() function."""
    
    def test_cleanup_old_uploads_removes_expired(self):
        """Should remove uploads that are older than UPLOAD_EXPIRY_SECONDS."""
        chunk_storage.clear()
        current_time = time.time()
        
        # Create expired upload (older than UPLOAD_EXPIRY_SECONDS)
        chunk_storage['expired_upload'] = {
            'chunks': {},
            'last_activity': current_time - UPLOAD_EXPIRY_SECONDS - 100
        }
        
        # Create active upload
        chunk_storage['active_upload'] = {
            'chunks': {},
            'last_activity': current_time
        }
        
        cleanup_old_uploads()
        
        assert 'expired_upload' not in chunk_storage
        assert 'active_upload' in chunk_storage
        
        chunk_storage.clear()
    
    def test_cleanup_old_uploads_keeps_active(self):
        """Should keep uploads that are still within expiry window."""
        chunk_storage.clear()
        current_time = time.time()
        
        # Create recent upload
        chunk_storage['recent_upload'] = {
            'chunks': {},
            'last_activity': current_time - 100  # 100 seconds ago
        }
        
        cleanup_old_uploads()
        
        assert 'recent_upload' in chunk_storage
        
        chunk_storage.clear()
    
    def test_cleanup_old_uploads_empty_storage(self):
        """Should handle empty chunk_storage gracefully."""
        chunk_storage.clear()
        
        # Should not raise any errors
        cleanup_old_uploads()
        
        assert len(chunk_storage) == 0
    
    def test_cleanup_old_uploads_mixed(self):
        """Should handle mix of expired and active uploads."""
        chunk_storage.clear()
        current_time = time.time()
        
        # Create 3 expired and 2 active uploads
        chunk_storage['expired_1'] = {
            'chunks': {},
            'last_activity': current_time - UPLOAD_EXPIRY_SECONDS - 500
        }
        chunk_storage['expired_2'] = {
            'chunks': {},
            'last_activity': current_time - UPLOAD_EXPIRY_SECONDS - 300
        }
        chunk_storage['expired_3'] = {
            'chunks': {},
            'last_activity': current_time - UPLOAD_EXPIRY_SECONDS - 100
        }
        chunk_storage['active_1'] = {
            'chunks': {},
            'last_activity': current_time - 100
        }
        chunk_storage['active_2'] = {
            'chunks': {},
            'last_activity': current_time
        }
        
        cleanup_old_uploads()
        
        assert 'expired_1' not in chunk_storage
        assert 'expired_2' not in chunk_storage
        assert 'expired_3' not in chunk_storage
        assert 'active_1' in chunk_storage
        assert 'active_2' in chunk_storage
        assert len(chunk_storage) == 2
        
        chunk_storage.clear()


class TestParseDatetimeColumn:
    """Test parse_datetime_column() function."""
    
    def test_parse_datetime_column_custom_format(self):
        """Should parse datetime column with custom format."""
        df = pd.DataFrame({
            'datetime': ['01.01.2024 10:00', '02.01.2024 11:00', '03.01.2024 12:00']
        })
        
        success, parsed_dates, error = parse_datetime_column(
            df, 
            'datetime', 
            custom_format='%d.%m.%Y %H:%M'
        )
        
        assert success is True
        assert parsed_dates is not None
        assert error is None
        assert len(parsed_dates) == 3
        assert not parsed_dates.isna().any()
    
    def test_parse_datetime_column_auto_detect(self):
        """Should auto-detect datetime format from supported formats."""
        df = pd.DataFrame({
            'datetime': ['2024-01-01 10:00:00', '2024-01-02 11:00:00', '2024-01-03 12:00:00']
        })
        
        success, parsed_dates, error = parse_datetime_column(df, 'datetime')
        
        assert success is True
        assert parsed_dates is not None
        assert error is None
        assert not parsed_dates.isna().any()
    
    def test_parse_datetime_column_all_formats_fail(self):
        """Should return error when no format matches."""
        df = pd.DataFrame({
            'datetime': ['invalid_format_xyz', 'not_a_date', 'abc123']
        })
        
        success, parsed_dates, error = parse_datetime_column(df, 'datetime')
        
        assert success is False
        assert parsed_dates is None
        assert error is not None
        assert "Format nicht unterstützt" in error
    
    def test_parse_datetime_column_empty_series(self):
        """Should handle empty or NaN series."""
        df = pd.DataFrame({
            'datetime': [None, None, None]
        })
        
        success, parsed_dates, error = parse_datetime_column(df, 'datetime')
        
        # Should fail because all values are NaN
        assert success is False
    
    def test_parse_datetime_column_multiple_formats(self):
        """Should successfully parse German date format."""
        df = pd.DataFrame({
            'datetime': ['31.12.2024 23:59:59', '01.01.2025 00:00:00']
        })
        
        success, parsed_dates, error = parse_datetime_column(df, 'datetime')
        
        assert success is True
        assert not parsed_dates.isna().any()


class TestParseDatetime:
    """Test parse_datetime() function for separate date/time columns."""
    
    def test_parse_datetime_separate_columns(self):
        """Should combine separate date and time columns."""
        df = pd.DataFrame({
            'date': ['2024-01-01', '2024-01-02'],
            'time': ['10:00:00', '11:00:00']
        })
        
        result = parse_datetime(df, 'date', 'time')
        
        assert isinstance(result, pd.DataFrame)
        assert 'datetime' in result.columns
        assert not result['datetime'].isna().any()
    
    def test_parse_datetime_custom_format(self):
        """Should use custom format for date/time parsing."""
        df = pd.DataFrame({
            'date': ['01.01.2024', '02.01.2024'],
            'time': ['10:00', '11:00']
        })
        
        result = parse_datetime(
            df, 
            'date', 
            'time', 
            custom_format='%d.%m.%Y %H:%M'
        )
        
        assert isinstance(result, pd.DataFrame)
        assert 'datetime' in result.columns
    
    def test_parse_datetime_german_format(self):
        """Should handle German date format (dd.mm.YYYY)."""
        df = pd.DataFrame({
            'date': ['31.12.2024', '01.01.2025'],
            'time': ['23:59:59', '00:00:00']
        })
        
        result = parse_datetime(df, 'date', 'time')
        
        assert isinstance(result, pd.DataFrame)
        assert 'datetime' in result.columns
        assert not result['datetime'].isna().any()
    
    def test_parse_datetime_with_spaces(self):
        """Should handle date/time values with extra whitespace."""
        df = pd.DataFrame({
            'date': ['  2024-01-01  ', '2024-01-02  '],
            'time': ['  10:00:00', '11:00:00  ']
        })
        
        result = parse_datetime(df, 'date', 'time')
        
        assert isinstance(result, pd.DataFrame)
        assert 'datetime' in result.columns


class TestConvertToUTC:
    """Test convert_to_utc() function."""
    
    def test_convert_to_utc_from_different_timezone(self):
        """Should convert from Europe/Vienna to UTC."""
        df = pd.DataFrame({
            'datetime': pd.date_range('2024-01-01 10:00:00', periods=3, freq='H')
        })
        
        result = convert_to_utc(df, 'datetime', timezone='Europe/Vienna')
        
        assert result['datetime'].dt.tz is not None
        assert str(result['datetime'].dt.tz) == 'UTC'
    
    def test_convert_to_utc_already_utc(self):
        """Should handle data that's already in UTC."""
        df = pd.DataFrame({
            'datetime': pd.date_range('2024-01-01 10:00:00', periods=3, freq='H', tz='UTC')
        })
        
        result = convert_to_utc(df, 'datetime', timezone='UTC')
        
        assert str(result['datetime'].dt.tz) == 'UTC'
    
    def test_convert_to_utc_timezone_naive(self):
        """Should localize timezone-naive datetime to specified timezone then convert to UTC."""
        df = pd.DataFrame({
            'datetime': pd.date_range('2024-01-01 10:00:00', periods=3, freq='H')
        })
        
        # Should localize to America/New_York then convert to UTC
        result = convert_to_utc(df, 'datetime', timezone='America/New_York')
        
        assert result['datetime'].dt.tz is not None
        assert str(result['datetime'].dt.tz) == 'UTC'
    
    def test_convert_to_utc_invalid_timezone(self):
        """Should raise ValueError for invalid timezone."""
        df = pd.DataFrame({
            'datetime': pd.date_range('2024-01-01 10:00:00', periods=3, freq='H')
        })
        
        with pytest.raises(UnsupportedTimezoneError):
            convert_to_utc(df, 'datetime', timezone='Invalid/Timezone')
        
    
    def test_convert_to_utc_non_datetime_column(self):
        """Should handle non-datetime column by converting it first."""
        df = pd.DataFrame({
            'datetime': ['2024-01-01 10:00:00', '2024-01-01 11:00:00', '2024-01-01 12:00:00']
        })
        
        result = convert_to_utc(df, 'datetime', timezone='Europe/Vienna')
        
        assert pd.api.types.is_datetime64_any_dtype(result['datetime'])
        assert str(result['datetime'].dt.tz) == 'UTC'


class TestParsingFunctionsIntegration:
    """Integration tests combining multiple parsing functions."""
    
    def test_full_datetime_parsing_pipeline(self):
        """Test complete pipeline: parse datetime column -> convert to UTC."""
        df = pd.DataFrame({
            'datetime': ['2024-01-01 10:00:00', '2024-01-02 11:00:00', '2024-01-03 12:00:00']
        })
        
        # Step 1: Parse datetime column
        success, parsed_dates, error = parse_datetime_column(df, 'datetime')
        assert success is True
        
        df['datetime'] = parsed_dates
        
        # Step 2: Convert to UTC
        result = convert_to_utc(df, 'datetime', timezone='Europe/Vienna')
        
        assert pd.api.types.is_datetime64_any_dtype(result['datetime'])
        assert str(result['datetime'].dt.tz) == 'UTC'
        assert len(result) == 3
    
    def test_separate_columns_to_utc_pipeline(self):
        """Test pipeline: parse separate date/time -> convert to UTC."""
        df = pd.DataFrame({
            'date': ['01.01.2024', '02.01.2024'],
            'time': ['10:00:00', '11:00:00']
        })
        
        # Step 1: Parse separate columns
        result = parse_datetime(df, 'date', 'time')
        assert isinstance(result, pd.DataFrame)
        
        # Step 2: Convert to UTC
        result = convert_to_utc(result, 'datetime', timezone='Europe/Vienna')
        
        assert str(result['datetime'].dt.tz) == 'UTC'
