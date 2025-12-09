"""
DateTime Parser for Upload Domain
Centralized datetime parsing with format detection and validation
"""
from typing import List, Optional, Tuple, Dict, Any
import pandas as pd

from domains.upload.config import SUPPORTED_DATE_FORMATS
from shared.exceptions import UnsupportedTimezoneError


class DateTimeParser:
    """
    Centralized datetime parsing with format detection and validation.

    Consolidates logic from multiple parsing functions to eliminate duplication
    and provide a single source of truth for datetime operations.
    """

    def __init__(self, supported_formats: Optional[List[str]] = None):
        """
        Initialize parser with supported formats.

        Args:
            supported_formats: List of datetime format strings.
                             If None, uses SUPPORTED_DATE_FORMATS constant.
        """
        self.formats = supported_formats or SUPPORTED_DATE_FORMATS

    def detect_format(self, sample: str) -> Optional[str]:
        """
        Detect which format matches the sample datetime string.

        Args:
            sample: Sample datetime string to check

        Returns:
            Matching format string, or None if no format matches
        """
        if not isinstance(sample, str):
            sample = str(sample)

        sample = sample.strip()

        for fmt in self.formats:
            try:
                pd.to_datetime(sample, format=fmt)
                return fmt
            except (ValueError, TypeError):
                continue

        return None

    def is_supported(self, sample: str) -> bool:
        """
        Check if datetime string format is supported.

        Args:
            sample: Datetime string to validate

        Returns:
            True if format is supported, False otherwise
        """
        return self.detect_format(sample) is not None

    def validate_format(self, sample: str) -> Tuple[bool, Optional[Dict[str, str]]]:
        """
        Validate datetime format with error information.

        Args:
            sample: Sample datetime string to validate

        Returns:
            Tuple of (is_valid, error_dict_if_invalid)
        """
        if self.is_supported(sample):
            return True, None

        return False, {
            "error": "UNSUPPORTED_DATE_FORMAT",
            "message": "Format nicht unterstützt. Beispielwert: %d.%m.%Y %H:%M:%S"
        }

    def parse_series(
        self,
        series: pd.Series,
        custom_format: Optional[str] = None
    ) -> Tuple[bool, Optional[pd.Series], Optional[str]]:
        """
        Parse datetime series using custom or auto-detected format.

        Args:
            series: Pandas Series containing datetime strings
            custom_format: Optional custom datetime format string

        Returns:
            Tuple of (success, parsed_series, error_message)
        """
        try:
            # Clean and prepare series
            clean_series = series.astype(str).str.strip()
            sample_value = clean_series.iloc[0]

            # Try custom format first if provided
            if custom_format:
                try:
                    parsed = pd.to_datetime(clean_series, format=custom_format, errors='coerce')
                    if not parsed.isna().all():
                        return True, parsed, None
                except Exception as e:
                    return False, None, f"Fehler mit custom Format: {str(e)}. Beispielwert: {sample_value}"

            # Try all supported formats
            for fmt in self.formats:
                try:
                    parsed = pd.to_datetime(clean_series, format=fmt, errors='coerce')
                    if not parsed.isna().all():
                        return True, parsed, None
                except Exception:
                    continue

            return False, None, "Format nicht unterstützt. Beispielwert: %d.%m.%Y %H:%M:%S"

        except Exception as e:
            return False, None, f"Fehler beim Parsen: {str(e)}"

    def parse_combined_columns(
        self,
        df: pd.DataFrame,
        date_column: str,
        time_column: str,
        custom_format: Optional[str] = None
    ) -> Tuple[bool, Optional[pd.Series], Optional[str]]:
        """
        Combine separate date and time columns and parse to datetime.

        Args:
            df: DataFrame containing date and time columns
            date_column: Name of the date column
            time_column: Name of the time column
            custom_format: Optional custom datetime format string

        Returns:
            Tuple of (success, parsed_series, error_message)
        """
        try:
            # Combine date + time columns
            combined = (
                df[date_column].astype(str).str.strip() + ' ' +
                df[time_column].astype(str).str.strip()
            )

            # Parse combined series
            return self.parse_series(combined, custom_format)

        except Exception as e:
            return False, None, f"Fehler beim Kombinieren von Datum/Zeit: {str(e)}"

    def convert_to_utc(
        self,
        series: pd.Series,
        source_timezone: str = 'UTC'
    ) -> pd.Series:
        """
        Convert datetime series to UTC timezone.

        If series has no timezone info, localizes to source_timezone first,
        then converts to UTC.

        Args:
            series: Pandas Series with datetime values
            source_timezone: Source timezone (default: 'UTC')

        Returns:
            Series with UTC-converted datetime values

        Raises:
            ValueError: If timezone is not supported
        """
        try:
            # Ensure series is datetime type
            if not pd.api.types.is_datetime64_any_dtype(series):
                series = pd.to_datetime(series, errors='coerce')

            # Localize if no timezone info
            if series.dt.tz is None:
                try:
                    series = series.dt.tz_localize(
                        source_timezone,
                        ambiguous='NaT',
                        nonexistent='NaT'
                    )
                except Exception as e:
                    raise UnsupportedTimezoneError(
                        timezone=source_timezone,
                        original_exception=e
                    )

                # Convert to UTC if not already
                if source_timezone.upper() != 'UTC':
                    series = series.dt.tz_convert('UTC')
            else:
                # Already has timezone, convert to UTC if needed
                if str(series.dt.tz) != 'UTC':
                    series = series.dt.tz_convert('UTC')

            return series

        except Exception:
            raise


# Global parser instance
datetime_parser = DateTimeParser()
