"""
CSV Utilities for Upload Domain
Helper functions for CSV parsing, delimiter detection, and data cleaning
"""
from typing import Any, List, Optional, Dict, Tuple

from domains.upload.config import DEFAULT_DELIMITER, SUPPORTED_DELIMITERS
from domains.upload.services.datetime_parser import datetime_parser


def detect_delimiter(file_content: str, sample_lines: int = 5) -> str:
    """
    Detect CSV delimiter from file content using consistency check.

    A proper delimiter should produce the same number of columns across all lines.

    Args:
        file_content: CSV file content as string
        sample_lines: Number of lines to sample for detection

    Returns:
        Detected delimiter character
    """
    lines = [l for l in file_content.splitlines()[:sample_lines] if l.strip()]

    if not lines:
        return DEFAULT_DELIMITER

    best_delimiter = DEFAULT_DELIMITER
    best_score = -1

    for delimiter in SUPPORTED_DELIMITERS:
        counts = [line.count(delimiter) + 1 for line in lines]

        if not counts or counts[0] < 2:
            continue

        if len(set(counts)) == 1:
            score = counts[0]
            if score > best_score:
                best_score = score
                best_delimiter = delimiter

    return best_delimiter


def clean_time(time_str: Any) -> Any:
    """
    Clean time string by removing invalid characters.

    Keeps only numbers and time separators (: - + . T / and space).
    Example: '00:00:00.000Kdd' -> '00:00:00.000'

    Args:
        time_str: Time string to clean

    Returns:
        Cleaned time string
    """
    if not isinstance(time_str, str):
        return time_str

    cleaned = ''.join(c for c in str(time_str) if c.isdigit() or c in ':-+.T/ ')
    return cleaned


def clean_file_content(file_content: str, delimiter: str) -> str:
    """
    Remove excess delimiters and whitespace from file content.

    Args:
        file_content: Raw CSV file content
        delimiter: CSV delimiter character

    Returns:
        Cleaned file content
    """
    cleaned_lines = [line.rstrip(f"{delimiter};,") for line in file_content.splitlines()]
    return "\n".join(cleaned_lines)


def check_date_format(sample_date: Any) -> Tuple[bool, Optional[Dict[str, str]]]:
    """
    Check if date format is supported.

    Args:
        sample_date: Sample date value to check

    Returns:
        Tuple of (is_supported, error_dict_if_not_supported)
    """
    if not isinstance(sample_date, str):
        sample_date = str(sample_date)

    return datetime_parser.validate_format(sample_date)


def is_format_supported(value: Any, formats: List[str]) -> Tuple[bool, Optional[str]]:
    """
    Check if value matches any of the provided formats.

    Args:
        value: Value to check
        formats: List of datetime format strings

    Returns:
        Tuple of (is_supported, matching_format)
    """
    if not isinstance(value, str):
        value = str(value)

    from domains.upload.services.datetime_parser import DateTimeParser
    parser = DateTimeParser(supported_formats=formats)
    detected_format = parser.detect_format(value)

    if detected_format:
        return True, detected_format
    return False, None


def validate_datetime_format(datetime_str: Any) -> bool:
    """
    Validate if datetime string format is supported.

    Args:
        datetime_str: Datetime string to validate

    Returns:
        True if format is supported, False otherwise
    """
    if not isinstance(datetime_str, str):
        datetime_str = str(datetime_str)

    return datetime_parser.is_supported(datetime_str)
