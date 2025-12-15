"""
Database configuration constants.

This module centralizes all configuration constants for database operations,
including retry settings, timeouts, regex patterns, and default values.
"""

import re


class DatabaseConfig:
    """Technical configuration constants for database operations."""

    # Retry settings
    DEFAULT_RETRY_ATTEMPTS = 3
    DEFAULT_INITIAL_DELAY = 1.0

    # Timeout settings (seconds)
    DEFAULT_TIMEOUT_CONNECT = 30.0
    DEFAULT_TIMEOUT_READ = 60.0
    DEFAULT_TIMEOUT_WRITE = 30.0
    DEFAULT_TIMEOUT_POOL = 30.0

    # Session ID validation patterns
    UUID_PATTERN = re.compile(
        r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$',
        re.IGNORECASE
    )
    SESSION_ID_PATTERN = r'^session_\d+_[a-zA-Z0-9]+$'
    SESSION_UUID_PATTERN = re.compile(
        r'^session_[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$',
        re.IGNORECASE
    )


class DomainDefaults:
    """Business domain default values."""

    SKALIERUNG = "nein"
    MITTELWERTBILDUNG = "nein"
    ZEITZONE = "UTC"
    FILE_TYPE = "input"
    LAND = "Deutschland"


class TableNames:
    """Database table names - centralized to avoid magic strings."""

    SESSIONS = "sessions"
    SESSION_MAPPINGS = "session_mappings"
    TIME_INFO = "time_info"
    ZEITSCHRITTE = "zeitschritte"
    FILES = "files"


class BucketNames:
    """Supabase Storage bucket names."""

    CSV_INPUT = "csv-files"
    CSV_OUTPUT = "aus-csv-files"

    @classmethod
    def get_bucket_for_type(cls, file_type: str) -> str:
        """Get the appropriate bucket name for a file type.

        Args:
            file_type: Either 'input' or 'output'

        Returns:
            str: The bucket name to use
        """
        return cls.CSV_OUTPUT if file_type == 'output' else cls.CSV_INPUT
