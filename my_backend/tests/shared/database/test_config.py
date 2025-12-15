"""Tests for shared.database.config module."""

import pytest
import re

from shared.database.config import (
    DatabaseConfig,
    DomainDefaults,
    TableNames,
    BucketNames
)


class TestDatabaseConfig:
    """Tests for DatabaseConfig class."""

    def test_retry_settings(self):
        """Test retry configuration constants."""
        assert DatabaseConfig.DEFAULT_RETRY_ATTEMPTS == 3
        assert DatabaseConfig.DEFAULT_INITIAL_DELAY == 1.0

    def test_timeout_settings(self):
        """Test timeout configuration constants."""
        assert DatabaseConfig.DEFAULT_TIMEOUT_CONNECT == 30.0
        assert DatabaseConfig.DEFAULT_TIMEOUT_READ == 60.0
        assert DatabaseConfig.DEFAULT_TIMEOUT_WRITE == 30.0
        assert DatabaseConfig.DEFAULT_TIMEOUT_POOL == 30.0

    def test_uuid_pattern(self):
        """Test UUID pattern validation."""
        valid_uuid = "b2be65df-ce96-4305-b4c7-6530c7bc7096"
        invalid_uuid = "not-a-uuid"

        assert DatabaseConfig.UUID_PATTERN.match(valid_uuid)
        assert DatabaseConfig.UUID_PATTERN.match(valid_uuid.upper())
        assert not DatabaseConfig.UUID_PATTERN.match(invalid_uuid)

    def test_session_id_pattern(self):
        """Test legacy session ID pattern validation."""
        valid_session = "session_1234567890_abc123"
        invalid_session = "not_a_session"

        assert re.match(DatabaseConfig.SESSION_ID_PATTERN, valid_session)
        assert not re.match(DatabaseConfig.SESSION_ID_PATTERN, invalid_session)

    def test_session_uuid_pattern(self):
        """Test session UUID pattern validation."""
        valid_session_uuid = "session_b2be65df-ce96-4305-b4c7-6530c7bc7096"
        invalid_session_uuid = "session_not-a-uuid"

        assert DatabaseConfig.SESSION_UUID_PATTERN.match(valid_session_uuid)
        assert not DatabaseConfig.SESSION_UUID_PATTERN.match(invalid_session_uuid)


class TestDomainDefaults:
    """Tests for DomainDefaults class."""

    def test_default_values(self):
        """Test default domain values."""
        assert DomainDefaults.SKALIERUNG == "nein"
        assert DomainDefaults.MITTELWERTBILDUNG == "nein"
        assert DomainDefaults.ZEITZONE == "UTC"
        assert DomainDefaults.FILE_TYPE == "input"
        assert DomainDefaults.LAND == "Deutschland"


class TestTableNames:
    """Tests for TableNames class."""

    def test_table_names(self):
        """Test table name constants."""
        assert TableNames.SESSIONS == "sessions"
        assert TableNames.SESSION_MAPPINGS == "session_mappings"
        assert TableNames.TIME_INFO == "time_info"
        assert TableNames.ZEITSCHRITTE == "zeitschritte"
        assert TableNames.FILES == "files"


class TestBucketNames:
    """Tests for BucketNames class."""

    def test_bucket_names(self):
        """Test bucket name constants."""
        assert BucketNames.CSV_INPUT == "csv-files"
        assert BucketNames.CSV_OUTPUT == "aus-csv-files"

    def test_get_bucket_for_type_input(self):
        """Test bucket selection for input files."""
        assert BucketNames.get_bucket_for_type("input") == "csv-files"

    def test_get_bucket_for_type_output(self):
        """Test bucket selection for output files."""
        assert BucketNames.get_bucket_for_type("output") == "aus-csv-files"

    def test_get_bucket_for_type_default(self):
        """Test bucket selection defaults to input bucket."""
        assert BucketNames.get_bucket_for_type("unknown") == "csv-files"
        assert BucketNames.get_bucket_for_type("") == "csv-files"
