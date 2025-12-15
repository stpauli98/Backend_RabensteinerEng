"""Tests for shared.database.validators module."""

import pytest

from shared.database.validators import (
    validate_session_id,
    validate_file_info,
    validate_time_info,
    sanitize_filename,
    _sanitize_filename
)


class TestValidateSessionId:
    """Tests for validate_session_id function."""

    def test_valid_pure_uuid(self):
        """Test validation of pure UUID format."""
        assert validate_session_id("b2be65df-ce96-4305-b4c7-6530c7bc7096") is True

    def test_valid_uuid_uppercase(self):
        """Test validation of uppercase UUID."""
        assert validate_session_id("B2BE65DF-CE96-4305-B4C7-6530C7BC7096") is True

    def test_valid_legacy_format(self):
        """Test validation of legacy session format."""
        assert validate_session_id("session_1234567890_abc123") is True

    def test_valid_session_uuid_format(self):
        """Test validation of session_UUID format."""
        assert validate_session_id("session_b2be65df-ce96-4305-b4c7-6530c7bc7096") is True

    def test_invalid_empty_string(self):
        """Test rejection of empty string."""
        assert validate_session_id("") is False

    def test_invalid_none(self):
        """Test rejection of None."""
        assert validate_session_id(None) is False

    def test_invalid_not_string(self):
        """Test rejection of non-string types."""
        assert validate_session_id(12345) is False
        assert validate_session_id([]) is False
        assert validate_session_id({}) is False

    def test_invalid_random_string(self):
        """Test rejection of random strings."""
        assert validate_session_id("not-a-valid-session") is False
        assert validate_session_id("random_string") is False


class TestValidateFileInfo:
    """Tests for validate_file_info function."""

    def test_valid_minimal_file_info(self):
        """Test validation with minimal required fields."""
        file_info = {"fileName": "test.csv"}
        assert validate_file_info(file_info) is True

    def test_valid_full_file_info(self):
        """Test validation with all fields."""
        file_info = {
            "fileName": "test.csv",
            "bezeichnung": "Test File",
            "min": "0",
            "max": "100",
            "type": "input"
        }
        assert validate_file_info(file_info) is True

    def test_invalid_missing_filename(self):
        """Test rejection when fileName is missing."""
        file_info = {"bezeichnung": "Test File"}
        assert validate_file_info(file_info) is False

    def test_invalid_not_dict(self):
        """Test rejection of non-dict types."""
        assert validate_file_info("not a dict") is False
        assert validate_file_info(None) is False
        assert validate_file_info([]) is False

    def test_invalid_empty_dict(self):
        """Test rejection of empty dict."""
        assert validate_file_info({}) is False


class TestValidateTimeInfo:
    """Tests for validate_time_info function."""

    def test_valid_empty_time_info(self):
        """Test validation of empty time info (valid)."""
        assert validate_time_info({}) is True

    def test_valid_full_time_info(self):
        """Test validation with all boolean fields."""
        time_info = {
            "jahr": True,
            "monat": False,
            "woche": True,
            "feiertag": False,
            "tag": True,
            "zeitzone": "UTC"
        }
        assert validate_time_info(time_info) is True

    def test_valid_partial_time_info(self):
        """Test validation with partial fields."""
        time_info = {"jahr": True, "monat": False}
        assert validate_time_info(time_info) is True

    def test_invalid_boolean_field_as_string(self):
        """Test rejection when boolean field is string."""
        time_info = {"jahr": "true"}
        assert validate_time_info(time_info) is False

    def test_invalid_boolean_field_as_int(self):
        """Test rejection when boolean field is int."""
        time_info = {"monat": 1}
        assert validate_time_info(time_info) is False

    def test_invalid_not_dict(self):
        """Test rejection of non-dict types."""
        assert validate_time_info("not a dict") is False
        assert validate_time_info(None) is False
        assert validate_time_info([]) is False


class TestSanitizeFilename:
    """Tests for sanitize_filename function."""

    def test_clean_filename_unchanged(self):
        """Test that clean filenames are unchanged."""
        assert sanitize_filename("test.csv") == "test.csv"
        assert sanitize_filename("my_file_2024.csv") == "my_file_2024.csv"

    def test_remove_forward_slash(self):
        """Test removal of forward slashes."""
        assert sanitize_filename("path/to/file.csv") == "pathtofile.csv"

    def test_remove_backslash(self):
        """Test removal of backslashes."""
        assert sanitize_filename("path\\to\\file.csv") == "pathtofile.csv"

    def test_remove_parent_directory(self):
        """Test removal of parent directory references."""
        assert sanitize_filename("../../../etc/passwd") == "etcpasswd"
        assert sanitize_filename("..\\..\\windows\\system32") == "windowssystem32"

    def test_remove_mixed_traversal(self):
        """Test removal of mixed path traversal attempts."""
        assert sanitize_filename("../../file/../test.csv") == "filetest.csv"

    def test_empty_string(self):
        """Test handling of empty string."""
        assert sanitize_filename("") == ""

    def test_none_input(self):
        """Test handling of None input."""
        assert sanitize_filename(None) == ""

    def test_alias_works(self):
        """Test that _sanitize_filename alias works."""
        assert _sanitize_filename("test/file.csv") == sanitize_filename("test/file.csv")
