"""Tests for shared.database.exceptions module."""

import pytest

from shared.database.exceptions import (
    DatabaseError,
    SessionNotFoundError,
    ValidationError,
    StorageError,
    ConfigurationError
)


class TestExceptionHierarchy:
    """Tests for exception class hierarchy."""

    def test_database_error_is_exception(self):
        """Test DatabaseError inherits from Exception."""
        assert issubclass(DatabaseError, Exception)

    def test_session_not_found_error_inherits_from_database_error(self):
        """Test SessionNotFoundError inherits from DatabaseError."""
        assert issubclass(SessionNotFoundError, DatabaseError)

    def test_validation_error_inherits_from_database_error(self):
        """Test ValidationError inherits from DatabaseError."""
        assert issubclass(ValidationError, DatabaseError)

    def test_storage_error_inherits_from_database_error(self):
        """Test StorageError inherits from DatabaseError."""
        assert issubclass(StorageError, DatabaseError)

    def test_configuration_error_inherits_from_database_error(self):
        """Test ConfigurationError inherits from DatabaseError."""
        assert issubclass(ConfigurationError, DatabaseError)


class TestExceptionInstantiation:
    """Tests for exception instantiation and messaging."""

    def test_database_error_with_message(self):
        """Test DatabaseError can be raised with message."""
        error = DatabaseError("Database operation failed")
        assert str(error) == "Database operation failed"

    def test_session_not_found_error_with_message(self):
        """Test SessionNotFoundError can be raised with message."""
        error = SessionNotFoundError("Session abc123 not found")
        assert str(error) == "Session abc123 not found"

    def test_validation_error_with_message(self):
        """Test ValidationError can be raised with message."""
        error = ValidationError("Invalid session_id format")
        assert str(error) == "Invalid session_id format"

    def test_storage_error_with_message(self):
        """Test StorageError can be raised with message."""
        error = StorageError("Failed to upload file")
        assert str(error) == "Failed to upload file"

    def test_configuration_error_with_message(self):
        """Test ConfigurationError can be raised with message."""
        error = ConfigurationError("Supabase client not available")
        assert str(error) == "Supabase client not available"


class TestExceptionCatching:
    """Tests for catching exceptions at different levels."""

    def test_catch_specific_exception(self):
        """Test catching specific exception type."""
        with pytest.raises(SessionNotFoundError):
            raise SessionNotFoundError("Session not found")

    def test_catch_base_database_error(self):
        """Test catching all database errors with base class."""
        with pytest.raises(DatabaseError):
            raise SessionNotFoundError("Session not found")

        with pytest.raises(DatabaseError):
            raise ValidationError("Invalid input")

        with pytest.raises(DatabaseError):
            raise StorageError("Upload failed")

        with pytest.raises(DatabaseError):
            raise ConfigurationError("Missing config")

    def test_differentiate_exception_types(self):
        """Test differentiating between exception types."""
        exceptions = [
            (SessionNotFoundError("test"), "SessionNotFoundError"),
            (ValidationError("test"), "ValidationError"),
            (StorageError("test"), "StorageError"),
            (ConfigurationError("test"), "ConfigurationError"),
        ]

        for exc, expected_name in exceptions:
            assert type(exc).__name__ == expected_name
