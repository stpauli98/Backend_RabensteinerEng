"""
Database exception classes.

This module defines all custom exceptions for database operations,
providing a clear hierarchy for error handling.
"""


class DatabaseError(Exception):
    """Base exception for database operations.

    All database-related exceptions inherit from this class,
    allowing for broad exception catching when needed.
    """
    pass


class SessionNotFoundError(DatabaseError):
    """Raised when a session cannot be found or created.

    This exception is raised when:
    - A session ID cannot be resolved to a UUID
    - A session mapping cannot be created
    - A session does not exist in the database
    """
    pass


class ValidationError(DatabaseError):
    """Raised when input validation fails.

    This exception is raised when:
    - Session ID format is invalid
    - File info structure is invalid
    - Time info structure is invalid
    - Required fields are missing
    """
    pass


class StorageError(DatabaseError):
    """Raised when file storage operations fail.

    This exception is raised when:
    - File upload to Supabase Storage fails
    - File download from Supabase Storage fails
    - Storage bucket operations fail
    """
    pass


class ConfigurationError(DatabaseError):
    """Raised when configuration is invalid.

    This exception is raised when:
    - Supabase client is not available
    - Required environment variables are missing
    - Database connection cannot be established
    """
    pass
