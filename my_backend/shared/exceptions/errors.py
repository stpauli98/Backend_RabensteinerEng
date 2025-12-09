"""Custom exceptions for the application.

This module provides a comprehensive exception hierarchy for handling
various error scenarios in the data upload and processing pipeline.

Exception Hierarchy:
    LoadDataException (base)
    ├── ValidationError
    │   ├── MissingParameterError
    │   ├── InvalidParameterError
    │   └── DelimiterMismatchError
    ├── ParsingError
    │   ├── CSVParsingError
    │   ├── DateTimeParsingError
    │   └── EncodingError
    ├── UploadError
    │   ├── ChunkUploadError
    │   ├── UploadNotFoundError
    │   └── IncompleteUploadError
    └── TimezoneError
        ├── UnsupportedTimezoneError
        └── TimezoneConversionError
"""

from typing import Dict, List, Optional, Any


class LoadDataException(Exception):
    """Base exception for all load_data operations.

    All custom exceptions inherit from this base class, providing consistent
    error handling with rich context information and actionable suggestions.

    Attributes:
        message: Human-readable error message
        error_code: Machine-readable error code for client handling
        details: Additional error context (dict)
        original_exception: Original exception if this is a wrapper
        suggestions: List of actionable suggestions for resolving the error
    """

    def __init__(
        self,
        message: str,
        error_code: str = "LOAD_DATA_ERROR",
        details: Optional[Dict[str, Any]] = None,
        original_exception: Optional[Exception] = None,
        suggestions: Optional[List[str]] = None
    ):
        """Initialize LoadDataException.

        Args:
            message: Human-readable error message
            error_code: Machine-readable error identifier
            details: Additional context information
            original_exception: Original exception if wrapping another error
            suggestions: List of suggested actions to resolve the error
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        self.original_exception = original_exception
        self.suggestions = suggestions or []

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for JSON response.

        Returns:
            Dictionary suitable for JSON serialization containing error details
        """
        result = {
            "error": self.message,
            "error_code": self.error_code,
            "details": self.details
        }
        if self.suggestions:
            result["suggestions"] = self.suggestions
        return result


# ============================================================================
# Validation Errors
# ============================================================================

class ValidationError(LoadDataException):
    """Base class for validation errors.

    Raised when input parameters or data fail validation checks.
    """

    def __init__(self, message: str, **kwargs):
        kwargs.setdefault('error_code', 'VALIDATION_ERROR')
        super().__init__(message, **kwargs)


class MissingParameterError(ValidationError):
    """Raised when a required parameter is missing from the request.

    Example:
        raise MissingParameterError('delimiter')
    """

    def __init__(self, parameter: str, **kwargs):
        message = f"Missing required parameter: {parameter}"
        kwargs.setdefault('error_code', 'MISSING_PARAMETER')
        kwargs.setdefault('details', {}).update({'parameter': parameter})
        kwargs.setdefault('suggestions', [
            f"Include '{parameter}' in the request parameters",
            "Check the API documentation for required parameters"
        ])
        super().__init__(message, **kwargs)


class InvalidParameterError(ValidationError):
    """Raised when a parameter value is invalid or out of range.

    Example:
        raise InvalidParameterError('delimiter', '|', 'Not detected in file')
    """

    def __init__(
        self,
        parameter: str,
        value: Any,
        reason: Optional[str] = None,
        **kwargs
    ):
        message = f"Invalid value for parameter '{parameter}': {value}"
        if reason:
            message += f" - {reason}"
        kwargs.setdefault('error_code', 'INVALID_PARAMETER')
        kwargs.setdefault('details', {}).update({
            'parameter': parameter,
            'value': str(value),
            'reason': reason
        })
        super().__init__(message, **kwargs)


class DelimiterMismatchError(ValidationError):
    """Raised when provided delimiter doesn't match the detected delimiter.

    Example:
        raise DelimiterMismatchError(provided=',', detected=';')
    """

    def __init__(self, provided: str, detected: str, **kwargs):
        message = f"Delimiter mismatch: provided '{provided}', detected '{detected}'"
        kwargs.setdefault('error_code', 'DELIMITER_MISMATCH')
        kwargs.setdefault('details', {}).update({
            'provided_delimiter': provided,
            'detected_delimiter': detected
        })
        kwargs.setdefault('suggestions', [
            f"Use the detected delimiter '{detected}'",
            "Check the file format and ensure correct delimiter is provided"
        ])
        super().__init__(message, **kwargs)


# ============================================================================
# Parsing Errors
# ============================================================================

class ParsingError(LoadDataException):
    """Base class for data parsing errors.

    Raised when file content cannot be parsed into expected format.
    """

    def __init__(self, message: str, **kwargs):
        kwargs.setdefault('error_code', 'PARSING_ERROR')
        super().__init__(message, **kwargs)


class CSVParsingError(ParsingError):
    """Raised when CSV file structure cannot be parsed.

    Example:
        raise CSVParsingError('No columns to parse from file')
    """

    def __init__(self, reason: str, **kwargs):
        message = f"Failed to parse CSV file: {reason}"
        kwargs.setdefault('error_code', 'CSV_PARSING_ERROR')
        kwargs.setdefault('details', {}).update({'reason': reason})
        kwargs.setdefault('suggestions', [
            "Verify the file is a valid CSV format",
            "Check for encoding issues (UTF-8 recommended)",
            "Ensure consistent delimiter usage throughout the file"
        ])
        super().__init__(message, **kwargs)


class DateTimeParsingError(ParsingError):
    """Raised when datetime values cannot be parsed.

    Example:
        raise DateTimeParsingError(column='timestamp', format_info='Format not supported')
    """

    def __init__(
        self,
        column: Optional[str] = None,
        format_info: Optional[str] = None,
        **kwargs
    ):
        message = "Failed to parse datetime"
        if column:
            message += f" in column '{column}'"
        if format_info:
            message += f": {format_info}"
        kwargs.setdefault('error_code', 'DATETIME_PARSING_ERROR')
        kwargs.setdefault('details', {}).update({
            'column': column,
            'format_info': format_info
        })
        kwargs.setdefault('suggestions', [
            "Check if datetime format is supported",
            "Ensure consistent datetime format throughout the column",
            "Consider using custom datetime format parameter"
        ])
        super().__init__(message, **kwargs)


class EncodingError(ParsingError):
    """Raised when file encoding cannot be determined or decoded.

    Example:
        raise EncodingError('Could not decode file as UTF-8')
    """

    def __init__(self, reason: str, **kwargs):
        message = f"File encoding error: {reason}"
        kwargs.setdefault('error_code', 'ENCODING_ERROR')
        kwargs.setdefault('details', {}).update({'reason': reason})
        kwargs.setdefault('suggestions', [
            "Ensure file is encoded in UTF-8",
            "Try converting file to UTF-8 before upload",
            "Check for special characters in the file"
        ])
        super().__init__(message, **kwargs)


# ============================================================================
# Upload Errors
# ============================================================================

class UploadError(LoadDataException):
    """Base class for file upload errors.

    Raised when chunked upload operations fail.
    """

    def __init__(self, message: str, **kwargs):
        kwargs.setdefault('error_code', 'UPLOAD_ERROR')
        super().__init__(message, **kwargs)


class ChunkUploadError(UploadError):
    """Raised when a specific chunk upload fails.

    Example:
        raise ChunkUploadError('upload-123', 5, 'Chunk data corrupted')
    """

    def __init__(
        self,
        upload_id: str,
        chunk_index: int,
        reason: str,
        **kwargs
    ):
        message = f"Failed to upload chunk {chunk_index} for upload {upload_id}: {reason}"
        kwargs.setdefault('error_code', 'CHUNK_UPLOAD_ERROR')
        kwargs.setdefault('details', {}).update({
            'upload_id': upload_id,
            'chunk_index': chunk_index,
            'reason': reason
        })
        kwargs.setdefault('suggestions', [
            "Retry the chunk upload",
            "Check network connectivity",
            "Verify chunk data integrity"
        ])
        super().__init__(message, **kwargs)


class UploadNotFoundError(UploadError):
    """Raised when an upload ID cannot be found.

    Example:
        raise UploadNotFoundError('upload-123')
    """

    def __init__(self, upload_id: str, **kwargs):
        message = f"Upload not found: {upload_id}"
        kwargs.setdefault('error_code', 'UPLOAD_NOT_FOUND')
        kwargs.setdefault('details', {}).update({'upload_id': upload_id})
        kwargs.setdefault('suggestions', [
            "Verify the upload ID is correct",
            "Check if upload has expired (timeout: 30 minutes)",
            "Start a new upload if necessary"
        ])
        super().__init__(message, **kwargs)


class IncompleteUploadError(UploadError):
    """Raised when attempting to process an incomplete upload.

    Example:
        raise IncompleteUploadError('upload-123', received=8, total=10)
    """

    def __init__(
        self,
        upload_id: str,
        received: int,
        total: int,
        **kwargs
    ):
        message = f"Incomplete upload {upload_id}: received {received}/{total} chunks"
        kwargs.setdefault('error_code', 'INCOMPLETE_UPLOAD')
        kwargs.setdefault('details', {}).update({
            'upload_id': upload_id,
            'received_chunks': received,
            'total_chunks': total,
            'missing_chunks': total - received
        })
        kwargs.setdefault('suggestions', [
            f"Upload remaining {total - received} chunks",
            "Check network connectivity",
            "Retry failed chunk uploads"
        ])
        super().__init__(message, **kwargs)


# ============================================================================
# Timezone Errors
# ============================================================================

class TimezoneError(LoadDataException):
    """Base class for timezone-related errors.

    Raised when timezone conversion or validation fails.
    """

    def __init__(self, message: str, **kwargs):
        kwargs.setdefault('error_code', 'TIMEZONE_ERROR')
        super().__init__(message, **kwargs)


class UnsupportedTimezoneError(TimezoneError):
    """Raised when a timezone identifier is not recognized.

    Example:
        raise UnsupportedTimezoneError('Invalid/Timezone')
    """

    def __init__(self, timezone: str, **kwargs):
        message = f"Unsupported timezone: {timezone}"
        kwargs.setdefault('error_code', 'UNSUPPORTED_TIMEZONE')
        kwargs.setdefault('details', {}).update({'timezone': timezone})
        kwargs.setdefault('suggestions', [
            "Use a standard IANA timezone name (e.g., 'Europe/Belgrade')",
            "Check pytz.all_timezones for list of supported timezones",
            "Use UTC offset format (e.g., '+02:00')"
        ])
        super().__init__(message, **kwargs)


class TimezoneConversionError(TimezoneError):
    """Raised when timezone conversion between zones fails.

    Example:
        raise TimezoneConversionError('UTC', 'Europe/Belgrade', 'Naive datetime')
    """

    def __init__(
        self,
        from_tz: str,
        to_tz: str,
        reason: str,
        **kwargs
    ):
        message = f"Failed to convert timezone from {from_tz} to {to_tz}: {reason}"
        kwargs.setdefault('error_code', 'TIMEZONE_CONVERSION_ERROR')
        kwargs.setdefault('details', {}).update({
            'from_timezone': from_tz,
            'to_timezone': to_tz,
            'reason': reason
        })
        kwargs.setdefault('suggestions', [
            "Verify both timezones are valid",
            "Check datetime values are timezone-aware",
            "Ensure proper datetime format before conversion"
        ])
        super().__init__(message, **kwargs)
