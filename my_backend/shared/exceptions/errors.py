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


class InvalidColumnIndexError(ValidationError):
    """Raised when a column index is out of range for the parsed DataFrame.

    Example:
        raise InvalidColumnIndexError(index=5, max_index=2)
    """

    def __init__(self, index: int, max_index: int, **kwargs):
        message = (
            f"Column index {index} is out of range "
            f"(file has {max_index + 1} columns, valid range: 0..{max_index})"
        )
        kwargs.setdefault('error_code', 'INVALID_COLUMN_INDEX')
        kwargs.setdefault('details', {}).update({
            'index': index,
            'max_index': max_index
        })
        kwargs.setdefault('suggestions', [
            "Re-select the column in the UI dropdown",
            "Verify the file has the expected number of columns"
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


# ============================================================================
# Anomaly Detection Exceptions (Anomalieerkennung domain — /api/adjustmentsOfData/*)
# ============================================================================

class AnomalyException(LoadDataException):
    """Base class for anomaly-pipeline errors.

    Raised when STL/LSTM/SBAD validation or processing fails. Inherits the
    full LoadDataException machinery (error_code, details, suggestions,
    to_dict()) so the route handler can return the same shape as W6 upload
    errors.
    """

    def __init__(self, message: str, **kwargs):
        kwargs.setdefault('error_code', 'ANOMALY_ERROR')
        super().__init__(message, **kwargs)


class STLPeriodAlignmentError(AnomalyException):
    """Raised when STL_PERIOD_H is not an integer multiple of dt_avg."""

    def __init__(self, period_h: float, dt_avg_h: float, **kwargs):
        message = (
            f"STL period ({period_h}h) is not aligned with data timestep "
            f"(dt_avg={dt_avg_h}h). Period must be an integer multiple of dt_avg."
        )
        kwargs.setdefault('error_code', 'STL_PERIOD_NOT_ALIGNED')
        details = kwargs.get('details', {}) or {}
        details.update({'period_h': period_h, 'dt_avg_h': dt_avg_h})
        kwargs['details'] = details
        kwargs.setdefault('suggestions', [
            f"Use a STL period that's a multiple of dt_avg ({dt_avg_h}h)",
            "Check data preprocessing — ensure consistent timestamps",
        ])
        super().__init__(message, **kwargs)


class LSTMHyperparameterCapError(AnomalyException):
    """Raised when an LSTM hyperparameter exceeds its cap."""

    def __init__(self, param: str, value: int, max_value: int, **kwargs):
        message = (
            f"LSTM {param} ({value}) exceeds the maximum allowed ({max_value})."
        )
        kwargs.setdefault('error_code', 'LSTM_HYPERPARAM_OUT_OF_RANGE')
        details = kwargs.get('details', {}) or {}
        details.update({'param': param, 'value': value, 'max': max_value})
        kwargs['details'] = details
        super().__init__(message, **kwargs)


class SBADSlopeRuleError(AnomalyException):
    """Raised when only one of SBAD chg_max / lg_max is provided.

    The SBAD slope rule requires BOTH or NEITHER.
    """

    def __init__(self, provided: str, **kwargs):
        message = (
            f"SBAD slope-based anomaly detection requires BOTH chg_max and "
            f"lg_max. Only '{provided}' was provided."
        )
        kwargs.setdefault('error_code', 'SBAD_SLOPE_REQUIRES_BOTH')
        details = kwargs.get('details', {}) or {}
        details.update({'provided': provided})
        kwargs['details'] = details
        super().__init__(message, **kwargs)


class ThresholdOutOfRangeError(AnomalyException):
    """Raised when STL/LSTM threshold is below the minimum allowed value."""

    def __init__(self, value: float, min_value: float = 0, **kwargs):
        message = (
            f"Threshold ({value}) must be at least {min_value}."
        )
        kwargs.setdefault('error_code', 'THRESHOLD_OUT_OF_RANGE')
        details = kwargs.get('details', {}) or {}
        details.update({'value': value, 'min': min_value})
        kwargs['details'] = details
        super().__init__(message, **kwargs)


class ParameterOutOfRangeError(AnomalyException):
    """Raised when a numeric parameter falls outside its allowed range."""

    def __init__(self, param: str, value: float, min_value: float = None, max_value: float = None, **kwargs):
        if min_value is not None and value < min_value:
            message = f"Parameter '{param}' value {value} is below minimum ({min_value})."
        elif max_value is not None and value > max_value:
            message = f"Parameter '{param}' value {value} exceeds maximum ({max_value})."
        else:
            message = f"Parameter '{param}' value {value} is out of range."
        kwargs.setdefault('error_code', 'PARAM_OUT_OF_RANGE')
        details_base = {'param': param, 'value': value}
        if min_value is not None:
            details_base['min'] = min_value
        if max_value is not None:
            details_base['max'] = max_value
        details = kwargs.get('details', {}) or {}
        details.update(details_base)
        kwargs['details'] = details
        super().__init__(message, **kwargs)


# ============================================================================
# Cloud / Datenwolke Exceptions (regression-based anomaly detection — /api/cloud/*)
# ============================================================================

class CloudException(LoadDataException):
    """Base class for data-cloud regression errors.

    Raised when validation, regression, or interpolation fails in the
    /api/cloud/* namespace. Inherits the full LoadDataException machinery
    (error_code, details, suggestions, to_dict()) so the route handler can
    return the same {ok, error, error_code, details, suggestions} shape as
    W6/W9 endpoints.
    """

    def __init__(self, message: str, **kwargs):
        kwargs.setdefault('error_code', 'CLOUD_ERROR')
        super().__init__(message, **kwargs)


class TimestampMismatchError(CloudException):
    """Raised when predictor and target file timestamps do not overlap."""

    def __init__(self, file1: str = '', file2: str = '', **kwargs):
        message = (
            f"No matching timestamps found between '{file1}' and '{file2}'. "
            f"Please ensure both files cover the same time range."
        )
        kwargs.setdefault('error_code', 'TIMESTAMP_MISMATCH')
        details = kwargs.get('details', {}) or {}
        details.update({'file1': file1, 'file2': file2})
        kwargs['details'] = details
        super().__init__(message, **kwargs)


class InsufficientMatchingPointsError(CloudException):
    """Raised when matched timestamp count is below minimum threshold for regression."""

    def __init__(self, matched: int, minimum: int, **kwargs):
        message = (
            f"Too few matching points: {matched} (minimum: {minimum}). "
            f"Regression requires at least {minimum} overlapping timestamps."
        )
        kwargs.setdefault('error_code', 'INSUFFICIENT_MATCHING_POINTS')
        details = kwargs.get('details', {}) or {}
        details.update({'matched': matched, 'minimum': minimum})
        kwargs['details'] = details
        super().__init__(message, **kwargs)


class ColumnDetectionError(CloudException):
    """Raised when required column (temperature/load/UTC) is missing from file."""

    def __init__(self, column_type: str, available: list = None, **kwargs):
        available_str = str(available) if available else '[]'
        message = (
            f"No valid '{column_type}' column found in file. "
            f"Available columns: {available_str}"
        )
        kwargs.setdefault('error_code', 'COLUMN_NOT_FOUND')
        details = kwargs.get('details', {}) or {}
        details.update({'column_type': column_type, 'available': available or []})
        kwargs['details'] = details
        super().__init__(message, **kwargs)


class ToleranceBoundsEmptyError(CloudException):
    """Raised when tolerance bounds produce no enclosed data points."""

    def __init__(self, tolerance_type: str, **kwargs):
        message = (
            f"No points within tolerance bounds (type: '{tolerance_type}'). "
            f"Try increasing the tolerance values."
        )
        kwargs.setdefault('error_code', 'TOLERANCE_BOUNDS_EMPTY')
        details = kwargs.get('details', {}) or {}
        details.update({'tolerance_type': tolerance_type})
        kwargs['details'] = details
        super().__init__(message, **kwargs)


class UnknownToleranceTypeError(CloudException):
    """Raised when tolerance type code (TR) is not recognized."""

    def __init__(self, provided: str, **kwargs):
        message = f"Unknown tolerance type: '{provided}'. Expected 'cnt' or 'dep'."
        kwargs.setdefault('error_code', 'UNKNOWN_TOLERANCE_TYPE')
        details = kwargs.get('details', {}) or {}
        details.update({'provided': provided})
        kwargs['details'] = details
        super().__init__(message, **kwargs)


class UploadIdValidationError(CloudException):
    """Raised when upload_id contains invalid characters, is empty, or has path separators."""

    def __init__(self, reason: str, **kwargs):
        message = f"Invalid upload ID: {reason}"
        kwargs.setdefault('error_code', 'INVALID_UPLOAD_ID')
        details = kwargs.get('details', {}) or {}
        details.update({'reason': reason})
        kwargs['details'] = details
        super().__init__(message, **kwargs)
