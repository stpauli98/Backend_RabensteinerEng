"""
Cloud Validation Services
Input validation and sanitization for cloud operations
"""
import os
import logging
import pandas as pd

from domains.cloud.config import (
    UPLOAD_ID_PATTERN,
    MAX_FILE_SIZE,
    MAX_ROWS,
    MAX_COLUMNS,
    FILE_BUFFER_SIZE,
    CHUNK_DIR
)
from shared.exceptions.errors import (
    CloudException,
    ColumnDetectionError,
    CSVParsingError,
    UploadIdValidationError,
)

logger = logging.getLogger(__name__)


def sanitize_upload_id(upload_id: str) -> str:
    """
    Sanitize upload ID to prevent path traversal attacks.

    Args:
        upload_id: The upload ID to sanitize

    Returns:
        Sanitized upload ID

    Raises:
        UploadIdValidationError: If upload_id contains invalid characters or is empty
    """
    if not upload_id:
        raise UploadIdValidationError(reason="empty")

    if not UPLOAD_ID_PATTERN.match(upload_id):
        logger.error(f"Invalid upload ID format: {upload_id}")
        raise UploadIdValidationError(
            reason="invalid characters (only alphanumeric, hyphen, underscore allowed; max 64 chars)"
        )

    if os.path.sep in upload_id or '/' in upload_id or '\\' in upload_id:
        logger.error(f"Upload ID contains path separators: {upload_id}")
        raise UploadIdValidationError(reason="path separator not allowed")

    return upload_id


def validate_csv_size(file_path: str):
    """
    Validate CSV file size to prevent resource exhaustion.

    Args:
        file_path: Path to the CSV file

    Raises:
        CSVParsingError: If file does not exist
        CloudException: If file exceeds size limits
    """
    if not os.path.exists(file_path):
        raise CSVParsingError(f"File not found: {file_path}")

    size = os.path.getsize(file_path)
    if size > MAX_FILE_SIZE:
        raise CloudException(
            f"File too large: {size / FILE_BUFFER_SIZE:.2f}MB (max {MAX_FILE_SIZE / FILE_BUFFER_SIZE:.0f}MB)",
            error_code='FILE_TOO_LARGE',
            details={'size_mb': size / FILE_BUFFER_SIZE, 'max_mb': MAX_FILE_SIZE / FILE_BUFFER_SIZE},
        )


def validate_dataframe(df: pd.DataFrame, name: str = "DataFrame"):
    """
    Validate DataFrame dimensions to prevent resource exhaustion.

    Args:
        df: DataFrame to validate
        name: Name of the DataFrame for error messages

    Raises:
        CloudException: If DataFrame exceeds size limits
    """
    if len(df) > MAX_ROWS:
        raise CloudException(
            f"{name} has too many rows: {len(df):,} (max {MAX_ROWS:,})",
            error_code='TOO_MANY_ROWS',
            details={'rows': len(df), 'max_rows': MAX_ROWS, 'name': name},
        )

    if len(df.columns) > MAX_COLUMNS:
        raise CloudException(
            f"{name} has too many columns: {len(df.columns)} (max {MAX_COLUMNS})",
            error_code='TOO_MANY_COLUMNS',
            details={'columns': len(df.columns), 'max_columns': MAX_COLUMNS, 'name': name},
        )


def validate_csv_columns(file_path: str, name: str = "File"):
    """
    Validate CSV has required columns by reading only the header line.
    Memory-efficient: does not load the file into a DataFrame.

    Returns:
        Tuple of (column_names, separator)

    Raises:
        CSVParsingError: If file does not exist or is empty
        ColumnDetectionError: If UTC column missing
        CloudException: If column count violates limits
    """
    if not os.path.exists(file_path):
        raise CSVParsingError(f"File not found: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        first_line = f.readline().strip()

    if not first_line:
        raise CSVParsingError(f"{name}: File is empty")

    # Auto-detect separator
    if ';' in first_line:
        sep = ';'
    elif '\t' in first_line:
        sep = '\t'
    else:
        sep = ','

    columns = [col.strip().strip('"') for col in first_line.split(sep)]

    if 'UTC' not in columns:
        raise ColumnDetectionError(
            column_type='UTC',
            available=columns,
        )
    if len(columns) < 2:
        raise CloudException(
            f"{name}: At least 2 columns required (UTC + data), found {len(columns)}",
            error_code='TOO_FEW_COLUMNS',
            details={'columns': columns, 'name': name},
        )
    if len(columns) > MAX_COLUMNS:
        raise CloudException(
            f"{name}: Too many columns: {len(columns)} (max {MAX_COLUMNS})",
            error_code='TOO_MANY_COLUMNS',
            details={'columns': len(columns), 'max_columns': MAX_COLUMNS, 'name': name},
        )

    return columns, sep


def count_csv_rows(file_path: str) -> int:
    """
    Count rows in a CSV file without loading into memory.

    Returns:
        Number of data rows (excludes header)
    """
    count = -1  # Subtract header line
    with open(file_path, 'r', encoding='utf-8') as f:
        for _ in f:
            count += 1
    return max(count, 0)


def get_chunk_dir(upload_id: str) -> str:
    """
    Create and return a directory path for storing chunks of a specific upload.

    Args:
        upload_id: Sanitized upload ID

    Returns:
        Path to chunk directory
    """
    sanitized_id = sanitize_upload_id(upload_id)
    chunk_dir = os.path.join(CHUNK_DIR, sanitized_id)
    os.makedirs(chunk_dir, exist_ok=True)
    return chunk_dir
