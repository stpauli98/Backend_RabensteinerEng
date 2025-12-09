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

logger = logging.getLogger(__name__)


def sanitize_upload_id(upload_id: str) -> str:
    """
    Sanitize upload ID to prevent path traversal attacks.

    Args:
        upload_id: The upload ID to sanitize

    Returns:
        Sanitized upload ID

    Raises:
        ValueError: If upload_id contains invalid characters or is empty
    """
    if not upload_id:
        raise ValueError("Upload ID cannot be empty")

    if not UPLOAD_ID_PATTERN.match(upload_id):
        logger.error(f"Invalid upload ID format: {upload_id}")
        raise ValueError(f"Invalid upload ID format. Only alphanumeric characters, hyphens, and underscores allowed (max 64 chars)")

    if os.path.sep in upload_id or '/' in upload_id or '\\' in upload_id:
        logger.error(f"Upload ID contains path separators: {upload_id}")
        raise ValueError("Upload ID cannot contain path separators")

    return upload_id


def validate_csv_size(file_path: str):
    """
    Validate CSV file size to prevent resource exhaustion.

    Args:
        file_path: Path to the CSV file

    Raises:
        ValueError: If file exceeds size limits
    """
    if not os.path.exists(file_path):
        raise ValueError(f"File not found: {file_path}")

    size = os.path.getsize(file_path)
    if size > MAX_FILE_SIZE:
        raise ValueError(f"File too large: {size / FILE_BUFFER_SIZE:.2f}MB (max {MAX_FILE_SIZE / FILE_BUFFER_SIZE:.0f}MB)")


def validate_dataframe(df: pd.DataFrame, name: str = "DataFrame"):
    """
    Validate DataFrame dimensions to prevent resource exhaustion.

    Args:
        df: DataFrame to validate
        name: Name of the DataFrame for error messages

    Raises:
        ValueError: If DataFrame exceeds size limits
    """
    if len(df) > MAX_ROWS:
        raise ValueError(f"{name} has too many rows: {len(df):,} (max {MAX_ROWS:,})")

    if len(df.columns) > MAX_COLUMNS:
        raise ValueError(f"{name} has too many columns: {len(df.columns)} (max {MAX_COLUMNS})")


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
