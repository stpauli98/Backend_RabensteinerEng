"""
Input validation functions for database operations.

This module provides validation functions for session IDs, file info,
time info, and filename sanitization.
"""

import re
import uuid
from typing import Dict, Any

from .config import DatabaseConfig


def validate_session_id(session_id: str) -> bool:
    """Validate session ID format.

    Accepts:
    - Pure UUID: b2be65df-ce96-4305-b4c7-6530c7bc7096
    - Legacy format: session_1234567890_abc123
    - UUID with prefix: session_b2be65df-ce96-4305-b4c7-6530c7bc7096

    Args:
        session_id: The session ID to validate

    Returns:
        bool: True if valid, False otherwise
    """
    if not session_id or not isinstance(session_id, str):
        return False

    # Check pure UUID
    try:
        uuid.UUID(session_id)
        return True
    except ValueError:
        pass

    # Check session_UUID format (e.g., session_b2be65df-ce96-4305-b4c7-6530c7bc7096)
    if DatabaseConfig.SESSION_UUID_PATTERN.match(session_id):
        return True

    # Check legacy format (e.g., session_1234567890_abc123)
    return bool(re.match(DatabaseConfig.SESSION_ID_PATTERN, session_id))


def validate_file_info(file_info: Dict[str, Any]) -> bool:
    """Validate file info dictionary structure.

    Args:
        file_info: Dictionary containing file information

    Returns:
        bool: True if valid (has required 'fileName' field), False otherwise
    """
    if not isinstance(file_info, dict):
        return False

    required_fields = ['fileName']
    return all(field in file_info for field in required_fields)


def validate_time_info(time_info: Dict[str, Any]) -> bool:
    """Validate time info dictionary structure.

    Checks that boolean fields are actually boolean values.

    Args:
        time_info: Dictionary containing time information

    Returns:
        bool: True if valid, False otherwise
    """
    if not isinstance(time_info, dict):
        return False

    boolean_fields = ['jahr', 'monat', 'woche', 'feiertag', 'tag']
    for field in boolean_fields:
        if field in time_info and not isinstance(time_info[field], bool):
            return False

    return True


def sanitize_filename(filename: str) -> str:
    """Sanitize filename to prevent path traversal attacks.

    Removes path separators and parent directory references.

    Args:
        filename: Original filename from user input

    Returns:
        str: Sanitized filename safe for storage paths
    """
    if not filename:
        return ""
    return re.sub(r'[/\\]|\.\.', '', filename)


# Alias for backward compatibility with internal usage
_sanitize_filename = sanitize_filename
