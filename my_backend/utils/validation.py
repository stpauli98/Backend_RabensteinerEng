"""
Validation utilities for API routes
Provides session ID validation and standardized response formatting
"""

import re
import uuid
from flask import jsonify


def validate_session_id(session_id):
    """
    Validate session ID format.

    Accepts:
    - Valid UUID format (e.g., "550e8400-e29b-41d4-a716-446655440000")
    - Session string format (e.g., "session_123456_abc123")
    - Session UUID format (e.g., "session_550e8400-e29b-41d4-a716-446655440000")

    Args:
        session_id: Session identifier string to validate

    Returns:
        bool: True if valid format, False otherwise
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
    session_uuid_pattern = r'^session_[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
    if re.match(session_uuid_pattern, session_id, re.IGNORECASE):
        return True

    # Check legacy format (e.g., session_1234567890_abc123)
    legacy_pattern = r'^session_\d+_[a-zA-Z0-9]+$'
    return bool(re.match(legacy_pattern, session_id))


def create_error_response(message, status_code=400):
    """
    Create standardized error response.

    Args:
        message: Error message to return
        status_code: HTTP status code (default: 400)

    Returns:
        tuple: (Flask JSON response, status_code)
    """
    return jsonify({
        'success': False,
        'error': message,
        'data': None
    }), status_code


def create_success_response(data=None, message=None):
    """
    Create standardized success response.

    Args:
        data: Data to return in response
        message: Optional success message

    Returns:
        Flask JSON response with success=True
    """
    response = {
        'success': True,
        'data': data
    }
    if message:
        response['message'] = message
    return jsonify(response)
