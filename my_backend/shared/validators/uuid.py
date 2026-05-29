"""Shared UUID validators.

Extracted from `domains/training/api/forecast_routes.py` (W12-SEC-2) for
reuse across cloud, training, and any future domain that accepts UUID
path params. Always validate UUIDs BEFORE DB lookup to avoid 500 leaks
from DB drivers.
"""

import uuid as uuid_module

from flask import jsonify


def validate_uuid_format(session_id):
    """Return (jsonify, status) tuple if session_id is malformed UUID, else None.

    Use at top of any route to fail fast with 400 BAD_UUID before reaching DB.

    Args:
        session_id: The session UUID string from path param.

    Returns:
        None if valid, else tuple of (Flask Response, 400).
    """
    try:
        uuid_module.UUID(session_id)
    except (ValueError, AttributeError, TypeError):
        return jsonify({
            'success': False,
            'code': 'BAD_UUID',
            'error': 'session_id is not a valid UUID',
        }), 400
    return None
