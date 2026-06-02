"""Shared helpers for the standardized W10/W11/W12 error response contract.

All 4xx/5xx error responses across the project should return:

    {
      "success": false,
      "code": "<MACHINE_READABLE_CODE>",
      "error": "<user-safe message>",
      "suggestion": "<optional follow-up>"   # optional
    }

Use :func:`error_response` to construct these payloads without
re-deriving the dict layout in every route handler.
"""

from typing import Optional

from flask import jsonify


def error_response(
    code: str,
    message: str,
    status: int,
    *,
    suggestion: Optional[str] = None,
):
    """Build a standardized error response tuple.

    Args:
        code: Machine-readable error code (e.g. ``BAD_UUID``, ``MISSING_FILE``).
        message: User-safe message string. Do NOT include exception text or PII.
        status: HTTP status code (400, 403, 404, 409, 422, 429, 500).
        suggestion: Optional follow-up the FE may surface (keyword-only).

    Returns:
        Flask ``(Response, status)`` tuple ready to be returned from a handler.
    """
    payload = {
        'success': False,
        'code': code,
        'error': message,
    }
    if suggestion:
        payload['suggestion'] = suggestion
    return jsonify(payload), status
