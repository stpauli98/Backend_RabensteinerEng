"""Keepalive endpoint for Cloud Run instance persistence."""
from flask import Blueprint, jsonify, g
from shared.auth import require_auth

bp = Blueprint('keepalive', __name__)


@bp.route('/<session_id>', methods=['GET'])
@require_auth
def keepalive(session_id):
    """
    Maintains Cloud Run instance activity.
    Called by frontend when browser tab is hidden during long operations.
    Silent endpoint - no logging except errors.
    """
    return jsonify({
        'status': 'alive',
        'session_id': session_id,
        'user_id': g.user_id
    })
