"""
API Key management routes.

Endpoints:
- POST /api-keys/<session_id> — generate new API key
- GET /api-keys/<session_id> — list keys for session
- DELETE /api-keys/<key_id> — revoke a key

W11-A T7: hardened with @limiter.limit, UUID guard (only on session_id
sites — NOT on /api-keys/<key_id> DELETE), and the standardized error
contract (shared.responses.errors.error_response).
"""

import re
import secrets
import hashlib
from datetime import datetime, timezone, timedelta

from .common import (
    request, jsonify, g,
    require_auth,
    get_logger, get_supabase_client,
    create_or_get_session_uuid
)
from flask import Blueprint

from core.rate_limits import limiter, training_limit_string
from shared.responses.errors import error_response as _err
from shared.validators.uuid import validate_training_session_format

bp = Blueprint('training_api_keys', __name__)
logger = get_logger(__name__)

API_KEY_PREFIX = "sk_fcst_"
MAX_KEYS_PER_SESSION = 5
KEY_NAME_PATTERN = re.compile(r'^[A-Za-z0-9 _\-()\.]{1,100}$')


def _generate_key():
    random_part = secrets.token_hex(16)
    plaintext = f"{API_KEY_PREFIX}{random_part}"
    key_hash = hashlib.sha256(plaintext.encode('utf-8')).hexdigest()
    prefix = plaintext[:12]
    return plaintext, key_hash, prefix


@bp.route('/api-keys/<session_id>', methods=['POST'])
@limiter.limit(training_limit_string)
@require_auth
def generate_api_key(session_id):
    """Generate a new API key for the session."""
    # W11-BE2: validate session_id BEFORE auth-ed work.
    err = validate_training_session_format(session_id)
    if err:
        return err

    data = request.get_json(silent=True) or {}
    name = data.get('name', '').strip()
    expires_in_days = data.get('expires_in_days')

    if not name:
        return _err('INVALID_API_KEY_NAME', 'Key name is required', 400)
    if len(name) > 100:
        return _err('INVALID_API_KEY_NAME', 'Key name too long (max 100 chars)', 400)
    if not KEY_NAME_PATTERN.match(name):
        return _err(
            'INVALID_API_KEY_NAME',
            'Key name may only contain letters, numbers, spaces, '
            'hyphens, underscores, parentheses, and dots.',
            400,
        )

    try:
        supabase = get_supabase_client(use_service_role=True)
        uuid_session_id = str(create_or_get_session_uuid(session_id, user_id=g.user_id))

        session = supabase.table('sessions') \
            .select('id, user_id, workflow_phase') \
            .eq('id', uuid_session_id) \
            .eq('user_id', g.user_id) \
            .limit(1) \
            .execute()

        if not session.data:
            return _err('SESSION_NOT_FOUND', 'Session not found', 404)

        if session.data[0]['workflow_phase'] not in ('phase4', 'completed'):
            return _err(
                'BAD_REQUEST',
                'Session must have a trained model',
                400,
            )

        existing = supabase.table('api_keys') \
            .select('id, name') \
            .eq('session_id', uuid_session_id) \
            .is_('revoked_at', 'null') \
            .execute()

        if len(existing.data) >= MAX_KEYS_PER_SESSION:
            return _err(
                'API_KEY_LIMIT_EXCEEDED',
                f'Maximum {MAX_KEYS_PER_SESSION} active keys per session',
                400,
            )

        if any(k['name'] == name for k in existing.data):
            return _err(
                'INVALID_API_KEY_NAME',
                f'Key name "{name}" already exists for this session',
                400,
            )

        plaintext, key_hash, prefix = _generate_key()

        expires_at = None
        if expires_in_days and int(expires_in_days) > 0:
            expires_at = (datetime.now(timezone.utc) + timedelta(days=int(expires_in_days))).isoformat()

        result = supabase.table('api_keys').insert({
            'session_id': uuid_session_id,
            'user_id': g.user_id,
            'key_hash': key_hash,
            'key_prefix': prefix,
            'name': name,
            'expires_at': expires_at,
        }).execute()

        logger.info(f"API key generated: session={uuid_session_id[:8]}..., name={name}")

        return jsonify({
            'success': True,
            'key': plaintext,
            'key_id': result.data[0]['id'],
            'name': name,
            'prefix': prefix,
            'expires_at': expires_at
        })

    except Exception:
        logger.exception("Failed to generate API key")
        return _err(
            'INTERNAL_ERROR',
            'Failed to generate API key',
            500,
            suggestion='Please try again. If the problem persists, contact support.',
        )


@bp.route('/api-keys/<session_id>', methods=['GET'])
@limiter.limit(training_limit_string)
@require_auth
def list_api_keys(session_id):
    """List API keys for the given session."""
    # W11-BE2: validate session_id BEFORE auth-ed work.
    err = validate_training_session_format(session_id)
    if err:
        return err

    try:
        supabase = get_supabase_client(use_service_role=True)
        uuid_session_id = str(create_or_get_session_uuid(session_id, user_id=g.user_id))

        result = supabase.table('api_keys') \
            .select('id, name, key_prefix, expires_at, last_used_at, revoked_at, created_at') \
            .eq('session_id', uuid_session_id) \
            .eq('user_id', g.user_id) \
            .order('created_at', desc=True) \
            .execute()

        return jsonify({
            'success': True,
            'keys': result.data
        })

    except Exception:
        logger.exception("Failed to list API keys")
        return _err(
            'INTERNAL_ERROR',
            'Failed to list API keys',
            500,
            suggestion='Please try again. If the problem persists, contact support.',
        )


@bp.route('/api-keys/<key_id>', methods=['DELETE'])
@limiter.limit(training_limit_string)
@require_auth
def revoke_api_key(key_id):
    """Revoke (soft-delete) the API key by row id.

    NOTE: ``key_id`` is the api_keys row id, NOT a session id, so no UUID
    guard is applied here. The DB query below acts as the implicit guard
    (unknown id → no rows → 404).
    """
    try:
        supabase = get_supabase_client(use_service_role=True)

        key_check = supabase.table('api_keys') \
            .select('id, user_id') \
            .eq('id', key_id) \
            .eq('user_id', g.user_id) \
            .is_('revoked_at', 'null') \
            .limit(1) \
            .execute()

        if not key_check.data:
            return _err('API_KEY_NOT_FOUND', 'Key not found or already revoked', 404)

        supabase.table('api_keys') \
            .update({'revoked_at': datetime.now(timezone.utc).isoformat()}) \
            .eq('id', key_id) \
            .execute()

        logger.info(f"API key revoked: {key_id}")

        return jsonify({'success': True, 'revoked': True})

    except Exception:
        logger.exception("Failed to revoke API key")
        return _err(
            'INTERNAL_ERROR',
            'Failed to revoke API key',
            500,
            suggestion='Please try again. If the problem persists, contact support.',
        )
