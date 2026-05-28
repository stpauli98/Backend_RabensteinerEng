"""
API Key authentication middleware for forecast endpoint.

Supports two auth methods:
1. API Key: Authorization: Bearer sk_fcst_...
2. JWT Token: Authorization: Bearer eyJ... (falls through to @require_auth)
"""

import hashlib
import logging
from datetime import datetime, timezone
from functools import wraps
from flask import request, jsonify, g

logger = logging.getLogger(__name__)

API_KEY_PREFIX = "sk_fcst_"


def _hash_key(key: str) -> str:
    return hashlib.sha256(key.encode('utf-8')).hexdigest()


def allow_api_key(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth_header = request.headers.get('Authorization', '')

        if not auth_header.startswith('Bearer '):
            return jsonify({'code': 'MISSING_AUTHORIZATION', 'error': 'Missing authorization header'}), 401

        token = auth_header[7:]

        if token.startswith(API_KEY_PREFIX):
            return _authenticate_api_key(token, f, *args, **kwargs)

        from shared.auth.jwt import require_auth
        return require_auth(f)(*args, **kwargs)

    return decorated


def _authenticate_api_key(key, f, *args, **kwargs):
    from shared.database.operations import get_supabase_client

    try:
        supabase = get_supabase_client(use_service_role=True)
        key_hash = _hash_key(key)

        result = supabase.table('api_keys') \
            .select('id, session_id, user_id, expires_at, last_used_at') \
            .eq('key_hash', key_hash) \
            .is_('revoked_at', 'null') \
            .limit(1) \
            .execute()

        if not result.data:
            return jsonify({'error': 'Invalid API key', 'code': 'INVALID_API_KEY'}), 401

        key_row = result.data[0]

        if key_row.get('expires_at'):
            from shared.datetime_utils import parse_iso_datetime
            expires_at = parse_iso_datetime(key_row['expires_at'])
            if expires_at < datetime.now(timezone.utc):
                return jsonify({'error': 'API key expired', 'code': 'API_KEY_EXPIRED'}), 401

        url_session_id = kwargs.get('session_id') or (args[0] if args else None)
        if url_session_id:
            from shared.database.operations import create_or_get_session_uuid
            try:
                uuid_session_id = str(create_or_get_session_uuid(url_session_id, user_id=key_row['user_id']))
            except Exception:
                uuid_session_id = url_session_id

            if str(key_row['session_id']) != uuid_session_id:
                return jsonify({'error': 'API key not valid for this session', 'code': 'KEY_SESSION_MISMATCH'}), 403

        user_check = supabase.table('sessions') \
            .select('user_id') \
            .eq('user_id', key_row['user_id']) \
            .limit(1) \
            .execute()

        if not user_check.data:
            supabase.table('api_keys') \
                .update({'revoked_at': datetime.now(timezone.utc).isoformat()}) \
                .eq('id', key_row['id']) \
                .execute()
            return jsonify({'error': 'User account no longer exists', 'code': 'USER_NOT_FOUND'}), 401

        sub_check = supabase.table('user_subscriptions') \
            .select('id, status') \
            .eq('user_id', key_row['user_id']) \
            .in_('status', ['active', 'trial']) \
            .limit(1) \
            .execute()

        if not sub_check.data:
            return jsonify({'error': 'No active subscription', 'code': 'SUBSCRIPTION_EXPIRED'}), 403

        g.user_id = key_row['user_id']
        g.auth_method = 'api_key'
        g.api_key_id = key_row['id']

        should_update = True
        if key_row.get('last_used_at'):
            from shared.datetime_utils import parse_iso_datetime
            last_used = parse_iso_datetime(key_row['last_used_at'])
            age = (datetime.now(timezone.utc) - last_used).total_seconds()
            should_update = age > 3600

        if should_update:
            supabase.table('api_keys') \
                .update({'last_used_at': datetime.now(timezone.utc).isoformat()}) \
                .eq('id', key_row['id']) \
                .execute()

        logger.info(f"API key auth: user={key_row['user_id'][:8]}..., session={key_row['session_id'][:8]}...")

        return f(*args, **kwargs)

    except Exception as e:
        logger.error(f"API key auth error: {e}")
        return jsonify({'error': 'Authentication failed', 'code': 'AUTH_ERROR'}), 500
