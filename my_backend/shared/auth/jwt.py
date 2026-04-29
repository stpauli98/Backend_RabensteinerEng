"""Authentication middleware for Supabase JWT validation.

JWTs are verified locally with HS256 against SUPABASE_JWT_SECRET (Supabase
Dashboard -> Settings -> API -> JWT Secret). This avoids a per-request
network round-trip to Supabase's /auth/v1/user endpoint and removes the
runtime dependency on Supabase availability for every authenticated call.
"""
import logging
import os
from functools import wraps

import jwt as pyjwt
from flask import request, jsonify, g
from jwt.exceptions import ExpiredSignatureError, InvalidTokenError, PyJWTError

logger = logging.getLogger(__name__)

SUPABASE_JWT_SECRET = os.environ.get('SUPABASE_JWT_SECRET')
if not SUPABASE_JWT_SECRET:
    # Deferred error: we surface this on the first auth request rather than
    # raising at import time, so localhost workflows that hit unauthenticated
    # routes still work even when the secret is missing from .env.
    logger.warning(
        "SUPABASE_JWT_SECRET is not set. Authenticated requests will fail "
        "until it is configured. Get it from Supabase Dashboard -> Settings "
        "-> API -> JWT Secret."
    )


def _verify_jwt_local(token: str) -> dict:
    """Verify a Supabase-issued JWT locally using HS256 + SUPABASE_JWT_SECRET.

    Raises:
        RuntimeError: if SUPABASE_JWT_SECRET is not configured.
        ExpiredSignatureError: if the token is expired.
        InvalidTokenError / PyJWTError: for any other validation failure
            (bad signature, missing claims, wrong audience, etc.).

    Returns:
        The decoded claims dict on success.
    """
    if not SUPABASE_JWT_SECRET:
        raise RuntimeError(
            "SUPABASE_JWT_SECRET is not set. Add it to .env from "
            "Supabase Dashboard -> Settings -> API -> JWT Secret."
        )
    return pyjwt.decode(
        token,
        SUPABASE_JWT_SECRET,
        algorithms=['HS256'],
        audience='authenticated',
        options={'require': ['exp', 'sub', 'aud']},
    )


def require_auth(f):
    """
    Decorator to require valid Supabase authentication

    Validates JWT token from Authorization header and adds user info to Flask g object

    Usage:
        @require_auth
        def protected_route():
            user_id = g.user_id
            user_email = g.user_email
            return jsonify({'message': 'Protected data'})
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # CORS preflight requests carry no auth header by design. Let them through
        # so Flask-CORS can attach the response headers; the real request that
        # follows will still be authenticated.
        if request.method == 'OPTIONS':
            return ('', 204)

        auth_header = request.headers.get('Authorization')

        if not auth_header:
            logger.warning("Missing Authorization header")
            return jsonify({'error': 'Missing authorization header'}), 401

        try:
            token_type, token = auth_header.split(' ', 1)
            if token_type.lower() != 'bearer':
                logger.warning(f"Invalid token type: {token_type}")
                return jsonify({'error': 'Invalid token type. Expected Bearer token'}), 401
        except ValueError:
            logger.warning("Malformed Authorization header")
            return jsonify({'error': 'Malformed authorization header'}), 401

        try:
            claims = _verify_jwt_local(token)
        except RuntimeError as e:
            # Misconfiguration — surface clearly for ops, generic for client
            logger.error("auth misconfiguration: %s", e)
            return jsonify({'error': 'Authentication misconfigured'}), 500
        except ExpiredSignatureError:
            return jsonify({'error': 'Token expired'}), 401
        except (InvalidTokenError, PyJWTError) as e:
            logger.warning("jwt verify failed: %s", e)
            return jsonify({'error': 'Authentication failed'}), 401

        g.user_id = claims['sub']
        g.user_email = claims.get('email')
        g.user_metadata = claims.get('user_metadata', {})
        g.user_role = claims.get('role', 'authenticated')
        g.access_token = token

        return f(*args, **kwargs)

    return decorated_function


def optional_auth(f):
    """
    Decorator to optionally validate authentication

    If token is present and valid, adds user info to g object
    If token is missing or invalid, continues without authentication

    Usage:
        @optional_auth
        def public_route():
            if hasattr(g, 'user_id'):
                # User is authenticated
                return jsonify({'message': f'Hello {g.user_email}'})
            else:
                # Anonymous user
                return jsonify({'message': 'Hello anonymous'})
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        auth_header = request.headers.get('Authorization')

        if not auth_header:
            return f(*args, **kwargs)

        try:
            token_type, token = auth_header.split(' ', 1)
            if token_type.lower() != 'bearer':
                return f(*args, **kwargs)

            claims = _verify_jwt_local(token)
            g.user_id = claims['sub']
            g.user_email = claims.get('email')
            g.user_metadata = claims.get('user_metadata', {})
            g.user_role = claims.get('role', 'authenticated')
            g.access_token = token

            logger.debug(f"Optional auth: Authenticated user {g.user_email}")

        except Exception as e:
            logger.debug(f"Optional auth failed (continuing anyway): {str(e)}")

        return f(*args, **kwargs)

    return decorated_function
