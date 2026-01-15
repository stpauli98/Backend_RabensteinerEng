"""Authentication middleware for Supabase JWT validation"""
import logging
from functools import wraps
from flask import request, jsonify, g
from shared.database.client import get_supabase_client

logger = logging.getLogger(__name__)

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
            supabase = get_supabase_client()

            response = supabase.auth.get_user(token)

            if not response or not response.user:
                logger.warning("Invalid or expired token")
                return jsonify({'error': 'Invalid or expired token'}), 401

            g.user_id = response.user.id
            g.user_email = response.user.email
            g.user_metadata = response.user.user_metadata
            g.access_token = token

            return f(*args, **kwargs)

        except Exception as e:
            logger.error(f"Authentication error: {str(e)}")
            return jsonify({'error': 'Authentication failed', 'details': str(e)}), 401

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

            supabase = get_supabase_client()
            response = supabase.auth.get_user(token)

            if response and response.user:
                g.user_id = response.user.id
                g.user_email = response.user.email
                g.user_metadata = response.user.user_metadata
                g.access_token = token

                logger.debug(f"Optional auth: Authenticated user {g.user_email}")

        except Exception as e:
            logger.debug(f"Optional auth failed (continuing anyway): {str(e)}")

        return f(*args, **kwargs)

    return decorated_function
