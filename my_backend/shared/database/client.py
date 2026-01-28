"""Supabase client singleton for backend operations"""
import os
import logging
from supabase import create_client, Client
from functools import lru_cache

logger = logging.getLogger(__name__)

# Try to import ClientOptions (available in supabase-py 2.x)
try:
    from supabase.client import ClientOptions
    HAS_CLIENT_OPTIONS = True
except ImportError:
    try:
        from supabase.lib.client_options import ClientOptions
        HAS_CLIENT_OPTIONS = True
    except ImportError:
        HAS_CLIENT_OPTIONS = False
        logger.warning("ClientOptions not available - using default timeouts")

@lru_cache(maxsize=1)
def get_supabase_client() -> Client:
    """
    Get Supabase client singleton with anon key

    Note: For user-specific operations, use get_supabase_user_client(token) instead
    Note: Timeout is controlled at database level via statement_timeout setting

    Returns:
        Client: Supabase client instance
    """
    supabase_url = os.getenv('SUPABASE_URL')
    supabase_key = os.getenv('SUPABASE_KEY')

    if not supabase_url or not supabase_key:
        raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in environment variables")

    logger.debug(f"Initializing Supabase client for URL: {supabase_url}")

    return create_client(supabase_url, supabase_key)


def get_supabase_user_client(access_token: str) -> Client:
    """
    Get Supabase client with user's JWT token

    This client will respect RLS policies based on the user's token

    Args:
        access_token: User's JWT access token

    Returns:
        Client: Supabase client instance with user context
    """
    supabase_url = os.getenv('SUPABASE_URL')
    supabase_key = os.getenv('SUPABASE_KEY')

    if not supabase_url or not supabase_key:
        raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in environment variables")

    client = create_client(supabase_url, supabase_key)

    client.postgrest.auth(access_token)

    return client


@lru_cache(maxsize=1)
def get_supabase_admin_client() -> Client:
    """
    Get Supabase admin client for privileged operations
    Uses service_role key instead of anon key

    Note: Timeout is handled at database level via statement_timeout = '0' for service_role
    This allows unlimited execution time for large operations like training result INSERTs

    Note: HTTP timeout is set to 60 seconds for large Storage uploads (chunk uploads)

    Returns:
        Client: Supabase admin client instance
    """
    supabase_url = os.getenv('SUPABASE_URL')
    supabase_service_key = os.getenv('SUPABASE_SERVICE_ROLE_KEY')

    if not supabase_url or not supabase_service_key:
        raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set for admin operations")

    logger.debug(f"Initializing Supabase admin client for URL: {supabase_url}")

    # Create with extended timeout for large storage uploads (120 seconds)
    # Supabase Storage can be slow under load, especially with concurrent uploads
    if HAS_CLIENT_OPTIONS:
        options = ClientOptions(
            postgrest_client_timeout=120,
            storage_client_timeout=120
        )
        return create_client(supabase_url, supabase_service_key, options=options)
    else:
        return create_client(supabase_url, supabase_service_key)
