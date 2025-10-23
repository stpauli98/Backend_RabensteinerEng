"""Supabase client singleton for backend operations"""
import os
import logging
from supabase import create_client, Client
from functools import lru_cache

logger = logging.getLogger(__name__)

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

    logger.info(f"Initializing Supabase client for URL: {supabase_url}")

    # Timeout is handled at database level (statement_timeout = '0' for service_role)
    # This allows long-running operations without client-side timeout
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

    # Create client
    client = create_client(supabase_url, supabase_key)

    # Set the user's access token for RLS
    client.postgrest.auth(access_token)

    return client


def get_supabase_admin_client() -> Client:
    """
    Get Supabase admin client for privileged operations
    Uses service_role key instead of anon key

    Note: Timeout is handled at database level via statement_timeout = '0' for service_role
    This allows unlimited execution time for large operations like training result INSERTs

    Returns:
        Client: Supabase admin client instance
    """
    supabase_url = os.getenv('SUPABASE_URL')
    supabase_service_key = os.getenv('SUPABASE_SERVICE_ROLE_KEY')

    if not supabase_url or not supabase_service_key:
        raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set for admin operations")

    logger.info(f"Initializing Supabase admin client for URL: {supabase_url}")

    # Timeout is handled at database level (statement_timeout = '0' for service_role)
    # This allows long-running INSERT operations without timeout errors
    return create_client(supabase_url, supabase_service_key)
