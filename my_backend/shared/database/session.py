"""
Session management functions for database operations.

This module handles session UUID conversion, creation, and retrieval.
These are the most critical path functions used throughout the application.
"""

import logging
import time
import uuid
from typing import Optional, TypeVar, Callable

import httpx
from supabase import Client

from .config import DatabaseConfig, TableNames
from .exceptions import (
    DatabaseError,
    SessionNotFoundError,
    ValidationError,
    ConfigurationError
)
from .validators import validate_session_id
from .client import get_supabase_client as get_shared_client, get_supabase_admin_client


# Configure httpx timeouts for Supabase client
httpx._config.DEFAULT_TIMEOUT_CONFIG = httpx.Timeout(
    connect=DatabaseConfig.DEFAULT_TIMEOUT_CONNECT,
    read=DatabaseConfig.DEFAULT_TIMEOUT_READ,
    write=DatabaseConfig.DEFAULT_TIMEOUT_WRITE,
    pool=DatabaseConfig.DEFAULT_TIMEOUT_POOL
)

logger = logging.getLogger(__name__)

T = TypeVar('T')


def get_supabase_client(use_service_role: bool = False) -> Client:
    """Get the Supabase client instance.

    This function wraps the centralized client from shared.database.client.

    Args:
        use_service_role: If True, always use service_role key (bypasses RLS).
                         If False, uses anon key for RLS enforcement.

    Returns:
        Client: Supabase client instance

    Raises:
        ConfigurationError: If client cannot be obtained
    """
    client = get_supabase_admin_client() if use_service_role else get_shared_client()
    if not client:
        raise ConfigurationError("Supabase client not available")
    return client


def retry_database_operation(
    operation_func: Callable[[], T],
    max_retries: int = DatabaseConfig.DEFAULT_RETRY_ATTEMPTS,
    initial_delay: float = DatabaseConfig.DEFAULT_INITIAL_DELAY
) -> Optional[T]:
    """Retry database operations with exponential backoff for DNS timeout issues.

    Args:
        operation_func: Function to retry
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds

    Returns:
        Result of operation_func

    Raises:
        DatabaseError: If all retries fail with non-timeout error
    """
    last_error = None

    for attempt in range(max_retries + 1):
        try:
            return operation_func()
        except Exception as e:
            last_error = e
            error_msg = str(e)
            if "Lookup timed out" in error_msg or "timeout" in error_msg.lower():
                if attempt < max_retries:
                    delay = initial_delay * (2 ** attempt)
                    logger.warning(
                        f"Database operation failed (attempt {attempt + 1}/{max_retries + 1}), "
                        f"retrying in {delay}s: {error_msg}"
                    )
                    time.sleep(delay)
                    continue
                else:
                    logger.error(
                        f"Database operation failed after {max_retries + 1} attempts: {error_msg}"
                    )
                    raise DatabaseError(
                        f"Database operation failed after {max_retries + 1} attempts: {error_msg}"
                    )
            else:
                logger.error(f"Database operation failed with non-timeout error: {error_msg}")
                raise DatabaseError(f"Database operation failed: {error_msg}")

    raise DatabaseError(f"Database operation failed: {last_error}")


def create_or_get_session_uuid(session_id: str, user_id: str = None) -> str:
    """Create or get UUID for a session from the session_mappings table.

    SECURITY: Validates session ownership when user_id is provided.

    Args:
        session_id: The string-based session ID from the frontend.
        user_id: The user ID to associate with the session (REQUIRED for creating new sessions).
                For existing sessions, validates that session belongs to this user.

    Returns:
        str: The UUID of the session.

    Raises:
        ValidationError: If session_id is invalid
        SessionNotFoundError: If session cannot be created
        ValueError: If user_id is not provided when creating a new session
        PermissionError: If session exists but doesn't belong to the user
        ConfigurationError: If Supabase client is not available
    """
    if not validate_session_id(session_id):
        raise ValidationError(f"Invalid session_id format: {session_id}")

    # If already a valid UUID, return as-is
    if DatabaseConfig.UUID_PATTERN.match(session_id):
        return session_id

    supabase = get_supabase_client(use_service_role=True)

    def check_existing_mapping():
        # Join with sessions table to get user_id for ownership validation
        response = supabase.table(TableNames.SESSION_MAPPINGS)\
            .select(f'uuid_session_id, {TableNames.SESSIONS}!inner(user_id)')\
            .eq('string_session_id', session_id)\
            .execute()

        if response.data and len(response.data) > 0:
            uuid_session_id = response.data[0]['uuid_session_id']
            session_owner = response.data[0][TableNames.SESSIONS]['user_id']

            # SECURITY: Validate ownership if user_id is provided
            if user_id and session_owner != user_id:
                logger.warning(
                    f"SECURITY: User {user_id} attempted to access session {session_id} "
                    f"owned by {session_owner}"
                )
                raise PermissionError(f'Session {session_id} does not belong to user')

            return uuid_session_id
        return None

    existing_uuid = retry_database_operation(check_existing_mapping)
    if existing_uuid:
        return existing_uuid

    # Validate user_id is provided for new sessions
    if not user_id:
        raise ValueError("user_id is required when creating a new session")

    def create_new_session_mapping():
        # Include user_id when creating session
        session_data = {'user_id': user_id}
        session_response = supabase.table(TableNames.SESSIONS).insert(session_data).execute()

        if getattr(session_response, 'error', None):
            raise DatabaseError(f"Error creating new session: {session_response.error}")

        if not session_response.data:
            raise DatabaseError("Insert operation into 'sessions' did not return data.")

        new_uuid_session_id = session_response.data[0]['id']

        mapping_response = supabase.table(TableNames.SESSION_MAPPINGS).insert({
            'string_session_id': session_id,
            'uuid_session_id': new_uuid_session_id
        }).execute()

        if getattr(mapping_response, 'error', None):
            raise DatabaseError(f"Error creating session mapping: {mapping_response.error}")

        return new_uuid_session_id

    new_uuid = retry_database_operation(create_new_session_mapping)
    if new_uuid:
        return new_uuid

    raise SessionNotFoundError(f"Failed to create session and mapping for {session_id} after retries")


def get_string_id_from_uuid(uuid_session_id: str) -> Optional[str]:
    """Get the string session ID from a UUID.

    Args:
        uuid_session_id: The UUID of the session.

    Returns:
        str: The string-based session ID, or None if not found.

    Raises:
        ConfigurationError: If Supabase client is not available
    """
    if not uuid_session_id:
        return None

    supabase = get_supabase_client()

    try:
        response = supabase.table(TableNames.SESSION_MAPPINGS)\
            .select('string_session_id')\
            .eq('uuid_session_id', uuid_session_id)\
            .execute()

        if response.data and len(response.data) > 0:
            return response.data[0]['string_session_id']
        return None
    except Exception as e:
        logger.error(f"Could not retrieve string_session_id for UUID {uuid_session_id}: {e}")
        return None


def get_session_uuid(session_id: str, user_id: str = None) -> str:
    """Convert session_id to UUID format if needed.

    This is a convenience wrapper that handles both UUID and string session IDs.

    Args:
        session_id: Original session ID (string or UUID format)
        user_id: User ID for ownership validation (optional for existing sessions)

    Returns:
        str: UUID format session ID

    Raises:
        SessionNotFoundError: If session cannot be found or created
        ValidationError: If session_id format is invalid
    """
    try:
        uuid.UUID(session_id)
        return session_id
    except (ValueError, TypeError):
        return create_or_get_session_uuid(session_id, user_id=user_id)


# Aliases for backward compatibility and convenience
_get_session_uuid = get_session_uuid
resolve_session_id = get_session_uuid  # Alias for instant upload functionality
