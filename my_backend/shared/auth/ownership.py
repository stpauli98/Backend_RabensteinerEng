"""
Session ownership verification helper for service-role backend queries.

Backend uses the Supabase service-role key which bypasses RLS, so manual
ownership checks are required before any query that uses session_id from
client input. Call assert_session_ownership(session_id) at the top of any
handler that takes session_id from URL or body.

Note: shared.database.session.create_or_get_session_uuid() validates
ownership only for string-format session IDs that already have a row in
session_mappings. When the caller passes a raw UUID directly, that helper
returns it as-is without any user_id check — the gap this module closes.
"""
from typing import Union
from uuid import UUID

from flask import g

from shared.database.operations import get_supabase_client


class SessionOwnershipError(Exception):
    """Raised when a session does not belong to the authenticated user."""


def assert_session_ownership(session_id: Union[str, UUID]) -> str:
    """Verify ``session_id`` belongs to ``g.user_id`` (set by ``@require_auth``).

    Returns the canonical string form of the session_id on success.
    Raises ``SessionOwnershipError`` if the session does not exist or does
    not belong to the caller. Caller should map this to a 403/404 response.
    """
    if not getattr(g, 'user_id', None):
        raise SessionOwnershipError('no authenticated user in context')

    sid = str(session_id)
    supabase = get_supabase_client(use_service_role=True)
    res = (
        supabase.table('sessions')
        .select('id')
        .eq('id', sid)
        .eq('user_id', g.user_id)
        .limit(1)
        .execute()
    )
    if not res.data:
        raise SessionOwnershipError(f'session {sid} not owned by {g.user_id}')
    return sid
