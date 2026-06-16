"""Delete a single user's data (keeps the account + subscription history)."""
import logging

from domains.training.services.session import delete_all_sessions

logger = logging.getLogger(__name__)

_USER_TABLES = ("api_keys", "usage_events", "usage_tracking")


def delete_user_data(supabase, user_id: str) -> None:
    """Idempotent: re-running on an already-purged user is a no-op (0 rows)."""
    # Sessions + their storage files + session-keyed tables.
    delete_all_sessions(confirm=True, user_id=user_id)
    # User-keyed tables.
    for table in _USER_TABLES:
        supabase.table(table).delete().eq("user_id", user_id).execute()
    logger.info("retention: purged data for user %s", user_id)
