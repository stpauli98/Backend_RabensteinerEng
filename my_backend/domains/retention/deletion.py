"""Delete a single user's data (keeps the account + subscription history)."""
import logging

from domains.training.services.session import delete_all_sessions

logger = logging.getLogger(__name__)

_USER_TABLES = ("api_keys", "usage_events", "usage_tracking")


def _count_remaining_storage(supabase, user_id: str) -> int:
    """Files still referenced by this user's sessions after deletion."""
    sess = (supabase.table("sessions").select("id").eq("user_id", user_id).execute().data or [])
    ids = [s["id"] for s in sess]
    if not ids:
        return 0
    files = (supabase.table("files").select("id").in_("session_id", ids).execute().data or [])
    return len(files)


def delete_user_data(supabase, user_id: str) -> dict:
    """Idempotent. Returns {'errors': [...], 'storage_files_remaining': N}.
    Caller stamps data_deleted_at only when errors == [] and remaining == 0."""
    errors = []
    result = delete_all_sessions(confirm=True, user_id=user_id)
    errors.extend(result.get("warnings", []))

    for table in _USER_TABLES:
        try:
            supabase.table(table).delete().eq("user_id", user_id).execute()
        except Exception as exc:
            errors.append(f"{table}: {exc}")

    remaining = _count_remaining_storage(supabase, user_id)
    if remaining:
        errors.append(f"{remaining} storage files still present")
    logger.info("retention: purge for user %s -> errors=%d remaining=%d",
                user_id, len(errors), remaining)
    return {"errors": errors, "storage_files_remaining": remaining}
