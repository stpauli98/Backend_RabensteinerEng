"""Data-access for retention_notices — source of truth for sent warnings."""
from datetime import datetime

_TABLE = "retention_notices"
_ACTIVE = ("sending", "sent", "delivered")


def fetch_notices(supabase) -> list:
    return supabase.table(_TABLE).select(
        "subscription_id,user_id,kind,resend_message_id,status,sent_at"
    ).execute().data or []


def claim_notice(supabase, subscription_id: str, user_id: str, kind: str) -> bool:
    """Insert a 'sending' notice. Returns False if a non-failed notice already exists."""
    existing = (supabase.table(_TABLE).select("status")
                .eq("subscription_id", subscription_id).eq("kind", kind).execute().data or [])
    if any(r.get("status") in _ACTIVE for r in existing):
        return False
    supabase.table(_TABLE).insert({
        "subscription_id": subscription_id, "user_id": user_id,
        "kind": kind, "status": "sending",
    }).execute()
    return True


def mark_sent(supabase, subscription_id: str, kind: str, message_id: str, now: datetime) -> None:
    (supabase.table(_TABLE).update({
        "resend_message_id": message_id, "status": "sent",
        "sent_at": now.isoformat(), "updated_at": now.isoformat(),
    }).eq("subscription_id", subscription_id).eq("kind", kind).execute())


def mark_status_by_message_id(supabase, message_id: str, status: str, now: datetime) -> bool:
    matched = (supabase.table(_TABLE).select("id")
               .eq("resend_message_id", message_id).execute().data or [])
    if not matched:
        return False
    (supabase.table(_TABLE).update({"status": status, "updated_at": now.isoformat()})
     .eq("resend_message_id", message_id).execute())
    return True
