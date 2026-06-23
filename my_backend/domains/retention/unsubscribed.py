"""Cleanup of users who uploaded data but never subscribed (#7)."""
import logging
from datetime import datetime, timedelta, timezone

from shared.datetime_utils import parse_iso_datetime
from domains.retention.deletion import delete_user_data

logger = logging.getLogger(__name__)


def _users_with_sessions_no_sub(supabase) -> dict:
    """Return {user_id: newest_session_created_at_iso} for users with sessions
    but no user_subscriptions row."""
    sub_users = {r["user_id"] for r in
                 (supabase.table("user_subscriptions").select("user_id").execute().data or [])}
    newest = {}        # {user_id: iso_string}
    newest_dt = {}     # {user_id: parsed datetime} — used only for comparison
    rows = supabase.table("sessions").select("user_id,created_at").execute().data or []
    for r in rows:
        uid = r.get("user_id")
        if not uid or uid in sub_users:
            continue
        raw = r.get("created_at")
        dt = parse_iso_datetime(raw) if raw else None
        cur_dt = newest_dt.get(uid)
        if dt is None:
            # Unparseable / null row: only win if we have nothing yet
            if uid not in newest:
                newest[uid] = raw
                newest_dt[uid] = None
        else:
            if cur_dt is None or dt > cur_dt:
                newest[uid] = raw
                newest_dt[uid] = dt
    return newest


def find_stale_unsubscribed(supabase, now: datetime, *, max_age_days: int = 180) -> list:
    cutoff = now - timedelta(days=max_age_days)
    stale = []
    for uid, newest in _users_with_sessions_no_sub(supabase).items():
        dt = parse_iso_datetime(newest) if newest else None
        if dt is not None and dt < cutoff:
            stale.append(uid)
    return stale


def sweep_unsubscribed(supabase, *, now=None, dry_run: bool, max_age_days: int = 180) -> dict:
    now = now or datetime.now(timezone.utc)
    stale = find_stale_unsubscribed(supabase, now, max_age_days=max_age_days)
    deleted = errors = 0
    for uid in stale:
        if dry_run:
            logger.info("unsubscribed DRY-RUN: would delete data for user %s", uid)
            continue
        try:
            outcome = delete_user_data(supabase, uid)
            if outcome["errors"]:
                raise RuntimeError(outcome["errors"])
            deleted += 1
        except Exception:
            errors += 1
            logger.exception("unsubscribed: delete failed for user %s", uid)
    return {"planned": len(stale), "deleted": deleted, "errors": errors}
