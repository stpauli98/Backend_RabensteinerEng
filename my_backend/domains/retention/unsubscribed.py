"""Cleanup of users who uploaded data but never subscribed (#7)."""
import logging
from datetime import datetime, timedelta, timezone

from shared.datetime_utils import parse_iso_datetime
from domains.retention.deletion import delete_user_data

logger = logging.getLogger(__name__)

_PAGE = 1000  # PostgREST default cap; paginate in chunks of this size


def _fetch_all_pages(supabase, table: str, select: str) -> list:
    """Fetch all rows from *table* using range-based pagination (PAGE=1000)."""
    rows = []
    start = 0
    while True:
        page = (
            supabase.table(table)
            .select(select)
            .range(start, start + _PAGE - 1)
            .execute()
            .data
            or []
        )
        rows.extend(page)
        if len(page) < _PAGE:
            break
        start += _PAGE
    return rows


def _users_with_sessions_no_sub(supabase) -> dict:
    """Return {user_id: newest_session_created_at_iso} for users with sessions
    but no user_subscriptions row.

    Both tables are fetched with range-based pagination so results are never
    silently truncated by the PostgREST 1000-row default cap.
    """
    sub_users = {r["user_id"] for r in _fetch_all_pages(supabase, "user_subscriptions", "user_id")}
    newest = {}        # {user_id: iso_string}
    newest_dt = {}     # {user_id: parsed datetime} — used only for comparison
    rows = _fetch_all_pages(supabase, "sessions", "user_id,created_at")
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


def _has_subscription(supabase, uid: str) -> bool:
    """Re-query user_subscriptions for a single user_id.

    This is a safety re-check immediately before deletion: even if the bulk
    discovery scan was complete, a race condition or future bug could cause a
    subscribed user to appear in the candidate list.  A single-row lookup is
    cheap and prevents any accidental deletion of paying users.
    """
    rows = (
        supabase.table("user_subscriptions")
        .select("id")
        .eq("user_id", uid)
        .limit(1)
        .execute()
        .data
        or []
    )
    return len(rows) > 0


def sweep_unsubscribed(supabase, *, now=None, dry_run: bool, max_age_days: int = 180) -> dict:
    now = now or datetime.now(timezone.utc)
    stale = find_stale_unsubscribed(supabase, now, max_age_days=max_age_days)
    deleted = errors = 0
    for uid in stale:
        if dry_run:
            logger.info("unsubscribed DRY-RUN: would delete data for user %s", uid)
            continue
        # Safety re-check: abort deletion if the user now has a subscription row.
        if _has_subscription(supabase, uid):
            logger.warning(
                "unsubscribed: skipping deletion for user %s — subscription row found on re-check",
                uid,
            )
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
