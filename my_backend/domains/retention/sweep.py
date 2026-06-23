"""Daily data-retention sweep orchestrator."""
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List

from domains.retention.constants import CLAIM_WINDOW, DELETE_AFTER, WARN1_WINDOW, login_redirect_url
from domains.retention.deletion import delete_user_data
from domains.retention.eligibility import RetentionAction, compute_actions
from domains.retention.email import send_warning
from domains.retention.notices import fetch_notices, claim_notice, mark_sent

logger = logging.getLogger(__name__)

_SUB_COLS = ("id,user_id,status,expires_at,scheduled_deletion_at,"
             "retention_warn1_sent_at,retention_warn2_sent_at,data_deleted_at")


def _claim_daily_lock(supabase, now: datetime) -> bool:
    """Atomically claim today's run. Returns True if this caller won the slot."""
    cutoff = (now - CLAIM_WINDOW).isoformat()
    resp = (
        supabase.table("retention_sweep_runs")
        .update({"last_started_at": now.isoformat()})
        .eq("id", 1)
        .lt("last_started_at", cutoff)
        .execute()
    )
    if resp.data:
        return True
    resp2 = (
        supabase.table("retention_sweep_runs")
        .update({"last_started_at": now.isoformat()})
        .eq("id", 1)
        .is_("last_started_at", "null")
        .execute()
    )
    return bool(resp2.data)


def _fetch_subscriptions(supabase) -> List[Dict[str, Any]]:
    return supabase.table("user_subscriptions").select(_SUB_COLS).execute().data or []


def _user_email_lang(supabase, user_id: str):
    """Return (email, lang) from auth.users via the admin client, or (None, 'de')."""
    res = supabase.auth.admin.get_user_by_id(user_id)
    user = getattr(res, "user", None) or res
    email = getattr(user, "email", None)
    meta = getattr(user, "user_metadata", None) or {}
    return email, (meta.get("lang") or "de")


def _stamp(supabase, subscription_id: str, column: str, now: datetime) -> None:
    (supabase.table("user_subscriptions")
     .update({column: now.isoformat()})
     .eq("id", subscription_id)
     .execute())


def _stamp_scheduled_deletion(supabase, subscription_id, row, now):
    from shared.datetime_utils import parse_iso_datetime
    lapsed = parse_iso_datetime(row['expires_at'])
    scheduled = max(lapsed + DELETE_AFTER, now + WARN1_WINDOW)
    (supabase.table("user_subscriptions")
     .update({"scheduled_deletion_at": scheduled.isoformat()})
     .eq("id", subscription_id).execute())


def _handle(supabase, action: RetentionAction, row: dict, now: datetime) -> None:
    if action.action in ("warn1", "warn2"):
        if not claim_notice(supabase, action.subscription_id, action.user_id, action.action):
            return  # already sent (idempotent)
        email, lang = _user_email_lang(supabase, action.user_id)
        if not email:
            raise RuntimeError(f"no email for user {action.user_id}")
        is_final = action.action == "warn2"
        message_id = send_warning(
            api_key=os.environ["RESEND_API_KEY"],
            from_addr=f'{os.environ.get("EMAIL_FROM_NAME", "Forecast Engine")} '
                      f'<{os.environ["EMAIL_FROM_ADDRESS"]}>',
            to=email, lang=lang,
            deletion_date=action.deletion_date.date().isoformat(),
            login_url=login_redirect_url(), is_final=is_final,
        )
        mark_sent(supabase, action.subscription_id, action.action, message_id, now)
        if action.action == "warn1":
            _stamp_scheduled_deletion(supabase, action.subscription_id, row, now)
    else:  # delete
        delete_user_data(supabase, action.user_id)
        _stamp(supabase, action.subscription_id, "data_deleted_at", now)


def run_sweep(supabase, *, now: datetime | None = None, dry_run: bool) -> Dict[str, Any]:
    now = now or datetime.now(timezone.utc)
    if not _claim_daily_lock(supabase, now):
        logger.info("retention sweep: lock not acquired, skipping")
        return {"ran": False, "planned": 0, "done": 0, "errors": 0}

    subs = _fetch_subscriptions(supabase)
    notices = fetch_notices(supabase)
    actions = compute_actions(subs, notices, now)
    rows_by_id = {s['id']: s for s in subs}
    done = errors = 0
    for action in actions:
        if dry_run:
            logger.info("retention DRY-RUN: would %s user %s (deletion %s)",
                        action.action, action.user_id, action.deletion_date.date())
            continue
        try:
            _handle(supabase, action, rows_by_id[action.subscription_id], now)
            done += 1
        except Exception:
            errors += 1
            logger.exception("retention: action %s failed for user %s",
                             action.action, action.user_id)
    return {"ran": True, "planned": len(actions), "done": done, "errors": errors}
