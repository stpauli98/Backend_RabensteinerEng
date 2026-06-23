"""Pure eligibility logic for the retention sweep.

Given all user_subscriptions rows, retention notices, and `now`, return the
single next action per lapsed user. No I/O — fully unit-testable.
"""
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from shared.datetime_utils import parse_iso_datetime
from domains.retention.constants import (
    DELETE_AFTER, WARN1_BEFORE, WARN2_BEFORE, MIN_GAP, WARN1_WINDOW,
    PROTECTED_STATUSES,
)

Action = Literal['warn1', 'warn2', 'delete']


@dataclass(frozen=True)
class RetentionAction:
    user_id: str
    subscription_id: str
    action: Action
    deletion_date: datetime


def _dt(value: Optional[str]) -> Optional[datetime]:
    return parse_iso_datetime(value) if value else None


def _latest_notice(notices, subscription_id, kind):
    rows = [n for n in notices
            if n.get("subscription_id") == subscription_id and n.get("kind") == kind]
    if not rows:
        return None
    return max(rows, key=lambda n: _dt(n.get("sent_at")) or datetime.min.replace(tzinfo=None))


def compute_actions(subscriptions: List[Dict[str, Any]], notices: List[Dict[str, Any]], now: datetime) -> List[RetentionAction]:
    by_user: Dict[str, List[Dict[str, Any]]] = {}
    for row in subscriptions:
        by_user.setdefault(row['user_id'], []).append(row)

    actions: List[RetentionAction] = []
    for user_id, rows in by_user.items():
        has_active = any(
            r.get('status') in PROTECTED_STATUSES
            and (_dt(r.get('expires_at')) or now) > now
            for r in rows
        )
        if has_active:
            continue

        row = max(rows, key=lambda r: _dt(r.get('expires_at'))
                  or datetime.min.replace(tzinfo=now.tzinfo))
        if row.get('data_deleted_at'):
            continue

        lapsed_at = _dt(row.get('expires_at'))
        if lapsed_at is None:
            continue

        sub_id = row['id']
        w1 = _latest_notice(notices, sub_id, 'warn1')
        w2 = _latest_notice(notices, sub_id, 'warn2')

        # Pause: a bounced/complained warning means we cannot prove notice.
        if (w1 and w1.get('status') in ('bounced', 'complained')) or \
           (w2 and w2.get('status') in ('bounced', 'complained')):
            continue

        w1_sent = _dt(w1.get('sent_at')) if w1 else None
        w2_sent = _dt(w2.get('sent_at')) if w2 else None

        # Anchor: once warn1 is sent, scheduled_deletion_at drives everything.
        scheduled = _dt(row.get('scheduled_deletion_at'))
        deletion_date = scheduled or (lapsed_at + DELETE_AFTER)

        action: Optional[Action] = None
        if w1_sent is None:
            if now >= (lapsed_at + DELETE_AFTER) - WARN1_BEFORE:
                action = 'warn1'
        elif w2_sent is None:
            if now >= max(deletion_date - WARN2_BEFORE, w1_sent + MIN_GAP):
                action = 'warn2'
        else:
            if now >= max(deletion_date, w2_sent + MIN_GAP):
                action = 'delete'

        if action:
            actions.append(RetentionAction(
                user_id=user_id, subscription_id=sub_id,
                action=action, deletion_date=deletion_date,
            ))
    return actions
