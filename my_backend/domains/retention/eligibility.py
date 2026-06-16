"""Pure eligibility logic for the retention sweep.

Given all user_subscriptions rows and `now`, return the single next action per
lapsed user. No I/O — fully unit-testable.
"""
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from shared.datetime_utils import parse_iso_datetime
from domains.retention.constants import (
    DELETE_AFTER, WARN1_BEFORE, WARN2_BEFORE, MIN_GAP,
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


def compute_actions(subscriptions: List[Dict[str, Any]], now: datetime) -> List[RetentionAction]:
    by_user: Dict[str, List[Dict[str, Any]]] = {}
    for row in subscriptions:
        by_user.setdefault(row['user_id'], []).append(row)

    actions: List[RetentionAction] = []
    for user_id, rows in by_user.items():
        has_active = any(
            r.get('status') == 'active' and (_dt(r.get('expires_at')) or now) > now
            for r in rows
        )
        if has_active:
            continue

        row = max(rows, key=lambda r: _dt(r.get('expires_at')) or datetime.min.replace(tzinfo=now.tzinfo))
        if row.get('data_deleted_at'):
            continue

        lapsed_at = _dt(row.get('expires_at'))
        if lapsed_at is None:
            continue
        deletion_date = lapsed_at + DELETE_AFTER
        w1 = _dt(row.get('retention_warn1_sent_at'))
        w2 = _dt(row.get('retention_warn2_sent_at'))

        action: Optional[Action] = None
        if w1 is None:
            if now >= deletion_date - WARN1_BEFORE:
                action = 'warn1'
        elif w2 is None:
            if now >= max(deletion_date - WARN2_BEFORE, w1 + MIN_GAP):
                action = 'warn2'
        else:
            if now >= max(deletion_date, w2 + MIN_GAP):
                action = 'delete'

        if action:
            actions.append(RetentionAction(
                user_id=user_id, subscription_id=row['id'],
                action=action, deletion_date=deletion_date,
            ))
    return actions
