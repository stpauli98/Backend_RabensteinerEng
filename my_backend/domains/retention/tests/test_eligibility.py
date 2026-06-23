from datetime import datetime, timedelta, timezone
from domains.retention.eligibility import compute_actions, RetentionAction

NOW = datetime(2026, 6, 23, 12, 0, tzinfo=timezone.utc)


def _sub(**kw):
    base = dict(id="s1", user_id="u1", status="cancelled",
                expires_at="2026-05-01T00:00:00+00:00",
                scheduled_deletion_at=None)
    base.update(kw); return base


def test_trialing_user_is_protected():
    subs = [_sub(status="trialing", expires_at="2026-12-01T00:00:00+00:00")]
    assert compute_actions(subs, [], NOW) == []


def test_past_due_user_is_protected():
    subs = [_sub(status="past_due", expires_at="2026-12-01T00:00:00+00:00")]
    assert compute_actions(subs, [], NOW) == []


def test_warn1_sets_no_action_when_outside_window():
    # expires 2026-06-20 -> deletion 2026-07-20 -> warn1 window opens 2026-07-13
    subs = [_sub(expires_at="2026-06-20T00:00:00+00:00")]
    assert compute_actions(subs, [], NOW) == []


def test_warn1_fires_inside_window():
    # lapsed long ago -> warn1 due now, no notice yet
    actions = compute_actions([_sub()], [], NOW)
    assert [a.action for a in actions] == ["warn1"]


def test_warn2_anchors_on_scheduled_deletion_at():
    # warn1 already sent; scheduled_deletion_at 24h from now -> warn2 due
    subs = [_sub(scheduled_deletion_at="2026-06-24T12:00:00+00:00")]
    notices = [{"subscription_id": "s1", "kind": "warn1", "status": "sent",
                "sent_at": "2026-06-22T12:00:00+00:00"}]
    actions = compute_actions(subs, notices, NOW)
    assert [a.action for a in actions] == ["warn2"]


def test_delete_after_warn2_and_scheduled_date():
    subs = [_sub(scheduled_deletion_at="2026-06-23T00:00:00+00:00")]
    notices = [
        {"subscription_id": "s1", "kind": "warn1", "status": "sent",
         "sent_at": "2026-06-15T12:00:00+00:00"},
        {"subscription_id": "s1", "kind": "warn2", "status": "sent",
         "sent_at": "2026-06-21T12:00:00+00:00"},
    ]
    actions = compute_actions(subs, notices, NOW)
    assert [a.action for a in actions] == ["delete"]


def test_bounced_warn_pauses_user():
    subs = [_sub(scheduled_deletion_at="2026-06-23T00:00:00+00:00")]
    notices = [{"subscription_id": "s1", "kind": "warn1", "status": "bounced",
                "sent_at": "2026-06-15T12:00:00+00:00"}]
    assert compute_actions(subs, notices, NOW) == []
