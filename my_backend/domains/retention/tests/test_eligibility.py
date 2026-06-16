from datetime import datetime, timedelta, timezone
from domains.retention.eligibility import compute_actions, RetentionAction

NOW = datetime(2026, 6, 16, 12, 0, tzinfo=timezone.utc)


def sub(uid, *, status='cancelled', expires_days_ago=None, expires_in_days=None,
        sub_id=None, w1=None, w2=None, deleted=None):
    if expires_days_ago is not None:
        exp = NOW - timedelta(days=expires_days_ago)
    else:
        exp = NOW + timedelta(days=expires_in_days)
    return {
        'id': sub_id or f'{uid}-sub',
        'user_id': uid,
        'status': status,
        'expires_at': exp.isoformat(),
        'retention_warn1_sent_at': w1.isoformat() if w1 else None,
        'retention_warn2_sent_at': w2.isoformat() if w2 else None,
        'data_deleted_at': deleted.isoformat() if deleted else None,
    }


def test_active_user_no_action():
    subs = [sub('u1', status='active', expires_in_days=10)]
    assert compute_actions(subs, NOW) == []


def test_lapsed_but_under_23_days_no_action():
    subs = [sub('u1', expires_days_ago=10)]
    assert compute_actions(subs, NOW) == []


def test_warn1_at_day_23():
    subs = [sub('u1', expires_days_ago=23)]
    actions = compute_actions(subs, NOW)
    assert actions == [RetentionAction(user_id='u1', subscription_id='u1-sub',
                                       action='warn1', deletion_date=NOW + timedelta(days=7))]


def test_warn2_after_warn1_and_24h():
    subs = [sub('u1', expires_days_ago=29, w1=NOW - timedelta(days=6))]
    actions = compute_actions(subs, NOW)
    assert [a.action for a in actions] == ['warn2']


def test_no_warn2_before_24h_gap():
    subs = [sub('u1', expires_days_ago=23, w1=NOW - timedelta(hours=1))]
    assert compute_actions(subs, NOW) == []


def test_delete_at_day_30_after_both_warns():
    subs = [sub('u1', expires_days_ago=30,
                w1=NOW - timedelta(days=7), w2=NOW - timedelta(days=1))]
    actions = compute_actions(subs, NOW)
    assert [a.action for a in actions] == ['delete']


def test_no_delete_without_warn2():
    subs = [sub('u1', expires_days_ago=40, w1=NOW - timedelta(days=5))]
    assert [a.action for a in compute_actions(subs, NOW)] == ['warn2']


def test_already_deleted_skipped():
    subs = [sub('u1', expires_days_ago=60, w1=NOW - timedelta(days=10),
                w2=NOW - timedelta(days=9), deleted=NOW - timedelta(days=8))]
    assert compute_actions(subs, NOW) == []


def test_resubscribe_excludes_user():
    subs = [
        sub('u1', sub_id='old', expires_days_ago=100),
        sub('u1', sub_id='new', status='active', expires_in_days=20),
    ]
    assert compute_actions(subs, NOW) == []


def test_backlog_first_run_sends_warn1():
    subs = [sub('u1', expires_days_ago=50)]
    assert [a.action for a in compute_actions(subs, NOW)] == ['warn1']
