import os
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch
from domains.retention.eligibility import RetentionAction
from domains.retention import sweep as sweep_mod

NOW = datetime(2026, 6, 16, 12, 0, tzinfo=timezone.utc)


def _supabase_with_lock(acquired=True):
    sb = MagicMock()
    lock_resp = MagicMock()
    lock_resp.data = [{'id': 1}] if acquired else []
    sb.table.return_value.update.return_value.eq.return_value.execute.return_value = lock_resp
    return sb


def test_skips_when_lock_not_acquired():
    sb = _supabase_with_lock(acquired=False)
    with patch('domains.retention.sweep._claim_daily_lock', return_value=False) as claim:
        result = sweep_mod.run_sweep(sb, now=NOW, dry_run=True)
    claim.assert_called_once()
    assert result['ran'] is False


def test_dry_run_does_not_send_or_delete():
    sb = MagicMock()
    action = RetentionAction(user_id='u1', subscription_id='s1', action='warn1',
                             deletion_date=NOW)
    with patch('domains.retention.sweep._claim_daily_lock', return_value=True), \
         patch('domains.retention.sweep._fetch_subscriptions', return_value=[]), \
         patch('domains.retention.sweep.compute_actions', return_value=[action]), \
         patch('domains.retention.sweep.send_warning') as send, \
         patch('domains.retention.sweep.delete_user_data') as delet:
        result = sweep_mod.run_sweep(sb, now=NOW, dry_run=True)
    send.assert_not_called()
    delet.assert_not_called()
    assert result['planned'] == 1


def test_warn1_sends_then_stamps():
    sb = MagicMock()
    row = {'id': 's1', 'user_id': 'u1', 'expires_at': '2026-01-01T00:00:00+00:00'}
    action = RetentionAction(user_id='u1', subscription_id='s1', action='warn1',
                             deletion_date=NOW)
    env = {"RESEND_API_KEY": "test-key", "EMAIL_FROM_ADDRESS": "from@x.test"}
    with patch.dict(os.environ, env, clear=False), \
         patch('domains.retention.sweep._claim_daily_lock', return_value=True), \
         patch('domains.retention.sweep._fetch_subscriptions', return_value=[row]), \
         patch('domains.retention.sweep.compute_actions', return_value=[action]), \
         patch('domains.retention.sweep.claim_notice', return_value=True), \
         patch('domains.retention.sweep._user_email_lang', return_value=('u@x', 'en')), \
         patch('domains.retention.sweep.send_warning', return_value='m1') as send, \
         patch('domains.retention.sweep.mark_sent'), \
         patch('domains.retention.sweep.delete_user_data') as delet:
        sweep_mod.run_sweep(sb, now=NOW, dry_run=False)
    send.assert_called_once()
    delet.assert_not_called()
    sb.table.assert_any_call('user_subscriptions')


def test_delete_action_purges_then_stamps():
    sb = MagicMock()
    row = {'id': 's1', 'user_id': 'u1', 'expires_at': '2026-01-01T00:00:00+00:00'}
    action = RetentionAction(user_id='u1', subscription_id='s1', action='delete',
                             deletion_date=NOW)
    with patch('domains.retention.sweep._claim_daily_lock', return_value=True), \
         patch('domains.retention.sweep._fetch_subscriptions', return_value=[row]), \
         patch('domains.retention.sweep.compute_actions', return_value=[action]), \
         patch('domains.retention.sweep.delete_user_data',
               return_value={"errors": [], "storage_files_remaining": 0}) as delet, \
         patch('domains.retention.sweep.send_warning') as send:
        sweep_mod.run_sweep(sb, now=NOW, dry_run=False)
    delet.assert_called_once_with(sb, 'u1')
    send.assert_not_called()


def test_one_user_error_does_not_block_others():
    sb = MagicMock()
    rows = [
        {'id': 's1', 'user_id': 'u1', 'expires_at': '2026-01-01T00:00:00+00:00'},
        {'id': 's2', 'user_id': 'u2', 'expires_at': '2026-01-01T00:00:00+00:00'},
    ]
    a1 = RetentionAction(user_id='u1', subscription_id='s1', action='delete', deletion_date=NOW)
    a2 = RetentionAction(user_id='u2', subscription_id='s2', action='delete', deletion_date=NOW)
    with patch('domains.retention.sweep._claim_daily_lock', return_value=True), \
         patch('domains.retention.sweep._fetch_subscriptions', return_value=rows), \
         patch('domains.retention.sweep.compute_actions', return_value=[a1, a2]), \
         patch('domains.retention.sweep.delete_user_data',
               side_effect=[RuntimeError('boom'), {"errors": [], "storage_files_remaining": 0}]) as delet:
        result = sweep_mod.run_sweep(sb, now=NOW, dry_run=False)
    assert delet.call_count == 2
    assert result['errors'] == 1
    assert result['done'] == 1


def test_warn1_stamps_scheduled_deletion_at(monkeypatch):
    from datetime import datetime, timezone
    from unittest.mock import MagicMock
    from domains.retention import sweep
    from domains.retention.eligibility import RetentionAction

    now = datetime(2026, 6, 23, 12, 0, tzinfo=timezone.utc)
    row = {"id": "s1", "user_id": "u1", "expires_at": "2026-01-01T00:00:00+00:00"}
    updates = []

    sb = MagicMock()
    def fake_update(patch):
        updates.append(patch)
        chain = MagicMock()
        chain.eq.return_value.execute.return_value = MagicMock(data=[])
        return chain
    sb.table.return_value.update.side_effect = fake_update

    monkeypatch.setattr(sweep, "_claim_daily_lock", lambda *a, **k: True)
    monkeypatch.setattr(sweep, "_fetch_subscriptions", lambda *a, **k: [row])
    monkeypatch.setattr(sweep, "fetch_notices", lambda *a, **k: [])
    monkeypatch.setattr(sweep, "compute_actions",
                        lambda *a, **k: [RetentionAction("u1", "s1", "warn1", now)])
    monkeypatch.setattr(sweep, "claim_notice", lambda *a, **k: True)
    monkeypatch.setattr(sweep, "_user_email_lang", lambda *a, **k: ("x@y.z", "de"))
    monkeypatch.setattr(sweep, "send_warning", lambda **k: "msg_1")
    monkeypatch.setattr(sweep, "mark_sent", lambda *a, **k: None)

    sweep.run_sweep(sb, now=now, dry_run=False)

    # expires(2026-01-01)+30d is well before now+7d, so scheduled = now + 7 days.
    sched = [u["scheduled_deletion_at"] for u in updates if "scheduled_deletion_at" in u]
    assert sched and sched[0].startswith("2026-06-30")
