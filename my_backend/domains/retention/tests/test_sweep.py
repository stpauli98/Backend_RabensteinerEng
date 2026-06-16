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
    action = RetentionAction(user_id='u1', subscription_id='s1', action='warn1',
                             deletion_date=NOW)
    env = {"RESEND_API_KEY": "test-key", "EMAIL_FROM_ADDRESS": "from@x.test"}
    with patch.dict(os.environ, env, clear=False), \
         patch('domains.retention.sweep._claim_daily_lock', return_value=True), \
         patch('domains.retention.sweep._fetch_subscriptions', return_value=[]), \
         patch('domains.retention.sweep.compute_actions', return_value=[action]), \
         patch('domains.retention.sweep._user_email_lang', return_value=('u@x', 'en')), \
         patch('domains.retention.sweep.send_warning', return_value='m1') as send, \
         patch('domains.retention.sweep.delete_user_data') as delet:
        sweep_mod.run_sweep(sb, now=NOW, dry_run=False)
    send.assert_called_once()
    delet.assert_not_called()
    sb.table.assert_any_call('user_subscriptions')


def test_delete_action_purges_then_stamps():
    sb = MagicMock()
    action = RetentionAction(user_id='u1', subscription_id='s1', action='delete',
                             deletion_date=NOW)
    with patch('domains.retention.sweep._claim_daily_lock', return_value=True), \
         patch('domains.retention.sweep._fetch_subscriptions', return_value=[]), \
         patch('domains.retention.sweep.compute_actions', return_value=[action]), \
         patch('domains.retention.sweep.delete_user_data') as delet, \
         patch('domains.retention.sweep.send_warning') as send:
        sweep_mod.run_sweep(sb, now=NOW, dry_run=False)
    delet.assert_called_once_with(sb, 'u1')
    send.assert_not_called()


def test_one_user_error_does_not_block_others():
    sb = MagicMock()
    a1 = RetentionAction(user_id='u1', subscription_id='s1', action='delete', deletion_date=NOW)
    a2 = RetentionAction(user_id='u2', subscription_id='s2', action='delete', deletion_date=NOW)
    with patch('domains.retention.sweep._claim_daily_lock', return_value=True), \
         patch('domains.retention.sweep._fetch_subscriptions', return_value=[]), \
         patch('domains.retention.sweep.compute_actions', return_value=[a1, a2]), \
         patch('domains.retention.sweep.delete_user_data',
               side_effect=[RuntimeError('boom'), None]) as delet:
        result = sweep_mod.run_sweep(sb, now=NOW, dry_run=False)
    assert delet.call_count == 2
    assert result['errors'] == 1
    assert result['done'] == 1
