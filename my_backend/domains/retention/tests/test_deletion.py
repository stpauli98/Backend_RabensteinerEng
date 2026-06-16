from unittest.mock import MagicMock, patch
from domains.retention.deletion import delete_user_data


def test_delete_user_data_calls_sessions_then_tables():
    supabase = MagicMock()
    with patch('domains.retention.deletion.delete_all_sessions') as das:
        delete_user_data(supabase, 'user-1')
    das.assert_called_once_with(confirm=True, user_id='user-1')

    deleted_tables = [c.args[0] for c in supabase.table.call_args_list]
    assert set(deleted_tables) == {'api_keys', 'usage_events', 'usage_tracking'}
    for tbl in ('api_keys', 'usage_events', 'usage_tracking'):
        supabase.table.assert_any_call(tbl)
