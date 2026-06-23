from unittest.mock import MagicMock, patch
from domains.retention import deletion
from domains.retention.deletion import delete_user_data


def test_delete_user_data_calls_sessions_then_tables():
    supabase = MagicMock()
    with patch('domains.retention.deletion.delete_all_sessions',
               return_value={"summary": {}}) as das, \
         patch.object(deletion, '_count_remaining_storage', return_value=0):
        delete_user_data(supabase, 'user-1')
    das.assert_called_once_with(confirm=True, user_id='user-1')

    deleted_tables = [c.args[0] for c in supabase.table.call_args_list]
    assert set(deleted_tables) == {'api_keys', 'usage_events', 'usage_tracking'}
    for tbl in ('api_keys', 'usage_events', 'usage_tracking'):
        supabase.table.assert_any_call(tbl)


def test_delete_returns_errors_when_sessions_have_warnings():
    sb = MagicMock()
    with patch.object(deletion, "delete_all_sessions",
                      return_value={"warnings": ["Table files: boom"], "summary": {}}):
        # api_keys/usage tables delete cleanly
        sb.table.return_value.delete.return_value.eq.return_value.execute.return_value.data = []
        result = deletion.delete_user_data(sb, "u1")
    assert result["errors"]  # non-empty -> caller must NOT stamp


def test_delete_clean_returns_no_errors():
    sb = MagicMock()
    sb.table.return_value.delete.return_value.eq.return_value.execute.return_value.data = []
    with patch.object(deletion, "delete_all_sessions", return_value={"summary": {}}), \
         patch.object(deletion, "_count_remaining_storage", return_value=0):
        result = deletion.delete_user_data(sb, "u1")
    assert result["errors"] == [] and result["storage_files_remaining"] == 0
