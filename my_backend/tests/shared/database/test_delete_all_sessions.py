"""Tests for delete_all_sessions user-scoped security fix.

Verifies that delete_all_sessions only deletes the authenticated user's
sessions and never touches other users' data.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock, call


# Helper to build a mock supabase client with chainable query builder
def _make_mock_supabase():
    """Create a mock Supabase client with chainable table/storage methods."""
    supabase = MagicMock()

    def make_query_builder():
        qb = MagicMock()
        qb.select.return_value = qb
        qb.delete.return_value = qb
        qb.eq.return_value = qb
        qb.in_.return_value = qb
        qb.execute.return_value = MagicMock(data=[], count=0)
        return qb

    builders = {}

    def table_side_effect(table_name):
        if table_name not in builders:
            builders[table_name] = make_query_builder()
        return builders[table_name]

    supabase.table.side_effect = table_side_effect
    supabase._builders = builders

    storage_bucket = MagicMock()
    storage_bucket.remove.return_value = None
    supabase.storage.from_.return_value = storage_bucket

    return supabase


class TestDeleteAllSessionsSecurity:
    """Critical: delete_all_sessions must ONLY delete the given user's data."""

    @patch('shared.database.operations.get_supabase_client')
    def test_requires_user_id(self, mock_get_client):
        """delete_all_sessions must raise TypeError if user_id is not provided."""
        from domains.training.services.session import delete_all_sessions

        with pytest.raises(TypeError):
            delete_all_sessions(confirm=True)

    @patch('os.path.exists', return_value=False)
    @patch('shared.database.operations.get_supabase_client')
    def test_only_deletes_own_sessions(self, mock_get_client, mock_exists):
        """Must filter session queries by user_id — never delete unfiltered."""
        from domains.training.services.session import delete_all_sessions

        supabase = _make_mock_supabase()
        mock_get_client.return_value = supabase

        user_a_id = 'user-a-uuid'
        user_a_sessions = [
            {'id': 'session-1-uuid'},
            {'id': 'session-2-uuid'},
        ]

        sessions_builder = MagicMock()
        sessions_builder.select.return_value = sessions_builder
        sessions_builder.eq.return_value = sessions_builder
        sessions_builder.delete.return_value = sessions_builder
        sessions_builder.in_.return_value = sessions_builder
        sessions_builder.execute.return_value = MagicMock(
            data=user_a_sessions, count=2
        )

        def table_effect(name):
            if name == 'sessions':
                return sessions_builder
            qb = MagicMock()
            qb.select.return_value = qb
            qb.delete.return_value = qb
            qb.eq.return_value = qb
            qb.in_.return_value = qb
            qb.execute.return_value = MagicMock(data=[], count=0)
            return qb

        supabase.table.side_effect = table_effect

        result = delete_all_sessions(confirm=True, user_id=user_a_id)

        sessions_builder.select.assert_any_call('id')
        sessions_builder.eq.assert_any_call('user_id', user_a_id)

        assert result['message'] is not None

    def test_no_unfiltered_delete_neq_trick(self):
        """Must NEVER use .delete().neq('id', '000...') — that deletes everything."""
        from domains.training.services.session import delete_all_sessions
        import inspect

        source = inspect.getsource(delete_all_sessions)
        assert '.neq(' not in source, (
            "delete_all_sessions must not use .neq() trick — "
            "it deletes ALL rows regardless of user"
        )
