"""E2E test: delete_all_sessions must never delete another user's data.

This test uses the REAL Supabase database with service_role client.
It creates test data for two users, deletes one user's sessions,
and verifies the other user's data is completely untouched.

Run with:
    docker run --rm --env-file .env my_backend python -m pytest tests/e2e/test_delete_isolation.py -v -s
"""

import uuid
import pytest
from shared.database.client import get_supabase_admin_client


# Use real user IDs from auth.users (FK constraint requires existing users)
# user_a = nmil322@icloud.com, user_b = dzonifu@gmail.com
USER_A_ID = '3e6c5335-af74-4903-9751-f51d10a2e5a9'
USER_B_ID = '172b1c4d-c6f3-47a6-8c13-8e69fff0a9a8'

# Unique session UUIDs
SESSION_A = str(uuid.uuid4())
SESSION_B = str(uuid.uuid4())


@pytest.fixture(autouse=True)
def setup_and_cleanup():
    """Insert test data before test, clean up after — even on failure."""
    supabase = get_supabase_admin_client()

    # ---- INSERT test data ----

    # Sessions for both users
    supabase.table('sessions').insert([
        {'id': SESSION_A, 'user_id': USER_A_ID, 'session_name': 'e2e_test_user_a', 'finalized': False, 'file_count': 1},
        {'id': SESSION_B, 'user_id': USER_B_ID, 'session_name': 'e2e_test_user_b', 'finalized': False, 'file_count': 1},
    ]).execute()

    # Files for both sessions
    supabase.table('files').insert([
        {
            'session_id': SESSION_A,
            'file_name': 'test_a.csv',
            'storage_path': f'{SESSION_A}/test_a.csv',
            'type': 'input',
            'bezeichnung': 'test_a_signal',
        },
        {
            'session_id': SESSION_B,
            'file_name': 'test_b.csv',
            'storage_path': f'{SESSION_B}/test_b.csv',
            'type': 'input',
            'bezeichnung': 'test_b_signal',
        },
    ]).execute()

    # time_info for both sessions
    supabase.table('time_info').insert([
        {'session_id': SESSION_A, 'jahr': True, 'monat': True, 'woche': False, 'tag': False, 'feiertag': False, 'zeitzone': 'UTC'},
        {'session_id': SESSION_B, 'jahr': True, 'monat': False, 'woche': True, 'tag': False, 'feiertag': False, 'zeitzone': 'UTC'},
    ]).execute()

    # zeitschritte for both sessions
    supabase.table('zeitschritte').insert([
        {'session_id': SESSION_A, 'eingabe': 96, 'ausgabe': 96, 'zeitschrittweite': 15, 'offset': 0},
        {'session_id': SESSION_B, 'eingabe': 48, 'ausgabe': 48, 'zeitschrittweite': 30, 'offset': 0},
    ]).execute()

    # session_mappings for both
    supabase.table('session_mappings').insert([
        {'string_session_id': f'session_e2e_a_{SESSION_A[:8]}', 'uuid_session_id': SESSION_A},
        {'string_session_id': f'session_e2e_b_{SESSION_B[:8]}', 'uuid_session_id': SESSION_B},
    ]).execute()

    yield  # ---- RUN TEST ----

    # ---- CLEANUP (always runs) ----
    for table, col in [
        ('files', 'session_id'),
        ('time_info', 'session_id'),
        ('zeitschritte', 'session_id'),
        ('session_mappings', 'uuid_session_id'),
        ('sessions', 'id'),
    ]:
        try:
            supabase.table(table).delete().in_(col, [SESSION_A, SESSION_B]).execute()
        except Exception:
            pass


class TestDeleteAllSessionsIsolation:
    """E2E: Verify user_a's delete does NOT touch user_b's data."""

    def test_delete_only_removes_own_sessions(self):
        """Call delete_all_sessions for user_a, assert user_b data survives."""
        from domains.training.services.session import delete_all_sessions

        # Act: delete all sessions for user_a
        result = delete_all_sessions(confirm=True, user_id=USER_A_ID)

        # Assert: deletion reported success
        assert 'Deleted 1 sessions' in result['message']

        # Assert: user_a's data is GONE
        supabase = get_supabase_admin_client()

        sessions_a = supabase.table('sessions').select('id').eq('id', SESSION_A).execute()
        assert len(sessions_a.data) == 0, "user_a session should be deleted"

        files_a = supabase.table('files').select('id').eq('session_id', SESSION_A).execute()
        assert len(files_a.data) == 0, "user_a files should be deleted"

        time_a = supabase.table('time_info').select('id').eq('session_id', SESSION_A).execute()
        assert len(time_a.data) == 0, "user_a time_info should be deleted"

        zeit_a = supabase.table('zeitschritte').select('id').eq('session_id', SESSION_A).execute()
        assert len(zeit_a.data) == 0, "user_a zeitschritte should be deleted"

        # Assert: user_b's data is UNTOUCHED
        sessions_b = supabase.table('sessions').select('id').eq('id', SESSION_B).execute()
        assert len(sessions_b.data) == 1, "user_b session must NOT be deleted"

        files_b = supabase.table('files').select('id').eq('session_id', SESSION_B).execute()
        assert len(files_b.data) == 1, "user_b files must NOT be deleted"

        time_b = supabase.table('time_info').select('id').eq('session_id', SESSION_B).execute()
        assert len(time_b.data) == 1, "user_b time_info must NOT be deleted"

        zeit_b = supabase.table('zeitschritte').select('id').eq('session_id', SESSION_B).execute()
        assert len(zeit_b.data) == 1, "user_b zeitschritte must NOT be deleted"

        mappings_b = supabase.table('session_mappings').select('id').eq('uuid_session_id', SESSION_B).execute()
        assert len(mappings_b.data) == 1, "user_b session_mappings must NOT be deleted"
