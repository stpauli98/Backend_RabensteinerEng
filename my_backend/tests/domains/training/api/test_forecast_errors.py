"""Forecast endpoint error contract tests (W12-F2, W12-F4, W12-F5, W12-F7)."""
from unittest.mock import patch, MagicMock
from flask import Flask

SESSION_UUID = '00000000-0000-0000-0000-000000000000'
TEST_USER_ID = 'aaaaaaaa-0000-0000-0000-000000000000'


def _make_client():
    """Register only forecast_routes' bp to avoid heavy imports from other blueprints."""
    from domains.training.api.forecast_routes import bp
    app = Flask(__name__)
    app.register_blueprint(bp, url_prefix='/api/training')
    return app.test_client()


def _mock_supabase():
    """Build a Supabase mock that satisfies all guard queries in the forecast view.

    Covers:
    - allow_api_key:  api_keys lookup, sessions (user-exists), user_subscriptions
    - assert_session_ownership: sessions (id + user_id match)
    - view body: files / time_info / zeitschritte return empty by default
      (view returns NO_CONFIG 404 if we get that far, but malformed-JSON guard
       must fire first once the fix is in place)
    """
    mock_supabase = MagicMock()

    api_key_row = {
        'id': 'key-id-1',
        'session_id': SESSION_UUID,
        'user_id': TEST_USER_ID,
        'expires_at': None,
        'last_used_at': None,
    }

    def table_side_effect(table_name):
        mock_query = MagicMock()
        mock_query.execute.return_value = MagicMock(data=[])

        if table_name == 'api_keys':
            mock_query.execute.return_value = MagicMock(data=[api_key_row])
        elif table_name == 'sessions':
            # Satisfies both the user-exists check in allow_api_key and
            # the ownership check in assert_session_ownership.
            mock_query.execute.return_value = MagicMock(
                data=[{'id': SESSION_UUID, 'user_id': TEST_USER_ID}]
            )
        elif table_name == 'user_subscriptions':
            mock_query.execute.return_value = MagicMock(
                data=[{'id': 'sub-1', 'status': 'active'}]
            )

        # Make every chained query method return the same mock so any
        # .select().eq().is_().limit().execute() chain works.
        for attr in ('select', 'eq', 'is_', 'in_', 'limit', 'update', 'insert', 'delete'):
            getattr(mock_query, attr).return_value = mock_query
        return mock_query

    mock_supabase.table.side_effect = table_side_effect
    return mock_supabase


def test_malformed_json_returns_400_with_code():
    """W12-F2: Malformed JSON body must return 400 + code MALFORMED_JSON.

    Approach: patch get_supabase_client at every module that holds a 'from X
    import Y' rebinding of the function, so all call sites in the auth layer
    and view body resolve to the same mock Supabase client.
    create_or_get_session_uuid is similarly patched to avoid real DB calls.
    """
    client = _make_client()
    mock_sb = _mock_supabase()

    # Modules that hold their own local reference via 'from … import …':
    #   shared.database.operations   — source + lazy imports inside api_key.py
    #   domains.training.api.common  — view body uses this binding
    #   shared.auth.ownership        — assert_session_ownership uses its own binding
    supabase_targets = [
        'shared.database.operations.get_supabase_client',
        'domains.training.api.common.get_supabase_client',
        'shared.auth.ownership.get_supabase_client',
    ]
    session_uuid_targets = [
        'shared.database.operations.create_or_get_session_uuid',
        'domains.training.api.common.create_or_get_session_uuid',
    ]

    patches = (
        [patch(t, return_value=mock_sb) for t in supabase_targets]
        + [patch(t, return_value=SESSION_UUID) for t in session_uuid_targets]
    )

    with patches[0], patches[1], patches[2], patches[3], patches[4]:
        resp = client.post(
            f'/api/training/forecast/{SESSION_UUID}',
            data='{this is not json',
            content_type='application/json',
            headers={'Authorization': 'Bearer sk_fcst_test_key'},
        )

    assert resp.status_code == 400, (
        f"Expected 400, got {resp.status_code}: {resp.get_data(as_text=True)}"
    )
    body = resp.get_json()
    assert body is not None, "Response body must be valid JSON"
    assert body.get('code') == 'MALFORMED_JSON', f"Expected MALFORMED_JSON code, got: {body}"
    assert body.get('success') is False

    # Must not leak Flask/werkzeug internals in the error message
    err_lower = body.get('error', '').lower()
    assert 'werkzeug' not in err_lower, f"Leaked werkzeug detail: {body}"
    assert 'expecting property name' not in err_lower, f"Leaked JSON parser detail: {body}"


def test_missing_authorization_returns_401_with_code():
    """W12-F4: Missing Authorization header should return 401 + code MISSING_AUTHORIZATION."""
    client = _make_client()
    # No mocks needed — auth decorator fires before any view body or DB lookup.
    resp = client.post(
        f'/api/training/forecast/{SESSION_UUID}',
        json={'user_data': {}},
        # NOTE: No Authorization header
    )
    assert resp.status_code == 401, f"Expected 401, got {resp.status_code}: {resp.get_data(as_text=True)}"
    body = resp.get_json()
    assert body.get('code') == 'MISSING_AUTHORIZATION', f"Expected MISSING_AUTHORIZATION code, got: {body}"
    assert 'missing' in body.get('error', '').lower()


def test_unknown_session_returns_404_not_403():
    """W12-F5: Unknown session UUID should return 404 SESSION_NOT_FOUND, not 403 KEY_SESSION_MISMATCH.

    Scenario: a user has a valid api key tied to SESSION_UUID, but they POST to
    an unknown/nonexistent session UUID in the URL. Before this fix the code
    would compare key.session_id != url_session_id and return KEY_SESSION_MISMATCH
    (403), misleading the user into thinking their key is wrong when really the
    session doesn't exist.

    The mock simulates a real key in api_keys, then returns empty data for any
    sessions table query (session doesn't exist), so the new existence check
    must fire before the mismatch check and return 404 SESSION_NOT_FOUND.
    """
    UNKNOWN_SESSION = 'ffffffff-ffff-ffff-ffff-ffffffffffff'
    client = _make_client()

    mock_supabase = MagicMock()

    # Valid key row — tied to SESSION_UUID (not UNKNOWN_SESSION)
    api_key_row = {
        'id': 'key-id-1',
        'session_id': SESSION_UUID,
        'user_id': TEST_USER_ID,
        'expires_at': None,
        'last_used_at': None,
    }

    def table_side_effect(table_name):
        mock_query = MagicMock()
        # Default: empty result (sessions table returns nothing → session not found)
        mock_query.execute.return_value = MagicMock(data=[])

        if table_name == 'api_keys':
            mock_query.execute.return_value = MagicMock(data=[api_key_row])
        # sessions table intentionally returns empty data (session does not exist)

        for attr in ('select', 'eq', 'is_', 'in_', 'limit', 'update', 'insert', 'delete'):
            getattr(mock_query, attr).return_value = mock_query
        return mock_query

    mock_supabase.table.side_effect = table_side_effect

    supabase_targets = [
        'shared.database.operations.get_supabase_client',
        'domains.training.api.common.get_supabase_client',
        'shared.auth.ownership.get_supabase_client',
    ]
    session_uuid_targets = [
        'shared.database.operations.create_or_get_session_uuid',
        'domains.training.api.common.create_or_get_session_uuid',
    ]

    patches = (
        [patch(t, return_value=mock_supabase) for t in supabase_targets]
        + [patch(t, return_value=UNKNOWN_SESSION) for t in session_uuid_targets]
    )

    with patches[0], patches[1], patches[2], patches[3], patches[4]:
        resp = client.post(
            f'/api/training/forecast/{UNKNOWN_SESSION}',
            json={'user_data': {}},
            headers={'Authorization': 'Bearer sk_fcst_test_key'},
        )

    body = resp.get_json()
    assert resp.status_code == 404, (
        f"Expected 404 SESSION_NOT_FOUND, got {resp.status_code}: {body}"
    )
    assert body.get('code') == 'SESSION_NOT_FOUND', (
        f"Expected code SESSION_NOT_FOUND, got: {body}"
    )
