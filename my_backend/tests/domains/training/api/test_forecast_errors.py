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


def test_empty_payload_returns_400_missing_user_data():
    """W12-F7: Empty {} payload should return 400 + MISSING_USER_DATA, not 422 INTERPOLATION_ERROR."""
    client = _make_client()
    mock_sb = _mock_supabase()

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
            json={},
            headers={'Authorization': 'Bearer sk_fcst_test_key'},
        )

    assert resp.status_code == 400, (
        f"Expected 400, got {resp.status_code}: {resp.get_data(as_text=True)}"
    )
    body = resp.get_json()
    assert body is not None, "Response body must be valid JSON"
    assert body.get('code') == 'MISSING_USER_DATA', (
        f"Expected MISSING_USER_DATA code, got: {body}"
    )
    assert body.get('success') is False
    assert 'user_data' in body.get('error', '').lower(), (
        f"Error message should mention user_data: {body}"
    )


def test_missing_user_data_key_returns_400():
    """W12-F7b: Payload without user_data key should return 400 + MISSING_USER_DATA."""
    client = _make_client()
    mock_sb = _mock_supabase()

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
            json={'other_key': 'value'},
            headers={'Authorization': 'Bearer sk_fcst_test_key'},
        )

    assert resp.status_code == 400, (
        f"Expected 400, got {resp.status_code}: {resp.get_data(as_text=True)}"
    )
    body = resp.get_json()
    assert body.get('code') == 'MISSING_USER_DATA', (
        f"Expected MISSING_USER_DATA code, got: {body}"
    )
    assert body.get('success') is False


def test_invalid_uuid_session_id_returns_400_bad_uuid(client=None):
    """SEC-W12-2: Malformed UUID in session_id should return 400 BAD_UUID, not 500."""
    from urllib.parse import quote
    test_client = client or _make_client()
    payloads = [
        "' OR 1=1--",
        "'; DROP TABLE sessions;--",
        "not-a-uuid",
        "../../etc/passwd",
        "",
    ]
    for p in payloads:
        # GET forecast-config (requires auth, but we want to test the UUID guard fires)
        resp = test_client.get(
            f'/api/training/forecast-config/{quote(p, safe="")}',
            headers={'Authorization': 'Bearer dummy-jwt'}
        )
        # Should be 400 BAD_UUID OR 401 (if auth fires first — that's also fine,
        # auth filter shouldn't crash on bad UUID either).
        assert resp.status_code in (400, 401, 404), (
            f"Payload {p!r}: expected 400/401/404, got {resp.status_code}: {resp.get_data(as_text=True)}"
        )
        # If we got 400, must have BAD_UUID code (not a generic 400 leak)
        if resp.status_code == 400:
            body = resp.get_json()
            assert body.get('code') in ('BAD_UUID', 'MISSING_AUTHORIZATION'), (
                f"Payload {p!r}: 400 without BAD_UUID code: {body}"
            )


def _make_client_with_limiter():
    """Minimal Flask app with forecast blueprint + limiter initialized.

    Used by rate-limit tests that need the limiter wired into the app so
    Flask-Limiter can track request counts. The regular _make_client() skips
    limiter.init_app(), so the @limiter.limit decorator would silently pass
    every request through (no app context = no storage).
    """
    import core.rate_limits as rate_limits_mod
    from domains.training.api.forecast_routes import bp
    app = Flask(__name__)
    app.register_blueprint(bp, url_prefix='/api/training')
    rate_limits_mod.limiter.init_app(app)
    return app.test_client()


def test_rate_limit_returns_429_after_burst():
    """SEC-W12-1: Many requests in burst should get 429 after limit.

    NOTE: This test temporarily forces production limits via monkey patch.
    """
    import core.rate_limits as rate_limits

    # Save original + override the testing detection to return False
    original = rate_limits._is_testing
    rate_limits._is_testing = lambda: False

    client = _make_client_with_limiter()

    # Reset limiter state for test isolation
    rate_limits.limiter.reset()

    try:
        # Send 65 requests in quick succession (limit is 60/min for forecast).
        # We expect at least one 429 once the limit is crossed.
        statuses = []
        for i in range(65):
            resp = client.post(
                '/api/training/forecast/a1b2c3d4-e5f6-4a7b-8c9d-0e1f2a3b4c5d',
                json={'user_data': {}},
                headers={'Authorization': f'Bearer sk_fcst_invalid_{i}'}
            )
            statuses.append(resp.status_code)

        # At least one 429 should appear once limit is exceeded.
        count_429 = sum(1 for s in statuses if s == 429)
        assert count_429 > 0, (
            f"Expected at least 1 HTTP 429 (rate limited) in 65 requests, got "
            f"{count_429} 429s. Statuses: {statuses[-15:]}..."
        )
    finally:
        # Restore original detection + clean state
        rate_limits._is_testing = original
        rate_limits.limiter.reset()
