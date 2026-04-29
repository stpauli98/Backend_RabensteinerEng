"""Regression tests for cross-tenant session ownership enforcement.

The Flask backend uses the Supabase service-role key everywhere, which
bypasses Row-Level Security. Route handlers that accept ``session_id`` from
the URL or request body and immediately query session-scoped tables are
therefore only as safe as their manual ownership checks.

A logged-in user (A) must not be able to read data belonging to another
user (B) by passing B's ``session_id`` to a session-scoped endpoint. The
``shared.auth.ownership.assert_session_ownership`` helper performs a
one-row lookup against the ``sessions`` table filtered by both id and
user_id; route handlers map a ``SessionOwnershipError`` to a 403 response.

These tests exercise that guard against the canonical example
``forecast_routes.get_forecast_config``. They run as unit tests with the
Flask request context populated manually, so they do not require Docker,
a live Supabase, or the JWT middleware.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch, MagicMock

import pytest
from flask import Flask, g


USER_A = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
USER_B = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"

# Canonical UUID for a session that, in our scenarios, belongs to USER_B.
SESSION_OF_B = "11111111-1111-1111-1111-111111111111"
SESSION_OF_A = "22222222-2222-2222-2222-222222222222"


@pytest.fixture
def app() -> Flask:
    """Minimal Flask app to provide request context for the view function."""
    app = Flask(__name__)
    app.config["TESTING"] = True
    return app


def _make_supabase_for_session(owner_user_id: str | None):
    """Build a MagicMock Supabase client.

    The mock is tailored for ``assert_session_ownership``: a query of the
    form ``table('sessions').select('id').eq('id', sid).eq('user_id', uid)
    .limit(1).execute()`` returns ``data=[{'id': sid}]`` when
    ``uid == owner_user_id``, else ``data=[]``. Any other table calls return
    ``data=[]`` so downstream queries that may run after the guard succeed
    do not raise.
    """
    supabase = MagicMock()

    def table_side_effect(name):
        chain = MagicMock()
        # Track the .eq() arguments so we can simulate the user_id filter.
        eq_filters: dict[str, str] = {}

        def select(*_a, **_kw):
            return chain

        def eq(col, val):
            eq_filters[col] = val
            return chain

        def order(*_a, **_kw):
            return chain

        def limit(*_a, **_kw):
            return chain

        def single():
            return chain

        def execute():
            if name == 'sessions' and 'id' in eq_filters and 'user_id' in eq_filters:
                if owner_user_id is not None and eq_filters['user_id'] == owner_user_id:
                    return SimpleNamespace(data=[{'id': eq_filters['id']}])
                return SimpleNamespace(data=[])
            # Default: empty for all other queries.
            return SimpleNamespace(data=[])

        chain.select.side_effect = select
        chain.eq.side_effect = eq
        chain.order.side_effect = order
        chain.limit.side_effect = limit
        chain.single.side_effect = single
        chain.execute.side_effect = execute
        return chain

    supabase.table.side_effect = table_side_effect
    return supabase


# ---------------------------------------------------------------------------
# assert_session_ownership unit tests
# ---------------------------------------------------------------------------


def test_assert_session_ownership_rejects_other_users_session(app):
    """User A asks the helper to confirm ownership of User B's session."""
    from shared.auth.ownership import assert_session_ownership, SessionOwnershipError

    supabase = _make_supabase_for_session(owner_user_id=USER_B)

    with app.test_request_context("/"):
        g.user_id = USER_A
        with patch(
            "shared.auth.ownership.get_supabase_client",
            return_value=supabase,
        ):
            with pytest.raises(SessionOwnershipError):
                assert_session_ownership(SESSION_OF_B)


def test_assert_session_ownership_allows_owner(app):
    """User A asks the helper to confirm ownership of their own session."""
    from shared.auth.ownership import assert_session_ownership

    supabase = _make_supabase_for_session(owner_user_id=USER_A)

    with app.test_request_context("/"):
        g.user_id = USER_A
        with patch(
            "shared.auth.ownership.get_supabase_client",
            return_value=supabase,
        ):
            result = assert_session_ownership(SESSION_OF_A)
            assert result == SESSION_OF_A


def test_assert_session_ownership_requires_authenticated_user(app):
    """No g.user_id means the helper must refuse, not silently pass."""
    from shared.auth.ownership import assert_session_ownership, SessionOwnershipError

    with app.test_request_context("/"):
        # g.user_id intentionally not set
        with pytest.raises(SessionOwnershipError):
            assert_session_ownership(SESSION_OF_A)


# ---------------------------------------------------------------------------
# Route-level integration: forecast_routes.get_forecast_config
# ---------------------------------------------------------------------------


def _call_get_forecast_config(app: Flask, caller_user_id: str, session_uuid: str):
    """Invoke ``get_forecast_config`` with @require_auth bypassed.

    Pre-populates ``g.user_id`` to mirror what the auth decorator would set
    after a successful JWT validation, then calls the unwrapped view fn.
    """
    from domains.training.api.forecast_routes import get_forecast_config

    inner = getattr(get_forecast_config, "__wrapped__", get_forecast_config)
    with app.test_request_context(f"/forecast-config/{session_uuid}"):
        g.user_id = caller_user_id
        return inner(session_id=session_uuid)


def _extract_status(response) -> int:
    if isinstance(response, tuple):
        return response[1]
    return getattr(response, "status_code", 200)


def test_get_forecast_config_rejects_other_users_session(app):
    """Caller A passes User B's session_id → handler must return 403.

    The 403 must short-circuit before any session-scoped table read happens.
    """
    supabase = _make_supabase_for_session(owner_user_id=USER_B)

    with patch(
        "domains.training.api.forecast_routes.get_supabase_client",
        return_value=supabase,
    ), patch(
        "domains.training.api.forecast_routes.create_or_get_session_uuid",
        return_value=SESSION_OF_B,
    ), patch(
        "shared.auth.ownership.get_supabase_client",
        return_value=supabase,
    ):
        response = _call_get_forecast_config(
            app, caller_user_id=USER_A, session_uuid=SESSION_OF_B
        )

    status = _extract_status(response)
    assert status == 403, f"expected 403, got {status}"

    # No data tables should have been queried beyond the ownership probe.
    queried_tables = [c.args[0] for c in supabase.table.call_args_list]
    # The only allowed table read is the 'sessions' lookup performed by the
    # ownership guard. Any 'files' / 'time_info' / 'zeitschritte' query
    # would mean we leaked data past the guard.
    assert "files" not in queried_tables
    assert "time_info" not in queried_tables
    assert "zeitschritte" not in queried_tables


def test_get_forecast_config_allows_owner(app):
    """Caller A passes their own session_id → handler proceeds (not 403).

    We only assert that the guard did NOT short-circuit. The downstream
    behaviour of the handler when the data tables are empty is not the
    subject of this test.
    """
    supabase = _make_supabase_for_session(owner_user_id=USER_A)

    with patch(
        "domains.training.api.forecast_routes.get_supabase_client",
        return_value=supabase,
    ), patch(
        "domains.training.api.forecast_routes.create_or_get_session_uuid",
        return_value=SESSION_OF_A,
    ), patch(
        "shared.auth.ownership.get_supabase_client",
        return_value=supabase,
    ), patch(
        "utils.model_storage.list_session_models",
        return_value=[],
    ):
        response = _call_get_forecast_config(
            app, caller_user_id=USER_A, session_uuid=SESSION_OF_A
        )

    status = _extract_status(response)
    assert status != 403, f"owner request must not be 403, got {status}"

    # Guard passed → handler proceeded to query session-scoped tables.
    queried_tables = [c.args[0] for c in supabase.table.call_args_list]
    assert "files" in queried_tables, (
        "owner request should reach the files query past the ownership guard"
    )
