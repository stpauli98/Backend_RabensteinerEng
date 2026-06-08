"""M-2: save-forecast-config input-type validation tests.

A malformed ``features`` value (a JSON object instead of a list, or a list of
scalars instead of a list of objects) must NOT reach the per-feature loop where
``feat.get(...)`` raises a TypeError/AttributeError that the broad
``except Exception`` previously converted into a 500 with ``str(e)`` in the
response body (info leak). It must return a clean 400 via the standardized
``_err`` contract, with no Python exception text / traceback leaked.
"""
from contextlib import ExitStack
from unittest.mock import patch, MagicMock

from flask import Flask, g

SESSION_UUID = '00000000-0000-0000-0000-000000000000'
TEST_USER_ID = 'aaaaaaaa-0000-0000-0000-000000000000'


def _app():
    app = Flask(__name__)
    return app


def _call_save(body):
    """Invoke save_forecast_config directly with the auth/DB layer stubbed out.

    Patches:
    - _validate_uuid_format → None (UUID passes the guard)
    - create_or_get_session_uuid → SESSION_UUID (no real DB)
    - assert_session_ownership → no-op (ownership passes)
    - get_supabase_client → MagicMock (any incidental query is inert)

    g.user_id is set so the handler reaches the features type check.
    The request carries ``body`` as JSON so request.get_json() returns it.
    """
    from domains.training.api import forecast_routes

    app = _app()
    targets = [
        patch.object(forecast_routes, '_validate_uuid_format', return_value=None),
        patch.object(forecast_routes, 'create_or_get_session_uuid', return_value=SESSION_UUID),
        patch.object(forecast_routes, 'assert_session_ownership', return_value=None),
        patch.object(forecast_routes, 'get_supabase_client', return_value=MagicMock()),
    ]
    with ExitStack() as stack:
        for t in targets:
            stack.enter_context(t)
        with app.test_request_context(
            f'/api/training/save-forecast-config/{SESSION_UUID}',
            method='POST',
            json=body,
        ):
            g.user_id = TEST_USER_ID
            # Bypass @require_auth (it performs a real JWT check) by invoking the
            # undecorated view; functools.wraps exposes it via __wrapped__.
            view = getattr(
                forecast_routes.save_forecast_config,
                '__wrapped__',
                forecast_routes.save_forecast_config,
            )
            resp = view(SESSION_UUID)
    return resp


def _status_and_body(resp):
    """Normalize a Flask view return (Response or (Response, status) tuple)."""
    if isinstance(resp, tuple):
        response, status = resp[0], resp[1]
    else:
        response, status = resp, resp.status_code
    body = response.get_json()
    return status, body


def _assert_clean_400(resp):
    status, body = _status_and_body(resp)
    assert status == 400, f"Expected 400, got {status}: {body}"
    assert body is not None and body.get('success') is False, f"Unexpected body: {body}"
    raw = str(body).lower()
    # No leaked Python exception internals / traceback in the response body.
    assert 'traceback' not in raw, f"Leaked traceback: {body}"
    assert 'attributeerror' not in raw, f"Leaked AttributeError: {body}"
    assert 'typeerror' not in raw, f"Leaked TypeError: {body}"
    assert "has no attribute" not in raw, f"Leaked attribute-access detail: {body}"
    assert "object is not" not in raw, f"Leaked iteration detail: {body}"
    return body


def test_features_as_object_returns_400_not_500():
    """features={"a": 1} (dict, not list) → 400 INVALID_FEATURES, no leak."""
    resp = _call_save({'features': {'a': 1}})
    body = _assert_clean_400(resp)
    assert body.get('code') == 'INVALID_FEATURES', f"Expected INVALID_FEATURES, got: {body}"


def test_features_as_scalar_list_returns_400_not_500():
    """features=[1,2,3] (list of scalars) → 400 INVALID_FEATURES, no leak."""
    resp = _call_save({'features': [1, 2, 3]})
    body = _assert_clean_400(resp)
    assert body.get('code') == 'INVALID_FEATURES', f"Expected INVALID_FEATURES, got: {body}"


def test_valid_empty_features_list_is_accepted():
    """features=[] (valid empty list) must pass the type guard → 200, not 400."""
    resp = _call_save({'features': []})
    status, body = _status_and_body(resp)
    assert status == 200, f"Expected 200 for valid empty list, got {status}: {body}"
    assert body.get('success') is True, f"Unexpected body: {body}"
