"""Test harness for payments HTTP routes.

Builds a minimal Flask app with only the stripe blueprint, mints a valid
HS256 Supabase-style JWT (require_auth verifies locally), and provides a
configurable fake Supabase that honors the queries the routes make.
"""
import time

import jwt as pyjwt
import pytest
from flask import Flask

import shared.auth.jwt as jwt_mod
from core.rate_limits import limiter
from domains.payments.api.stripe import bp

_JWT_SECRET = "test-jwt-secret-for-payments-0123456789abcdef"  # >=32 bytes (HS256)


@pytest.fixture(autouse=True)
def _auth_secret(monkeypatch):
    # require_auth reads the module-global SUPABASE_JWT_SECRET captured at
    # import time; patch the global so _verify_jwt_local uses our test key.
    monkeypatch.setattr(jwt_mod, "SUPABASE_JWT_SECRET", _JWT_SECRET)


@pytest.fixture
def auth_header():
    token = pyjwt.encode(
        {
            "sub": "u1",
            "email": "u1@example.com",
            "aud": "authenticated",
            "exp": int(time.time()) + 3600,
        },
        _JWT_SECRET,
        algorithm="HS256",
    )
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def client():
    app = Flask(__name__)
    app.config["RATELIMIT_ENABLED"] = False
    limiter.init_app(app)
    app.register_blueprint(bp, url_prefix="/api/stripe")
    return app.test_client()


class _Resp:
    def __init__(self, data):
        self.data = data


class _Query:
    def __init__(self, rows):
        self._rows = list(rows)
        self._single = False

    def select(self, *a, **k):
        return self

    def eq(self, col, val):
        self._rows = [r for r in self._rows if r.get(col) == val]
        return self

    def in_(self, col, vals):
        self._rows = [r for r in self._rows if r.get(col) in vals]
        return self

    def limit(self, n):
        self._rows = self._rows[:n]
        return self

    def single(self):
        self._single = True
        return self

    def execute(self):
        if self._single:
            return _Resp(self._rows[0] if self._rows else None)
        return _Resp(self._rows)


class FakeSupabase:
    def __init__(self, tables):
        self._tables = tables

    def table(self, name):
        return _Query(self._tables.get(name, []))


@pytest.fixture
def fake_supabase():
    return FakeSupabase
