"""Centralized rate-limit decorators.

We use Flask-Limiter with the remote IP address as the key. For
authenticated endpoints we could key on user_id / api_key, but IP-keying
catches both legitimate and adversarial traffic without requiring DB
lookups before the limit fires.

Limits (per SEC-W12-1 plan):
- forecast_limit: 60 req/min per IP on /forecast/* (matches GeoSphere
  240/h hint at the strictest end).
- auth_strict_limit: 10 invalid auth attempts/min per IP before a 5-min
  block. Applied inside the auth decorator on the failed path.

In testing (FLASK_ENV=testing or pytest), limits default to 1000/min so
test runs don't get throttled.
"""

import os
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address


def _is_testing() -> bool:
    return (
        os.environ.get('FLASK_ENV') == 'testing'
        or os.environ.get('PYTEST_CURRENT_TEST') is not None
    )


# Module-level Limiter instance — bound to app in app_factory.py.
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=[],  # no global default; explicit per route
)


def forecast_limit_string() -> str:
    """Rate limit for forecast endpoints."""
    if _is_testing():
        return "1000 per minute"
    return "60 per minute"


def auth_strict_limit_string() -> str:
    """Rate limit for failed-auth attempts (brute force defense)."""
    if _is_testing():
        return "1000 per minute"
    return "10 per minute"


def cloud_limit_string() -> str:
    """Rate limit for /api/cloud/* endpoints."""
    if _is_testing():
        return "1000 per minute"
    return "60 per minute"
