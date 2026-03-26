"""Datetime parsing utilities for Python 3.9 compatibility.

Python 3.9's datetime.fromisoformat() only supports 0, 3, or 6 fractional
second digits. Supabase/PostgreSQL often returns 5 digits (e.g. '.74731'),
which causes ValueError. This module normalizes fractional seconds before parsing.
"""
from datetime import datetime, timezone


def parse_iso_datetime(value: str) -> datetime:
    """Parse ISO 8601 timestamp with any number of fractional second digits.

    Normalizes fractional seconds to 6 digits (microseconds) for Python 3.9
    compatibility, then delegates to datetime.fromisoformat().

    Args:
        value: ISO 8601 datetime string, optionally with 'Z' suffix.

    Returns:
        Timezone-aware datetime object.
    """
    ts = value.replace('Z', '+00:00')
    if '.' in ts:
        dot = ts.index('.')
        # Find timezone offset after fractional seconds
        plus = ts.find('+', dot)
        if plus == -1:
            plus = ts.find('-', dot + 1)
        if plus != -1:
            frac = ts[dot + 1:plus].ljust(6, '0')[:6]
            ts = ts[:dot + 1] + frac + ts[plus:]
    return datetime.fromisoformat(ts)
