"""
Anomaly detection backend debug logger.

Toggled by DEBUG_ANOMALY env var (true | false). When unset, all helpers
are no-ops with negligible overhead. When set, emits structured DEBUG-level
log lines via the `domains.adjustments.debug` logger.
"""
import os
import time
import logging
from functools import wraps
from typing import Any, Callable

DEBUG = os.getenv("DEBUG_ANOMALY", "false").lower() == "true"
_dlogger = logging.getLogger("domains.adjustments.debug")


def _short(s: Any, n: int = 8) -> str:
    """Truncate long IDs (uploadId, user_id) for log readability."""
    s = str(s)
    return s[:n] + "…" if len(s) > n + 1 else s


def dlog(label: str, **fields: Any) -> None:
    """Structured debug log; no-op when DEBUG=false.

    Usage:
        dlog("CSV_PARSED", rows=1234, cols=["UTC", "value"])
    """
    if not DEBUG:
        return
    extras = " ".join(f"{k}={v!r}" for k, v in fields.items())
    _dlogger.debug(f"[{label}] {extras}")


def log_request(handler: Callable) -> Callable:
    """Decorator: log Flask request entry + response status + duration.

    Place CLOSEST to the function so g.user_id (set by @require_auth) is
    available when this decorator runs.
    """
    @wraps(handler)
    def wrapper(*args, **kwargs):
        if not DEBUG:
            return handler(*args, **kwargs)
        from flask import request, g
        body = request.get_json(silent=True) or {}
        user_id = getattr(g, "user_id", "anon")
        dlog("REQ",
             method=request.method,
             path=request.path,
             user=_short(user_id),
             body_keys=list(body.keys()) if isinstance(body, dict) else "<non-dict>")
        t0 = time.time()
        try:
            result = handler(*args, **kwargs)
            dt_ms = (time.time() - t0) * 1000
            status = result[1] if isinstance(result, tuple) and len(result) > 1 else 200
            dlog("RESP",
                 method=request.method,
                 path=request.path,
                 status=status,
                 duration_ms=f"{dt_ms:.1f}")
            return result
        except Exception as e:
            dt_ms = (time.time() - t0) * 1000
            dlog("RESP_ERR",
                 method=request.method,
                 path=request.path,
                 exc=type(e).__name__,
                 duration_ms=f"{dt_ms:.1f}")
            raise
    return wrapper


def log_phase(phase: str) -> Callable:
    """Decorator factory: time + log a pipeline phase.

    Usage:
        @log_phase("preprocess_and_sbad")
        def run_preprocess_and_sbad(...): ...
    """
    def deco(fn: Callable) -> Callable:
        @wraps(fn)
        def wrapper(*args, **kwargs):
            if not DEBUG:
                return fn(*args, **kwargs)
            dlog("PHASE_START", phase=phase)
            t0 = time.time()
            try:
                result = fn(*args, **kwargs)
                dlog("PHASE_END", phase=phase, duration_s=f"{time.time()-t0:.2f}")
                return result
            except Exception as e:
                dlog("PHASE_FAIL", phase=phase, exc=type(e).__name__, duration_s=f"{time.time()-t0:.2f}")
                raise
        return wrapper
    return deco
