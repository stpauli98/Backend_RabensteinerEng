"""Per-user in-memory sliding-window rate limit for the chatbot endpoint.

Single-process, best-effort: under multiple Cloud Run instances each holds its own
window, which is acceptable as a soft cost guard on an already auth-gated endpoint.
"""
import time

_WINDOW_SECONDS = 5 * 60
_MAX_PER_WINDOW = 20
_calls: dict[str, list[float]] = {}


def check_and_record(user_id: str) -> bool:
    """Return True and record the call if the user is under the limit; else False."""
    now = time.monotonic()
    cutoff = now - _WINDOW_SECONDS
    times = [t for t in _calls.get(user_id, []) if t > cutoff]
    if len(times) >= _MAX_PER_WINDOW:
        _calls[user_id] = times
        return False
    times.append(now)
    _calls[user_id] = times
    return True
