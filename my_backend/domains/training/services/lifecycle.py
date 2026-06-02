"""Training lifecycle safety helpers (FIX-5).

Two route-layer guards share the same logic:

1. ``POST /train-models/<sid>`` must reject a new training run when one is
   already in flight for the same session. Two parallel threading.Thread
   workers race on the shared Supabase HTTP/2 pool → RST_STREAM kills both
   → double-billed training_count.

2. ``POST /session/<sid>/delete`` must reject when a training thread is
   mid-flight. The thread continues uploading 0.9-12MB orphan storage
   objects with no FK, cascading FK violations for ~30s after delete.

Both guards consult ``training_progress`` (written by
``TrainingProgressTracker`` and ``ModelTrainingProgressTracker``). A row
with ``status='running'`` AND ``updated_at`` within
``TRAINING_HEARTBEAT_WINDOW`` is treated as live. Outside the window we
assume the worker crashed (OOM, container restart) and allow the operation
so users aren't permanently locked out of their own sessions.

The trackers heartbeat every 10s (HEARTBEAT_INTERVAL_SECONDS in
training_tracker.py), so a 5-minute window gives ~30x slack against
transient DB hiccups.
"""

from datetime import datetime, timezone, timedelta

from shared.database.operations import get_supabase_client
from shared.datetime_utils import parse_iso_datetime

import logging

logger = logging.getLogger(__name__)


# Heartbeat tolerance: 5 minutes ≫ 10s heartbeat interval. Tune downward if
# the heartbeat cadence ever shortens to <2 min. Larger windows risk locking
# users out longer after a worker crash; smaller windows risk false-negatives
# during a slow DB write.
TRAINING_HEARTBEAT_WINDOW = timedelta(minutes=5)


def is_training_in_flight(uuid_session_id: str) -> bool:
    """Return True iff a training run is currently live for this session.

    "Live" = ``training_progress.status == 'running'`` AND
    ``updated_at`` is within :data:`TRAINING_HEARTBEAT_WINDOW`.

    On DB errors we deliberately return False (do NOT fail-closed): a
    Supabase outage would otherwise permanently lock every session's
    /train-models and /delete endpoints. The training thread itself will
    re-hit the same DB and surface a visible error if persistence is
    genuinely down.

    Args:
        uuid_session_id: Resolved UUID of the session (after
            ``assert_session_ownership``).

    Returns:
        True if a heartbeat-fresh running row exists, else False.
    """
    try:
        supabase = get_supabase_client(use_service_role=True)
        result = (
            supabase.table('training_progress')
            .select('status, updated_at')
            .eq('session_id', str(uuid_session_id))
            .order('updated_at', desc=True)
            .limit(1)
            .execute()
        )
        if not result.data:
            return False
        latest = result.data[0]
        if latest.get('status') != 'running':
            return False
        updated_at_raw = latest.get('updated_at')
        if not updated_at_raw:
            # Row says running but has no heartbeat — be conservative and
            # treat as in-flight. The alternative is allowing a double-start.
            return True
        try:
            updated = parse_iso_datetime(updated_at_raw)
        except ValueError:
            logger.warning(
                "[FIX-5] Unparseable training_progress.updated_at=%r for session %s — treating as in-flight",
                updated_at_raw,
                uuid_session_id,
            )
            return True
        return (datetime.now(timezone.utc) - updated) < TRAINING_HEARTBEAT_WINDOW
    except Exception:
        # Do NOT fail-closed (see module docstring). Log loudly so on-call
        # sees the underlying DB error.
        logger.exception(
            "[FIX-5] Failed to query training_progress for in-flight check on session %s",
            uuid_session_id,
        )
        return False
