"""Tuning + config for the data-retention sweep."""
import os
from datetime import timedelta

# Timing (anchored on the deletion date = lapse + DELETE_AFTER).
DELETE_AFTER = timedelta(days=30)
WARN1_BEFORE = timedelta(days=7)    # first email: 7 days before deletion
WARN2_BEFORE = timedelta(hours=24)  # second email: 24h before deletion
MIN_GAP = timedelta(hours=24)       # min spacing between warn1->warn2 and warn2->delete

# Daily-claim window: if the sweep started within this window, another instance
# already ran today.
CLAIM_WINDOW = timedelta(hours=23)


def sweep_enabled() -> bool:
    return os.environ.get("RETENTION_SWEEP_ENABLED", "false").lower() == "true"


def dry_run() -> bool:
    return os.environ.get("RETENTION_DRY_RUN", "true").lower() == "true"


def login_redirect_url() -> str:
    base = os.environ.get("FRONTEND_URL", "").rstrip("/")
    return f"{base}/login?redirect=/pricing"
