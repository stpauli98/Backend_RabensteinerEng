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


# Subscription statuses that protect a user from deletion (#5).
PROTECTED_STATUSES = frozenset({"active", "trialing", "past_due"})

# Post-warn1 guaranteed notice window (#1): deletion is always >= this after warn1.
WARN1_WINDOW = timedelta(days=7)

# Email language fallback when user_metadata.lang is missing/unsupported (#8).
RETENTION_DEFAULT_LANG = os.environ.get("RETENTION_DEFAULT_LANG", "de")


def admin_alert_email() -> str:
    """Where bounce/complaint alerts go."""
    return os.environ.get("RETENTION_ADMIN_EMAIL", "")


def resend_webhook_secret() -> str:
    return os.environ.get("RESEND_WEBHOOK_SECRET", "")


def admin_secret() -> str:
    return os.environ.get("RETENTION_ADMIN_SECRET", "")
