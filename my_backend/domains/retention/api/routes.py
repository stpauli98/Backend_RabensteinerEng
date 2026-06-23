"""Resend webhook + admin sweep trigger for the retention domain."""
import json
import logging
import os
from datetime import datetime, timezone

from flask import Blueprint, request, jsonify

from domains.retention.webhook import verify_svix
from domains.retention.notices import mark_status_by_message_id
from domains.retention.constants import (
    resend_webhook_secret, admin_alert_email, admin_secret, dry_run,
)
from domains.auth_emails.services.resend_client import send_email
from shared.database.client import get_supabase_admin_client

logger = logging.getLogger(__name__)
retention_bp = Blueprint("retention", __name__)

_STATUS_MAP = {
    "email.delivered": "delivered",
    "email.bounced": "bounced",
    "email.complained": "complained",
}


def _alert_admin(subject: str, html: str) -> None:
    to = admin_alert_email()
    if not to:
        logger.warning("retention: RETENTION_ADMIN_EMAIL unset; cannot send alert")
        return
    try:
        send_email(api_key=os.environ["RESEND_API_KEY"],
                   from_addr=f'{os.environ.get("EMAIL_FROM_NAME", "Forecast Engine")} '
                             f'<{os.environ["EMAIL_FROM_ADDRESS"]}>',
                   to=to, subject=subject, html=html)
    except Exception:
        logger.exception("retention: failed to send admin alert")


@retention_bp.route("/resend-webhook", methods=["POST"])
def resend_webhook():
    body = request.get_data(as_text=True)
    if not verify_svix(resend_webhook_secret(), dict(request.headers), body):
        return jsonify({"error": "invalid signature"}), 401
    event = json.loads(body)
    event_type = event.get("type")
    status = _STATUS_MAP.get(event_type)
    if not status:
        return jsonify({"status": "ignored"}), 200
    message_id = (event.get("data") or {}).get("email_id") \
        or (event.get("data") or {}).get("id")
    now = datetime.now(timezone.utc)
    supabase = get_supabase_admin_client()
    matched = mark_status_by_message_id(supabase, message_id, status, now)
    if matched and status in ("bounced", "complained"):
        _alert_admin(
            subject="[Forecast Engine] Retention warning bounced",
            html=f"A retention warning email had status <b>{status}</b> "
                 f"(message {message_id}). The user's data deletion is paused "
                 f"pending manual review.",
        )
    return jsonify({"status": "ok", "matched": matched}), 200


@retention_bp.route("/run-sweep", methods=["POST"])
def run_sweep_now():
    if request.headers.get("X-Retention-Secret") != admin_secret() or not admin_secret():
        return jsonify({"error": "unauthorized"}), 401
    from domains.retention.sweep import run_sweep
    result = run_sweep(get_supabase_admin_client(), dry_run=dry_run())
    return jsonify(result), 200
