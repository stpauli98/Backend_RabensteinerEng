"""Supabase Send Email Hook endpoint.

POST /api/auth/send-email
  Body: Supabase webhook payload (Standard Webhooks signed)
  Returns:
    200 {"status": "ok"}             on success
    400 {"error": "..."}              on unknown action / malformed payload
    401 {"error": "..."}              on signature failure / stale webhook
    502 {"error": "..."}              on Resend failure (Supabase will retry)
"""

import logging
import os

from flask import Blueprint, jsonify, request

from domains.auth_emails.services.resend_client import ResendError, send_email
from domains.auth_emails.services.subjects import UnknownAction, subject_for
from domains.auth_emails.services.template_renderer import (
    UnknownTemplate,
    render_email,
)
from domains.auth_emails.services.webhook_verifier import (
    InvalidSignature,
    StaleWebhook,
    verify_webhook,
)


log = logging.getLogger(__name__)
auth_emails_bp = Blueprint("auth_emails", __name__)

_DEFAULT_LANG = "de"


@auth_emails_bp.route("/send-email", methods=["POST"])
def send_email_hook():
    secret = os.environ.get("AUTH_HOOK_SECRET")
    api_key = os.environ.get("RESEND_API_KEY")
    from_address = os.environ.get("EMAIL_FROM_ADDRESS")
    from_name = os.environ.get("EMAIL_FROM_NAME", "Forecast Engine")
    if not (secret and api_key and from_address):
        log.error("auth_emails: missing env config (secret/api_key/from)")
        return jsonify({"error": "Server misconfigured"}), 500

    body = request.get_data()

    try:
        verify_webhook(
            {k.lower(): v for k, v in request.headers.items()},
            body,
            secret,
        )
    except (InvalidSignature, StaleWebhook) as exc:
        log.warning("auth_emails: rejected webhook (%s)", exc)
        return jsonify({"error": str(exc)}), 401

    payload = request.get_json(silent=True) or {}
    user = payload.get("user") or {}
    email_data = payload.get("email_data") or {}

    to_address = user.get("email")
    lang = (user.get("user_metadata") or {}).get("lang") or _DEFAULT_LANG
    action = email_data.get("email_action_type")
    token_hash = email_data.get("token_hash") or ""
    redirect_to = email_data.get("redirect_to") or ""
    site_url = email_data.get("site_url") or ""

    if not to_address or not action:
        return jsonify({"error": "Malformed payload"}), 400

    try:
        subject = subject_for(action, lang)
        html = render_email(
            action=action,
            lang=lang,
            token_hash=token_hash,
            redirect_to=redirect_to,
            site_url=site_url,
        )
    except (UnknownAction, UnknownTemplate) as exc:
        log.warning("auth_emails: unknown action %r (%s)", action, exc)
        return jsonify({"error": f"Unsupported action: {action}"}), 400

    try:
        msg_id = send_email(
            api_key=api_key,
            from_addr=f"{from_name} <{from_address}>",
            to=to_address,
            subject=subject,
            html=html,
        )
        log.info(
            "auth_emails: sent %s to %s lang=%s id=%s",
            action, to_address, lang, msg_id,
        )
    except ResendError as exc:
        log.error("auth_emails: Resend failure: %s", exc)
        return jsonify({"error": "Resend failure"}), 502

    return jsonify({"status": "ok"}), 200
