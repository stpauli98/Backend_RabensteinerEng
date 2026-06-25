"""Chatbot API — POST /api/chatbot/message (auth-gated in-app assistant)."""
import logging

from flask import Blueprint, g, jsonify, request

from shared.auth.jwt import require_auth
from domains.chatbot.services import rate_limit
from domains.chatbot.services.chat_service import ChatUnavailable, generate_reply

logger = logging.getLogger(__name__)
chatbot_bp = Blueprint("chatbot", __name__)

_MAX_MESSAGE_CHARS = 1500
_MAX_HISTORY = 8
_VALID_ROLES = {"user", "assistant"}


@chatbot_bp.route("/message", methods=["POST"])
@require_auth
def message():
    body = request.get_json(silent=True) or {}
    raw = body.get("messages")
    if not isinstance(raw, list) or not raw:
        return jsonify({"error": "messages must be a non-empty list"}), 400

    messages = []
    for m in raw[-_MAX_HISTORY:]:
        if not isinstance(m, dict):
            return jsonify({"error": "each message must be an object"}), 400
        role = m.get("role")
        content = m.get("content")
        if role not in _VALID_ROLES or not isinstance(content, str) or not content.strip():
            return jsonify({"error": "invalid message"}), 400
        if len(content) > _MAX_MESSAGE_CHARS:
            return jsonify({"error": "message too long"}), 400
        messages.append({"role": role, "content": content})

    if not rate_limit.check_and_record(g.user_id):
        return jsonify({"error": "rate limited"}), 429

    lang = body.get("lang")
    if lang not in ("de", "en"):
        lang = "en"
    step = body.get("step")
    if not isinstance(step, str):
        step = None

    try:
        reply = generate_reply(messages=messages, step=step, lang=lang)
    except ChatUnavailable:
        return jsonify({"error": "assistant temporarily unavailable"}), 503

    return jsonify({"reply": reply}), 200
