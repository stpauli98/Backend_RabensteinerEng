"""Tests for the chatbot domain."""
import os

# Force testing mode before any app import.
os.environ.setdefault('FLASK_ENV', 'testing')


def test_product_knowledge_covers_the_advisory_topics():
    from domains.chatbot.knowledge import PRODUCT_KNOWLEDGE
    assert isinstance(PRODUCT_KNOWLEDGE, str)
    assert len(PRODUCT_KNOWLEDGE) > 1500
    lower = PRODUCT_KNOWLEDGE.lower()
    for topic in [
        "data preparation", "model", "lstm", "cnn", "lgbmr",
        "time information", "forecast horizon", "resolution",
        "scaling", "downloaded model",
    ]:
        assert topic in lower, f"knowledge missing topic: {topic}"


from unittest.mock import patch, MagicMock


def test_rate_limit_blocks_after_the_cap(monkeypatch):
    from domains.chatbot.services import rate_limit
    monkeypatch.setattr(rate_limit, "_MAX_PER_WINDOW", 3)
    monkeypatch.setattr(rate_limit, "_calls", {})
    uid = "user-A"
    assert rate_limit.check_and_record(uid) is True
    assert rate_limit.check_and_record(uid) is True
    assert rate_limit.check_and_record(uid) is True
    assert rate_limit.check_and_record(uid) is False  # 4th in window blocked


def test_system_blocks_cache_stable_and_carry_step_and_lang():
    from domains.chatbot.services import chat_service
    blocks = chat_service.build_system_blocks(step="training", lang="de")
    # first block is the cached knowledge prefix
    assert blocks[0]["cache_control"] == {"type": "ephemeral"}
    assert "Forecast Engine" in blocks[0]["text"]
    assert "info@forecast-engine.com" in blocks[0]["text"]
    # response-style guidelines are part of the cached preamble
    assert "Markdown" in blocks[0]["text"]
    assert "large headings" in blocks[0]["text"]
    # last block is the volatile tail with step + language, NOT cached
    tail = blocks[-1]
    assert "cache_control" not in tail
    assert "training" in tail["text"]
    assert "German" in tail["text"]


def test_generate_reply_calls_claude_and_returns_text():
    from domains.chatbot.services import chat_service
    fake_block = MagicMock(type="text", text="Use an LSTM for strong seasonality.")
    fake_resp = MagicMock(content=[fake_block])
    fake_client = MagicMock()
    fake_client.messages.create.return_value = fake_resp
    with patch("domains.chatbot.services.chat_service.anthropic.Anthropic", return_value=fake_client), \
         patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test"}):
        reply = chat_service.generate_reply(
            messages=[{"role": "user", "content": "Which model?"}], step="training", lang="en")
    assert reply == "Use an LSTM for strong seasonality."
    kwargs = fake_client.messages.create.call_args.kwargs
    assert kwargs["model"] == "claude-sonnet-4-6"
    assert kwargs["max_tokens"] == 700


def test_generate_reply_raises_when_api_key_missing():
    from domains.chatbot.services import chat_service
    with patch.dict(os.environ, {}, clear=True):
        os.environ["FLASK_ENV"] = "testing"
        try:
            chat_service.generate_reply(messages=[{"role": "user", "content": "hi"}], step=None, lang="en")
            assert False, "expected ChatUnavailable"
        except chat_service.ChatUnavailable:
            pass


def test_generate_reply_wraps_anthropic_errors():
    from domains.chatbot.services import chat_service
    fake_client = MagicMock()
    fake_client.messages.create.side_effect = RuntimeError("boom")
    with patch("domains.chatbot.services.chat_service.anthropic.Anthropic", return_value=fake_client), \
         patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test"}):
        try:
            chat_service.generate_reply(messages=[{"role": "user", "content": "hi"}], step=None, lang="en")
            assert False, "expected ChatUnavailable"
        except chat_service.ChatUnavailable:
            pass


import importlib
import json
from functools import wraps

import pytest
from flask import Flask, g


def _stub_require_auth(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        g.user_id = "user-id-123"
        g.user_email = "test@example.com"
        return f(*args, **kwargs)
    return wrapper


def _build_app():
    with patch("shared.auth.jwt.require_auth", side_effect=_stub_require_auth):
        import domains.chatbot.api as chatbot_api
        importlib.reload(chatbot_api)
        app = Flask(__name__)
        app.config["TESTING"] = True
        app.register_blueprint(chatbot_api.chatbot_bp, url_prefix="/api/chatbot")
    import domains.chatbot.api as chatbot_api
    importlib.reload(chatbot_api)
    return app


@pytest.fixture
def client():
    app = _build_app()
    with app.test_client() as c:
        yield c


def _post(client, body):
    return client.post("/api/chatbot/message", data=json.dumps(body),
                       content_type="application/json",
                       headers={"Authorization": "Bearer test-token"})


def test_requires_auth():
    # Build an app WITHOUT the auth stub → the real decorator should 401.
    import domains.chatbot.api as chatbot_api
    importlib.reload(chatbot_api)
    app = Flask(__name__)
    app.register_blueprint(chatbot_api.chatbot_bp, url_prefix="/api/chatbot")
    with app.test_client() as c:
        r = c.post("/api/chatbot/message", json={"messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 401


def test_rejects_oversized_message(client, monkeypatch):
    from domains.chatbot.services import rate_limit
    monkeypatch.setattr(rate_limit, "_calls", {})
    r = _post(client, {"messages": [{"role": "user", "content": "x" * 1501}], "lang": "en"})
    assert r.status_code == 400


def test_rejects_empty_messages(client, monkeypatch):
    from domains.chatbot.services import rate_limit
    monkeypatch.setattr(rate_limit, "_calls", {})
    r = _post(client, {"messages": [], "lang": "en"})
    assert r.status_code == 400


def test_returns_reply_on_success(client, monkeypatch):
    from domains.chatbot.services import rate_limit
    monkeypatch.setattr(rate_limit, "_calls", {})
    with patch("domains.chatbot.api.generate_reply", return_value="Try LGBMR first."):
        r = _post(client, {"messages": [{"role": "user", "content": "Which model?"}],
                           "step": "training", "lang": "en"})
    assert r.status_code == 200
    assert r.get_json()["reply"] == "Try LGBMR first."


def test_returns_429_when_rate_limited(client, monkeypatch):
    from domains.chatbot.services import rate_limit
    monkeypatch.setattr(rate_limit, "check_and_record", lambda uid: False)
    r = _post(client, {"messages": [{"role": "user", "content": "hi"}], "lang": "en"})
    assert r.status_code == 429


def test_returns_503_when_service_unavailable(client, monkeypatch):
    from domains.chatbot.services import rate_limit
    from domains.chatbot.services.chat_service import ChatUnavailable
    monkeypatch.setattr(rate_limit, "_calls", {})
    with patch("domains.chatbot.api.generate_reply", side_effect=ChatUnavailable("x")):
        r = _post(client, {"messages": [{"role": "user", "content": "hi"}], "lang": "en"})
    assert r.status_code == 503
