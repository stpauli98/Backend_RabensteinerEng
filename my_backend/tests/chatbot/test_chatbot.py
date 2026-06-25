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
