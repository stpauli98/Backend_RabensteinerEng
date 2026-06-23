"""Test environment defaults for retention domain tests."""
import os

os.environ.setdefault("RESEND_API_KEY", "re_test_dummy")
os.environ.setdefault("EMAIL_FROM_ADDRESS", "noreply@test.example")
os.environ.setdefault("EMAIL_FROM_NAME", "Test Engine")
