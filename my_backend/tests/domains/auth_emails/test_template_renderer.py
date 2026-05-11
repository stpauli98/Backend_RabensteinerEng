import pytest

from domains.auth_emails.services.template_renderer import (
    UnknownTemplate,
    render_email,
)


def test_renders_signup_de_with_link():
    html = render_email(
        action="signup",
        lang="de",
        token_hash="abc123",
        redirect_to="https://forecast-engine.com/dashboard",
        site_url="https://api.forecast-engine.com",
    )
    assert "Bestätigen Sie Ihre Registrierung" in html
    assert "token_hash=abc123" in html
    assert "type=signup" in html
    assert "redirect_to=https://forecast-engine.com/dashboard" in html


def test_renders_signup_en():
    html = render_email(
        action="signup", lang="en",
        token_hash="x", redirect_to="https://app/", site_url="https://api/",
    )
    assert "Confirm your email" in html


def test_renders_recovery_both_langs():
    de = render_email(action="recovery", lang="de", token_hash="t",
                     redirect_to="https://a/", site_url="https://b/")
    en = render_email(action="recovery", lang="en", token_hash="t",
                     redirect_to="https://a/", site_url="https://b/")
    assert "Passwort zurücksetzen" in de
    assert "Reset your password" in en


def test_renders_magiclink_both_langs():
    de = render_email(action="magiclink", lang="de", token_hash="t",
                     redirect_to="https://a/", site_url="https://b/")
    en = render_email(action="magiclink", lang="en", token_hash="t",
                     redirect_to="https://a/", site_url="https://b/")
    assert "Anmelde-Link" in de
    assert "sign-in link" in en


def test_unknown_lang_falls_back_to_de():
    html = render_email(action="signup", lang="fr", token_hash="t",
                        redirect_to="https://a/", site_url="https://b/")
    assert "Bestätigen Sie Ihre Registrierung" in html


def test_unknown_action_raises():
    with pytest.raises(UnknownTemplate):
        render_email(action="invite", lang="de", token_hash="t",
                     redirect_to="https://a/", site_url="https://b/")


def test_html_escaping_prevents_injection():
    """If Supabase ever delivered a redirect_to with HTML, autoescape must neutralise it."""
    html = render_email(
        action="signup", lang="en",
        token_hash="t",
        redirect_to="<script>alert(1)</script>",
        site_url="https://a/",
    )
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html
