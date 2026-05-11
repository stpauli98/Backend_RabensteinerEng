import pytest

from domains.auth_emails.services.subjects import UnknownAction, subject_for


def test_signup_de():
    assert subject_for("signup", "de") == "Bestätigen Sie Ihre E-Mail – Forecast Engine"


def test_signup_en():
    assert subject_for("signup", "en") == "Confirm your email – Forecast Engine"


def test_recovery_de():
    assert subject_for("recovery", "de") == "Passwort zurücksetzen – Forecast Engine"


def test_recovery_en():
    assert subject_for("recovery", "en") == "Reset your password – Forecast Engine"


def test_magiclink_de():
    assert subject_for("magiclink", "de") == "Ihr Anmelde-Link – Forecast Engine"


def test_magiclink_en():
    assert subject_for("magiclink", "en") == "Your sign-in link – Forecast Engine"


def test_unknown_lang_falls_back_to_de():
    assert subject_for("signup", "fr") == "Bestätigen Sie Ihre E-Mail – Forecast Engine"


def test_unknown_action_raises():
    with pytest.raises(UnknownAction):
        subject_for("invite", "de")
