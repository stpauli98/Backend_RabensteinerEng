"""Email subject lines per (action, language) pair.

Action keys mirror Supabase Auth's `email_action_type` values.
Falls back to German when the requested language is not configured.
"""


class UnknownAction(Exception):
    """Raised when called with an action key we do not support."""


_SUBJECTS = {
    "signup": {
        "de": "Bestätigen Sie Ihre E-Mail – Forecast Engine",
        "en": "Confirm your email – Forecast Engine",
    },
    "recovery": {
        "de": "Passwort zurücksetzen – Forecast Engine",
        "en": "Reset your password – Forecast Engine",
    },
    "magiclink": {
        "de": "Ihr Anmelde-Link – Forecast Engine",
        "en": "Your sign-in link – Forecast Engine",
    },
}


def subject_for(action: str, lang: str) -> str:
    if action not in _SUBJECTS:
        raise UnknownAction(f"No subject configured for action {action!r}")
    by_lang = _SUBJECTS[action]
    return by_lang.get(lang, by_lang["de"])
