"""Jinja2-based email renderer.

Maps Supabase email_action_type values to template filenames in
`domains/auth_emails/templates/`. Filename convention: <basename>_<lang>.html.j2.
Falls back to German template when the requested language is unconfigured.
"""

import os

from jinja2 import Environment, FileSystemLoader, TemplateNotFound, select_autoescape


class UnknownTemplate(Exception):
    """Raised when no template exists for the requested action."""


_TEMPLATES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "templates")

_ACTION_TO_BASENAME = {
    "signup": "confirm_signup",
    "recovery": "recovery",
    "magiclink": "magic_link",
}

_env = Environment(
    loader=FileSystemLoader(_TEMPLATES_DIR),
    autoescape=select_autoescape(["html", "j2"]),
    keep_trailing_newline=True,
)


def render_email(
    *,
    action: str,
    lang: str,
    token_hash: str,
    redirect_to: str,
    site_url: str,
    **extra,
) -> str:
    """Render the HTML body for a given auth email."""
    if action not in _ACTION_TO_BASENAME:
        raise UnknownTemplate(f"No template configured for action {action!r}")
    basename = _ACTION_TO_BASENAME[action]

    candidate = f"{basename}_{lang}.html.j2"
    try:
        template = _env.get_template(candidate)
    except TemplateNotFound:
        try:
            template = _env.get_template(f"{basename}_de.html.j2")
        except TemplateNotFound as inner:
            raise UnknownTemplate(
                f"Missing fallback template for action {action!r}"
            ) from inner

    return template.render(
        token_hash=token_hash,
        redirect_to=redirect_to,
        site_url=site_url,
        **extra,
    )
