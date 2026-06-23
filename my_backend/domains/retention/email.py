"""Render + send the data-deletion warning emails (reuses the Resend client)."""
import os

from jinja2 import Environment, FileSystemLoader, select_autoescape

from domains.auth_emails.services.resend_client import send_email
from domains.retention.constants import RETENTION_DEFAULT_LANG

_TEMPLATES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "auth_emails", "templates"
)
_env = Environment(
    loader=FileSystemLoader(_TEMPLATES_DIR),
    autoescape=select_autoescape(["html", "j2"]),
)
_SUPPORTED = {"de", "en"}

_SUBJECTS = {
    ("de", False): "Ihre Daten werden in 7 Tagen gelöscht",
    ("de", True): "Letzte Warnung: Datenlöschung in 24 Stunden",
    ("en", False): "Your data will be deleted in 7 days",
    ("en", True): "Final warning: data deletion in 24 hours",
}


def _lang(lang: str) -> str:
    return lang if lang in _SUPPORTED else RETENTION_DEFAULT_LANG


def render_warning_html(*, lang: str, deletion_date: str, login_url: str, is_final: bool) -> str:
    template = _env.get_template(f"data_deletion_warning_{_lang(lang)}.html.j2")
    return template.render(deletion_date=deletion_date, login_url=login_url, is_final=is_final)


def retention_subject(lang: str, is_final: bool) -> str:
    return _SUBJECTS[(_lang(lang), is_final)]


def send_warning(*, api_key: str, from_addr: str, to: str, lang: str,
                 deletion_date: str, login_url: str, is_final: bool) -> str:
    html = render_warning_html(lang=lang, deletion_date=deletion_date,
                               login_url=login_url, is_final=is_final)
    return send_email(api_key=api_key, from_addr=from_addr, to=to,
                      subject=retention_subject(lang, is_final), html=html)
