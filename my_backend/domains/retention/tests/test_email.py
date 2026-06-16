from unittest.mock import patch
from domains.retention.email import render_warning_html, retention_subject, send_warning


def test_render_de_final_contains_link_and_date():
    html = render_warning_html(lang='de', deletion_date='2026-07-16',
                               login_url='https://app/login?redirect=/pricing', is_final=True)
    assert 'Letzte Warnung' in html
    assert '2026-07-16' in html
    assert 'https://app/login?redirect=/pricing' in html


def test_render_unknown_lang_falls_back_to_german():
    html = render_warning_html(lang='fr', deletion_date='2026-07-16',
                               login_url='https://app/x', is_final=False)
    assert 'gelöscht' in html  # German fallback


def test_subject_differs_by_finality():
    assert retention_subject('en', is_final=False) != retention_subject('en', is_final=True)


def test_send_warning_calls_resend_with_rendered_html():
    with patch('domains.retention.email.send_email', return_value='msg-1') as send:
        msg_id = send_warning(api_key='k', from_addr='F <f@x>', to='u@x', lang='en',
                              deletion_date='2026-07-16',
                              login_url='https://app/login?redirect=/pricing', is_final=True)
    assert msg_id == 'msg-1'
    kwargs = send.call_args.kwargs
    assert kwargs['to'] == 'u@x'
    assert 'Final warning' in kwargs['html']
    assert kwargs['subject'] == retention_subject('en', is_final=True)
