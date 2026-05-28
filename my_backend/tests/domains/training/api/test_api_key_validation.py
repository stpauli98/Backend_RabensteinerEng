"""Key name validation tests (SEC-W12-4)."""


def test_KEY_NAME_PATTERN_rejects_html():
    """SEC-W12-4: HTML/script tags and control chars must be rejected by the regex."""
    from domains.training.api.api_key_routes import KEY_NAME_PATTERN

    bad = [
        '<script>alert(1)</script>',
        '"><img src=x onerror=alert(1)>',
        'javascript:alert(1)',
        '<svg/onload=alert(1)>',
        'name\x00with\x00null',
        'name\r\nwith\r\nbreaks',
        'a&b',
        'quote"here',
        "single'quote",
    ]
    for b in bad:
        assert not KEY_NAME_PATTERN.match(b), f"{b!r} should fail regex"


def test_KEY_NAME_PATTERN_accepts_normal():
    """Sanity check — normal key names pass the regex."""
    from domains.training.api.api_key_routes import KEY_NAME_PATTERN

    good = [
        'production',
        'test-key 1',
        'My Key (2026)',
        'staging_v2',
        'API.Key.v3',
        'a',
    ]
    for g in good:
        assert KEY_NAME_PATTERN.match(g), f"{g!r} should match regex"
