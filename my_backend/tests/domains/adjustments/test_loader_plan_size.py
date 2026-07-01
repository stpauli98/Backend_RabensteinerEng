"""Tests for per-plan file size limit support in load_and_validate_csv.

`max_size_bytes=None` must disable the size cap entirely (unlimited plan),
while an explicit integer limit must still reject oversized files.
"""
import pytest

from domains.adjustments.data.loader import load_and_validate_csv


def test_loader_rejects_over_limit(tmp_path):
    p = tmp_path / "big.csv"
    p.write_bytes(b"a" * (2 * 1024 * 1024))
    with pytest.raises(ValueError):
        load_and_validate_csv(p, max_size_bytes=1 * 1024 * 1024, allowed_root=tmp_path)


def test_loader_unlimited_none_skips_size(tmp_path):
    # None disables the size cap; a valid small CSV loads fine.
    p = tmp_path / "ok.csv"
    p.write_text(
        "UTC;v\n"
        "2025-01-01 00:00:00;1.0\n"
        "2025-01-01 00:03:00;2.0\n"
    )
    df, dt = load_and_validate_csv(p, max_size_bytes=None, allowed_root=tmp_path)
    assert len(df) == 2
