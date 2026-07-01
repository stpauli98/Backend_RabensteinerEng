import pytest
from domains.cloud.services.validation import validate_csv_size
from shared.exceptions.errors import CloudException


def _write(tmp_path, mb):
    p = tmp_path / "f.csv"
    p.write_bytes(b"x" * (mb * 1024 * 1024))
    return str(p)


def test_unlimited_skips_check(tmp_path):
    validate_csv_size(_write(tmp_path, 2), max_size_bytes=None)  # must not raise


def test_rejects_over_plan_limit(tmp_path):
    with pytest.raises(CloudException):
        validate_csv_size(_write(tmp_path, 2), max_size_bytes=1 * 1024 * 1024)


def test_accepts_under_plan_limit(tmp_path):
    validate_csv_size(_write(tmp_path, 1), max_size_bytes=5 * 1024 * 1024)
