"""Tests for shared.storage.errors.is_storage_not_found."""

import pytest

from shared.storage.errors import is_storage_not_found


def test_filenotfounderror_is_storage_not_found():
    assert is_storage_not_found(FileNotFoundError("file.txt"))


def test_runtimeerror_is_not_storage_not_found():
    assert not is_storage_not_found(RuntimeError("connection failed"))


def test_message_substring_match_not_found():
    assert is_storage_not_found(Exception("File not found in bucket"))


def test_message_substring_match_no_such():
    assert is_storage_not_found(Exception("ERROR: no such file"))


def test_message_substring_match_object_not_found():
    assert is_storage_not_found(Exception("S3 object not found at path"))


def test_generic_message_is_not_storage_not_found():
    assert not is_storage_not_found(Exception("authentication failed"))


def test_storage_exception_with_404_status(monkeypatch):
    """If storage3 is available, StorageException with .status=404 is detected."""
    try:
        from storage3.utils import StorageException
    except ImportError:
        pytest.skip("storage3 not installed")
    exc = StorageException("not found")
    exc.status = 404
    assert is_storage_not_found(exc)


def test_storage_exception_with_500_status_is_not_not_found(monkeypatch):
    try:
        from storage3.utils import StorageException
    except ImportError:
        pytest.skip("storage3 not installed")
    exc = StorageException("internal server error")
    exc.status = 500
    # message doesn't contain "not found", so this should NOT be detected
    assert not is_storage_not_found(exc)
