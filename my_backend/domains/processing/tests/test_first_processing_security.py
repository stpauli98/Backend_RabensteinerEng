"""Security tests for first_processing endpoints (IDOR + path traversal + decorators)."""
import os
import sys

# Ensure the backend root is on path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from domains.processing.api.first_processing import _file_id_is_owned_by_user


def test_file_id_is_owned_by_user_accepts_own_prefix():
    """file_id starting with '{user_id}_' is owned by that user."""
    assert _file_id_is_owned_by_user("abc123_deadbeef", "abc123") is True


def test_file_id_is_owned_by_user_rejects_different_owner():
    """file_id starting with a different uuid is not owned."""
    assert _file_id_is_owned_by_user("xyz999_deadbeef", "abc123") is False


def test_file_id_is_owned_by_user_rejects_path_traversal():
    """Path traversal payloads must be rejected even if they happen to contain user_id."""
    assert _file_id_is_owned_by_user("../abc123_evil", "abc123") is False
    assert _file_id_is_owned_by_user("..%2Fabc123_evil", "abc123") is False
    assert _file_id_is_owned_by_user("/etc/passwd", "abc123") is False
    assert _file_id_is_owned_by_user("abc123_../etc/passwd", "abc123") is False


def test_file_id_is_owned_by_user_rejects_empty_or_none():
    """Empty or None file_id is never owned."""
    assert _file_id_is_owned_by_user("", "abc123") is False
    assert _file_id_is_owned_by_user(None, "abc123") is False


def test_file_id_is_owned_by_user_rejects_prefix_without_underscore():
    """An exact-uuid match without underscore must NOT pass."""
    assert _file_id_is_owned_by_user("abc123", "abc123") is False
    assert _file_id_is_owned_by_user("abc1234567", "abc123") is False
