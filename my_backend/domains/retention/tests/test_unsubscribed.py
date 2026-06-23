from datetime import datetime, timezone
from unittest.mock import MagicMock, patch
from domains.retention import unsubscribed

NOW = datetime(2026, 6, 23, 12, 0, tzinfo=timezone.utc)

def test_recent_user_not_stale():
    # newest session 10 days ago -> not stale at 180d
    sb = MagicMock()
    with patch.object(unsubscribed, "_users_with_sessions_no_sub",
                      return_value={"u1": "2026-06-13T00:00:00+00:00"}):
        assert unsubscribed.find_stale_unsubscribed(sb, NOW, max_age_days=180) == []

def test_old_user_is_stale():
    sb = MagicMock()
    with patch.object(unsubscribed, "_users_with_sessions_no_sub",
                      return_value={"u1": "2025-01-01T00:00:00+00:00"}):
        assert unsubscribed.find_stale_unsubscribed(sb, NOW, max_age_days=180) == ["u1"]

def test_sweep_dry_run_does_not_delete():
    sb = MagicMock()
    with patch.object(unsubscribed, "find_stale_unsubscribed", return_value=["u1"]), \
         patch.object(unsubscribed, "delete_user_data") as del_mock:
        out = unsubscribed.sweep_unsubscribed(sb, now=NOW, dry_run=True)
    del_mock.assert_not_called()
    assert out["planned"] == 1 and out["deleted"] == 0

def test_sweep_delete_user_data_errors_path():
    """When delete_user_data returns errors, sweep counts errors and not deleted."""
    sb = MagicMock()
    error_outcome = {"errors": ["boom"], "storage_files_remaining": 0}
    with patch.object(unsubscribed, "find_stale_unsubscribed", return_value=["u1"]), \
         patch.object(unsubscribed, "delete_user_data", return_value=error_outcome):
        out = unsubscribed.sweep_unsubscribed(sb, now=NOW, dry_run=False)
    assert out["errors"] == 1
    assert out["deleted"] == 0
