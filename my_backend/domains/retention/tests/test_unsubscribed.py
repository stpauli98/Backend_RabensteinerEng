from datetime import datetime, timezone
from unittest.mock import MagicMock, call, patch
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

def test_sweep_skips_user_with_subscription_on_recheck():
    """A candidate returned by find_stale_unsubscribed must NOT be deleted if
    the per-user subscription re-check finds an existing row.  This guards
    against paginated-scan truncation or race conditions."""
    sb = MagicMock()
    # Re-check query returns one subscription row for "u1"
    recheck_result = MagicMock()
    recheck_result.data = [{"id": "sub-row-1"}]
    (
        sb.table.return_value
        .select.return_value
        .eq.return_value
        .limit.return_value
        .execute.return_value
    ) = recheck_result

    with patch.object(unsubscribed, "find_stale_unsubscribed", return_value=["u1"]), \
         patch.object(unsubscribed, "delete_user_data") as del_mock:
        out = unsubscribed.sweep_unsubscribed(sb, now=NOW, dry_run=False)

    del_mock.assert_not_called()
    # Planned still reflects the discovery scan result; skipped user is neither
    # counted as deleted nor as an error.
    assert out["planned"] == 1
    assert out["deleted"] == 0
    assert out["errors"] == 0

def test_fetch_all_pages_paginates():
    """_fetch_all_pages must issue multiple requests until a short page is returned."""
    full_page = [{"user_id": f"u{i}"} for i in range(1000)]
    short_page = [{"user_id": "u1000"}]

    sb = MagicMock()
    execute_mock = MagicMock()
    execute_mock.data = full_page
    execute_mock2 = MagicMock()
    execute_mock2.data = short_page

    chain = sb.table.return_value.select.return_value.range.return_value.execute
    chain.side_effect = [execute_mock, execute_mock2]

    rows = unsubscribed._fetch_all_pages(sb, "user_subscriptions", "user_id")
    assert len(rows) == 1001
    # Two range calls: [0,999] then [1000,1999]
    assert chain.call_count == 2
