"""get_period_start_for_user must resolve the period from the SQL RPC, not the 1st of month."""
import datetime
from unittest.mock import MagicMock, patch


def test_get_period_start_for_user_uses_rpc():
    fake_client = MagicMock()
    fake_client.rpc.return_value.execute.return_value = MagicMock(data="2026-06-15")

    with patch("shared.tracking.usage.get_supabase_admin_client", return_value=fake_client):
        from shared.tracking.usage import get_period_start_for_user
        result = get_period_start_for_user("user-123")

    fake_client.rpc.assert_called_once_with(
        "get_current_period_start", {"p_user_id": "user-123"}
    )
    assert result == datetime.date(2026, 6, 15)


def test_get_period_start_for_user_falls_back_on_error():
    fake_client = MagicMock()
    fake_client.rpc.side_effect = RuntimeError("rpc down")

    with patch("shared.tracking.usage.get_supabase_admin_client", return_value=fake_client):
        from shared.tracking.usage import get_period_start_for_user
        result = get_period_start_for_user("user-123")

    today = datetime.datetime.now(datetime.timezone.utc).date()
    assert result == today.replace(day=1)


def test_anniversary_period_end_matches_sql_convention():
    from shared.tracking.usage import anniversary_period_end
    from datetime import date
    # mid-month anchor -> day before next anniversary
    assert anniversary_period_end(date(2026, 6, 15)) == date(2026, 7, 14)
    # month-end clamp (Jan 31 -> Feb has 28): + 1 month clamped (Feb 28) - 1 day
    assert anniversary_period_end(date(2026, 1, 31)) == date(2026, 2, 27)
    # leap year (Feb 29) clamp
    assert anniversary_period_end(date(2024, 1, 31)) == date(2024, 2, 28)
    # December -> January year rollover
    assert anniversary_period_end(date(2026, 12, 10)) == date(2027, 1, 9)
