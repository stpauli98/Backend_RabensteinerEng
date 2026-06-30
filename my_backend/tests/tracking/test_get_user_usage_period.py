"""get_user_usage must filter usage_tracking by the anniversary period from the no-arg RPC."""
from unittest.mock import MagicMock, patch


def test_get_user_usage_filters_by_anniversary_period():
    user_client = MagicMock()
    # RPC returns the anniversary date for the caller
    user_client.rpc.return_value.execute.return_value = MagicMock(data="2026-06-15")
    # Table query chain returns one usage row
    table = user_client.table.return_value
    chain = table.select.return_value.eq.return_value.gte.return_value.order.return_value.limit.return_value
    chain.execute.return_value = MagicMock(data=[{"training_runs_count": 4}])

    with patch("shared.auth.subscription.get_supabase_user_client", return_value=user_client):
        from shared.auth.subscription import get_user_usage
        usage = get_user_usage("user-123", "access-token")

    user_client.rpc.assert_called_once_with("get_current_period_start")
    table.select.return_value.eq.return_value.gte.assert_called_once()
    args = table.select.return_value.eq.return_value.gte.call_args.args
    assert args[1] == "2026-06-15"
    assert usage["training_runs_count"] == 4
