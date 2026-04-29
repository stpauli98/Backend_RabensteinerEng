"""
Coverage for cancel_active_subscription helper.

Pins the contract of the new "cancel subscription" flow that replaced
the legacy "downgrade to Free" flow:
  1) Calls the cancel_active_subscription_transaction RPC with p_user_id.
  2) Returns the RPC payload verbatim when the RPC returns data.
  3) Returns a zero-count fallback dict (and logs a warning) when the
     RPC returns empty data.
  4) Does NOT query the subscription_plans table looking for a Free row.
"""
from unittest.mock import patch, MagicMock
import logging


def _make_supabase_with_rpc_data(data):
    """Build a MagicMock supabase client whose .rpc(...).execute() returns `data`."""
    fake_supabase = MagicMock()
    fake_rpc = MagicMock()
    fake_supabase.rpc.return_value = fake_rpc
    fake_rpc.execute.return_value = MagicMock(data=data)
    return fake_supabase


def test_cancel_active_subscription_calls_rpc_with_user_id():
    from shared.payments.stripe import cancel_active_subscription

    user_id = '4633c88e-36fb-446d-a17e-90374359875c'
    fake_supabase = _make_supabase_with_rpc_data(
        {'cancelled_count': 1, 'user_id': user_id}
    )

    with patch(
        'shared.payments.stripe.get_supabase_admin_client',
        return_value=fake_supabase,
    ):
        result = cancel_active_subscription(user_id)

    fake_supabase.rpc.assert_called_once_with(
        'cancel_active_subscription_transaction',
        {'p_user_id': user_id},
    )
    assert result == {'cancelled_count': 1, 'user_id': user_id}


def test_cancel_active_subscription_returns_zero_when_rpc_empty(caplog):
    from shared.payments.stripe import cancel_active_subscription

    user_id = 'aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee'
    fake_supabase = _make_supabase_with_rpc_data(None)

    with patch(
        'shared.payments.stripe.get_supabase_admin_client',
        return_value=fake_supabase,
    ):
        with caplog.at_level(logging.WARNING, logger='shared.payments.stripe'):
            result = cancel_active_subscription(user_id)

    assert result == {'cancelled_count': 0, 'user_id': user_id}
    assert any(
        'cancel_active_subscription returned no data' in rec.message
        for rec in caplog.records
    ), f"expected a WARNING about empty RPC data; saw {[r.message for r in caplog.records]}"


def test_cancel_active_subscription_does_not_query_subscription_plans():
    """
    Regression guard: the legacy downgrade_to_free_plan fell back to a
    SELECT against subscription_plans WHERE name='Free' when the RPC was
    missing. The new helper must NOT do that — Free is no longer a plan.
    """
    from shared.payments.stripe import cancel_active_subscription

    user_id = '11111111-2222-3333-4444-555555555555'
    fake_supabase = _make_supabase_with_rpc_data(
        {'cancelled_count': 0, 'user_id': user_id}
    )

    with patch(
        'shared.payments.stripe.get_supabase_admin_client',
        return_value=fake_supabase,
    ):
        cancel_active_subscription(user_id)

    fake_supabase.table.assert_not_called()
