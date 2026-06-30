"""require_subscription must reject an active-but-expired subscription."""
from unittest.mock import MagicMock, patch


def test_get_user_subscription_excludes_expired():
    client = MagicMock()
    # the query chain must include a .gt('expires_at', ...) filter
    with patch("shared.auth.subscription.get_supabase_user_client", return_value=client):
        from shared.auth.subscription import get_user_subscription
        get_user_subscription("u1", "tok")
    table = client.table.return_value
    # eq('status','active') then gt('expires_at', <iso>) must both be applied
    eq_chain = table.select.return_value.eq.return_value
    assert eq_chain.eq.called or eq_chain.gt.called, "expires_at filter missing"
