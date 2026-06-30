"""increment_training_count must use a single atomic upsert (INSERT ... ON CONFLICT)."""
from unittest.mock import MagicMock, patch
import datetime


def test_increment_training_is_atomic_upsert():
    client = MagicMock()
    client.rpc.return_value.execute.return_value = MagicMock(data="2026-06-15")
    with patch("shared.tracking.usage.get_supabase_admin_client", return_value=client):
        from shared.tracking.usage import increment_training_count
        increment_training_count("u1")
    # must call the atomic SQL upsert RPC, not table().select()/.update()
    names = [c.args[0] for c in client.rpc.call_args_list]
    assert "increment_usage" in names, f"expected atomic increment_usage RPC, got {names}"
