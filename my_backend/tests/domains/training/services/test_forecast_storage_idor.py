"""IDOR regression test for forecast CSV download (fix/forecast-storage-idor).

Threat model
------------
Legit ``files.storage_path`` values are always server-minted as
``{session_id}/{filename}`` (see upload_routes.py:464, shared/database/storage.py:64,
shared/database/persistence.py:240). However the ``POST /csv-files`` endpoint
accepts a client-supplied ``fileData`` dict and persistence.py reads
``storage_path = file_info.get("storagePath", "")`` verbatim, so a client can
plant an arbitrary path (e.g. another user's ``{other_session}/secret.csv``) on a
files row of a session they own.

The forecast pipeline downloads that path with the SERVICE-ROLE client, which
bypasses Storage RLS. ``execute_forecast`` asserts ownership of the *session* but
not that the path prefix belongs to that session — so the planted path is
fetched cross-tenant.

Fix: ``_download_user_csv`` must verify the path is owned by the session it runs
for (prefix ``{session_id}/``) before the service-role download.
"""
from unittest.mock import patch, MagicMock

import pytest

from domains.training.services.forecast_service import _download_user_csv

OWNER_SESSION = '11111111-1111-1111-1111-111111111111'
VICTIM_SESSION = '22222222-2222-2222-2222-222222222222'


def test_cross_session_storage_path_is_rejected():
    """A storage_path under another session's prefix must be rejected
    BEFORE any service-role storage download happens."""
    mock_sb = MagicMock()

    with patch(
        'shared.database.operations.get_supabase_client',
        return_value=mock_sb,
    ):
        with pytest.raises(ValueError):
            _download_user_csv(
                f'{VICTIM_SESSION}/secret.csv',
                session_id=OWNER_SESSION,
            )

    # The download must never be attempted for a non-owned path.
    mock_sb.storage.from_.assert_not_called()


def test_owned_storage_path_is_downloaded():
    """A storage_path under the caller's own session prefix proceeds to download."""
    mock_sb = MagicMock()
    csv_bytes = b'UTC;value\n2024-01-01 00:00:00;1.0\n2024-01-01 01:00:00;2.0\n'
    mock_sb.storage.from_.return_value.download.return_value = csv_bytes

    with patch(
        'shared.database.operations.get_supabase_client',
        return_value=mock_sb,
    ):
        df = _download_user_csv(
            f'{OWNER_SESSION}/mydata.csv',
            session_id=OWNER_SESSION,
        )

    mock_sb.storage.from_.assert_called_once_with('csv-files')
    mock_sb.storage.from_.return_value.download.assert_called_once_with(
        f'{OWNER_SESSION}/mydata.csv'
    )
    assert len(df) == 2
