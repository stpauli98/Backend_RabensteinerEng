"""IDOR regression tests for CSV file record update/delete (finding C-1).

The `files` table is mutated by service-role client (RLS bypassed) keyed only
on `id`. Without an ownership check, any authenticated user can edit/delete any
other user's file. Ownership is transitive: files.session_id -> sessions.user_id,
enforced via `assert_session_ownership(session_id)`.

These tests assert that when `assert_session_ownership` raises
`SessionOwnershipError`, the update/delete functions treat the file as
not-found (raise ValueError) and DO NOT execute the mutating query — no
cross-tenant leak. A positive path confirms the mutation proceeds when
ownership passes.
"""

import uuid as uuid_lib

import pytest

import domains.training.services.upload as upload
from shared.auth.ownership import SessionOwnershipError

FILE_ID = str(uuid_lib.uuid4())
SESSION_ID = str(uuid_lib.uuid4())


class _Result:
    def __init__(self, data):
        self.data = data


class _FilesQuery:
    """Fake Supabase query builder for the `files` table.

    Records whether update()/delete() were invoked so tests can assert that
    mutating operations were (not) reached.
    """

    def __init__(self, table):
        self.table = table

    def select(self, *_args, **_kwargs):
        return self

    def update(self, *_args, **_kwargs):
        self.table.update_called = True
        return self

    def delete(self, *_args, **_kwargs):
        self.table.delete_called = True
        return self

    def eq(self, *_args, **_kwargs):
        return self

    def execute(self):
        if self.table.update_called:
            return _Result([{'id': FILE_ID, 'file_name': 'f.csv'}])
        if self.table.delete_called:
            return _Result([{'id': FILE_ID, 'file_name': 'f.csv'}])
        # SELECT of the file row
        return _Result([{
            'id': FILE_ID,
            'session_id': SESSION_ID,
            'file_name': 'f.csv',
            'type': 'input',
            'storage_path': '',
        }])


class _FilesTable:
    def __init__(self):
        self.update_called = False
        self.delete_called = False

    def _q(self):
        return _FilesQuery(self)


class _FakeStorageBucket:
    def remove(self, *_args, **_kwargs):
        return None


class _FakeStorage:
    def from_(self, *_args, **_kwargs):
        return _FakeStorageBucket()


class _FakeClient:
    def __init__(self):
        self._files = _FilesTable()
        self.storage = _FakeStorage()

    def table(self, name):
        assert name == 'files'
        return self._files._q()


@pytest.fixture
def fake_client(monkeypatch):
    """Patch get_supabase_client (imported inside the upload functions)."""
    client = _FakeClient()
    monkeypatch.setattr(
        'shared.database.operations.get_supabase_client',
        lambda *a, **k: client,
    )
    return client


def _deny_ownership(monkeypatch):
    def _raise(_session_id):
        raise SessionOwnershipError('not owned')
    monkeypatch.setattr(upload, 'assert_session_ownership', _raise)


def _allow_ownership(monkeypatch):
    monkeypatch.setattr(upload, 'assert_session_ownership', lambda sid: str(sid))


# ─── delete_csv_file_record ───────────────────────────────────────────────────

def test_delete_denied_for_foreign_file(fake_client, monkeypatch):
    _deny_ownership(monkeypatch)

    with pytest.raises(ValueError):
        upload.delete_csv_file_record(FILE_ID)

    assert fake_client._files.delete_called is False


def test_delete_proceeds_for_owned_file(fake_client, monkeypatch):
    _allow_ownership(monkeypatch)

    result = upload.delete_csv_file_record(FILE_ID)

    assert fake_client._files.delete_called is True
    assert result['deleted_file']['id'] == FILE_ID


# ─── update_csv_file_record ───────────────────────────────────────────────────

def test_update_denied_for_foreign_file(fake_client, monkeypatch):
    _deny_ownership(monkeypatch)

    with pytest.raises(ValueError):
        upload.update_csv_file_record(FILE_ID, {'bezeichnung': 'hacked'})

    assert fake_client._files.update_called is False


def test_update_proceeds_for_owned_file(fake_client, monkeypatch):
    _allow_ownership(monkeypatch)

    result = upload.update_csv_file_record(FILE_ID, {'bezeichnung': 'mine'})

    assert fake_client._files.update_called is True
    assert result['id'] == FILE_ID
