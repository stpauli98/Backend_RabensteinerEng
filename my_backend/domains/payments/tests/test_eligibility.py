"""Unit tests for the API-Only eligibility helper.

The fake Supabase honors .eq()/.in_()/.limit() against canned rows, so a
missing `status='completed'` filter or a missing session-ownership join
would make these tests fail (not silently pass).
"""
from types import SimpleNamespace

from domains.payments.services.eligibility import has_trained_model


class _Query:
    def __init__(self, rows):
        self._rows = list(rows)

    def select(self, *a, **k):
        return self

    def eq(self, col, val):
        self._rows = [r for r in self._rows if r.get(col) == val]
        return self

    def in_(self, col, vals):
        self._rows = [r for r in self._rows if r.get(col) in vals]
        return self

    def limit(self, n):
        self._rows = self._rows[:n]
        return self

    def execute(self):
        return SimpleNamespace(data=self._rows)


class _FakeSupabase:
    def __init__(self, tables):
        self._tables = tables

    def table(self, name):
        return _Query(self._tables.get(name, []))


def _sb(sessions, training_results):
    return _FakeSupabase({'sessions': sessions, 'training_results': training_results})


def test_true_when_user_has_a_completed_model():
    sb = _sb(
        sessions=[{'id': 's1', 'user_id': 'u1'}],
        training_results=[{'id': 't1', 'session_id': 's1', 'status': 'completed'}],
    )
    assert has_trained_model(sb, 'u1') is True


def test_false_when_only_failed_trainings():
    sb = _sb(
        sessions=[{'id': 's1', 'user_id': 'u1'}],
        training_results=[{'id': 't1', 'session_id': 's1', 'status': 'failed'}],
    )
    assert has_trained_model(sb, 'u1') is False


def test_false_when_completed_model_belongs_to_another_user():
    sb = _sb(
        sessions=[{'id': 's2', 'user_id': 'u2'}],
        training_results=[{'id': 't1', 'session_id': 's2', 'status': 'completed'}],
    )
    assert has_trained_model(sb, 'u1') is False


def test_false_when_user_has_no_sessions():
    sb = _sb(sessions=[], training_results=[])
    assert has_trained_model(sb, 'u1') is False
