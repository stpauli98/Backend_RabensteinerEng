from datetime import datetime, timezone
from unittest.mock import MagicMock
from domains.retention import notices

NOW = datetime(2026, 6, 23, 12, 0, tzinfo=timezone.utc)

class FakeTable:
    def __init__(self, store): self.store = store; self._f = {}
    def insert(self, row): self.store.append(row); return self
    def select(self, *_): return self
    def eq(self, k, v): self._f[k] = v; return self
    def execute(self):
        rows = [r for r in self.store if all(r.get(k) == v for k, v in self._f.items())]
        self._f = {}; return _Resp(rows)
    def update(self, patch):
        for r in self.store:
            if all(r.get(k) == v for k, v in self._f.items()): r.update(patch)
        self._f = {}; return self

class _Resp:
    def __init__(self, data): self.data = data

class FakeSupabase:
    def __init__(self): self.store = []; self._t = FakeTable(self.store)
    def table(self, _): return self._t

def test_claim_notice_first_time_returns_true():
    sb = FakeSupabase()
    assert notices.claim_notice(sb, "s1", "u1", "warn1") is True
    assert sb.store[0]["status"] == "sending"

def test_claim_notice_duplicate_returns_false():
    sb = FakeSupabase()
    sb.store.append({"subscription_id": "s1", "kind": "warn1", "status": "sent"})
    assert notices.claim_notice(sb, "s1", "u1", "warn1") is False

def test_mark_sent_updates_row():
    sb = FakeSupabase()
    sb.store.append({"subscription_id": "s1", "kind": "warn1", "status": "sending"})
    notices.mark_sent(sb, "s1", "warn1", "msg_123", NOW)
    assert sb.store[0]["status"] == "sent"
    assert sb.store[0]["resend_message_id"] == "msg_123"

def test_mark_status_by_message_id_matches():
    sb = FakeSupabase()
    sb.store.append({"resend_message_id": "msg_123", "status": "sent"})
    assert notices.mark_status_by_message_id(sb, "msg_123", "bounced", NOW) is True
    assert sb.store[0]["status"] == "bounced"

def test_mark_status_by_message_id_no_match():
    sb = FakeSupabase()
    assert notices.mark_status_by_message_id(sb, "nope", "bounced", NOW) is False
