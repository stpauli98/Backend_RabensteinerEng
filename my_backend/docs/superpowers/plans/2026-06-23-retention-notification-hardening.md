# Retention Notification & Deletion Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Harden the live retention notify→delete lifecycle so data is never deleted without a fair, delivered warning, and the sweep's health is observable.

**Architecture:** A new `retention_notices` table becomes the source of truth for warning emails (id, status, Resend message_id). `eligibility.py` anchors deletion on a new `scheduled_deletion_at` column (guaranteeing ≥7 days post-warn1), protects `active/trialing/past_due`, and skips users whose warning bounced. A Resend webhook updates notice status and alerts the admin on bounce. Deletion only stamps `data_deleted_at` when every step verifiably succeeds. A GCP log-absence alert watches sweep health.

**Tech Stack:** Python 3.11, Flask blueprints, Supabase (PostgREST + Storage) via `supabase-py`, APScheduler, Resend HTTP API (Svix-signed webhooks), pytest, Docker Compose, GCP Cloud Run + Cloud Monitoring.

## Global Constraints

- Backend tests run **in Docker built with `.env`**: `docker compose run --rm -T backend python -m pytest <path> -q -p no:cacheprovider` (scope to `domains/retention/tests/` — a bare `pytest` collects legacy `OriginalTraining/.../test_plots.py` which aborts collection).
- Use `shared.datetime_utils.parse_iso_datetime()` for Supabase timestamps, never `datetime.fromisoformat()` (Python 3.9 fractional-seconds bug).
- Supabase admin client: `from shared.database.client import get_supabase_admin_client`.
- Prod Supabase project ref: `luvjebsltuttakatnzaa`. Cloud Run service `entropia`, GCP project `entropia-460611`, region `europe-west1`.
- New env vars must be added to Cloud Run via `gcloud run services update entropia --update-env-vars KEY=VALUE` (preserves all other vars). Never `--set-env-vars` (wipes Stripe live/Supabase/Resend keys).
- Branch: `feature/retention-notification-hardening`. Commit after every passing step.
- No raw secrets in commits or logs.

---

## File Structure

- `supabase/migrations/20260623_retention_hardening.sql` — **create**: schema migration (also applied to prod via Supabase MCP `apply_migration`).
- `domains/retention/constants.py` — **modify**: add `PROTECTED_STATUSES`, `RETENTION_DEFAULT_LANG`, `WARN1_WINDOW`, admin/webhook env accessors.
- `domains/retention/notices.py` — **create**: data-access for `retention_notices` (source of truth for sent warnings).
- `domains/retention/eligibility.py` — **modify**: protected statuses, anchor on `scheduled_deletion_at`, notice-driven state, skip bounced.
- `domains/retention/sweep.py` — **modify**: idempotent notice send, stamp `scheduled_deletion_at`, robust-deletion gating.
- `domains/retention/deletion.py` — **modify**: return structured result; verify storage.
- `domains/retention/email.py` — **modify**: render `scheduled_deletion_at`; language fallback.
- `domains/retention/webhook.py` — **create**: Svix signature verification (stdlib HMAC).
- `domains/retention/api/__init__.py` + `domains/retention/api/routes.py` — **create**: `retention_bp` (Resend webhook + admin run-sweep).
- `domains/retention/unsubscribed.py` — **create**: never-subscribed cleanup (#7).
- `core/blueprints.py` — **modify**: register `retention_bp` at `/api/retention`.
- `core/app_factory.py` — **modify**: wire never-subscribed pass into the sweep job.
- `domains/retention/tests/` — **modify/create**: `test_eligibility.py`, `test_notices.py`, `test_webhook.py`, `test_deletion.py`, `test_unsubscribed.py`.
- `scripts/provision_retention_alert.sh` — **create**: idempotent gcloud Cloud Monitoring alert provisioning (#4).

---

## Phase 1 — Schema

### Task 1: Migration — `scheduled_deletion_at` + `retention_notices` + backfill

**Files:**
- Create: `supabase/migrations/20260623_retention_hardening.sql`

**Interfaces:**
- Produces: column `user_subscriptions.scheduled_deletion_at timestamptz`; table `retention_notices(id, subscription_id, user_id, kind, resend_message_id, status, sent_at, created_at, updated_at)` with `unique(subscription_id, kind)`, indexes on `resend_message_id` and `user_id`.

- [ ] **Step 1: Write the migration SQL**

Create `supabase/migrations/20260623_retention_hardening.sql`:

```sql
-- Retention notification & deletion hardening (2026-06-23)

ALTER TABLE user_subscriptions
  ADD COLUMN IF NOT EXISTS scheduled_deletion_at timestamptz;

CREATE TABLE IF NOT EXISTS retention_notices (
  id                uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  subscription_id   uuid NOT NULL REFERENCES user_subscriptions(id) ON DELETE CASCADE,
  user_id           uuid NOT NULL,
  kind              text NOT NULL CHECK (kind IN ('warn1','warn2')),
  resend_message_id text,
  status            text NOT NULL DEFAULT 'sending'
                    CHECK (status IN ('sending','sent','delivered','bounced','complained','failed')),
  sent_at           timestamptz,
  created_at        timestamptz NOT NULL DEFAULT now(),
  updated_at        timestamptz NOT NULL DEFAULT now(),
  UNIQUE (subscription_id, kind)
);

CREATE INDEX IF NOT EXISTS idx_retention_notices_message_id
  ON retention_notices (resend_message_id);
CREATE INDEX IF NOT EXISTS idx_retention_notices_user_id
  ON retention_notices (user_id);

-- Backfill existing warn timestamps into notices (status 'sent', no message_id).
INSERT INTO retention_notices (subscription_id, user_id, kind, status, sent_at, created_at)
SELECT id, user_id, 'warn1', 'sent', retention_warn1_sent_at, retention_warn1_sent_at
FROM user_subscriptions
WHERE retention_warn1_sent_at IS NOT NULL
ON CONFLICT (subscription_id, kind) DO NOTHING;

INSERT INTO retention_notices (subscription_id, user_id, kind, status, sent_at, created_at)
SELECT id, user_id, 'warn2', 'sent', retention_warn2_sent_at, retention_warn2_sent_at
FROM user_subscriptions
WHERE retention_warn2_sent_at IS NOT NULL
ON CONFLICT (subscription_id, kind) DO NOTHING;

-- Backfill scheduled_deletion_at for already-warned rows: max(expires_at + 30d, warn1 + 7d).
UPDATE user_subscriptions
SET scheduled_deletion_at = GREATEST(expires_at + interval '30 days',
                                     retention_warn1_sent_at + interval '7 days')
WHERE retention_warn1_sent_at IS NOT NULL
  AND scheduled_deletion_at IS NULL;
```

- [ ] **Step 2: Apply to prod via Supabase MCP**

Use the Supabase MCP `apply_migration` tool with `project_id=luvjebsltuttakatnzaa`, name `retention_hardening_20260623`, and the SQL above.

- [ ] **Step 3: Verify schema**

Run via Supabase MCP `execute_sql`:
```sql
SELECT column_name FROM information_schema.columns
WHERE table_name='user_subscriptions' AND column_name='scheduled_deletion_at';
SELECT count(*) FROM retention_notices;
```
Expected: one column row; `retention_notices` count = number of existing warn1+warn2 stamps (currently 1 — the test account's warn1).

- [ ] **Step 4: Commit**

```bash
git add supabase/migrations/20260623_retention_hardening.sql
git commit -m "feat(retention): schema for notices + scheduled_deletion_at"
```

---

## Phase 2 — Eligibility

### Task 2: Constants

**Files:**
- Modify: `domains/retention/constants.py`

**Interfaces:**
- Produces: `PROTECTED_STATUSES: frozenset[str]`; `WARN1_WINDOW: timedelta`; `RETENTION_DEFAULT_LANG: str`; `admin_alert_email() -> str`; `resend_webhook_secret() -> str`; `admin_secret() -> str`.

- [ ] **Step 1: Add constants and accessors**

Append to `domains/retention/constants.py` (keep existing `DELETE_AFTER`, `WARN1_BEFORE`, etc.):

```python
# Subscription statuses that protect a user from deletion (#5).
PROTECTED_STATUSES = frozenset({"active", "trialing", "past_due"})

# Post-warn1 guaranteed notice window (#1): deletion is always >= this after warn1.
WARN1_WINDOW = timedelta(days=7)

# Email language fallback when user_metadata.lang is missing/unsupported (#8).
RETENTION_DEFAULT_LANG = os.environ.get("RETENTION_DEFAULT_LANG", "de")


def admin_alert_email() -> str:
    """Where bounce/complaint alerts go."""
    return os.environ.get("RETENTION_ADMIN_EMAIL", "")


def resend_webhook_secret() -> str:
    return os.environ.get("RESEND_WEBHOOK_SECRET", "")


def admin_secret() -> str:
    return os.environ.get("RETENTION_ADMIN_SECRET", "")
```

- [ ] **Step 2: Commit**

```bash
git add domains/retention/constants.py
git commit -m "feat(retention): protected statuses + config accessors"
```

### Task 3: Eligibility — anchor, protected statuses, skip bounced

**Files:**
- Modify: `domains/retention/eligibility.py`
- Test: `domains/retention/tests/test_eligibility.py`

**Interfaces:**
- Consumes: `PROTECTED_STATUSES`, `WARN1_WINDOW`, `DELETE_AFTER`, `WARN2_BEFORE`, `MIN_GAP` from constants.
- Produces: `compute_actions(subscriptions: list[dict], notices: list[dict], now: datetime) -> list[RetentionAction]`. `RetentionAction` unchanged (`user_id, subscription_id, action, deletion_date`); `action in {'warn1','warn2','delete'}`. Notice rows are dicts with keys `subscription_id, kind, status, sent_at`. Users whose latest warn notice is `bounced`/`complained` yield **no** action (paused).

- [ ] **Step 1: Write the failing tests**

Replace the body of `domains/retention/tests/test_eligibility.py` test cases with these (keep existing imports; add `from datetime import datetime, timedelta, timezone`):

```python
NOW = datetime(2026, 6, 23, 12, 0, tzinfo=timezone.utc)

def _sub(**kw):
    base = dict(id="s1", user_id="u1", status="cancelled",
                expires_at="2026-05-01T00:00:00+00:00",
                scheduled_deletion_at=None)
    base.update(kw); return base

def test_trialing_user_is_protected():
    subs = [_sub(status="trialing", expires_at="2026-12-01T00:00:00+00:00")]
    assert compute_actions(subs, [], NOW) == []

def test_past_due_user_is_protected():
    subs = [_sub(status="past_due", expires_at="2026-12-01T00:00:00+00:00")]
    assert compute_actions(subs, [], NOW) == []

def test_warn1_sets_no_action_when_outside_window():
    # expires 2026-06-20 -> deletion 2026-07-20 -> warn1 window opens 2026-07-13
    subs = [_sub(expires_at="2026-06-20T00:00:00+00:00")]
    assert compute_actions(subs, [], NOW) == []

def test_warn1_fires_inside_window():
    # lapsed long ago -> warn1 due now, no notice yet
    actions = compute_actions([_sub()], [], NOW)
    assert [a.action for a in actions] == ["warn1"]

def test_warn2_anchors_on_scheduled_deletion_at():
    # warn1 already sent; scheduled_deletion_at 24h from now -> warn2 due
    subs = [_sub(scheduled_deletion_at="2026-06-24T12:00:00+00:00")]
    notices = [{"subscription_id": "s1", "kind": "warn1", "status": "sent",
                "sent_at": "2026-06-22T12:00:00+00:00"}]
    actions = compute_actions(subs, notices, NOW)
    assert [a.action for a in actions] == ["warn2"]

def test_delete_after_warn2_and_scheduled_date():
    subs = [_sub(scheduled_deletion_at="2026-06-23T00:00:00+00:00")]
    notices = [
        {"subscription_id": "s1", "kind": "warn1", "status": "sent",
         "sent_at": "2026-06-15T12:00:00+00:00"},
        {"subscription_id": "s1", "kind": "warn2", "status": "sent",
         "sent_at": "2026-06-21T12:00:00+00:00"},
    ]
    actions = compute_actions(subs, notices, NOW)
    assert [a.action for a in actions] == ["delete"]

def test_bounced_warn_pauses_user():
    subs = [_sub(scheduled_deletion_at="2026-06-23T00:00:00+00:00")]
    notices = [{"subscription_id": "s1", "kind": "warn1", "status": "bounced",
                "sent_at": "2026-06-15T12:00:00+00:00"}]
    assert compute_actions(subs, notices, NOW) == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `docker compose run --rm -T backend python -m pytest domains/retention/tests/test_eligibility.py -q -p no:cacheprovider`
Expected: FAIL (signature mismatch — `compute_actions` takes 2 args, not 3).

- [ ] **Step 3: Rewrite `compute_actions`**

Replace `compute_actions` in `domains/retention/eligibility.py`:

```python
from domains.retention.constants import (
    DELETE_AFTER, WARN1_BEFORE, WARN2_BEFORE, MIN_GAP, WARN1_WINDOW,
    PROTECTED_STATUSES,
)

def _latest_notice(notices, subscription_id, kind):
    rows = [n for n in notices
            if n.get("subscription_id") == subscription_id and n.get("kind") == kind]
    if not rows:
        return None
    return max(rows, key=lambda n: _dt(n.get("sent_at")) or datetime.min.replace(tzinfo=None))

def compute_actions(subscriptions, notices, now):
    by_user = {}
    for row in subscriptions:
        by_user.setdefault(row['user_id'], []).append(row)

    actions = []
    for user_id, rows in by_user.items():
        has_active = any(
            r.get('status') in PROTECTED_STATUSES
            and (_dt(r.get('expires_at')) or now) > now
            for r in rows
        )
        if has_active:
            continue

        row = max(rows, key=lambda r: _dt(r.get('expires_at'))
                  or datetime.min.replace(tzinfo=now.tzinfo))
        if row.get('data_deleted_at'):
            continue

        lapsed_at = _dt(row.get('expires_at'))
        if lapsed_at is None:
            continue

        sub_id = row['id']
        w1 = _latest_notice(notices, sub_id, 'warn1')
        w2 = _latest_notice(notices, sub_id, 'warn2')

        # Pause: a bounced/complained warning means we cannot prove notice.
        if (w1 and w1.get('status') in ('bounced', 'complained')) or \
           (w2 and w2.get('status') in ('bounced', 'complained')):
            continue

        w1_sent = _dt(w1.get('sent_at')) if w1 else None
        w2_sent = _dt(w2.get('sent_at')) if w2 else None

        # Anchor: once warn1 is sent, scheduled_deletion_at drives everything.
        scheduled = _dt(row.get('scheduled_deletion_at'))
        deletion_date = scheduled or (lapsed_at + DELETE_AFTER)

        action = None
        if w1_sent is None:
            if now >= (lapsed_at + DELETE_AFTER) - WARN1_BEFORE:
                action = 'warn1'
        elif w2_sent is None:
            if now >= max(deletion_date - WARN2_BEFORE, w1_sent + MIN_GAP):
                action = 'warn2'
        else:
            if now >= max(deletion_date, w2_sent + MIN_GAP):
                action = 'delete'

        if action:
            actions.append(RetentionAction(
                user_id=user_id, subscription_id=sub_id,
                action=action, deletion_date=deletion_date,
            ))
    return actions
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `docker compose run --rm -T backend python -m pytest domains/retention/tests/test_eligibility.py -q -p no:cacheprovider`
Expected: PASS (all eligibility tests).

- [ ] **Step 5: Commit**

```bash
git add domains/retention/eligibility.py domains/retention/tests/test_eligibility.py
git commit -m "feat(retention): anchor deletion on scheduled_deletion_at, protect statuses, skip bounced"
```

---

## Phase 3 — Notices & Email

### Task 4: Notices data-access module

**Files:**
- Create: `domains/retention/notices.py`
- Test: `domains/retention/tests/test_notices.py`

**Interfaces:**
- Produces:
  - `fetch_notices(supabase) -> list[dict]` — all rows from `retention_notices`.
  - `claim_notice(supabase, subscription_id, user_id, kind) -> bool` — insert a `status='sending'` row; returns `False` if a non-failed notice of that kind already exists (idempotency guard, #10).
  - `mark_sent(supabase, subscription_id, kind, message_id, now)` — set status `sent`, `resend_message_id`, `sent_at`.
  - `mark_status_by_message_id(supabase, message_id, status, now) -> bool` — webhook updates status; returns `True` if a row matched.

- [ ] **Step 1: Write the failing tests**

Create `domains/retention/tests/test_notices.py`:

```python
from datetime import datetime, timezone
from unittest.mock import MagicMock
from domains.retention import notices

NOW = datetime(2026, 6, 23, 12, 0, tzinfo=timezone.utc)

class FakeTable:
    def __init__(self, store): self.store = store; self._f = {}
    def insert(self, row): self.store.append(row); return _Resp([row])
    def select(self, *_): return self
    def eq(self, k, v): self._f[k] = v; return self
    def execute(self):
        rows = [r for r in self.store if all(r.get(k) == v for k, v in self._f.items())]
        self._f = {}; return _Resp(rows)
    def update(self, patch):
        for r in self.store:
            if all(r.get(k) == v for k, v in self._f.items()): r.update(patch)
        self._f = {}; return _Resp(self.store)

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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `docker compose run --rm -T backend python -m pytest domains/retention/tests/test_notices.py -q -p no:cacheprovider`
Expected: FAIL ("No module named 'domains.retention.notices'").

- [ ] **Step 3: Implement `notices.py`**

Create `domains/retention/notices.py`:

```python
"""Data-access for retention_notices — source of truth for sent warnings."""
from datetime import datetime

_TABLE = "retention_notices"
_ACTIVE = ("sending", "sent", "delivered")


def fetch_notices(supabase) -> list:
    return supabase.table(_TABLE).select(
        "subscription_id,user_id,kind,resend_message_id,status,sent_at"
    ).execute().data or []


def claim_notice(supabase, subscription_id: str, user_id: str, kind: str) -> bool:
    """Insert a 'sending' notice. Returns False if a non-failed notice already exists."""
    existing = (supabase.table(_TABLE).select("status")
                .eq("subscription_id", subscription_id).eq("kind", kind).execute().data or [])
    if any(r.get("status") in _ACTIVE for r in existing):
        return False
    supabase.table(_TABLE).insert({
        "subscription_id": subscription_id, "user_id": user_id,
        "kind": kind, "status": "sending",
    }).execute()
    return True


def mark_sent(supabase, subscription_id: str, kind: str, message_id: str, now: datetime) -> None:
    (supabase.table(_TABLE).update({
        "resend_message_id": message_id, "status": "sent",
        "sent_at": now.isoformat(), "updated_at": now.isoformat(),
    }).eq("subscription_id", subscription_id).eq("kind", kind).execute())


def mark_status_by_message_id(supabase, message_id: str, status: str, now: datetime) -> bool:
    matched = (supabase.table(_TABLE).select("id")
               .eq("resend_message_id", message_id).execute().data or [])
    if not matched:
        return False
    (supabase.table(_TABLE).update({"status": status, "updated_at": now.isoformat()})
     .eq("resend_message_id", message_id).execute())
    return True
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `docker compose run --rm -T backend python -m pytest domains/retention/tests/test_notices.py -q -p no:cacheprovider`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add domains/retention/notices.py domains/retention/tests/test_notices.py
git commit -m "feat(retention): retention_notices data-access (idempotent claim + status)"
```

### Task 5: Sweep — idempotent send, stamp scheduled_deletion_at, fetch notices

**Files:**
- Modify: `domains/retention/sweep.py`
- Test: `domains/retention/tests/test_sweep.py`

**Interfaces:**
- Consumes: `fetch_notices`, `claim_notice`, `mark_sent` from `notices`; `compute_actions(subscriptions, notices, now)`; `WARN1_WINDOW`, `DELETE_AFTER`.
- Produces: `run_sweep(supabase, *, now=None, dry_run)` unchanged signature; on a `warn1`/`warn2` action it claims+sends+marks the notice and (for warn1) stamps `user_subscriptions.scheduled_deletion_at = max(expires_at+30d, now+7d)`.

- [ ] **Step 1: Write the failing test**

Add to `domains/retention/tests/test_sweep.py` (self-contained via monkeypatch on the names `sweep.py` imports after Task 5):

```python
def test_warn1_stamps_scheduled_deletion_at(monkeypatch):
    from datetime import datetime, timezone
    from unittest.mock import MagicMock
    from domains.retention import sweep
    from domains.retention.eligibility import RetentionAction

    now = datetime(2026, 6, 23, 12, 0, tzinfo=timezone.utc)
    row = {"id": "s1", "user_id": "u1", "expires_at": "2026-01-01T00:00:00+00:00"}
    updates = []

    sb = MagicMock()
    def fake_update(patch):
        updates.append(patch)
        chain = MagicMock()
        chain.eq.return_value.execute.return_value = MagicMock(data=[])
        return chain
    sb.table.return_value.update.side_effect = fake_update

    monkeypatch.setattr(sweep, "_claim_daily_lock", lambda *a, **k: True)
    monkeypatch.setattr(sweep, "_fetch_subscriptions", lambda *a, **k: [row])
    monkeypatch.setattr(sweep, "fetch_notices", lambda *a, **k: [])
    monkeypatch.setattr(sweep, "compute_actions",
                        lambda *a, **k: [RetentionAction("u1", "s1", "warn1", now)])
    monkeypatch.setattr(sweep, "claim_notice", lambda *a, **k: True)
    monkeypatch.setattr(sweep, "_user_email_lang", lambda *a, **k: ("x@y.z", "de"))
    monkeypatch.setattr(sweep, "send_warning", lambda **k: "msg_1")
    monkeypatch.setattr(sweep, "mark_sent", lambda *a, **k: None)

    sweep.run_sweep(sb, now=now, dry_run=False)

    # expires(2026-01-01)+30d is well before now+7d, so scheduled = now + 7 days.
    sched = [u["scheduled_deletion_at"] for u in updates if "scheduled_deletion_at" in u]
    assert sched and sched[0].startswith("2026-06-30")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `docker compose run --rm -T backend python -m pytest domains/retention/tests/test_sweep.py -q -p no:cacheprovider`
Expected: FAIL.

- [ ] **Step 3: Update `run_sweep` and `_handle`**

In `domains/retention/sweep.py`: change the `compute_actions` call to pass notices, and rework `_handle`:

```python
from domains.retention.notices import fetch_notices, claim_notice, mark_sent
from domains.retention.constants import WARN1_WINDOW, DELETE_AFTER

def _stamp_scheduled_deletion(supabase, subscription_id, row, now):
    from shared.datetime_utils import parse_iso_datetime
    lapsed = parse_iso_datetime(row['expires_at'])
    scheduled = max(lapsed + DELETE_AFTER, now + WARN1_WINDOW)
    (supabase.table("user_subscriptions")
     .update({"scheduled_deletion_at": scheduled.isoformat()})
     .eq("id", subscription_id).execute())

def _handle(supabase, action, row, now):
    if action.action in ("warn1", "warn2"):
        if not claim_notice(supabase, action.subscription_id, action.user_id, action.action):
            return  # already sent (idempotent)
        email, lang = _user_email_lang(supabase, action.user_id)
        if not email:
            raise RuntimeError(f"no email for user {action.user_id}")
        is_final = action.action == "warn2"
        message_id = send_warning(
            api_key=os.environ["RESEND_API_KEY"],
            from_addr=f'{os.environ.get("EMAIL_FROM_NAME", "Forecast Engine")} '
                      f'<{os.environ["EMAIL_FROM_ADDRESS"]}>',
            to=email, lang=lang,
            deletion_date=action.deletion_date.date().isoformat(),
            login_url=login_redirect_url(), is_final=is_final,
        )
        mark_sent(supabase, action.subscription_id, action.action, message_id, now)
        if action.action == "warn1":
            _stamp_scheduled_deletion(supabase, action.subscription_id, row, now)
    else:  # delete
        delete_user_data(supabase, action.user_id)
        _stamp(supabase, action.subscription_id, "data_deleted_at", now)
```

Then in `run_sweep`, fetch subscriptions + notices and pass both; pass the matching `row` to `_handle` (look it up from the fetched subscriptions by `action.subscription_id`):

```python
    subs = _fetch_subscriptions(supabase)
    notices = fetch_notices(supabase)
    actions = compute_actions(subs, notices, now)
    rows_by_id = {s['id']: s for s in subs}
    ...
    for action in actions:
        if dry_run:
            logger.info("retention DRY-RUN: would %s user %s (deletion %s)",
                        action.action, action.user_id, action.deletion_date.date())
            continue
        try:
            _handle(supabase, action, rows_by_id[action.subscription_id], now)
            done += 1
        except Exception:
            errors += 1
            logger.exception("retention: action %s failed for user %s",
                             action.action, action.user_id)
```

Update `_SUB_COLS` to include `scheduled_deletion_at`:
```python
_SUB_COLS = ("id,user_id,status,expires_at,scheduled_deletion_at,"
             "retention_warn1_sent_at,retention_warn2_sent_at,data_deleted_at")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `docker compose run --rm -T backend python -m pytest domains/retention/tests/test_sweep.py -q -p no:cacheprovider`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add domains/retention/sweep.py domains/retention/tests/test_sweep.py
git commit -m "feat(retention): idempotent notice send + stamp scheduled_deletion_at"
```

### Task 6: Email — render scheduled date + language fallback

**Files:**
- Modify: `domains/retention/email.py`
- Test: `domains/retention/tests/test_email.py`

**Interfaces:**
- Consumes: `RETENTION_DEFAULT_LANG`.
- Produces: `send_warning(...) -> str` (now returns the Resend message id — already does, via `send_email`); `_lang()` falls back to `RETENTION_DEFAULT_LANG`.

- [ ] **Step 1: Write the failing test**

Add to `domains/retention/tests/test_email.py`:

```python
def test_lang_falls_back_to_configured_default(monkeypatch):
    monkeypatch.setenv("RETENTION_DEFAULT_LANG", "en")
    import importlib
    from domains.retention import constants, email
    importlib.reload(constants); importlib.reload(email)
    assert email._lang("fr") == "en"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `docker compose run --rm -T backend python -m pytest domains/retention/tests/test_email.py -q -p no:cacheprovider`
Expected: FAIL (`_lang("fr")` returns `"de"`).

- [ ] **Step 3: Update `_lang`**

In `domains/retention/email.py` replace the default:

```python
from domains.retention.constants import RETENTION_DEFAULT_LANG

def _lang(lang: str) -> str:
    return lang if lang in _SUPPORTED else RETENTION_DEFAULT_LANG
```

(The `deletion_date` already comes from `action.deletion_date` which now equals `scheduled_deletion_at` — #9 is satisfied by Task 5; no change needed in the render path.)

- [ ] **Step 4: Run test to verify it passes**

Run: `docker compose run --rm -T backend python -m pytest domains/retention/tests/test_email.py -q -p no:cacheprovider`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add domains/retention/email.py domains/retention/tests/test_email.py
git commit -m "feat(retention): configurable email language fallback"
```

---

## Phase 4 — Resend Webhook

### Task 7: Svix signature verification + webhook + admin route

**Files:**
- Create: `domains/retention/webhook.py`
- Create: `domains/retention/api/__init__.py`, `domains/retention/api/routes.py`
- Modify: `core/blueprints.py`
- Test: `domains/retention/tests/test_webhook.py`

**Interfaces:**
- Consumes: `mark_status_by_message_id`, `resend_webhook_secret`, `admin_alert_email`, `admin_secret`, `run_sweep`, `dry_run`.
- Produces: `verify_svix(secret, headers, body) -> bool`; blueprint `retention_bp` with `POST /resend-webhook` and `POST /run-sweep`.

- [ ] **Step 1: Write the failing tests**

Create `domains/retention/tests/test_webhook.py`:

```python
import base64, hashlib, hmac, json, time
from domains.retention.webhook import verify_svix

def _sign(secret_b64, msg_id, ts, body):
    key = base64.b64decode(secret_b64)
    signed = f"{msg_id}.{ts}.{body}".encode()
    sig = base64.b64encode(hmac.new(key, signed, hashlib.sha256).digest()).decode()
    return f"v1,{sig}"

def test_verify_svix_accepts_valid_signature():
    secret = base64.b64encode(b"0123456789abcdef").decode()
    body = json.dumps({"type": "email.bounced"})
    ts = str(int(time.time()))
    headers = {"svix-id": "msg_1", "svix-timestamp": ts,
               "svix-signature": _sign(secret, "msg_1", ts, body)}
    assert verify_svix(f"whsec_{secret}", headers, body) is True

def test_verify_svix_rejects_bad_signature():
    secret = base64.b64encode(b"0123456789abcdef").decode()
    body = json.dumps({"type": "email.bounced"})
    ts = str(int(time.time()))
    headers = {"svix-id": "msg_1", "svix-timestamp": ts,
               "svix-signature": "v1,AAAA"}
    assert verify_svix(f"whsec_{secret}", headers, body) is False

def test_verify_svix_missing_headers_returns_false():
    assert verify_svix("whsec_x", {}, "{}") is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `docker compose run --rm -T backend python -m pytest domains/retention/tests/test_webhook.py -q -p no:cacheprovider`
Expected: FAIL ("No module named 'domains.retention.webhook'").

- [ ] **Step 3: Implement `webhook.py`**

Create `domains/retention/webhook.py`:

```python
"""Svix webhook signature verification (Resend uses Svix). Stdlib only."""
import base64
import hashlib
import hmac


def verify_svix(secret: str, headers: dict, body: str) -> bool:
    msg_id = headers.get("svix-id")
    timestamp = headers.get("svix-timestamp")
    sig_header = headers.get("svix-signature")
    if not (msg_id and timestamp and sig_header and secret):
        return False
    raw = secret.split("_", 1)[1] if secret.startswith("whsec_") else secret
    try:
        key = base64.b64decode(raw)
    except Exception:
        return False
    signed = f"{msg_id}.{timestamp}.{body}".encode()
    expected = base64.b64encode(hmac.new(key, signed, hashlib.sha256).digest()).decode()
    # svix-signature can carry multiple space-separated "vN,sig" entries.
    for part in sig_header.split(" "):
        _, _, sig = part.partition(",")
        if sig and hmac.compare_digest(sig, expected):
            return True
    return False
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `docker compose run --rm -T backend python -m pytest domains/retention/tests/test_webhook.py -q -p no:cacheprovider`
Expected: PASS.

- [ ] **Step 5: Implement the routes**

Create `domains/retention/api/__init__.py`:

```python
from domains.retention.api.routes import retention_bp  # noqa: F401
```

Create `domains/retention/api/routes.py`:

```python
"""Resend webhook + admin sweep trigger for the retention domain."""
import json
import logging
import os

from flask import Blueprint, request, jsonify

from shared.datetime_utils import parse_iso_datetime  # noqa: F401 (consistency)
from datetime import datetime, timezone

from domains.retention.webhook import verify_svix
from domains.retention.notices import mark_status_by_message_id
from domains.retention.constants import (
    resend_webhook_secret, admin_alert_email, admin_secret, dry_run,
)
from domains.retention.email import send_warning  # reused only for admin alert text? no
from domains.auth_emails.services.resend_client import send_email
from shared.database.client import get_supabase_admin_client

logger = logging.getLogger(__name__)
retention_bp = Blueprint("retention", __name__)

_STATUS_MAP = {
    "email.delivered": "delivered",
    "email.bounced": "bounced",
    "email.complained": "complained",
}


def _alert_admin(subject: str, html: str) -> None:
    to = admin_alert_email()
    if not to:
        logger.warning("retention: RETENTION_ADMIN_EMAIL unset; cannot send alert")
        return
    try:
        send_email(api_key=os.environ["RESEND_API_KEY"],
                   from_addr=f'{os.environ.get("EMAIL_FROM_NAME", "Forecast Engine")} '
                             f'<{os.environ["EMAIL_FROM_ADDRESS"]}>',
                   to=to, subject=subject, html=html)
    except Exception:
        logger.exception("retention: failed to send admin alert")


@retention_bp.route("/resend-webhook", methods=["POST"])
def resend_webhook():
    body = request.get_data(as_text=True)
    if not verify_svix(resend_webhook_secret(), dict(request.headers), body):
        return jsonify({"error": "invalid signature"}), 401
    event = json.loads(body)
    event_type = event.get("type")
    status = _STATUS_MAP.get(event_type)
    if not status:
        return jsonify({"status": "ignored"}), 200
    message_id = (event.get("data") or {}).get("email_id") \
        or (event.get("data") or {}).get("id")
    now = datetime.now(timezone.utc)
    supabase = get_supabase_admin_client()
    matched = mark_status_by_message_id(supabase, message_id, status, now)
    if matched and status in ("bounced", "complained"):
        _alert_admin(
            subject="[Forecast Engine] Retention warning bounced",
            html=f"A retention warning email had status <b>{status}</b> "
                 f"(message {message_id}). The user's data deletion is paused "
                 f"pending manual review.",
        )
    return jsonify({"status": "ok", "matched": matched}), 200


@retention_bp.route("/run-sweep", methods=["POST"])
def run_sweep_now():
    if request.headers.get("X-Retention-Secret") != admin_secret() or not admin_secret():
        return jsonify({"error": "unauthorized"}), 401
    from domains.retention.sweep import run_sweep
    result = run_sweep(get_supabase_admin_client(), dry_run=dry_run())
    return jsonify(result), 200
```

> Note: remove the unused `send_warning`/`parse_iso_datetime` imports if your linter is strict; they are listed for clarity but only `send_email` and `datetime` are used.

- [ ] **Step 6: Register the blueprint**

In `core/blueprints.py` add inside `register_blueprints`:

```python
    from domains.retention.api import retention_bp
    ...
    app.register_blueprint(retention_bp, url_prefix='/api/retention')
```

- [ ] **Step 7: Verify app boots in Docker**

Run: `docker compose run --rm -T backend python -c "from core.app_factory import create_app; create_app(); print('OK')"`
Expected: prints `OK` (blueprint imports resolve).

- [ ] **Step 8: Commit**

```bash
git add domains/retention/webhook.py domains/retention/api/ core/blueprints.py domains/retention/tests/test_webhook.py
git commit -m "feat(retention): Resend webhook (svix verify) + admin run-sweep endpoint"
```

---

## Phase 5 — Robust Deletion

### Task 8: Verify-before-stamp deletion

**Files:**
- Modify: `domains/retention/deletion.py`
- Modify: `domains/retention/sweep.py`
- Test: `domains/retention/tests/test_deletion.py`

**Interfaces:**
- Produces: `delete_user_data(supabase, user_id) -> dict` returning `{"errors": list[str], "storage_files_remaining": int}`. Empty `errors` and zero remaining ⇒ success. The sweep stamps `data_deleted_at` **only** when `delete_user_data` reports no errors and no remaining files.

- [ ] **Step 1: Write the failing tests**

Add to `domains/retention/tests/test_deletion.py`:

```python
from unittest.mock import patch, MagicMock
from domains.retention import deletion

def test_delete_returns_errors_when_sessions_have_warnings():
    sb = MagicMock()
    with patch.object(deletion, "delete_all_sessions",
                      return_value={"warnings": ["Table files: boom"], "summary": {}}):
        # api_keys/usage tables delete cleanly
        sb.table.return_value.delete.return_value.eq.return_value.execute.return_value.data = []
        result = deletion.delete_user_data(sb, "u1")
    assert result["errors"]  # non-empty -> caller must NOT stamp

def test_delete_clean_returns_no_errors():
    sb = MagicMock()
    sb.table.return_value.delete.return_value.eq.return_value.execute.return_value.data = []
    with patch.object(deletion, "delete_all_sessions", return_value={"summary": {}}), \
         patch.object(deletion, "_count_remaining_storage", return_value=0):
        result = deletion.delete_user_data(sb, "u1")
    assert result["errors"] == [] and result["storage_files_remaining"] == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `docker compose run --rm -T backend python -m pytest domains/retention/tests/test_deletion.py -q -p no:cacheprovider`
Expected: FAIL (`delete_user_data` returns `None`).

- [ ] **Step 3: Update `delete_user_data`**

Replace `domains/retention/deletion.py`:

```python
"""Delete a single user's data (keeps the account + subscription history)."""
import logging

from domains.training.services.session import delete_all_sessions

logger = logging.getLogger(__name__)

_USER_TABLES = ("api_keys", "usage_events", "usage_tracking")


def _count_remaining_storage(supabase, user_id: str) -> int:
    """Files still referenced by this user's sessions after deletion."""
    sess = (supabase.table("sessions").select("id").eq("user_id", user_id).execute().data or [])
    ids = [s["id"] for s in sess]
    if not ids:
        return 0
    files = (supabase.table("files").select("id").in_("session_id", ids).execute().data or [])
    return len(files)


def delete_user_data(supabase, user_id: str) -> dict:
    """Idempotent. Returns {'errors': [...], 'storage_files_remaining': N}.
    Caller stamps data_deleted_at only when errors == [] and remaining == 0."""
    errors = []
    result = delete_all_sessions(confirm=True, user_id=user_id)
    errors.extend(result.get("warnings", []))

    for table in _USER_TABLES:
        try:
            supabase.table(table).delete().eq("user_id", user_id).execute()
        except Exception as exc:
            errors.append(f"{table}: {exc}")

    remaining = _count_remaining_storage(supabase, user_id)
    if remaining:
        errors.append(f"{remaining} storage files still present")
    logger.info("retention: purge for user %s -> errors=%d remaining=%d",
                user_id, len(errors), remaining)
    return {"errors": errors, "storage_files_remaining": remaining}
```

- [ ] **Step 4: Gate the stamp in the sweep**

In `domains/retention/sweep.py` `_handle`, replace the `else: # delete` branch:

```python
    else:  # delete
        outcome = delete_user_data(supabase, action.user_id)
        if outcome["errors"]:
            raise RuntimeError(f"deletion incomplete for {action.user_id}: {outcome['errors']}")
        _stamp(supabase, action.subscription_id, "data_deleted_at", now)
```

(The `raise` routes to the existing `except` in `run_sweep`, incrementing `errors` and retrying next sweep without stamping.)

- [ ] **Step 5: Run tests to verify they pass**

Run: `docker compose run --rm -T backend python -m pytest domains/retention/tests/test_deletion.py domains/retention/tests/test_sweep.py -q -p no:cacheprovider`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add domains/retention/deletion.py domains/retention/sweep.py domains/retention/tests/test_deletion.py
git commit -m "feat(retention): verify-before-stamp deletion (no data_deleted_at on partial failure)"
```

---

## Phase 6 — Operations

### Task 9: Cloud Monitoring sweep-staleness alert (#4)

**Files:**
- Create: `scripts/provision_retention_alert.sh`

**Interfaces:**
- Produces: an idempotent script that creates (or no-ops) a log-based alert policy firing when no `Retention sweep result` log appears for 36h, notifying an email channel.

- [ ] **Step 1: Write the provisioning script**

Create `scripts/provision_retention_alert.sh`:

```bash
#!/usr/bin/env bash
# Idempotent: creates a Cloud Monitoring log-absence alert for the retention sweep.
set -euo pipefail
PROJECT=entropia-460611
EMAIL="${1:?usage: provision_retention_alert.sh <alert-email>}"

# 1) Log-based metric counting successful sweep results.
gcloud logging metrics describe retention_sweep_ran --project "$PROJECT" >/dev/null 2>&1 || \
gcloud logging metrics create retention_sweep_ran --project "$PROJECT" \
  --description="Retention sweep completed runs" \
  --log-filter='resource.type="cloud_run_revision" AND resource.labels.service_name="entropia" AND textPayload:"Retention sweep result"'

# 2) Notification channel (email).
CH=$(gcloud beta monitoring channels list --project "$PROJECT" \
      --filter="type=email AND labels.email_address=$EMAIL" --format='value(name)' | head -1)
if [ -z "$CH" ]; then
  CH=$(gcloud beta monitoring channels create --project "$PROJECT" \
        --display-name="Retention alerts" --type=email \
        --channel-labels="email_address=$EMAIL" --format='value(name)')
fi

# 3) Alert policy: metric absent for 36h.
gcloud alpha monitoring policies list --project "$PROJECT" \
  --filter='displayName="Retention sweep stale"' --format='value(name)' | grep -q . || \
gcloud alpha monitoring policies create --project "$PROJECT" \
  --display-name="Retention sweep stale" \
  --notification-channels="$CH" \
  --condition-display-name="No sweep result in 36h" \
  --condition-filter='metric.type="logging.googleapis.com/user/retention_sweep_ran" AND resource.type="cloud_run_revision"' \
  --condition-threshold-comparison=COMPARISON_LT \
  --condition-threshold-value=1 \
  --condition-threshold-duration=129600s \
  --aggregation='{"alignmentPeriod":"3600s","perSeriesAligner":"ALIGN_COUNT"}'
echo "Provisioned retention staleness alert."
```

- [ ] **Step 2: Run it (manual verification)**

Run: `bash scripts/provision_retention_alert.sh markusrabensteiner@gmx.at`
Expected: prints `Provisioned retention staleness alert.`; the policy appears in `gcloud alpha monitoring policies list --project entropia-460611`.

- [ ] **Step 3: Commit**

```bash
git add scripts/provision_retention_alert.sh
git commit -m "feat(retention): Cloud Monitoring sweep-staleness alert provisioning"
```

### Task 10: Configure new env vars + Resend webhook (manual ops)

**Files:** none (operational).

- [ ] **Step 1: Set Cloud Run env vars (preserving all others)**

Run:
```bash
gcloud run services update entropia --project entropia-460611 --region europe-west1 \
  --update-env-vars RETENTION_ADMIN_EMAIL=markusrabensteiner@gmx.at,RETENTION_ADMIN_SECRET=<gen>,RETENTION_WEBHOOK_PLACEHOLDER=1
```
Then set `RESEND_WEBHOOK_SECRET` after creating the Resend webhook (next step).

- [ ] **Step 2: Create the Resend webhook**

In the Resend dashboard, add a webhook → URL `https://entropia-rhkwpfnuza-ew.a.run.app/api/retention/resend-webhook`, events `email.delivered`, `email.bounced`, `email.complained`. Copy the signing secret (`whsec_...`) and set it:
```bash
gcloud run services update entropia --project entropia-460611 --region europe-west1 \
  --update-env-vars RESEND_WEBHOOK_SECRET=whsec_xxx
```

- [ ] **Step 3: Smoke-test the webhook**

Send a test event from the Resend dashboard; confirm Cloud Run logs show a `200` and (for a bounce test on a known message) the `retention_notices.status` updates. No commit (ops only).

---

## Phase 7 — Never-Subscribed Cleanup (#7)

### Task 11: Stale never-subscribed data cleanup

**Files:**
- Create: `domains/retention/unsubscribed.py`
- Modify: `core/app_factory.py`
- Test: `domains/retention/tests/test_unsubscribed.py`

**Interfaces:**
- Consumes: `delete_user_data`; `get_supabase_admin_client`.
- Produces: `find_stale_unsubscribed(supabase, now, *, max_age_days=180) -> list[str]` (user_ids with sessions but no `user_subscriptions` row and newest session older than `max_age_days`); `sweep_unsubscribed(supabase, *, now=None, dry_run, max_age_days=180) -> dict`.

> Threshold decision (locked here): **180 days of inactivity** (no new session) for users who never had any `user_subscriptions` row. Conservative; Markus approved the policy in principle.

- [ ] **Step 1: Write the failing tests**

Create `domains/retention/tests/test_unsubscribed.py`:

```python
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `docker compose run --rm -T backend python -m pytest domains/retention/tests/test_unsubscribed.py -q -p no:cacheprovider`
Expected: FAIL ("No module named 'domains.retention.unsubscribed'").

- [ ] **Step 3: Implement `unsubscribed.py`**

Create `domains/retention/unsubscribed.py`:

```python
"""Cleanup of users who uploaded data but never subscribed (#7)."""
import logging
from datetime import datetime, timedelta, timezone

from shared.datetime_utils import parse_iso_datetime
from domains.retention.deletion import delete_user_data

logger = logging.getLogger(__name__)


def _users_with_sessions_no_sub(supabase) -> dict:
    """Return {user_id: newest_session_created_at_iso} for users with sessions
    but no user_subscriptions row."""
    sub_users = {r["user_id"] for r in
                 (supabase.table("user_subscriptions").select("user_id").execute().data or [])}
    newest = {}
    rows = supabase.table("sessions").select("user_id,created_at").execute().data or []
    for r in rows:
        uid = r.get("user_id")
        if not uid or uid in sub_users:
            continue
        cur = newest.get(uid)
        if cur is None or (r.get("created_at") or "") > cur:
            newest[uid] = r.get("created_at")
    return newest


def find_stale_unsubscribed(supabase, now: datetime, *, max_age_days: int = 180) -> list:
    cutoff = now - timedelta(days=max_age_days)
    stale = []
    for uid, newest in _users_with_sessions_no_sub(supabase).items():
        dt = parse_iso_datetime(newest) if newest else None
        if dt is not None and dt < cutoff:
            stale.append(uid)
    return stale


def sweep_unsubscribed(supabase, *, now=None, dry_run: bool, max_age_days: int = 180) -> dict:
    now = now or datetime.now(timezone.utc)
    stale = find_stale_unsubscribed(supabase, now, max_age_days=max_age_days)
    deleted = errors = 0
    for uid in stale:
        if dry_run:
            logger.info("unsubscribed DRY-RUN: would delete data for user %s", uid)
            continue
        try:
            outcome = delete_user_data(supabase, uid)
            if outcome["errors"]:
                raise RuntimeError(outcome["errors"])
            deleted += 1
        except Exception:
            errors += 1
            logger.exception("unsubscribed: delete failed for user %s", uid)
    return {"planned": len(stale), "deleted": deleted, "errors": errors}
```

- [ ] **Step 4: Wire into the scheduled sweep**

In `core/app_factory.py` `run_data_retention_sweep`, after the existing `run_sweep(...)` call add:

```python
            from domains.retention.unsubscribed import sweep_unsubscribed
            unsub = sweep_unsubscribed(get_supabase_admin_client(), dry_run=dry_run())
            logger.info(f"Unsubscribed cleanup result: {unsub}")
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `docker compose run --rm -T backend python -m pytest domains/retention/tests/test_unsubscribed.py -q -p no:cacheprovider`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add domains/retention/unsubscribed.py core/app_factory.py domains/retention/tests/test_unsubscribed.py
git commit -m "feat(retention): never-subscribed stale-data cleanup (180d)"
```

---

## Final verification

- [ ] **Step 1: Full retention suite in Docker**

Run: `docker compose run --rm -T backend python -m pytest domains/retention/tests/ -q -p no:cacheprovider`
Expected: all retention tests PASS (original 20 + new), no regressions.

- [ ] **Step 2: App boots**

Run: `docker compose run --rm -T backend python -c "from core.app_factory import create_app; create_app(); print('OK')"`
Expected: `OK`.

- [ ] **Step 3: Open PR**

```bash
git push -u origin feature/retention-notification-hardening
gh pr create --title "Retention notification & deletion hardening" --body "Implements docs/superpowers/specs/2026-06-23-retention-notification-hardening-design.md"
```

> Deployment note: merging to `main` auto-deploys via Cloud Build. Before/after merge, ensure the new env vars (`RETENTION_ADMIN_EMAIL`, `RETENTION_ADMIN_SECRET`, `RESEND_WEBHOOK_SECRET`) are set on Cloud Run (Task 10) and the Resend webhook is configured, or the webhook route returns 401 and bounces won't pause deletions.

## Notes on sequencing
- Tasks 1→6 are strictly sequential (schema → eligibility → notices → sweep → email).
- Task 7 (webhook) depends on Task 4 (`mark_status_by_message_id`).
- Task 8 (deletion) depends on Task 5 (sweep `_handle`).
- Task 9 (alert) is independent — can be done any time.
- Task 10 (ops) must follow Task 7 and precede relying on bounce-pausing in prod.
- Task 11 (#7) depends on Task 8 (`delete_user_data` return shape).
