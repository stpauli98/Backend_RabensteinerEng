# Data Retention After Subscription Expiry — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Automatically delete a lapsed user's data 30 days after their subscription expired, sending two warning emails (7 days and 24h before) with a re-subscribe link, while keeping the account and subscription history.

**Architecture:** A daily backend APScheduler sweep claims a daily lock row (single-runner across Cloud Run instances), computes per-user actions from a pure eligibility function, then sends a warning email (Resend) or deletes the user's data (reusing `delete_all_sessions` + `api_keys`/`usage_*` deletes). State is tracked by three columns on `user_subscriptions`. A `RETENTION_DRY_RUN` mode logs without acting. A small frontend change makes the login page honor a post-login `?redirect=`.

**Tech Stack:** Python/Flask, APScheduler, Supabase (PostgREST via supabase-py), Resend, Jinja2; React/TypeScript (frontend); pytest in Docker.

**Spec:** `my_backend/docs/superpowers/specs/2026-06-16-data-retention-after-expiry-design.md`

> **Spec refinement:** the backend talks to Postgres via PostgREST (supabase-py), not raw SQL, so the spec's "advisory lock" is implemented as a **daily-claim lock row** (`retention_sweep_runs`). One instance claims the day; others skip. This keeps the single-runner guarantee while preserving stamp-after-success.

---

## File Structure

All backend paths are relative to the repo root (the worktree); the package lives under `my_backend/`. Run all backend commands from `my_backend/`.

**Backend — create:**
- `my_backend/supabase/migrations/20260616_data_retention.sql` — 3 columns on `user_subscriptions` + `retention_sweep_runs` lock table.
- `my_backend/domains/retention/__init__.py`
- `my_backend/domains/retention/constants.py` — thresholds + dry-run/enabled flags + email env.
- `my_backend/domains/retention/eligibility.py` — pure function: subscriptions + now → actions.
- `my_backend/domains/retention/email.py` — render + send the two warning emails.
- `my_backend/domains/retention/deletion.py` — delete one user's data.
- `my_backend/domains/retention/sweep.py` — orchestrator (lock, fetch, compute, execute, stamp).
- `my_backend/domains/auth_emails/templates/data_deletion_warning_de.html.j2`
- `my_backend/domains/auth_emails/templates/data_deletion_warning_en.html.j2`
- Tests: `my_backend/domains/retention/tests/__init__.py`, `test_eligibility.py`, `test_email.py`, `test_deletion.py`, `test_sweep.py`.

**Backend — modify:**
- `my_backend/core/app_factory.py` — register the daily sweep job.

**Frontend — modify (separate repo `RabensteinerEngineering`):**
- `Frontend/src/pages/auth/LoginPage.tsx` — honor `?redirect=`.
- Create: `Frontend/src/pages/auth/safeRedirect.ts` + `Frontend/src/pages/auth/safeRedirect.test.ts`.

Backend tests run in Docker (project rule):
```
docker build --build-arg ENV_FILE=.env -t my_backend . && \
docker run --rm --env-file .env --entrypoint pytest my_backend <path> -v
```

---

## Task 1: DB migration — tracking columns + lock table

**Files:**
- Create: `my_backend/supabase/migrations/20260616_data_retention.sql`

- [ ] **Step 1: Write the migration SQL**

Create `my_backend/supabase/migrations/20260616_data_retention.sql`:

```sql
-- Data retention after subscription expiry (Trello #101)

ALTER TABLE public.user_subscriptions
  ADD COLUMN IF NOT EXISTS retention_warn1_sent_at timestamptz,
  ADD COLUMN IF NOT EXISTS retention_warn2_sent_at timestamptz,
  ADD COLUMN IF NOT EXISTS data_deleted_at         timestamptz;

-- Single-row daily-claim lock so only one Cloud Run instance runs the sweep/day.
CREATE TABLE IF NOT EXISTS public.retention_sweep_runs (
  id              integer PRIMARY KEY DEFAULT 1,
  last_started_at timestamptz,
  CONSTRAINT retention_sweep_runs_singleton CHECK (id = 1)
);
INSERT INTO public.retention_sweep_runs (id, last_started_at)
  VALUES (1, NULL)
  ON CONFLICT (id) DO NOTHING;
```

- [ ] **Step 2: Apply the migration to Supabase**

Apply via the Supabase tool (project `luvjebsltuttakatnzaa`) using the SQL above (the controller runs `apply_migration` with name `data_retention` and this SQL). Verify the columns exist:
```sql
SELECT column_name FROM information_schema.columns
WHERE table_schema='public' AND table_name='user_subscriptions'
  AND column_name IN ('retention_warn1_sent_at','retention_warn2_sent_at','data_deleted_at');
```
Expected: 3 rows. And `SELECT * FROM retention_sweep_runs;` → one row `(1, null)`.

- [ ] **Step 3: Commit**

```bash
git add supabase/migrations/20260616_data_retention.sql
git commit -m "feat(retention): add tracking columns + sweep lock table"
```

---

## Task 2: Constants

**Files:**
- Create: `my_backend/domains/retention/__init__.py` (empty)
- Create: `my_backend/domains/retention/constants.py`

- [ ] **Step 1: Create the package + constants**

Create empty `my_backend/domains/retention/__init__.py`.

Create `my_backend/domains/retention/constants.py`:

```python
"""Tuning + config for the data-retention sweep."""
import os
from datetime import timedelta

# Timing (anchored on the deletion date = lapse + DELETE_AFTER).
DELETE_AFTER = timedelta(days=30)
WARN1_BEFORE = timedelta(days=7)    # first email: 7 days before deletion
WARN2_BEFORE = timedelta(hours=24)  # second email: 24h before deletion
MIN_GAP = timedelta(hours=24)       # min spacing between warn1→warn2 and warn2→delete

# Daily-claim window: if the sweep started within this window, another instance
# already ran today.
CLAIM_WINDOW = timedelta(hours=23)


def sweep_enabled() -> bool:
    return os.environ.get("RETENTION_SWEEP_ENABLED", "false").lower() == "true"


def dry_run() -> bool:
    return os.environ.get("RETENTION_DRY_RUN", "true").lower() == "true"


def login_redirect_url() -> str:
    base = os.environ.get("FRONTEND_URL", "").rstrip("/")
    return f"{base}/login?redirect=/pricing"
```

- [ ] **Step 2: Commit**

```bash
git add domains/retention/__init__.py domains/retention/constants.py
git commit -m "feat(retention): constants and config flags"
```

---

## Task 3: Eligibility (pure function) — TDD

**Files:**
- Create: `my_backend/domains/retention/eligibility.py`
- Test: `my_backend/domains/retention/tests/__init__.py` (empty), `my_backend/domains/retention/tests/test_eligibility.py`

- [ ] **Step 1: Write the failing test**

Create empty `my_backend/domains/retention/tests/__init__.py`.

Create `my_backend/domains/retention/tests/test_eligibility.py`:

```python
from datetime import datetime, timedelta, timezone
from domains.retention.eligibility import compute_actions, RetentionAction

NOW = datetime(2026, 6, 16, 12, 0, tzinfo=timezone.utc)


def sub(uid, *, status='cancelled', expires_days_ago=None, expires_in_days=None,
        sub_id=None, w1=None, w2=None, deleted=None):
    if expires_days_ago is not None:
        exp = NOW - timedelta(days=expires_days_ago)
    else:
        exp = NOW + timedelta(days=expires_in_days)
    return {
        'id': sub_id or f'{uid}-sub',
        'user_id': uid,
        'status': status,
        'expires_at': exp.isoformat(),
        'retention_warn1_sent_at': w1.isoformat() if w1 else None,
        'retention_warn2_sent_at': w2.isoformat() if w2 else None,
        'data_deleted_at': deleted.isoformat() if deleted else None,
    }


def test_active_user_no_action():
    subs = [sub('u1', status='active', expires_in_days=10)]
    assert compute_actions(subs, NOW) == []


def test_lapsed_but_under_23_days_no_action():
    subs = [sub('u1', expires_days_ago=10)]
    assert compute_actions(subs, NOW) == []


def test_warn1_at_day_23():
    subs = [sub('u1', expires_days_ago=23)]
    actions = compute_actions(subs, NOW)
    assert actions == [RetentionAction(user_id='u1', subscription_id='u1-sub',
                                       action='warn1', deletion_date=NOW + timedelta(days=7))]


def test_warn2_after_warn1_and_24h():
    subs = [sub('u1', expires_days_ago=29, w1=NOW - timedelta(days=6))]
    actions = compute_actions(subs, NOW)
    assert [a.action for a in actions] == ['warn2']


def test_no_warn2_before_24h_gap():
    # warn1 sent 1h ago and not yet at deletion_date-24h → wait
    subs = [sub('u1', expires_days_ago=23, w1=NOW - timedelta(hours=1))]
    assert compute_actions(subs, NOW) == []


def test_delete_at_day_30_after_both_warns():
    subs = [sub('u1', expires_days_ago=30,
                w1=NOW - timedelta(days=7), w2=NOW - timedelta(days=1))]
    actions = compute_actions(subs, NOW)
    assert [a.action for a in actions] == ['delete']


def test_no_delete_without_warn2():
    subs = [sub('u1', expires_days_ago=40, w1=NOW - timedelta(days=5))]
    # warn1 set, warn2 missing, way past 30d → must NOT delete; next is warn2
    assert [a.action for a in compute_actions(subs, NOW)] == ['warn2']


def test_already_deleted_skipped():
    subs = [sub('u1', expires_days_ago=60, w1=NOW - timedelta(days=10),
                w2=NOW - timedelta(days=9), deleted=NOW - timedelta(days=8))]
    assert compute_actions(subs, NOW) == []


def test_resubscribe_excludes_user():
    # latest active sub means no action even if an old cancelled one is ancient
    subs = [
        sub('u1', sub_id='old', expires_days_ago=100),
        sub('u1', sub_id='new', status='active', expires_in_days=20),
    ]
    assert compute_actions(subs, NOW) == []


def test_backlog_first_run_sends_warn1():
    # expired 50 days ago, nothing sent yet → first action is warn1 (never jumps to delete)
    subs = [sub('u1', expires_days_ago=50)]
    assert [a.action for a in compute_actions(subs, NOW)] == ['warn1']
```

- [ ] **Step 2: Run the test to verify it fails**

Run:
```
docker build --build-arg ENV_FILE=.env -t my_backend . && \
docker run --rm --env-file .env --entrypoint pytest my_backend \
  domains/retention/tests/test_eligibility.py -v
```
Expected: FAIL — module `domains.retention.eligibility` not found.

- [ ] **Step 3: Implement the eligibility function**

Create `my_backend/domains/retention/eligibility.py`:

```python
"""Pure eligibility logic for the retention sweep.

Given all user_subscriptions rows and `now`, return the single next action per
lapsed user. No I/O — fully unit-testable.
"""
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from shared.datetime_utils import parse_iso_datetime
from domains.retention.constants import (
    DELETE_AFTER, WARN1_BEFORE, WARN2_BEFORE, MIN_GAP,
)

Action = Literal['warn1', 'warn2', 'delete']


@dataclass(frozen=True)
class RetentionAction:
    user_id: str
    subscription_id: str
    action: Action
    deletion_date: datetime


def _dt(value: Optional[str]) -> Optional[datetime]:
    return parse_iso_datetime(value) if value else None


def compute_actions(subscriptions: List[Dict[str, Any]], now: datetime) -> List[RetentionAction]:
    by_user: Dict[str, List[Dict[str, Any]]] = {}
    for row in subscriptions:
        by_user.setdefault(row['user_id'], []).append(row)

    actions: List[RetentionAction] = []
    for user_id, rows in by_user.items():
        # Active access anywhere → not lapsed.
        has_active = any(
            r.get('status') == 'active' and (_dt(r.get('expires_at')) or now) > now
            for r in rows
        )
        if has_active:
            continue

        # The lapsed row = latest expires_at (when access actually ended).
        row = max(rows, key=lambda r: _dt(r.get('expires_at')) or datetime.min.replace(tzinfo=now.tzinfo))
        if row.get('data_deleted_at'):
            continue

        lapsed_at = _dt(row.get('expires_at'))
        if lapsed_at is None:
            continue
        deletion_date = lapsed_at + DELETE_AFTER
        w1 = _dt(row.get('retention_warn1_sent_at'))
        w2 = _dt(row.get('retention_warn2_sent_at'))

        action: Optional[Action] = None
        if w1 is None:
            if now >= deletion_date - WARN1_BEFORE:
                action = 'warn1'
        elif w2 is None:
            if now >= max(deletion_date - WARN2_BEFORE, w1 + MIN_GAP):
                action = 'warn2'
        else:
            if now >= max(deletion_date, w2 + MIN_GAP):
                action = 'delete'

        if action:
            actions.append(RetentionAction(
                user_id=user_id, subscription_id=row['id'],
                action=action, deletion_date=deletion_date,
            ))
    return actions
```

> Note: `parse_iso_datetime` is the project helper for Supabase timestamps (`shared/datetime_utils.py`) — use it, not `datetime.fromisoformat` (Python 3.9 fractional-seconds bug, per backend CLAUDE.md).

- [ ] **Step 4: Run the test to verify it passes**

Run:
```
docker run --rm --env-file .env --entrypoint pytest my_backend \
  domains/retention/tests/test_eligibility.py -v
```
Expected: PASS (10 tests). If a timestamp parse error appears, confirm `parse_iso_datetime` accepts ISO strings with `+00:00`; adjust the import path only if the helper lives elsewhere (grep `def parse_iso_datetime`).

- [ ] **Step 5: Commit**

```bash
git add domains/retention/eligibility.py domains/retention/tests/__init__.py domains/retention/tests/test_eligibility.py
git commit -m "feat(retention): pure eligibility state machine + tests"
```

---

## Task 4: Warning email (render + send) — TDD

**Files:**
- Create: `my_backend/domains/auth_emails/templates/data_deletion_warning_de.html.j2`
- Create: `my_backend/domains/auth_emails/templates/data_deletion_warning_en.html.j2`
- Create: `my_backend/domains/retention/email.py`
- Test: `my_backend/domains/retention/tests/test_email.py`

- [ ] **Step 1: Create the templates**

Create `my_backend/domains/auth_emails/templates/data_deletion_warning_en.html.j2`:

```html
<!DOCTYPE html>
<html lang="en">
  <body style="font-family: Arial, sans-serif; color: #1a1a1a;">
    {% if is_final %}
      <h2>Final warning: your data will be deleted in 24 hours</h2>
    {% else %}
      <h2>Your data will be deleted in 7 days</h2>
    {% endif %}
    <p>Your subscription has expired. On <strong>{{ deletion_date }}</strong> we will
       permanently delete your stored data: training sessions, uploaded and processed
       CSV files, trained models and training results.</p>
    <p>To keep your data, log in and re-subscribe:</p>
    <p><a href="{{ login_url }}"
          style="background:#2563eb;color:#fff;padding:10px 18px;border-radius:6px;text-decoration:none;">
       Log in &amp; re-subscribe</a></p>
    <p style="color:#666;font-size:12px;">If you do nothing, your data will be deleted on {{ deletion_date }}.</p>
  </body>
</html>
```

Create `my_backend/domains/auth_emails/templates/data_deletion_warning_de.html.j2`:

```html
<!DOCTYPE html>
<html lang="de">
  <body style="font-family: Arial, sans-serif; color: #1a1a1a;">
    {% if is_final %}
      <h2>Letzte Warnung: Ihre Daten werden in 24 Stunden gelöscht</h2>
    {% else %}
      <h2>Ihre Daten werden in 7 Tagen gelöscht</h2>
    {% endif %}
    <p>Ihr Abonnement ist abgelaufen. Am <strong>{{ deletion_date }}</strong> löschen wir
       Ihre gespeicherten Daten endgültig: Trainingssitzungen, hochgeladene und verarbeitete
       CSV-Dateien, trainierte Modelle und Trainingsergebnisse.</p>
    <p>Um Ihre Daten zu behalten, melden Sie sich an und abonnieren Sie erneut:</p>
    <p><a href="{{ login_url }}"
          style="background:#2563eb;color:#fff;padding:10px 18px;border-radius:6px;text-decoration:none;">
       Anmelden &amp; erneut abonnieren</a></p>
    <p style="color:#666;font-size:12px;">Wenn Sie nichts tun, werden Ihre Daten am {{ deletion_date }} gelöscht.</p>
  </body>
</html>
```

- [ ] **Step 2: Write the failing test**

Create `my_backend/domains/retention/tests/test_email.py`:

```python
from unittest.mock import patch
from domains.retention.email import render_warning_html, retention_subject, send_warning


def test_render_de_final_contains_link_and_date():
    html = render_warning_html(lang='de', deletion_date='2026-07-16',
                               login_url='https://app/login?redirect=/pricing', is_final=True)
    assert 'Letzte Warnung' in html
    assert '2026-07-16' in html
    assert 'https://app/login?redirect=/pricing' in html


def test_render_unknown_lang_falls_back_to_german():
    html = render_warning_html(lang='fr', deletion_date='2026-07-16',
                               login_url='https://app/x', is_final=False)
    assert 'gelöscht' in html  # German fallback


def test_subject_differs_by_finality():
    assert retention_subject('en', is_final=False) != retention_subject('en', is_final=True)


def test_send_warning_calls_resend_with_rendered_html():
    with patch('domains.retention.email.send_email', return_value='msg-1') as send:
        msg_id = send_warning(api_key='k', from_addr='F <f@x>', to='u@x', lang='en',
                              deletion_date='2026-07-16',
                              login_url='https://app/login?redirect=/pricing', is_final=True)
    assert msg_id == 'msg-1'
    kwargs = send.call_args.kwargs
    assert kwargs['to'] == 'u@x'
    assert 'Final warning' in kwargs['html']
    assert kwargs['subject'] == retention_subject('en', is_final=True)
```

- [ ] **Step 3: Run the test to verify it fails**

Run:
```
docker run --rm --env-file .env --entrypoint pytest my_backend \
  domains/retention/tests/test_email.py -v
```
Expected: FAIL — `domains.retention.email` not found.

- [ ] **Step 4: Implement the email module**

Create `my_backend/domains/retention/email.py`:

```python
"""Render + send the data-deletion warning emails (reuses the Resend client)."""
import os

from jinja2 import Environment, FileSystemLoader, select_autoescape

from domains.auth_emails.services.resend_client import send_email

_TEMPLATES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "auth_emails", "templates"
)
_env = Environment(
    loader=FileSystemLoader(_TEMPLATES_DIR),
    autoescape=select_autoescape(["html", "j2"]),
)
_DEFAULT_LANG = "de"
_SUPPORTED = {"de", "en"}

_SUBJECTS = {
    ("de", False): "Ihre Daten werden in 7 Tagen gelöscht",
    ("de", True): "Letzte Warnung: Datenlöschung in 24 Stunden",
    ("en", False): "Your data will be deleted in 7 days",
    ("en", True): "Final warning: data deletion in 24 hours",
}


def _lang(lang: str) -> str:
    return lang if lang in _SUPPORTED else _DEFAULT_LANG


def render_warning_html(*, lang: str, deletion_date: str, login_url: str, is_final: bool) -> str:
    template = _env.get_template(f"data_deletion_warning_{_lang(lang)}.html.j2")
    return template.render(deletion_date=deletion_date, login_url=login_url, is_final=is_final)


def retention_subject(lang: str, is_final: bool) -> str:
    return _SUBJECTS[(_lang(lang), is_final)]


def send_warning(*, api_key: str, from_addr: str, to: str, lang: str,
                 deletion_date: str, login_url: str, is_final: bool) -> str:
    html = render_warning_html(lang=lang, deletion_date=deletion_date,
                               login_url=login_url, is_final=is_final)
    return send_email(api_key=api_key, from_addr=from_addr, to=to,
                      subject=retention_subject(lang, is_final), html=html)
```

- [ ] **Step 5: Run the test to verify it passes**

Run:
```
docker run --rm --env-file .env --entrypoint pytest my_backend \
  domains/retention/tests/test_email.py -v
```
Expected: PASS (4 tests).

- [ ] **Step 6: Commit**

```bash
git add domains/auth_emails/templates/data_deletion_warning_de.html.j2 \
        domains/auth_emails/templates/data_deletion_warning_en.html.j2 \
        domains/retention/email.py domains/retention/tests/test_email.py
git commit -m "feat(retention): warning email templates + send helper"
```

---

## Task 5: Delete one user's data — TDD

**Files:**
- Create: `my_backend/domains/retention/deletion.py`
- Test: `my_backend/domains/retention/tests/test_deletion.py`

- [ ] **Step 1: Write the failing test**

Create `my_backend/domains/retention/tests/test_deletion.py`:

```python
from unittest.mock import MagicMock, patch
from domains.retention.deletion import delete_user_data


def test_delete_user_data_calls_sessions_then_tables():
    supabase = MagicMock()
    with patch('domains.retention.deletion.delete_all_sessions') as das:
        delete_user_data(supabase, 'user-1')
    das.assert_called_once_with(confirm=True, user_id='user-1')

    deleted_tables = [c.args[0] for c in supabase.table.call_args_list]
    assert set(deleted_tables) == {'api_keys', 'usage_events', 'usage_tracking'}
    # each .delete().eq('user_id', 'user-1').execute() chain was invoked
    for tbl in ('api_keys', 'usage_events', 'usage_tracking'):
        supabase.table.assert_any_call(tbl)
```

- [ ] **Step 2: Run the test to verify it fails**

Run:
```
docker run --rm --env-file .env --entrypoint pytest my_backend \
  domains/retention/tests/test_deletion.py -v
```
Expected: FAIL — `domains.retention.deletion` not found.

- [ ] **Step 3: Implement the deletion module**

Create `my_backend/domains/retention/deletion.py`:

```python
"""Delete a single user's data (keeps the account + subscription history)."""
import logging

from domains.training.services.session import delete_all_sessions

logger = logging.getLogger(__name__)

_USER_TABLES = ("api_keys", "usage_events", "usage_tracking")


def delete_user_data(supabase, user_id: str) -> None:
    """Idempotent: re-running on an already-purged user is a no-op (0 rows)."""
    # Sessions + their storage files + session-keyed tables.
    delete_all_sessions(confirm=True, user_id=user_id)
    # User-keyed tables.
    for table in _USER_TABLES:
        supabase.table(table).delete().eq("user_id", user_id).execute()
    logger.info("retention: purged data for user %s", user_id)
```

- [ ] **Step 4: Run the test to verify it passes**

Run:
```
docker run --rm --env-file .env --entrypoint pytest my_backend \
  domains/retention/tests/test_deletion.py -v
```
Expected: PASS (1 test).

- [ ] **Step 5: Commit**

```bash
git add domains/retention/deletion.py domains/retention/tests/test_deletion.py
git commit -m "feat(retention): delete one user's data (reuses delete_all_sessions)"
```

---

## Task 6: Sweep orchestrator — TDD

**Files:**
- Create: `my_backend/domains/retention/sweep.py`
- Test: `my_backend/domains/retention/tests/test_sweep.py`

The sweep: claim the daily lock; if not claimed, return. Fetch all subscriptions, compute actions, and for each: in dry-run just log; otherwise send the warning (then stamp `retention_warnN_sent_at`) or delete the user's data (then stamp `data_deleted_at`). Email/lang/address come from `auth.users` via the admin client. Stamp only after success.

- [ ] **Step 1: Write the failing test**

Create `my_backend/domains/retention/tests/test_sweep.py`:

```python
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch
from domains.retention.eligibility import RetentionAction
from domains.retention import sweep as sweep_mod

NOW = datetime(2026, 6, 16, 12, 0, tzinfo=timezone.utc)


def _supabase_with_lock(acquired=True):
    sb = MagicMock()
    # claim lock: update(...).eq(...).lt(...)/is_(...).execute() returns rows when acquired
    lock_resp = MagicMock()
    lock_resp.data = [{'id': 1}] if acquired else []
    sb.table.return_value.update.return_value.eq.return_value.execute.return_value = lock_resp
    return sb


def test_skips_when_lock_not_acquired():
    sb = _supabase_with_lock(acquired=False)
    with patch('domains.retention.sweep._claim_daily_lock', return_value=False) as claim:
        result = sweep_mod.run_sweep(sb, now=NOW, dry_run=True)
    claim.assert_called_once()
    assert result['ran'] is False


def test_dry_run_does_not_send_or_delete():
    sb = MagicMock()
    action = RetentionAction(user_id='u1', subscription_id='s1', action='warn1',
                             deletion_date=NOW)
    with patch('domains.retention.sweep._claim_daily_lock', return_value=True), \
         patch('domains.retention.sweep._fetch_subscriptions', return_value=[]), \
         patch('domains.retention.sweep.compute_actions', return_value=[action]), \
         patch('domains.retention.sweep.send_warning') as send, \
         patch('domains.retention.sweep.delete_user_data') as delet:
        result = sweep_mod.run_sweep(sb, now=NOW, dry_run=True)
    send.assert_not_called()
    delet.assert_not_called()
    assert result['planned'] == 1


def test_warn1_sends_then_stamps():
    sb = MagicMock()
    action = RetentionAction(user_id='u1', subscription_id='s1', action='warn1',
                             deletion_date=NOW)
    with patch('domains.retention.sweep._claim_daily_lock', return_value=True), \
         patch('domains.retention.sweep._fetch_subscriptions', return_value=[]), \
         patch('domains.retention.sweep.compute_actions', return_value=[action]), \
         patch('domains.retention.sweep._user_email_lang', return_value=('u@x', 'en')), \
         patch('domains.retention.sweep.send_warning', return_value='m1') as send, \
         patch('domains.retention.sweep.delete_user_data') as delet:
        sweep_mod.run_sweep(sb, now=NOW, dry_run=False)
    send.assert_called_once()
    delet.assert_not_called()
    # stamped retention_warn1_sent_at on subscription s1
    sb.table.assert_any_call('user_subscriptions')


def test_delete_action_purges_then_stamps():
    sb = MagicMock()
    action = RetentionAction(user_id='u1', subscription_id='s1', action='delete',
                             deletion_date=NOW)
    with patch('domains.retention.sweep._claim_daily_lock', return_value=True), \
         patch('domains.retention.sweep._fetch_subscriptions', return_value=[]), \
         patch('domains.retention.sweep.compute_actions', return_value=[action]), \
         patch('domains.retention.sweep.delete_user_data') as delet, \
         patch('domains.retention.sweep.send_warning') as send:
        sweep_mod.run_sweep(sb, now=NOW, dry_run=False)
    delet.assert_called_once_with(sb, 'u1')
    send.assert_not_called()


def test_one_user_error_does_not_block_others():
    sb = MagicMock()
    a1 = RetentionAction(user_id='u1', subscription_id='s1', action='delete', deletion_date=NOW)
    a2 = RetentionAction(user_id='u2', subscription_id='s2', action='delete', deletion_date=NOW)
    with patch('domains.retention.sweep._claim_daily_lock', return_value=True), \
         patch('domains.retention.sweep._fetch_subscriptions', return_value=[]), \
         patch('domains.retention.sweep.compute_actions', return_value=[a1, a2]), \
         patch('domains.retention.sweep.delete_user_data',
               side_effect=[RuntimeError('boom'), None]) as delet:
        result = sweep_mod.run_sweep(sb, now=NOW, dry_run=False)
    assert delet.call_count == 2
    assert result['errors'] == 1
    assert result['done'] == 1
```

- [ ] **Step 2: Run the test to verify it fails**

Run:
```
docker run --rm --env-file .env --entrypoint pytest my_backend \
  domains/retention/tests/test_sweep.py -v
```
Expected: FAIL — `domains.retention.sweep` not found.

- [ ] **Step 3: Implement the sweep**

Create `my_backend/domains/retention/sweep.py`:

```python
"""Daily data-retention sweep orchestrator."""
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List

from domains.retention.constants import CLAIM_WINDOW, login_redirect_url
from domains.retention.deletion import delete_user_data
from domains.retention.eligibility import RetentionAction, compute_actions
from domains.retention.email import send_warning

logger = logging.getLogger(__name__)

_SUB_COLS = ("id,user_id,status,expires_at,"
             "retention_warn1_sent_at,retention_warn2_sent_at,data_deleted_at")


def _claim_daily_lock(supabase, now: datetime) -> bool:
    """Atomically claim today's run. Returns True if this caller won the slot."""
    cutoff = (now - CLAIM_WINDOW).isoformat()
    resp = (
        supabase.table("retention_sweep_runs")
        .update({"last_started_at": now.isoformat()})
        .eq("id", 1)
        .lt("last_started_at", cutoff)
        .execute()
    )
    if resp.data:
        return True
    # Cover the very first run where last_started_at IS NULL (lt() skips NULLs).
    resp2 = (
        supabase.table("retention_sweep_runs")
        .update({"last_started_at": now.isoformat()})
        .eq("id", 1)
        .is_("last_started_at", "null")
        .execute()
    )
    return bool(resp2.data)


def _fetch_subscriptions(supabase) -> List[Dict[str, Any]]:
    return supabase.table("user_subscriptions").select(_SUB_COLS).execute().data or []


def _user_email_lang(supabase, user_id: str):
    """Return (email, lang) from auth.users via the admin client, or (None, 'de')."""
    res = supabase.auth.admin.get_user_by_id(user_id)
    user = getattr(res, "user", None) or res
    email = getattr(user, "email", None)
    meta = getattr(user, "user_metadata", None) or {}
    return email, (meta.get("lang") or "de")


def _stamp(supabase, subscription_id: str, column: str, now: datetime) -> None:
    (supabase.table("user_subscriptions")
     .update({column: now.isoformat()})
     .eq("id", subscription_id)
     .execute())


def _handle(supabase, action: RetentionAction, now: datetime) -> None:
    if action.action in ("warn1", "warn2"):
        email, lang = _user_email_lang(supabase, action.user_id)
        if not email:
            raise RuntimeError(f"no email for user {action.user_id}")
        is_final = action.action == "warn2"
        send_warning(
            api_key=os.environ["RESEND_API_KEY"],
            from_addr=f'{os.environ.get("EMAIL_FROM_NAME", "Forecast Engine")} '
                      f'<{os.environ["EMAIL_FROM_ADDRESS"]}>',
            to=email, lang=lang,
            deletion_date=action.deletion_date.date().isoformat(),
            login_url=login_redirect_url(), is_final=is_final,
        )
        col = "retention_warn2_sent_at" if is_final else "retention_warn1_sent_at"
        _stamp(supabase, action.subscription_id, col, now)
    else:  # delete
        delete_user_data(supabase, action.user_id)
        _stamp(supabase, action.subscription_id, "data_deleted_at", now)


def run_sweep(supabase, *, now: datetime | None = None, dry_run: bool) -> Dict[str, Any]:
    now = now or datetime.now(timezone.utc)
    if not _claim_daily_lock(supabase, now):
        logger.info("retention sweep: lock not acquired, skipping")
        return {"ran": False, "planned": 0, "done": 0, "errors": 0}

    actions = compute_actions(_fetch_subscriptions(supabase), now)
    done = errors = 0
    for action in actions:
        if dry_run:
            logger.info("retention DRY-RUN: would %s user %s (deletion %s)",
                        action.action, action.user_id, action.deletion_date.date())
            continue
        try:
            _handle(supabase, action, now)
            done += 1
        except Exception:
            errors += 1
            logger.exception("retention: action %s failed for user %s",
                             action.action, action.user_id)
    return {"ran": True, "planned": len(actions), "done": done, "errors": errors}
```

- [ ] **Step 4: Run the test to verify it passes**

Run:
```
docker run --rm --env-file .env --entrypoint pytest my_backend \
  domains/retention/tests/test_sweep.py -v
```
Expected: PASS (5 tests). If `is_("last_started_at", "null")` raises on the mock, that's fine — the tests patch `_claim_daily_lock`; only the real run uses it.

- [ ] **Step 5: Commit**

```bash
git add domains/retention/sweep.py domains/retention/tests/test_sweep.py
git commit -m "feat(retention): daily sweep orchestrator with lock + dry-run"
```

---

## Task 7: Register the daily APScheduler job

**Files:**
- Modify: `my_backend/core/app_factory.py` (the scheduler block, near the existing `scheduler.add_job(...)` calls around line 230)

- [ ] **Step 1: Add the job runner + registration**

In `core/app_factory.py`, immediately BEFORE the line `scheduler.start()`, add:

```python
    def run_data_retention_sweep():
        """Daily: warn + delete data of users whose subscription lapsed >30 days ago."""
        try:
            from domains.retention.constants import sweep_enabled, dry_run
            if not sweep_enabled():
                return
            from domains.retention.sweep import run_sweep
            from shared.database.client import get_supabase_admin_client
            result = run_sweep(get_supabase_admin_client(), dry_run=dry_run())
            logger.info(f"Retention sweep result: {result}")
        except Exception as e:
            logger.error(f"Error in data retention sweep: {str(e)}")

    scheduler.add_job(run_data_retention_sweep, 'interval', hours=24, id='data_retention_sweep_job')
```

(Place it alongside the other `scheduler.add_job(...)` lines so it is registered before `scheduler.start()`.)

- [ ] **Step 2: Verify the app boots + job registers**

Run:
```
docker run --rm --env-file .env --entrypoint python my_backend -c \
  "from core.app_factory import create_app; create_app(); print('boot-ok')"
```
Expected: prints `boot-ok` with no traceback. (If `create_app` has a different name/signature, grep `def create_app` in `core/app_factory.py` and adjust the call.)

- [ ] **Step 3: Commit**

```bash
git add core/app_factory.py
git commit -m "feat(retention): register daily retention sweep job (gated by RETENTION_SWEEP_ENABLED)"
```

---

## Task 8: Frontend — login honors `?redirect=`

**Files (separate repo `RabensteinerEngineering`, run from `Frontend/`):**
- Create: `Frontend/src/pages/auth/safeRedirect.ts`
- Create: `Frontend/src/pages/auth/safeRedirect.test.ts`
- Modify: `Frontend/src/pages/auth/LoginPage.tsx`

- [ ] **Step 1: Write the failing test**

Create `Frontend/src/pages/auth/safeRedirect.test.ts`:

```ts
import { describe, it, expect } from 'vitest';
import { safeRedirectTarget } from './safeRedirect';

describe('safeRedirectTarget', () => {
  it('returns the default when param is missing', () => {
    expect(safeRedirectTarget(null)).toBe('/rohdaten-laden');
  });
  it('allows an internal absolute path', () => {
    expect(safeRedirectTarget('/pricing')).toBe('/pricing');
  });
  it('rejects protocol-relative (open redirect)', () => {
    expect(safeRedirectTarget('//evil.com')).toBe('/rohdaten-laden');
  });
  it('rejects absolute URLs', () => {
    expect(safeRedirectTarget('https://evil.com')).toBe('/rohdaten-laden');
  });
  it('rejects non-rooted paths', () => {
    expect(safeRedirectTarget('pricing')).toBe('/rohdaten-laden');
  });
});
```

- [ ] **Step 2: Run the test to verify it fails**

Run (from `Frontend/`): `npx vitest run src/pages/auth/safeRedirect.test.ts`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement the helper**

Create `Frontend/src/pages/auth/safeRedirect.ts`:

```ts
const DEFAULT_TARGET = '/rohdaten-laden';

/** Only allow internal, single-slash-rooted paths to prevent open redirects. */
export function safeRedirectTarget(param: string | null): string {
  if (param && param.startsWith('/') && !param.startsWith('//')) {
    return param;
  }
  return DEFAULT_TARGET;
}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `npx vitest run src/pages/auth/safeRedirect.test.ts`
Expected: PASS (5 tests).

- [ ] **Step 5: Wire it into LoginPage**

Replace the contents of `Frontend/src/pages/auth/LoginPage.tsx` with:

```tsx
import { useNavigate, useSearchParams } from 'react-router-dom';
import LoginForm from '../../features/auth/components/LoginForm';
import { useAuth } from '../../features/auth/contexts/AuthContext';
import { safeRedirectTarget } from './safeRedirect';

const LoginPage = () => {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const { login, error } = useAuth();

  const handleLogin = async (email: string, password: string) => {
    try {
      await login(email, password);
      navigate(safeRedirectTarget(searchParams.get('redirect')));
    } catch {
      // Error is handled by the AuthContext
    }
  };

  return <LoginForm onLogin={handleLogin} error={error} />;
};

export default LoginPage;
```

- [ ] **Step 6: Type-check + lint**

Run: `npx tsc --noEmit -p tsconfig.app.json && npm run lint`
Expected: no new errors.

- [ ] **Step 7: Commit**

```bash
git add src/pages/auth/safeRedirect.ts src/pages/auth/safeRedirect.test.ts src/pages/auth/LoginPage.tsx
git commit -m "feat(auth): login honors safe ?redirect= param"
```

---

## Self-Review

**Spec coverage:**
- §2 delete at 30d, warn at 23d/29d, link, only-re-subscribe, decision C + safety gate → Task 3 (state machine) + Task 6 (execution). ✓
- §3 deletion scope (sessions+files, api_keys, usage_events, usage_tracking; keep account+subscription) → Task 5. ✓
- §4 daily APScheduler sweep + single-runner lock → Task 1 (lock table) + Task 6 (`_claim_daily_lock`) + Task 7 (job). ✓
- §5 eligibility state machine (normal + backlog + re-subscribe + already-deleted + delete gate) → Task 3 tests. ✓
- §6 three columns → Task 1. ✓
- §7 emails (Resend reuse, de/en templates, `is_final`, lang fallback, link) → Task 4 + Task 6 (`_user_email_lang`, `login_redirect_url`). ✓
- §8 idempotency/error handling (stamp-after-success, per-user isolation, one-action-per-run) → Task 6. ✓
- §9 config/rollout (`RETENTION_SWEEP_ENABLED`, `RETENTION_DRY_RUN`, thresholds) → Task 2 + Task 7. ✓
- §10 testing → Tasks 3-6 tests. ✓
- §11 FE `/login?redirect=/pricing` dependency → Task 8. ✓

**Placeholder scan:** No TBD/TODO. Two grep-and-adjust escape hatches (parse_iso_datetime import path; `create_app` signature) are explicit verification steps, not vague placeholders. All modules ship with complete code.

**Type consistency:** `RetentionAction(user_id, subscription_id, action, deletion_date)` defined in Task 3, used identically in Task 6. `send_warning(...)`/`render_warning_html(...)`/`retention_subject(...)` signatures match between Task 4 and Task 6. `delete_user_data(supabase, user_id)` matches between Task 5 and Task 6. `run_sweep(supabase, *, now=None, dry_run)` matches between Task 6 and Task 7. Constants (`DELETE_AFTER`, `WARN1_BEFORE`, `WARN2_BEFORE`, `MIN_GAP`, `CLAIM_WINDOW`, `login_redirect_url`, `sweep_enabled`, `dry_run`) defined in Task 2, used in Tasks 3/6/7.
