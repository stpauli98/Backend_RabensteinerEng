# Data Retention After Subscription Expiry — Design

> Date: 2026-06-16
> Status: Approved design, pre-implementation
> Repo: Backend (Flask) + Supabase (`luvjebsltuttakatnzaa`)
> Trello: card #101 "Daten werden ein Monat nach dem Ablauf noch gespeichert"

## 1. Problem

When a user's subscription lapses, their data (training sessions, uploaded/processed
CSVs, trained models, training results) is currently **kept indefinitely** — there
is no automatic deletion. Verified: no backend code, no APScheduler job, and no
Supabase pg_cron/function deletes user data after expiry (only
`delete-unverified-users-daily` for unverified accounts, and transient temp-file
cleanups).

## 2. Requirements

- **Delete a lapsed user's data 30 days after their access expired.**
- **Two warning emails** before deletion: one **7 days before** (day 23) and one
  **24 hours before** (day 29), each sent exactly once.
- Email contains what will be deleted, the deletion date, and a CTA **"log in and
  re-subscribe to keep your data"** linking to `{FRONTEND_URL}/login?redirect=/pricing`.
- **Only re-subscribing** (a new active subscription) cancels deletion. Logging in
  alone does not.
- **Applies to all lapsed users by `expires_at`** (no grandfathering — decision C),
  **but** with a hard safety guarantee: data is never deleted unless both warning
  emails were actually sent and ≥24h passed since the second.

## 3. What is deleted vs kept

**Deleted** (data only):
- Sessions and all session-keyed data via the existing
  `delete_all_sessions(user_id, confirm=True)`: tables `sessions`, `files`,
  `time_info`, `zeitschritte`, `training_results`, `session_mappings`; storage in
  `training-results`, `trained-models`, `csv-files`, `aus-csv-files` buckets.
- `api_keys` (by `user_id`) — credentials for now-deleted data.
- `usage_events` (by `user_id`) — action log.
- `usage_tracking` (by `user_id`) — monthly counters (stale after long expiry;
  re-subscribe starts a fresh period).

**Kept**:
- The auth user account (so they can log in and re-subscribe).
- `user_subscriptions` history. The lapsed row is stamped `data_deleted_at` as an
  audit trail; the row itself is not deleted.

**Not touched** (already auto-cleaned by existing jobs): `temp-chunks` and
`processed-files` buckets — transient artifacts cleaned by the 30-min / 24h jobs.

## 4. Architecture

A **daily APScheduler job** `run_data_retention_sweep`, added alongside the existing
cleanup jobs in `core/app_factory.py`. On start it acquires a Postgres
`pg_try_advisory_lock(<const>)`; if not acquired (another Cloud Run instance is
sweeping), the run is skipped. The tracking columns are the deeper guarantee — even
without the lock there is no duplicate email or delete.

For each eligible user the sweep performs **at most one action per run**
(warn1 → warn2 → delete), preserving spacing.

Storage deletion and Resend email sending require Python (a pg_cron SQL function
cannot remove S3 storage objects nor call Resend), so all logic lives in the backend
and reuses existing services (`delete_all_sessions`, `resend_client`).

## 5. Eligibility & state machine

**Lapsed user** = a user with **no active subscription** (`status='active' AND
expires_at > now()`). For such a user, take the subscription row with the **latest
`expires_at`** = `lapsed_at` (when access actually ended). All retention stamps live
on that row.

Anchor on the deletion date: `deletion_date = lapsed_at + 30 days`.

| Action | Condition (in addition to "lapsed") |
|--------|-------------------------------------|
| **warn1** | `now ≥ deletion_date − 7d` AND `retention_warn1_sent_at IS NULL` |
| **warn2** | `retention_warn1_sent_at` set AND `retention_warn2_sent_at IS NULL` AND `now ≥ max(deletion_date − 24h, retention_warn1_sent_at + 24h)` |
| **delete** | `retention_warn2_sent_at` set AND `data_deleted_at IS NULL` AND `now ≥ max(deletion_date, retention_warn2_sent_at + 24h)` |

- **Normal cadence**: warn1 day 23, warn2 day 29, delete day 30 (exactly the spec).
- **Backlog** (already far past expiry when first processed, e.g. cron was off):
  warn1 on first run, warn2 ~24h later, delete ~24h after that — so everyone always
  gets both emails with spacing, regardless of how long ago they lapsed. This is the
  safety gate enforcing decision C without deleting un-warned data.
- **Re-subscribe**: a new active subscription means the user is no longer "lapsed" →
  excluded from the sweep. Old-row stamps remain as history.
- **Already deleted**: `data_deleted_at` set → skipped.

## 6. Data model

Three nullable columns on `user_subscriptions` (stamped on the lapsed row):

```sql
ALTER TABLE user_subscriptions
  ADD COLUMN IF NOT EXISTS retention_warn1_sent_at timestamptz,
  ADD COLUMN IF NOT EXISTS retention_warn2_sent_at timestamptz,
  ADD COLUMN IF NOT EXISTS data_deleted_at         timestamptz;
```

Applied via Supabase `apply_migration` (idempotent). No index needed (small table,
cheap daily query). Rationale for columns-on-`user_subscriptions` rather than a
separate table: the lapsed row *is* the expiry record; it carries its own retention
state with no sync logic. Re-subscribe creates a new row with empty stamps.

## 7. Emails

Reuse the existing `resend_client` and the `auth_emails` Jinja2 environment. Because
these are custom transactional emails (not Supabase auth actions with a
`token_hash`), add a dedicated helper rather than extending `render_email`'s
action map:

```python
send_data_deletion_warning(email, lang, *, deletion_date, login_url, is_final: bool)
```

- **Templates** (same `<basename>_<lang>.html.j2` convention):
  `data_deletion_warning_de.html.j2` + `_en.html.j2`, one per language,
  parameterized by `is_final`:
  - `is_final=False` (warn1): "Your data will be deleted in 7 days ({{ deletion_date }})…"
  - `is_final=True` (warn2): "Final warning — your data will be deleted in 24 hours…"
- **Subjects**: add warn1/warn2 subjects per language alongside the existing
  `subjects.py` mechanism.
- **Language**: from `auth.users.user_metadata.lang`, fallback **German** (same
  convention as existing emails). Email address + lang are fetched from `auth.users`
  via the admin client by `user_id`.
- **Link**: `{FRONTEND_URL}/login?redirect=/pricing`.

## 8. Idempotency & error handling

- **Stamp-after-success**: a tracking column is written only after the email is sent
  (or deletion completes). Any failure leaves the column NULL → retried next run; no
  half-states.
- **Email failure** → warn column stays NULL → retried; deletion still waits for both
  warnings, so a failed warning safely delays deletion.
- **Deletion failure** (partial) → `data_deleted_at` stays NULL → idempotent retry
  next day (`delete_all_sessions` and the `usage_*`/`api_keys` deletes are no-ops on
  already-removed rows).
- **Per-user isolation**: each user processed in `try/except`; one user's error logs
  and does not block the rest.
- **At most one action per user per run** preserves the warn→warn→delete spacing.

## 9. Configuration & rollout

- Thresholds (`30d`, `7d`, `24h`) as named constants for easy tuning.
- `RETENTION_SWEEP_ENABLED` (default **off** until validated) gates the whole job.
- `RETENTION_DRY_RUN` logs who *would* be warned/deleted without sending or deleting.
  Rollout: deploy with dry-run on, inspect logs against the known lapsed users
  (currently 8 cancelled, 0 past 30 days), then enable for real.

## 10. Testing (Docker — project rule)

- **Eligibility as a pure function** `(subscription rows, now) → [(user_id, action)]`,
  tested with a frozen `now`:
  - normal cadence (warn1 day 23, warn2 day 29, delete day 30);
  - backlog compression (warn1 → warn2 → delete on consecutive runs);
  - re-subscribe excludes the user;
  - already-deleted is skipped;
  - delete gate requires both warnings + ≥24h since warn2 (never delete un-warned).
- **Email helper** (mock Resend): correct template, language fallback, and link.
- **Deletion orchestration** (mock `delete_all_sessions` + Supabase): deletes
  `api_keys`/`usage_events`/`usage_tracking`, calls `delete_all_sessions`, stamps
  `data_deleted_at`, keeps the account + subscription.

## 11. Dependencies / out of scope

- **Frontend**: the email link uses `{FRONTEND_URL}/login?redirect=/pricing`. The
  login page must honor a `redirect` query param (redirect to `/pricing` after auth).
  If it does not already, a small frontend change is required — verify during
  planning; tracked as a dependency, not part of the backend sweep.
- Out of scope: anonymized usage analytics/roll-up before deletion; any change to the
  subscription/billing lifecycle itself.
