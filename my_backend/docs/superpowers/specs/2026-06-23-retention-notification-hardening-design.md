# Retention Notification & Deletion Hardening — Design

**Date:** 2026-06-23
**Status:** Approved (brainstorming) → ready for implementation plan
**Scope:** `my_backend/domains/retention/` (`eligibility.py`, `sweep.py`, `email.py`, `deletion.py`, `constants.py`) + `core/app_factory.py` + new webhook/admin routes + GCP Cloud Monitoring.
**Predecessor:** `2026-06-16-data-retention-after-expiry-design.md` (original retention design). The retention sweep is live in production (`RETENTION_SWEEP_ENABLED=true`, `RETENTION_DRY_RUN=false`) since 2026-06-23.

## Problem statement

A post-go-live analysis of the live retention/deletion-notification flow found 11 gaps. The system can currently delete a user's data without a fair warning window, without confirming the warning was actually delivered, and without any alarm if the sweep silently stops running. This design hardens the full notify→delete lifecycle.

### Findings being addressed
1. **Compressed warning window** — `deletion_date` is anchored on `expires_at`, not on when `warn1` was actually sent. For accounts that lapsed long ago (or whenever the sweep was down), the 7-day policy collapses to ~2 days.
2. **No delivery confirmation** — `send_email` only checks Resend HTTP 2xx ("accepted for sending"), never detects bounce/spam. A bounced warning still stamps "sent" and the data is deleted anyway.
3. **No proof of notice** — the Resend `message_id` is discarded; only a bare timestamp survives.
4. **No sweep-staleness alarm** — a week-long silent gap (2026-06-16 → 2026-06-23) was found only by manual inspection.
5. **`past_due`/`trialing` not protected** — only `status == 'active'` protects a user from deletion.
6. **Deletion is not transactional** — `delete_all_sessions` is best-effort; storage orphans can remain while `data_deleted_at` is stamped as success.
7. **Never-subscribed users never cleaned** — retention only looks at `user_subscriptions`.
8. **Email language** — defaults to `de` when `user_metadata.lang` is unset.
9. **Inaccurate email text on drift** — fixed subject strings and a date that may not match the actual deletion time.
10. **Possible duplicate warn1** — the stamp happens after the send; a crash between send and stamp re-sends.
11. **No admin trigger** — operators must hand-edit `retention_sweep_runs` in the DB to force a run.

### Locked decisions (from brainstorming)
- **#1/#9:** new `scheduled_deletion_at` column, set when `warn1` is sent = `max(expires_at + 30d, warn1_sent + 7d)`; `warn2`, deletion, and email date all anchor on it.
- **#2:** Resend webhook (`email.bounced`/`email.complained`) → **pause that user's deletion** + alert Markus. (Not just alert; not just record.)
- **#3:** persist Resend `message_id` + delivery status.
- **#5:** protected statuses = `{active, trialing, past_due}` with `expires_at` in the future or null.
- **#4:** GCP Cloud Monitoring **log-absence** alert — no `Retention sweep result` log for >36h → email Markus.
- **A (data model):** dedicated `retention_notices` table is the source of truth for notice state (long-term: audit history, webhook mapping by `message_id`, extensibility). `scheduled_deletion_at` + `data_deleted_at` stay on `user_subscriptions`.
- **#7:** included as a full phase — Markus confirmed the policy.

## Data model (Component A)

### Migration on `user_subscriptions`
- Add `scheduled_deletion_at timestamptz NULL` — stamped when `warn1` is sent.

`data_deleted_at` already exists and stays. The legacy `retention_warn1_sent_at` / `retention_warn2_sent_at` columns are backfilled into `retention_notices` by the migration; eligibility stops reading them (they may be dropped in a later migration once the notices path is proven).

### New table `retention_notices`
One row per notice sent. **Source of truth** for "was warnN sent, and what happened to it."

```
id               uuid pk default gen_random_uuid()
subscription_id  uuid  fk -> user_subscriptions(id) on delete cascade
user_id          uuid  (denormalized for fast lookup; not FK to auth.users)
kind             text  check (kind in ('warn1','warn2'))
resend_message_id text null
status           text  check (status in ('sending','sent','delivered','bounced','complained','failed')) default 'sending'
sent_at          timestamptz null
created_at       timestamptz default now()
updated_at       timestamptz default now()

unique (subscription_id, kind)          -- idempotency guard (#10)
index  (resend_message_id)              -- webhook lookup (#2)
index  (user_id)                        -- eligibility lookup
```

The `unique (subscription_id, kind)` constraint is the structural guarantee against duplicate warn1/warn2 (#10): a second send attempt collides instead of double-emailing.

## Eligibility logic (Component B) — `eligibility.py`

`compute_actions` changes:
- **Protected set:** treat `status in {active, trialing, past_due}` AND (`expires_at` is null OR `expires_at > now`) as active → skip user (#5). Set is a module constant for easy tuning.
- **Anchor on `scheduled_deletion_at`:** once `warn1` is sent, `scheduled_deletion_at` is the single anchor for `warn2` (24h before) and `delete` (at/after it). `warn1` itself still triggers on `now >= (expires_at + 30d) - 7d` (the pre-notice window is unchanged), but upon sending it sets `scheduled_deletion_at = max(expires_at + 30d, now + 7d)` so the post-warn1 window is always ≥7 days (#1).
- **Notice state from `retention_notices`:** derive "warn1 sent / warn2 sent" by querying the notices table (latest `sent`/`delivered` row per kind), not the legacy columns.
- **Pause on bounce (#2):** if the user's latest relevant notice has status `bounced` or `complained`, the user enters a **paused** state: `compute_actions` returns neither a further warn nor a `delete` for them, and the sweep emits a one-time admin alert. The sweep does **not** auto-resend to the same (failed) address. Resolution is a deliberate admin action — fix the address and reset the notice (or manually delete) via the admin tooling — not an automatic retry. This prevents both silent deletion-without-notice and an email loop to a dead address.

`MIN_GAP` (24h) between steps is retained.

## Notice sending (Component C) — `sweep.py` + `email.py`

- **Idempotent create-then-send:** before calling Resend, upsert a `retention_notices` row `(subscription_id, kind, status='sending')`. If a `sent`/`delivered` row already exists for that `(subscription_id, kind)` → skip (crash-safe #10).
- **After send:** update the row with `resend_message_id`, `status='sent'`, `sent_at=now()`. On `warn1`, also stamp `user_subscriptions.scheduled_deletion_at`.
- **Email date (#9):** templates render `scheduled_deletion_at` (the stored, authoritative date) instead of a recomputed one.
- **Language (#8):** resolve from `user_metadata.lang`; fall back to a configurable `RETENTION_DEFAULT_LANG` (default `de`, matching current behavior). Documented as a modest fix — only `de`/`en` templates exist and lang data is unreliable; this makes the fallback explicit and configurable rather than hard-coded.
- `send_email` still raises on non-2xx; a raise leaves the notice at `status='sending'` and is retried next sweep (no stamp, no scheduled_deletion_at on failure).

## Resend webhook (Component D) — new route

- `POST /api/retention/resend-webhook`
  - Verify the Resend (svix) signature using `RESEND_WEBHOOK_SECRET`. Reject unsigned/invalid.
  - Parse event type: `email.delivered`, `email.bounced`, `email.complained`.
  - Map by `resend_message_id` → update `retention_notices.status` (+ `updated_at`).
  - On `bounced`/`complained`: send a one-time alert email to Markus (admin address via env) and rely on Component B to pause that user's deletion.
- Requires configuring the webhook endpoint in the Resend dashboard and setting `RESEND_WEBHOOK_SECRET` on Cloud Run.
- Registered as a new blueprint under the `retention` domain (the domain currently has no `api/`).

## Robust deletion (Component E) — `deletion.py` + `sweep.py` (#6)

- `delete_user_data` returns a structured result `{sessions, storage_files, tables, errors: [...]}` instead of `None`.
- The sweep stamps `data_deleted_at` **only when `errors` is empty** (all storage + all user-keyed tables succeeded). On any error → leave unstamped, count as error, retry next sweep (idempotent re-run).
- **Storage verification:** after deletion, re-query for files belonging to the user's sessions; any remaining file is treated as an error (blocks the stamp).
- True cross-store transactions aren't possible (Supabase REST + Storage); the guarantee is "stamp = verified-complete, otherwise retry," not atomicity.

## Operations (Component F)

- **Admin trigger (#11):** `POST /api/retention/run-sweep`, protected by `RETENTION_ADMIN_SECRET` header. Calls `run_sweep(...)` honoring the live `dry_run` setting. Replaces hand-editing `retention_sweep_runs`.
- **Cloud Monitoring alarm (#4):** a log-based alert policy in GCP project `entropia-460611` that fires when no log entry matching `Retention sweep result` appears for >36h, with an email notification channel to Markus. Provisioned via `gcloud` and documented as an infra step (idempotent script).

## Never-subscribed cleanup (Component G) — #7

- A separate pass (or extension of the sweep) that identifies users with uploaded data but **no** `user_subscriptions` row (or only free), applies the same warn1→warn2→delete lifecycle anchored on data age (definition of "stale" to be set in the plan; e.g., last activity + N days).
- Reuses `retention_notices`, the email templates, and `delete_user_data`.
- Markus has approved the policy. Exact age threshold and which users qualify are settled during plan-writing.

## Delivery phases

```
Phase 1: Schema migration (A) — scheduled_deletion_at + retention_notices + backfill
Phase 2: Eligibility + unit tests (B) — protected statuses, anchor, pause-on-bounce
Phase 3: Notice send path + email (C) — idempotent notices, message_id, scheduled date, language
Phase 4: Resend webhook + admin alert (D)
Phase 5: Robust deletion (E) — verify-before-stamp
Phase 6: Admin endpoint + Cloud Monitoring alarm (F)
Phase 7: Never-subscribed cleanup (G) — #7
```

Phases 1→5 are sequential (each builds on the prior). Phase 6 is independent of 2–5. Phase 7 depends on 1–3.

## Error handling principles
- Email send failure → notice stays `sending`, retried next sweep; never stamps progress.
- Deletion partial failure → never stamps `data_deleted_at`; retried (idempotent).
- Webhook signature failure → 401, no state change.
- All money/PII-affecting actions are idempotent and safe to re-run.

## Testing strategy
All backend tests run in Docker (built with `.env`), per project rule.
- **Unit:** eligibility across all statuses + anchor + pause-on-bounce; idempotent notice creation (unique-constraint collision); webhook status mapping per event type; robust deletion (partial failure → no stamp); language resolution.
- **Integration:** webhook end-to-end with a mock Resend payload (valid + invalid signature); full warn1→warn2→delete sequence against a seeded test subscription with time injection (`now` param already supported by `compute_actions`/`run_sweep`).
- **Regression:** existing 20 retention tests must stay green.

## Out of scope
- Full GDPR "right to erasure" of the account itself (auth.users PII) — this flow is data-minimization on lapse, not erasure-on-request. Flagged separately.
- Switching the sweep trigger away from request-driven startup-fire (current mechanism is verified working; Cloud Monitoring alarm covers the failure mode).
