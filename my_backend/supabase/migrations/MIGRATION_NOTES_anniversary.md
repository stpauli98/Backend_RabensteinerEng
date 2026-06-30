# Anniversary quota migrations — repo vs prod

The `20260630_100..135` files (plus `20260701_*` security fixes) were applied to
prod `luvjebsltuttakatnzaa` via the Supabase MCP `apply_migration` under
**descriptive names**, not the `20260630_NNN_*` version prefixes used in these
filenames. So the repo file version prefixes are **NOT** present in prod's
`supabase_migrations.schema_migrations`.

## DO NOT run `supabase db push` against prod for these without reconciling first

A `db push` that does not recognize these versions as applied would replay them in
order. To reconcile, get the prod versions (`supabase migration list` or the
`schema_migrations` table) and mark each as applied without re-running:

```
supabase migration repair --status applied <version>
```

## Seed safety

`_seed_anniversary_usage_once()` is a one-shot cutover. It was **DROPPED** on prod
by `20260630_135`. The repo `20260630_130` is now **guarded**: it only runs the
seed when no active subscriber yet has an anniversary-keyed `usage_tracking` row,
so a replay on a live database is a no-op (re-running would otherwise overwrite
live anniversary counts with stale 1st-of-month calendar values).

## Accepted non-fixes (audit 2026-06-30)

- Compute-gate **fail-open** on RPC error in `check_processing_limit` is
  intentional (don't lock out paying users on transient DB errors). The real
  exploit (active-but-expired sub → unlimited compute) is closed by the
  `expires_at > now()` check added to `require_subscription`.
- `usage_tracking.period_end` is informational only — no read path enforces
  against it. It may be a few days short for 29–31 day anchors; do not trust it
  as the reset date (use `get_next_period_reset()`).
- Reset is anchored to **midnight UTC of the subscription start day** (time-of-day
  discarded) by design — consistent month to month.
- Pre-existing perf-advisor items (RLS-initplan `auth.uid()` re-eval, multiple
  permissive policies, unused/duplicate `idx_usage_tracking_user_period`) are
  general DB hygiene, not introduced by this work.
