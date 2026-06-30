-- Assertion tests for the anniversary-period SQL functions.
-- Run against a Postgres that has the subscription tables + these functions:
--   psql "$DATABASE_URL" -f this_file.sql
-- Raises an exception on the first failed assertion; prints success NOTICEs otherwise.

-- ============================================================
-- Task 1: pure period math
-- ============================================================
DO $$
BEGIN
  -- same-month, anniversary day not reached again -> current window
  ASSERT compute_period_start('2026-06-15 07:49:00+00','2026-06-30 00:00:00+00') = DATE '2026-06-15', 'A1';
  -- day before next anniversary -> still current window
  ASSERT compute_period_start('2026-06-15 07:49:00+00','2026-07-14 23:00:00+00') = DATE '2026-06-15', 'A2';
  -- exactly on the anniversary (midnight UTC) -> new window
  ASSERT compute_period_start('2026-06-15 07:49:00+00','2026-07-15 00:00:00+00') = DATE '2026-07-15', 'A3';
  -- month-end clamp: Jan 31 anchor, in early March -> Feb 28
  ASSERT compute_period_start('2026-01-31 12:00:00+00','2026-03-01 00:00:00+00') = DATE '2026-02-28', 'A4';
  -- leap year: Jan 31 anchor -> Feb 29
  ASSERT compute_period_start('2024-01-31 12:00:00+00','2024-03-01 00:00:00+00') = DATE '2024-02-29', 'A5';
  -- yearly subscriber, many months later
  ASSERT compute_period_start('2026-06-15 07:49:00+00','2027-01-20 10:00:00+00') = DATE '2027-01-15', 'A6';
  -- prior-year anchor (internal_unlimited style)
  ASSERT compute_period_start('2025-11-07 20:26:00+00','2026-06-30 00:00:00+00') = DATE '2026-06-07', 'A7';
  -- NEXT window (reset date)
  ASSERT compute_next_period_start('2026-06-15 07:49:00+00','2026-06-30 00:00:00+00') = DATE '2026-07-15', 'N1';
  -- NEXT after a Jan31 anchor while still in the Jan window (Feb 10) -> Feb 28 (clamped anniversary)
  ASSERT compute_next_period_start('2026-01-31 12:00:00+00','2026-02-10 00:00:00+00') = DATE '2026-02-28', 'N2';
  -- NEXT re-expands once inside the clamped Feb window (Mar 10) -> Mar 31
  ASSERT compute_next_period_start('2026-01-31 12:00:00+00','2026-03-10 00:00:00+00') = DATE '2026-03-31', 'N3';
  RAISE NOTICE 'ALL PERIOD ASSERTIONS PASSED';
END $$;

-- ============================================================
-- Task 2: user-facing period functions (overload + no-arg via auth.uid + next reset)
-- ============================================================
DO $$
DECLARE
  v_uid uuid := '00000000-0000-0000-0000-0000000000a2';
  v_plan uuid;
  v_expected date;
  v_noarg date;
BEGIN
  SELECT id INTO v_plan FROM subscription_plans WHERE slug='standard' LIMIT 1;
  DELETE FROM user_subscriptions WHERE user_id = v_uid;
  -- active sub anchored on the 15th, two-ish months ago, not yet expired
  INSERT INTO user_subscriptions (user_id, plan_id, billing_cycle, status, started_at, expires_at, created_at)
  VALUES (v_uid, v_plan, 'yearly', 'active',
          (date_trunc('month', now()) - INTERVAL '2 months' + INTERVAL '14 days'),
          now() + INTERVAL '300 days', now() - INTERVAL '2 months');

  v_expected := compute_period_start(
    (date_trunc('month', now()) - INTERVAL '2 months' + INTERVAL '14 days')::timestamptz, now());

  -- overload (service-role path) matches the pure function
  ASSERT get_current_period_start(v_uid) = v_expected, 'OVERLOAD_MATCHES_COMPUTE';

  -- no-arg version resolves the same value when auth.uid() is the caller
  PERFORM set_config('request.jwt.claim.sub', v_uid::text, true);
  v_noarg := get_current_period_start();
  ASSERT v_noarg = v_expected, format('NOARG_MATCHES_OVERLOAD expected %s got %s', v_expected, v_noarg);

  -- next reset is exactly one anniversary step after the current period start
  ASSERT get_next_period_reset() = compute_next_period_start(
    (date_trunc('month', now()) - INTERVAL '2 months' + INTERVAL '14 days')::timestamptz, now()),
    'NEXT_RESET_MATCHES';

  -- fallback: unknown caller -> 1st of current month (unchanged behavior)
  PERFORM set_config('request.jwt.claim.sub', '00000000-0000-0000-0000-0000000000ff', true);
  ASSERT get_current_period_start() = date_trunc('month', (now() AT TIME ZONE 'UTC'))::date, 'NOSUB_FALLBACK';

  DELETE FROM user_subscriptions WHERE user_id = v_uid;
  RAISE NOTICE 'PERIOD FUNCTION ASSERTIONS PASSED';
END $$;

-- ============================================================
-- Task 3: compute hours window = since current anniversary
-- ============================================================
DO $$
DECLARE
  v_uid uuid := '00000000-0000-0000-0000-0000000000aa';
  v_plan uuid;
  v_total bigint;
BEGIN
  SELECT id INTO v_plan FROM subscription_plans WHERE slug='standard' LIMIT 1;
  DELETE FROM usage_events WHERE user_id = v_uid;
  DELETE FROM user_subscriptions WHERE user_id = v_uid;

  -- started_at 2 months + 5 days before "now" so the current window opened recently
  INSERT INTO user_subscriptions (user_id, plan_id, billing_cycle, status, started_at, expires_at, created_at)
  VALUES (v_uid, v_plan, 'monthly', 'active', now() - INTERVAL '2 months 5 days', now() + INTERVAL '25 days', now() - INTERVAL '2 months 5 days');

  -- 100s logged BEFORE the current anniversary (should be excluded)
  INSERT INTO usage_events (user_id, event_type, resource_type, processing_duration_sec, created_at)
  VALUES (v_uid, 'processing', 'training', 100, now() - INTERVAL '40 days');
  -- 60s logged AFTER the current anniversary (should be counted)
  INSERT INTO usage_events (user_id, event_type, resource_type, processing_duration_sec, created_at)
  VALUES (v_uid, 'processing', 'training', 60, now() - INTERVAL '2 days');

  SELECT get_total_compute_seconds(v_uid) INTO v_total;
  ASSERT v_total = 60, format('Expected 60, got %s', v_total);

  DELETE FROM usage_events WHERE user_id = v_uid;
  DELETE FROM user_subscriptions WHERE user_id = v_uid;
  RAISE NOTICE 'COMPUTE WINDOW ASSERTION PASSED';
END $$;

-- ============================================================
-- Task 4: cutover seed carries existing counts into the anniversary row
-- ============================================================
DO $$
DECLARE
  v_uid uuid := '00000000-0000-0000-0000-0000000000bb';
  v_plan uuid;
  v_cal date := date_trunc('month', (now() AT TIME ZONE 'UTC'))::date;
  v_anchor timestamptz := (v_cal + INTERVAL '14 days');   -- anchor day = 15th
  v_anniv date;
  v_seeded int;
BEGIN
  SELECT id INTO v_plan FROM subscription_plans WHERE slug='standard' LIMIT 1;
  DELETE FROM usage_tracking WHERE user_id = v_uid;
  DELETE FROM user_subscriptions WHERE user_id = v_uid;

  INSERT INTO user_subscriptions (user_id, plan_id, billing_cycle, status, started_at, expires_at, created_at)
  VALUES (v_uid, v_plan, 'monthly', 'active', v_anchor, now() + INTERVAL '20 days', now());
  v_anniv := compute_period_start(v_anchor, now());

  -- existing calendar-month usage row with real counts
  INSERT INTO usage_tracking (user_id, period_start, period_end, training_runs_count, processing_jobs_count, uploads_count, storage_used_gb)
  VALUES (v_uid, v_cal, (v_cal + INTERVAL '1 month' - INTERVAL '1 day')::date, 3, 2, 1, 0.5);

  PERFORM public._seed_anniversary_usage_once();

  SELECT training_runs_count INTO v_seeded
  FROM usage_tracking WHERE user_id = v_uid AND period_start = v_anniv;
  ASSERT v_seeded = 3, format('Expected seeded training=3 at anniversary %s, got %s', v_anniv, v_seeded);

  -- idempotency: a second run must not duplicate or alter
  PERFORM public._seed_anniversary_usage_once();
  ASSERT (SELECT count(*) FROM usage_tracking WHERE user_id = v_uid AND period_start = v_anniv) = 1, 'SEED_IDEMPOTENT';

  DELETE FROM usage_tracking WHERE user_id = v_uid;
  DELETE FROM user_subscriptions WHERE user_id = v_uid;
  RAISE NOTICE 'SEED ASSERTION PASSED';
END $$;

-- ============================================================
-- Part 2: retire trigger + seed overwrites vestigial anniversary rows
-- ============================================================
-- Part 2 validation: simulate prod's vestigial dual-write, then prove the
-- revised seed (DO UPDATE from calendar) overwrites the zeros without reset.

-- Simulate the production trigger (anniversary-keyed, training never logged).
CREATE OR REPLACE FUNCTION public.update_usage_tracking() RETURNS trigger
LANGUAGE plpgsql AS $$
DECLARE v_ps date; v_created date;
BEGIN
  SELECT created_at::date INTO v_created FROM user_subscriptions
   WHERE user_id=NEW.user_id AND status='active' ORDER BY created_at DESC LIMIT 1;
  IF v_created IS NULL THEN v_ps := date_trunc('month', current_date);
  ELSE
    IF extract(day from current_date) < extract(day from v_created)
      THEN v_ps := (date_trunc('month', current_date) - interval '1 month' + ((extract(day from v_created)-1)||' days')::interval)::date;
      ELSE v_ps := (date_trunc('month', current_date) + ((extract(day from v_created)-1)||' days')::interval)::date;
    END IF;
  END IF;
  INSERT INTO usage_tracking (user_id, period_start, period_end, processing_jobs_count, training_runs_count, uploads_count, storage_used_gb)
  VALUES (NEW.user_id, v_ps, (v_ps + interval '30 days' - interval '1 day')::date,
          CASE WHEN NEW.event_type='processing' THEN 1 ELSE 0 END, 0, 0, 0)
  ON CONFLICT (user_id, period_start) DO UPDATE SET
    processing_jobs_count = usage_tracking.processing_jobs_count + CASE WHEN NEW.event_type='processing' THEN 1 ELSE 0 END;
  RETURN NEW;
END $$;
CREATE TRIGGER trigger_update_usage_tracking AFTER INSERT ON public.usage_events
  FOR EACH ROW EXECUTE FUNCTION update_usage_tracking();

DO $$
DECLARE
  v_uid uuid := '00000000-0000-0000-0000-0000000000c2';
  v_plan uuid;
  v_cal date := date_trunc('month', now() at time zone 'UTC')::date;
  v_anchor timestamptz := v_cal + interval '14 days';  -- 15th
  v_anniv date;
  v_train int; v_proc int; v_end date;
BEGIN
  SELECT id INTO v_plan FROM subscription_plans WHERE slug='standard' LIMIT 1;
  DELETE FROM usage_tracking WHERE user_id=v_uid;
  DELETE FROM usage_events WHERE user_id=v_uid;
  DELETE FROM user_subscriptions WHERE user_id=v_uid;

  INSERT INTO user_subscriptions (user_id, plan_id, billing_cycle, status, started_at, expires_at, created_at)
  VALUES (v_uid, v_plan, 'monthly', 'active', v_anchor, now()+interval '20 days', v_anchor);
  v_anniv := compute_period_start(v_anchor, now());

  -- Authoritative app calendar row: training=5, processing=10
  INSERT INTO usage_tracking (user_id, period_start, period_end, training_runs_count, processing_jobs_count, uploads_count, storage_used_gb)
  VALUES (v_uid, v_cal, (v_cal + interval '1 month' - interval '1 day')::date, 5, 10, 2, 0.30);

  -- Vestigial trigger row: fire 3 processing events -> anniversary row training=0, proc=3
  INSERT INTO usage_events (user_id, event_type, resource_type, processing_duration_sec, created_at)
    SELECT v_uid, 'processing', 'x', 10, now() from generate_series(1,3);

  -- precondition: vestigial anniversary row exists with training=0
  SELECT training_runs_count, processing_jobs_count INTO v_train, v_proc
    FROM usage_tracking WHERE user_id=v_uid AND period_start=v_anniv;
  ASSERT v_train = 0 AND v_proc = 3, format('precondition: expected training0/proc3, got %s/%s', v_train, v_proc);

  -- DEPLOY STEP 1: drop trigger
  DROP TRIGGER trigger_update_usage_tracking ON public.usage_events;

  -- DEPLOY STEP 2: seed (DO UPDATE from calendar)
  PERFORM public._seed_anniversary_usage_once();

  -- assert anniversary row now carries the AUTHORITATIVE calendar counts (no reset)
  SELECT training_runs_count, processing_jobs_count, period_end INTO v_train, v_proc, v_end
    FROM usage_tracking WHERE user_id=v_uid AND period_start=v_anniv;
  ASSERT v_train = 5, format('seed overwrite: expected training=5, got %s', v_train);
  ASSERT v_proc = 10, format('seed overwrite: expected proc=10, got %s', v_proc);
  ASSERT v_end = (v_anniv + interval '1 month' - interval '1 day')::date, format('period_end fixed: got %s', v_end);

  -- idempotency: second run keeps values, single row
  PERFORM public._seed_anniversary_usage_once();
  ASSERT (SELECT count(*) FROM usage_tracking WHERE user_id=v_uid AND period_start=v_anniv)=1, 'idempotent rows';
  SELECT training_runs_count INTO v_train FROM usage_tracking WHERE user_id=v_uid AND period_start=v_anniv;
  ASSERT v_train = 5, 'idempotent value';

  -- DEPLOY STEP 4 effect: read path keys on anniversary -> sees 5 (not the 0 it would have)
  ASSERT get_current_period_start(v_uid) = v_anniv, 'period flip resolves to anniversary';

  DELETE FROM usage_tracking WHERE user_id=v_uid;
  DELETE FROM usage_events WHERE user_id=v_uid;
  DELETE FROM user_subscriptions WHERE user_id=v_uid;
  RAISE NOTICE 'PART 2 SEED/RETIRE ASSERTIONS PASSED';
END $$;
