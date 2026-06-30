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
