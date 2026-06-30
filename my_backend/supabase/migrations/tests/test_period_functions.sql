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
