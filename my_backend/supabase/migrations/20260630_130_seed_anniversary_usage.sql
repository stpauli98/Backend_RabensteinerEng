-- One-time cutover: carry each active subscriber's CURRENT calendar-month usage
-- into a row keyed at their CURRENT anniversary period. Idempotent + re-runnable.
-- Uses DO UPDATE (not DO NOTHING) because production ALREADY has vestigial
-- anniversary rows written by the retired trigger update_usage_tracking() (with
-- training_runs_count=0); we overwrite them with the authoritative calendar-row
-- counts so no user is reset. Old 1st-of-month rows are left as history.
-- MUST run AFTER the trigger is dropped (see 20260630_125_retire_usage_tracking_trigger.sql).
CREATE OR REPLACE FUNCTION public._seed_anniversary_usage_once()
RETURNS integer
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path TO 'public', 'pg_temp'
AS $$
DECLARE
  v_cal date := date_trunc('month', (now() AT TIME ZONE 'UTC'))::date;
  v_count int;
BEGIN
  WITH active_sub AS (
    SELECT DISTINCT ON (user_id) user_id, started_at
    FROM user_subscriptions
    WHERE status = 'active' AND expires_at > now()
    ORDER BY user_id, created_at DESC
  ),
  ins AS (
    INSERT INTO usage_tracking (
      user_id, period_start, period_end,
      uploads_count, processing_jobs_count, training_runs_count, storage_used_gb
    )
    SELECT
      ut.user_id,
      compute_period_start(a.started_at, now()) AS anniv_start,
      (compute_period_start(a.started_at, now()) + INTERVAL '1 month' - INTERVAL '1 day')::date,
      ut.uploads_count, ut.processing_jobs_count, ut.training_runs_count, ut.storage_used_gb
    FROM active_sub a
    JOIN usage_tracking ut
      ON ut.user_id = a.user_id
     AND ut.period_start = v_cal
    WHERE compute_period_start(a.started_at, now()) <> v_cal
    ON CONFLICT (user_id, period_start) DO UPDATE SET
      uploads_count         = EXCLUDED.uploads_count,
      processing_jobs_count = EXCLUDED.processing_jobs_count,
      training_runs_count   = EXCLUDED.training_runs_count,
      storage_used_gb       = EXCLUDED.storage_used_gb,
      period_end            = EXCLUDED.period_end
    RETURNING 1
  )
  SELECT count(*) INTO v_count FROM ins;
  RETURN v_count;
END;
$$;

-- Run the cutover once at deploy.
SELECT public._seed_anniversary_usage_once();
