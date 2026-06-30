-- Atomic additive storage update (replaces the racy SELECT-then-write in
-- shared/tracking/usage.py:update_storage_usage). ON CONFLICT upsert so concurrent
-- writes can't lose updates.
CREATE OR REPLACE FUNCTION public.increment_storage_usage(p_user_id uuid, p_period_start date, p_storage_gb numeric)
RETURNS void LANGUAGE plpgsql SECURITY DEFINER SET search_path TO 'public', 'pg_temp'
AS $$
BEGIN
  INSERT INTO usage_tracking (user_id, period_start, period_end, storage_used_gb)
  VALUES (p_user_id, p_period_start, (p_period_start + INTERVAL '1 month' - INTERVAL '1 day')::date, p_storage_gb)
  ON CONFLICT (user_id, period_start) DO UPDATE SET
    storage_used_gb = usage_tracking.storage_used_gb + p_storage_gb, updated_at = now();
END; $$;
REVOKE ALL ON FUNCTION public.increment_storage_usage(uuid, date, numeric) FROM public;
GRANT EXECUTE ON FUNCTION public.increment_storage_usage(uuid, date, numeric) TO service_role;
