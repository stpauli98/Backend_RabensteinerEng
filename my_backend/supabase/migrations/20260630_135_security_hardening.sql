-- Part 2 security hardening (CodeRabbit review + ACL audit on prod).
-- Supabase default privileges auto-grant EXECUTE to anon/authenticated on new
-- functions, which a bare `REVOKE ... FROM public` does NOT remove.

-- 1) Drop the one-shot cutover seed. It has already run; leaving it is dangerous:
--    it was PUBLIC-executable and its DO UPDATE would clobber LIVE anniversary rows
--    with stale calendar counts on any re-run, resetting users.
DROP FUNCTION IF EXISTS public._seed_anniversary_usage_once();

-- 2) Lock the explicit-user period overload to service_role only.
REVOKE EXECUTE ON FUNCTION public.get_current_period_start(uuid) FROM anon, authenticated;

-- 3) get_total_compute_seconds(uuid) stays authenticated (the frontend calls it with
--    the user's own id), but enforces caller ownership so an authenticated user cannot
--    read another user's usage via a different uuid. service_role (auth.uid() IS NULL)
--    and same-user calls pass.
CREATE OR REPLACE FUNCTION public.get_total_compute_seconds(p_user_id uuid)
RETURNS bigint
LANGUAGE plpgsql STABLE SECURITY DEFINER SET search_path TO 'public', 'pg_temp'
AS $$
DECLARE total BIGINT; v_started TIMESTAMPTZ; v_window_start TIMESTAMPTZ;
BEGIN
  IF auth.uid() IS NOT NULL AND auth.uid() <> p_user_id THEN
    RAISE EXCEPTION 'forbidden: cannot read another user''s usage';
  END IF;
  SELECT started_at INTO v_started FROM user_subscriptions
   WHERE user_id = p_user_id AND status = 'active' AND expires_at > NOW()
   ORDER BY created_at DESC LIMIT 1;
  IF v_started IS NULL THEN RETURN 0; END IF;
  v_window_start := (compute_period_start(v_started, NOW())::timestamp AT TIME ZONE 'UTC');
  SELECT COALESCE(SUM(processing_duration_sec), 0) INTO total FROM usage_events
   WHERE user_id = p_user_id AND processing_duration_sec IS NOT NULL AND created_at >= v_window_start;
  RETURN total;
END; $$;
REVOKE ALL ON FUNCTION public.get_total_compute_seconds(uuid) FROM public;
GRANT EXECUTE ON FUNCTION public.get_total_compute_seconds(uuid) TO authenticated, service_role;
