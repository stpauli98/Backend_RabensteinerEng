-- Replace the calendar-month period with anniversary-of-started_at.
-- No-arg version: resolves the caller via auth.uid() (frontend + backend read path).
CREATE OR REPLACE FUNCTION public.get_current_period_start()
RETURNS date
LANGUAGE plpgsql
STABLE
SECURITY DEFINER
SET search_path TO 'public', 'pg_temp'
AS $$
DECLARE
  v_started timestamptz;
BEGIN
  SELECT started_at INTO v_started
  FROM user_subscriptions
  WHERE user_id = auth.uid() AND status = 'active' AND expires_at > now()
  ORDER BY created_at DESC
  LIMIT 1;

  IF v_started IS NULL THEN
    RETURN date_trunc('month', (now() AT TIME ZONE 'UTC'))::date;  -- fallback (unchanged behavior)
  END IF;
  RETURN compute_period_start(v_started, now());
END;
$$;

-- Explicit-user overload for the service-role backend write path.
CREATE OR REPLACE FUNCTION public.get_current_period_start(p_user_id uuid)
RETURNS date
LANGUAGE plpgsql
STABLE
SECURITY DEFINER
SET search_path TO 'public', 'pg_temp'
AS $$
DECLARE
  v_started timestamptz;
BEGIN
  SELECT started_at INTO v_started
  FROM user_subscriptions
  WHERE user_id = p_user_id AND status = 'active' AND expires_at > now()
  ORDER BY created_at DESC
  LIMIT 1;

  IF v_started IS NULL THEN
    RETURN date_trunc('month', (now() AT TIME ZONE 'UTC'))::date;
  END IF;
  RETURN compute_period_start(v_started, now());
END;
$$;

-- Next reset date for the current caller (UI display).
CREATE OR REPLACE FUNCTION public.get_next_period_reset()
RETURNS date
LANGUAGE plpgsql
STABLE
SECURITY DEFINER
SET search_path TO 'public', 'pg_temp'
AS $$
DECLARE
  v_started timestamptz;
BEGIN
  SELECT started_at INTO v_started
  FROM user_subscriptions
  WHERE user_id = auth.uid() AND status = 'active' AND expires_at > now()
  ORDER BY created_at DESC
  LIMIT 1;

  IF v_started IS NULL THEN
    RETURN (date_trunc('month', (now() AT TIME ZONE 'UTC')) + INTERVAL '1 month')::date;
  END IF;
  RETURN compute_next_period_start(v_started, now());
END;
$$;

-- Grants (mirror supabase/migrations/20260429_010/020_*.sql conventions).
REVOKE ALL ON FUNCTION public.compute_period_start(timestamptz, timestamptz) FROM public;
REVOKE ALL ON FUNCTION public.compute_next_period_start(timestamptz, timestamptz) FROM public;
REVOKE ALL ON FUNCTION public.get_current_period_start() FROM public;
REVOKE ALL ON FUNCTION public.get_current_period_start(uuid) FROM public;
REVOKE ALL ON FUNCTION public.get_next_period_reset() FROM public;

GRANT EXECUTE ON FUNCTION public.get_current_period_start() TO authenticated, service_role;
GRANT EXECUTE ON FUNCTION public.get_next_period_reset() TO authenticated, service_role;
GRANT EXECUTE ON FUNCTION public.compute_period_start(timestamptz, timestamptz) TO authenticated, service_role;
GRANT EXECUTE ON FUNCTION public.compute_next_period_start(timestamptz, timestamptz) TO authenticated, service_role;
-- Explicit-user overload: backend only; a normal user must not query another user's period.
GRANT EXECUTE ON FUNCTION public.get_current_period_start(uuid) TO service_role;
