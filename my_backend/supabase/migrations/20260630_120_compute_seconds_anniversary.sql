-- Compute hours reset monthly: sum events from the CURRENT anniversary boundary,
-- not cumulatively from started_at.
CREATE OR REPLACE FUNCTION public.get_total_compute_seconds(p_user_id uuid)
RETURNS bigint
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path TO 'public'
AS $$
DECLARE
  total BIGINT;
  v_started TIMESTAMPTZ;
  v_window_start TIMESTAMPTZ;
BEGIN
  SELECT started_at INTO v_started
  FROM user_subscriptions
  WHERE user_id = p_user_id
    AND status = 'active'
    AND expires_at > NOW()
  ORDER BY created_at DESC
  LIMIT 1;

  IF v_started IS NULL THEN
    RETURN 0;
  END IF;

  -- Midnight UTC of the current anniversary day.
  v_window_start := (compute_period_start(v_started, NOW())::timestamp AT TIME ZONE 'UTC');

  SELECT COALESCE(SUM(processing_duration_sec), 0)
  INTO total
  FROM usage_events
  WHERE user_id = p_user_id
    AND processing_duration_sec IS NOT NULL
    AND created_at >= v_window_start;

  RETURN total;
END;
$$;
