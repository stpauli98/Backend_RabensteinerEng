-- Pure, deterministic period math anchored to a subscription start.
-- Anchor = midnight UTC of the start day; monthly stepping; Postgres clamps month-end.

CREATE OR REPLACE FUNCTION public.compute_period_start(
  p_started_at timestamptz,
  p_now timestamptz
) RETURNS date
LANGUAGE plpgsql
STABLE
SET search_path TO 'public', 'pg_temp'
AS $$
DECLARE
  v_started timestamp := date_trunc('day', (p_started_at AT TIME ZONE 'UTC'));
  v_now     timestamp := (p_now AT TIME ZONE 'UTC');
  v_diff    int;
  v_candidate timestamp;
BEGIN
  v_diff := (date_part('year', v_now) - date_part('year', v_started)) * 12
          + (date_part('month', v_now) - date_part('month', v_started));
  v_candidate := v_started + (v_diff || ' months')::interval;
  IF v_candidate > v_now THEN
    v_diff := v_diff - 1;
    v_candidate := v_started + (v_diff || ' months')::interval;
  END IF;
  RETURN v_candidate::date;
END;
$$;

CREATE OR REPLACE FUNCTION public.compute_next_period_start(
  p_started_at timestamptz,
  p_now timestamptz
) RETURNS date
LANGUAGE plpgsql
STABLE
SET search_path TO 'public', 'pg_temp'
AS $$
DECLARE
  v_started timestamp := date_trunc('day', (p_started_at AT TIME ZONE 'UTC'));
  v_now     timestamp := (p_now AT TIME ZONE 'UTC');
  v_diff    int;
BEGIN
  v_diff := (date_part('year', v_now) - date_part('year', v_started)) * 12
          + (date_part('month', v_now) - date_part('month', v_started));
  IF (v_started + (v_diff || ' months')::interval) > v_now THEN
    v_diff := v_diff - 1;
  END IF;
  RETURN (v_started + ((v_diff + 1) || ' months')::interval)::date;
END;
$$;
