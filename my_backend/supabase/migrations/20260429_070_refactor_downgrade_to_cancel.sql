-- 20260429_070_refactor_downgrade_to_cancel.sql
-- Replace the legacy "downgrade to Free" pattern with "cancel active subscription".
-- Free plan is being removed; cancelled users land on /pricing on their next request.

CREATE OR REPLACE FUNCTION public.cancel_active_subscription_transaction(p_user_id uuid)
RETURNS json
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public, pg_temp
AS $$
DECLARE
  v_cancelled_count int;
BEGIN
  IF p_user_id IS NULL THEN
    RAISE EXCEPTION 'p_user_id is required';
  END IF;

  UPDATE public.user_subscriptions
  SET status = 'cancelled',
      expires_at = LEAST(expires_at, NOW())
  WHERE user_id = p_user_id
    AND status = 'active';
  GET DIAGNOSTICS v_cancelled_count = ROW_COUNT;

  RETURN json_build_object(
    'cancelled_count', v_cancelled_count,
    'user_id', p_user_id
  );
END;
$$;

REVOKE EXECUTE ON FUNCTION public.cancel_active_subscription_transaction(uuid) FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.cancel_active_subscription_transaction(uuid) TO service_role;
