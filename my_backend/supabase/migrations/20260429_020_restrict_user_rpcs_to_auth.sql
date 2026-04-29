-- 20260429_020_restrict_user_rpcs_to_auth.sql
-- Remove anonymous access to user-scoped quota/usage RPCs.
-- These are called from AuthContext after sign-in; anon should never reach them.

REVOKE EXECUTE ON FUNCTION public.check_quota(uuid, character varying)
  FROM PUBLIC, anon;
GRANT EXECUTE ON FUNCTION public.check_quota(uuid, character varying)
  TO authenticated;

REVOKE EXECUTE ON FUNCTION public.get_current_period_start()
  FROM PUBLIC, anon;
GRANT EXECUTE ON FUNCTION public.get_current_period_start()
  TO authenticated;

REVOKE EXECUTE ON FUNCTION public.get_total_compute_seconds(uuid)
  FROM PUBLIC, anon;
GRANT EXECUTE ON FUNCTION public.get_total_compute_seconds(uuid)
  TO authenticated;

REVOKE EXECUTE ON FUNCTION public.calculate_user_storage(uuid)
  FROM PUBLIC, anon;
GRANT EXECUTE ON FUNCTION public.calculate_user_storage(uuid)
  TO authenticated;

REVOKE EXECUTE ON FUNCTION public.is_email_verified()
  FROM PUBLIC, anon;
GRANT EXECUTE ON FUNCTION public.is_email_verified()
  TO authenticated;

REVOKE EXECUTE ON FUNCTION public.user_owns_storage_path(text)
  FROM PUBLIC, anon;
GRANT EXECUTE ON FUNCTION public.user_owns_storage_path(text)
  TO authenticated;
