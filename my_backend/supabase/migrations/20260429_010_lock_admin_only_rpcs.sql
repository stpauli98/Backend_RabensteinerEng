-- 20260429_010_lock_admin_only_rpcs.sql
-- Restrict admin/maintenance functions to service_role only.
-- Frontend audit (2026-04-29) confirmed no client callers.

-- delete_unverified_users: cleanup utility, must never be reachable from anon
REVOKE EXECUTE ON FUNCTION public.delete_unverified_users() FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.delete_unverified_users() TO service_role;

-- update_usage_tracking: trigger function only, never called directly
REVOKE EXECUTE ON FUNCTION public.update_usage_tracking() FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.update_usage_tracking() TO service_role;

-- log_usage_event: only the backend should log usage; usageLogger.ts is dead code
REVOKE EXECUTE ON FUNCTION public.log_usage_event(uuid, text, text, text, numeric, integer, jsonb)
  FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.log_usage_event(uuid, text, text, text, numeric, integer, jsonb)
  TO service_role;

-- increment_usage / update_storage_usage: backend-driven, never called from React
REVOKE EXECUTE ON FUNCTION public.increment_usage(uuid, date, character varying)
  FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.increment_usage(uuid, date, character varying)
  TO service_role;

REVOKE EXECUTE ON FUNCTION public.update_storage_usage(uuid, date, numeric)
  FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.update_storage_usage(uuid, date, numeric)
  TO service_role;

-- atomic_check_and_increment_quota: backend-only quota gate
REVOKE EXECUTE ON FUNCTION public.atomic_check_and_increment_quota(uuid, character varying, date)
  FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.atomic_check_and_increment_quota(uuid, character varying, date)
  TO service_role;
