-- 20260429_030_pin_search_paths.sql
-- Pin search_path on all flagged SECURITY DEFINER and trigger functions.
-- Prevents catalog injection via user-created temp objects.

ALTER FUNCTION public.update_evaluation_tables_updated_at() SET search_path = public, pg_temp;
ALTER FUNCTION public.update_notification_preferences_updated_at() SET search_path = public, pg_temp;
ALTER FUNCTION public.downgrade_to_free_plan_transaction(uuid) SET search_path = public, pg_temp;

-- User-RPCs flagged with mutable paths
ALTER FUNCTION public.check_email_exists(text) SET search_path = public, pg_temp;
ALTER FUNCTION public.check_quota(uuid, character varying) SET search_path = public, pg_temp;
ALTER FUNCTION public.is_email_verified() SET search_path = public, pg_temp;
ALTER FUNCTION public.calculate_user_storage(uuid) SET search_path = public, pg_temp;
ALTER FUNCTION public.get_current_period_start() SET search_path = public, pg_temp;
ALTER FUNCTION public.user_owns_storage_path(text) SET search_path = public, pg_temp;
ALTER FUNCTION public.delete_unverified_users() SET search_path = public, auth, pg_temp;
ALTER FUNCTION public.increment_usage(uuid, date, character varying) SET search_path = public, pg_temp;
ALTER FUNCTION public.update_storage_usage(uuid, date, numeric) SET search_path = public, pg_temp;
