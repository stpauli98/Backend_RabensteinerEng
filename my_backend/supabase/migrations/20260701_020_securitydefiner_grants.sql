-- Advisor WARN (authenticated_security_definer_function_executable): these
-- SECURITY DEFINER functions took an arbitrary uuid and were authenticated-callable
-- → cross-user reads. Neither has a production caller (check_quota: only the dead
-- frontend checkQuota() wrapper; calculate_user_storage: none, and its body ignores
-- p_user_id). Lock both to service_role.
-- NOTE: check_email_exists(text) is intentionally NOT revoked — the frontend
-- registration flow calls it (anon was already revoked in 20260429_040).
REVOKE EXECUTE ON FUNCTION public.check_quota(uuid, character varying) FROM anon, authenticated;
REVOKE EXECUTE ON FUNCTION public.calculate_user_storage(uuid) FROM anon, authenticated;
