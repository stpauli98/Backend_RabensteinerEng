-- 20260429_040_remove_check_email_exists_anon.sql
-- Frontend no longer calls this from anon context (RegisterForm pre-check removed
-- in commit 668b00a). Close the email-enumeration vector.

REVOKE EXECUTE ON FUNCTION public.check_email_exists(text) FROM PUBLIC, anon;
GRANT EXECUTE ON FUNCTION public.check_email_exists(text) TO authenticated;
