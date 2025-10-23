-- Migration: Remove timeout limit for service_role
-- Purpose: Allow long-running training operations to complete without timeout
--
-- The service_role is used by the backend with SUPABASE_SERVICE_ROLE_KEY
-- for operations like inserting large training results
--
-- Default service_role timeout: 8 seconds (inherited from authenticator role)
-- New timeout: 0 (unlimited)

-- Set service_role timeout to unlimited (0 means no timeout)
ALTER ROLE service_role SET statement_timeout = '0';

-- Reload PostgREST to apply the timeout change
NOTIFY pgrst, 'reload config';

-- Verification query (run manually to check):
-- SELECT rolname, rolconfig FROM pg_roles WHERE rolname = 'service_role';
