-- Advisor ERROR (rls_disabled_in_public): retention_sweep_runs / retention_notices
-- are exposed to PostgREST without RLS. They are written only by the backend
-- retention sweep (service_role, which bypasses RLS); no anon/authenticated client
-- should touch them. Enable RLS with NO public policy → public API gets no access,
-- backend (service_role) unaffected.
ALTER TABLE public.retention_sweep_runs ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.retention_notices   ENABLE ROW LEVEL SECURITY;
