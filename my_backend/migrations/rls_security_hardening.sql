-- RLS Security Hardening Migration
-- Applied: 2026-04-01
-- Fixes: overpermissive USING(true) policies on public role across all tables
-- Audit report: Frontend/claudedocs/SECURITY_AUDIT_RLS_2026-04-02.md

-- ============================================================================
-- 1. Drop overpermissive "Service role" policies on public role
-- ============================================================================
DROP POLICY IF EXISTS "Service role can access all csv_file_refs" ON csv_file_refs;
DROP POLICY IF EXISTS "Service role can access all files" ON files;
DROP POLICY IF EXISTS "Service role can access all session_mappings" ON session_mappings;
DROP POLICY IF EXISTS "Service role can access all time_info" ON time_info;
DROP POLICY IF EXISTS "Service role can access all training_progress" ON training_progress;
DROP POLICY IF EXISTS "Service role can access all zeitschritte" ON zeitschritte;
DROP POLICY IF EXISTS "Anyone can read api_parameters" ON api_parameters;
DROP POLICY IF EXISTS "Public read access to subscription plans" ON subscription_plans;

-- ============================================================================
-- 2. Recreate as service_role only (not public)
-- ============================================================================
CREATE POLICY "Service role full access csv_file_refs" ON csv_file_refs FOR ALL TO service_role USING (true) WITH CHECK (true);
CREATE POLICY "Service role full access files" ON files FOR ALL TO service_role USING (true) WITH CHECK (true);
CREATE POLICY "Service role full access session_mappings" ON session_mappings FOR ALL TO service_role USING (true) WITH CHECK (true);
CREATE POLICY "Service role full access time_info" ON time_info FOR ALL TO service_role USING (true) WITH CHECK (true);
CREATE POLICY "Service role full access training_progress" ON training_progress FOR ALL TO service_role USING (true) WITH CHECK (true);
CREATE POLICY "Service role full access zeitschritte" ON zeitschritte FOR ALL TO service_role USING (true) WITH CHECK (true);

-- ============================================================================
-- 3. Reference tables: authenticated read, service_role write
-- ============================================================================
CREATE POLICY "Authenticated can read api_parameters" ON api_parameters FOR SELECT TO authenticated USING (true);
CREATE POLICY "Service role full access api_parameters" ON api_parameters FOR ALL TO service_role USING (true) WITH CHECK (true);
CREATE POLICY "Authenticated can read subscription_plans" ON subscription_plans FOR SELECT TO authenticated USING (true);

-- ============================================================================
-- 4. training_progress: add missing user-scoped policies
-- ============================================================================
CREATE POLICY "Users can select own training_progress" ON training_progress FOR SELECT TO authenticated
USING (EXISTS (SELECT 1 FROM sessions s WHERE s.id = training_progress.session_id AND s.user_id = auth.uid()));
CREATE POLICY "Users can update own training_progress" ON training_progress FOR UPDATE TO authenticated
USING (EXISTS (SELECT 1 FROM sessions s WHERE s.id = training_progress.session_id AND s.user_id = auth.uid()));
CREATE POLICY "Users can insert own training_progress" ON training_progress FOR INSERT TO authenticated
WITH CHECK (EXISTS (SELECT 1 FROM sessions s WHERE s.id = training_progress.session_id AND s.user_id = auth.uid()));

-- ============================================================================
-- 5. Missing service_role + user write policies
-- ============================================================================
CREATE POLICY "Service role full access evaluation_tables" ON evaluation_tables FOR ALL TO service_role USING (true) WITH CHECK (true);
CREATE POLICY "Service role full access forecasts" ON forecasts FOR ALL TO service_role USING (true) WITH CHECK (true);
CREATE POLICY "Service role full access saved_models" ON saved_models FOR ALL TO service_role USING (true) WITH CHECK (true);

CREATE POLICY "Users can insert own evaluation_tables" ON evaluation_tables FOR INSERT TO authenticated
WITH CHECK (EXISTS (SELECT 1 FROM sessions s WHERE s.id = evaluation_tables.session_id AND s.user_id = auth.uid()));
CREATE POLICY "Users can update own evaluation_tables" ON evaluation_tables FOR UPDATE TO authenticated
USING (EXISTS (SELECT 1 FROM sessions s WHERE s.id = evaluation_tables.session_id AND s.user_id = auth.uid()));
CREATE POLICY "Users can delete own evaluation_tables" ON evaluation_tables FOR DELETE TO authenticated
USING (EXISTS (SELECT 1 FROM sessions s WHERE s.id = evaluation_tables.session_id AND s.user_id = auth.uid()));

-- ============================================================================
-- 6. Fix sessions INSERT policy (add user_id check)
-- ============================================================================
DROP POLICY IF EXISTS "Users can insert own sessions" ON sessions;
CREATE POLICY "Users can insert own sessions" ON sessions FOR INSERT TO authenticated
WITH CHECK (auth.uid() = user_id);
