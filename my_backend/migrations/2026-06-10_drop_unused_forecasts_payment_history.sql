-- 2026-06-10 audit cleanup: drop two tables that are defined but never written
-- by any code path (0 rows, no FK/view dependents). Applied to Supabase project
-- luvjebsltuttakatnzaa via the MCP migration
-- `drop_unused_forecasts_and_payment_history_tables`.
--
-- The `forecasts` table was created in migrations/add_forecast_columns.sql but no
-- code ever inserts into it (the forecast endpoint builds its response in-memory).
-- `payment_history` was likewise never written.
--
-- INTENTIONALLY KEPT (live, in use by the Forecast API flow): the api_parameters
-- table and the files.fcst_var / data_source / api_source / latitude / longitude /
-- feature_index columns from add_forecast_columns.sql.

DROP TABLE IF EXISTS public.forecasts;
DROP TABLE IF EXISTS public.payment_history;
