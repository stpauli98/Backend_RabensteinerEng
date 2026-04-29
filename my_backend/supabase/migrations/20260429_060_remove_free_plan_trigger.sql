-- 20260429_060_remove_free_plan_trigger.sql
-- New users no longer receive an auto Free subscription. They land on /pricing
-- and must purchase a paid plan to access the app.
DROP TRIGGER IF EXISTS on_user_created_assign_free_plan ON auth.users;
-- assign_free_plan() function stays for now (harmless without trigger);
-- it is dropped in migration 20260429_090.
