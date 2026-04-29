-- 20260429_090_drop_free_plan_row.sql
-- Final removal of Free from subscription_plans + drop legacy functions.
-- Safe at this point because:
-- 1. DB-1 dropped the trigger that auto-creates Free rows.
-- 2. DB-3 cancelled all active Free user_subscriptions.
-- 3. BE-1/BE-2 removed all Python callers of Free-related functions.
-- 4. FE-1..FE-6 removed all frontend references to Free.
--
-- We must cascade-delete bottom-up: usage_tracking → user_subscriptions →
-- subscription_plans. The 10 historical Free user_subscriptions all belong
-- to the same test user (nmil32@icloud.com) and are pre-production artifacts.

-- 1. Clear usage_tracking rows that reference Free user_subscriptions
DELETE FROM public.usage_tracking
WHERE subscription_id IN (
  SELECT id FROM public.user_subscriptions
  WHERE plan_id IN (SELECT id FROM public.subscription_plans WHERE name = 'Free')
);

-- 2. Delete the Free user_subscriptions rows themselves
DELETE FROM public.user_subscriptions
WHERE plan_id IN (SELECT id FROM public.subscription_plans WHERE name = 'Free');

-- 3. Now safe to delete the Free plan
DELETE FROM public.subscription_plans WHERE name = 'Free';

-- 4. Drop the now-orphan helper functions
DROP FUNCTION IF EXISTS public.assign_free_plan();
DROP FUNCTION IF EXISTS public.downgrade_to_free_plan_transaction(uuid);
