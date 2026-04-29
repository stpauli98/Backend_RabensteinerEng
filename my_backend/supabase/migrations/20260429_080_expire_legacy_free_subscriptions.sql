-- 20260429_080_expire_legacy_free_subscriptions.sql
-- Cancel any remaining active Free user_subscriptions so the Free row in
-- subscription_plans has no FK references and can be safely deleted.
-- Idempotent: rows already cancelled are unaffected.

UPDATE public.user_subscriptions
SET status = 'cancelled',
    expires_at = LEAST(expires_at, NOW())
WHERE status = 'active'
  AND plan_id IN (SELECT id FROM public.subscription_plans WHERE name = 'Free');
