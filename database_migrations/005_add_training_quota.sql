-- =====================================================
-- ADD TRAINING QUOTA COLUMN TO SUBSCRIPTION PLANS
-- Date: 2025-10-22
-- Description: Adds max_training_runs_per_month column for training quota management
-- =====================================================

-- Add max_training_runs_per_month column to subscription_plans table
ALTER TABLE subscription_plans
ADD COLUMN IF NOT EXISTS max_training_runs_per_month INTEGER DEFAULT 0;

-- Add comment
COMMENT ON COLUMN subscription_plans.max_training_runs_per_month IS 'Maximum number of training runs per month. -1 means unlimited, 0 means not allowed';

-- Update existing plans with training limits
UPDATE subscription_plans
SET max_training_runs_per_month = CASE
  WHEN name = 'Free' THEN 0        -- Free plan: no training
  WHEN name = 'Pro' THEN 5          -- Pro plan: 5 runs per month
  WHEN name = 'Enterprise' THEN -1  -- Enterprise: unlimited
  ELSE 0                            -- Default: no training
END
WHERE max_training_runs_per_month = 0 OR max_training_runs_per_month IS NULL;

-- Verify the migration
SELECT
  name,
  can_use_training,
  max_training_runs_per_month,
  CASE
    WHEN max_training_runs_per_month = -1 THEN 'Unlimited'
    WHEN max_training_runs_per_month = 0 THEN 'Not allowed'
    ELSE CAST(max_training_runs_per_month AS TEXT) || ' runs/month'
  END as training_quota
FROM subscription_plans
ORDER BY price_monthly;
