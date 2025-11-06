-- ============================================================================
-- SUBSCRIPTION TRANSACTION FUNCTIONS
-- ============================================================================
-- These functions provide atomic database operations for subscription management
-- Run this SQL in your Supabase SQL Editor to enable transaction support
-- ============================================================================

-- ============================================================================
-- Function: handle_successful_payment_transaction
-- Purpose: Atomically cancel old subscriptions and create new one
-- Returns: JSON with success status and new subscription ID
-- ============================================================================
CREATE OR REPLACE FUNCTION handle_successful_payment_transaction(
    p_user_id UUID,
    p_plan_id UUID,
    p_billing_cycle VARCHAR(20),
    p_started_at TIMESTAMPTZ,
    p_expires_at TIMESTAMPTZ,
    p_next_billing_date TIMESTAMPTZ,
    p_stripe_customer_id VARCHAR(255),
    p_stripe_subscription_id VARCHAR(255)
)
RETURNS JSON
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
DECLARE
    v_new_subscription_id UUID;
    v_cancelled_count INT;
BEGIN
    -- Cancel all existing active subscriptions for this user
    WITH cancelled AS (
        UPDATE user_subscriptions
        SET
            status = 'cancelled',
            cancelled_at = NOW(),
            updated_at = NOW()
        WHERE
            user_id = p_user_id
            AND status = 'active'
        RETURNING id
    )
    SELECT COUNT(*) INTO v_cancelled_count FROM cancelled;

    -- Create new subscription
    INSERT INTO user_subscriptions (
        user_id,
        plan_id,
        billing_cycle,
        status,
        started_at,
        expires_at,
        next_billing_date,
        stripe_customer_id,
        stripe_subscription_id,
        payment_method,
        is_trial,
        created_at,
        updated_at
    ) VALUES (
        p_user_id,
        p_plan_id,
        p_billing_cycle,
        'active',
        p_started_at,
        p_expires_at,
        p_next_billing_date,
        p_stripe_customer_id,
        p_stripe_subscription_id,
        'stripe',
        FALSE,
        NOW(),
        NOW()
    )
    RETURNING id INTO v_new_subscription_id;

    -- Return success with details
    RETURN json_build_object(
        'success', TRUE,
        'new_subscription_id', v_new_subscription_id,
        'cancelled_count', v_cancelled_count
    );

EXCEPTION WHEN OTHERS THEN
    -- Rollback happens automatically
    RAISE EXCEPTION 'Transaction failed: %', SQLERRM;
END;
$$;

-- ============================================================================
-- Function: downgrade_to_free_plan_transaction
-- Purpose: Atomically cancel paid subscriptions and create Free plan
-- Returns: JSON with success status and subscription ID
-- ============================================================================
CREATE OR REPLACE FUNCTION downgrade_to_free_plan_transaction(
    p_user_id UUID
)
RETURNS JSON
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
DECLARE
    v_free_plan_id UUID;
    v_new_subscription_id UUID;
    v_cancelled_count INT;
    v_existing_free_count INT;
BEGIN
    -- Get Free plan ID
    SELECT id INTO v_free_plan_id
    FROM subscription_plans
    WHERE name = 'Free'
    LIMIT 1;

    IF v_free_plan_id IS NULL THEN
        RAISE EXCEPTION 'Free plan not found in database';
    END IF;

    -- Check if user already has active Free plan
    SELECT COUNT(*) INTO v_existing_free_count
    FROM user_subscriptions
    WHERE user_id = p_user_id
        AND plan_id = v_free_plan_id
        AND status = 'active';

    IF v_existing_free_count > 0 THEN
        RETURN json_build_object(
            'success', TRUE,
            'message', 'User already has active Free plan',
            'new_subscription_id', NULL,
            'cancelled_count', 0
        );
    END IF;

    -- Cancel ALL active subscriptions
    WITH cancelled AS (
        UPDATE user_subscriptions
        SET
            status = 'cancelled',
            cancelled_at = NOW(),
            updated_at = NOW()
        WHERE
            user_id = p_user_id
            AND status = 'active'
        RETURNING id
    )
    SELECT COUNT(*) INTO v_cancelled_count FROM cancelled;

    -- Create Free plan subscription
    INSERT INTO user_subscriptions (
        user_id,
        plan_id,
        status,
        billing_cycle,
        started_at,
        payment_method,
        is_trial,
        created_at,
        updated_at
    ) VALUES (
        p_user_id,
        v_free_plan_id,
        'active',
        'monthly',
        NOW(),
        'none',
        FALSE,
        NOW(),
        NOW()
    )
    RETURNING id INTO v_new_subscription_id;

    RETURN json_build_object(
        'success', TRUE,
        'message', 'User downgraded to Free plan',
        'new_subscription_id', v_new_subscription_id,
        'cancelled_count', v_cancelled_count
    );

EXCEPTION WHEN OTHERS THEN
    RAISE EXCEPTION 'Transaction failed: %', SQLERRM;
END;
$$;

-- ============================================================================
-- Grant execute permissions (adjust role as needed)
-- ============================================================================
GRANT EXECUTE ON FUNCTION handle_successful_payment_transaction TO service_role;
GRANT EXECUTE ON FUNCTION downgrade_to_free_plan_transaction TO service_role;

-- ============================================================================
-- DEPLOYMENT INSTRUCTIONS
-- ============================================================================
-- 1. Open your Supabase Dashboard
-- 2. Go to SQL Editor
-- 3. Create a new query
-- 4. Paste this entire file
-- 5. Click "Run"
-- 6. Verify functions created successfully
-- ============================================================================
