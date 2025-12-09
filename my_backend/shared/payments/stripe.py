"""Stripe integration utilities"""
import os
import logging
import stripe
from shared.database.client import get_supabase_admin_client

logger = logging.getLogger(__name__)

# Initialize Stripe
stripe.api_key = os.getenv('STRIPE_SECRET_KEY')


def is_webhook_processed(event_id: str) -> bool:
    """
    Check if webhook event has already been processed

    Args:
        event_id: Stripe event ID (evt_...)

    Returns:
        bool: True if event was already processed, False otherwise
    """
    try:
        supabase = get_supabase_admin_client()

        result = supabase.table('processed_webhook_events') \
            .select('id') \
            .eq('event_id', event_id) \
            .execute()

        return result.data and len(result.data) > 0

    except Exception as e:
        logger.error(f"Error checking webhook processed status: {str(e)}")
        # If check fails, assume not processed to avoid blocking
        return False


def mark_webhook_processed(event_id: str, event_type: str) -> None:
    """
    Mark webhook event as processed

    Args:
        event_id: Stripe event ID (evt_...)
        event_type: Type of event (checkout.session.completed, etc)
    """
    try:
        supabase = get_supabase_admin_client()
        from datetime import datetime, timezone

        supabase.table('processed_webhook_events').insert({
            'event_id': event_id,
            'event_type': event_type,
            'processed_at': datetime.now(timezone.utc).isoformat()
        }).execute()

        logger.info(f"Marked webhook {event_id} as processed")

    except Exception as e:
        # Log error but don't raise - webhook was processed successfully
        logger.error(f"Error marking webhook as processed: {str(e)}")

def get_or_create_stripe_customer(user_id: str, email: str) -> str:
    """
    Get existing Stripe customer or create new one

    Args:
        user_id: Supabase user ID
        email: User email

    Returns:
        str: Stripe customer ID
    """
    try:
        supabase = get_supabase_admin_client()

        # Check if user already has stripe_customer_id
        response = supabase.table('user_subscriptions') \
            .select('stripe_customer_id') \
            .eq('user_id', user_id) \
            .execute()

        # Get first result if exists
        if response.data and len(response.data) > 0 and response.data[0].get('stripe_customer_id'):
            logger.info(f"Found existing Stripe customer: {response.data[0]['stripe_customer_id']}")
            return response.data[0]['stripe_customer_id']

        # Create new Stripe customer
        customer = stripe.Customer.create(
            email=email,
            metadata={'supabase_user_id': user_id}
        )

        logger.info(f"Created new Stripe customer: {customer.id}")

        # Update stripe_customer_id - check if record exists
        existing = supabase.table('user_subscriptions') \
            .select('id') \
            .eq('user_id', user_id) \
            .execute()

        if existing.data and len(existing.data) > 0:
            # Update existing record
            supabase.table('user_subscriptions') \
                .update({'stripe_customer_id': customer.id}) \
                .eq('user_id', user_id) \
                .execute()
        else:
            # Get Free plan ID
            free_plan = supabase.table('subscription_plans') \
                .select('id') \
                .eq('name', 'Free') \
                .single() \
                .execute()

            # Create new record with Free plan
            from datetime import datetime, timezone
            supabase.table('user_subscriptions') \
                .insert({
                    'user_id': user_id,
                    'plan_id': free_plan.data['id'],
                    'stripe_customer_id': customer.id,
                    'status': 'active',
                    'billing_cycle': 'monthly',
                    'started_at': datetime.now(timezone.utc).isoformat()
                }) \
                .execute()

        return customer.id

    except Exception as e:
        logger.error(f"Error getting/creating Stripe customer: {str(e)}")
        raise


def get_stripe_price_id(plan_id: str, billing_cycle: str) -> str:
    """
    Get Stripe Price ID from database based on plan and billing cycle

    CRITICAL FIX: Reads from database instead of hardcoded mapping.
    This allows dynamic plan management without code changes.

    Prerequisites: subscription_plans table must have stripe_price_id_monthly
    and stripe_price_id_yearly columns populated.

    Args:
        plan_id: Subscription plan UUID from database
        billing_cycle: 'monthly' or 'yearly'

    Returns:
        str: Stripe price ID

    Raises:
        ValueError: If plan not found or price ID not configured
    """
    supabase = get_supabase_admin_client()

    # Get plan with Stripe price IDs
    plan = supabase.table('subscription_plans') \
        .select('name, stripe_price_id_monthly, stripe_price_id_yearly') \
        .eq('id', plan_id) \
        .single() \
        .execute()

    if not plan.data:
        raise ValueError(f"Plan {plan_id} not found in database")

    # Get appropriate price ID based on billing cycle
    if billing_cycle == 'monthly':
        price_id = plan.data.get('stripe_price_id_monthly')
    elif billing_cycle == 'yearly':
        price_id = plan.data.get('stripe_price_id_yearly')
    else:
        raise ValueError(f"Invalid billing cycle: {billing_cycle}")

    if not price_id:
        plan_name = plan.data.get('name', 'Unknown')
        raise ValueError(
            f"No Stripe price ID configured for {plan_name} plan with {billing_cycle} billing. "
            f"Please update subscription_plans table with stripe_price_id_{billing_cycle} value."
        )

    return price_id


def handle_successful_payment(session: stripe.checkout.Session) -> None:
    """
    Handle successful Stripe checkout session

    CRITICAL FIX: Uses PostgreSQL transaction function to atomically:
    1. Cancel old subscriptions
    2. Create new subscription
    This prevents data corruption if insert fails after update.

    Prerequisites: Run sql/subscription_transactions.sql in Supabase SQL Editor

    Args:
        session: Stripe checkout session object
    """
    try:
        supabase = get_supabase_admin_client()

        # Extract metadata
        user_id = session.metadata.get('user_id')
        plan_id = session.metadata.get('plan_id')
        billing_cycle = session.metadata.get('billing_cycle')

        if not all([user_id, plan_id, billing_cycle]):
            raise ValueError("Missing required metadata in session")

        # Get subscription details from Stripe
        subscription = stripe.Subscription.retrieve(session.subscription)

        # Calculate dates
        from datetime import datetime, timezone
        started_at = datetime.fromtimestamp(subscription.current_period_start, tz=timezone.utc)
        expires_at = datetime.fromtimestamp(subscription.current_period_end, tz=timezone.utc)
        next_billing_date = expires_at

        # CRITICAL FIX: Use atomic database transaction via PostgreSQL function
        # This ensures either BOTH operations succeed or BOTH fail (no partial state)
        try:
            result = supabase.rpc('handle_successful_payment_transaction', {
                'p_user_id': user_id,
                'p_plan_id': plan_id,
                'p_billing_cycle': billing_cycle,
                'p_started_at': started_at.isoformat(),
                'p_expires_at': expires_at.isoformat(),
                'p_next_billing_date': next_billing_date.isoformat(),
                'p_stripe_customer_id': session.customer,
                'p_stripe_subscription_id': session.subscription
            }).execute()

            if result.data:
                cancelled_count = result.data.get('cancelled_count', 0)
                logger.info(f"Subscription activated for user {user_id}, plan {plan_id} (cancelled {cancelled_count} old subscription(s))")
            else:
                logger.info(f"Subscription activated for user {user_id}, plan {plan_id}")

        except Exception as rpc_error:
            # If RPC function doesn't exist, fall back to non-transactional approach
            # This happens if sql/subscription_transactions.sql was not deployed
            logger.warning(f"RPC function not found, using fallback non-transactional approach: {str(rpc_error)}")
            logger.warning("DEPLOY sql/subscription_transactions.sql for atomic transactions!")

            # Fallback: Non-transactional (RISKY but better than nothing)
            supabase.table('user_subscriptions') \
                .update({
                    'status': 'cancelled',
                    'cancelled_at': datetime.now(timezone.utc).isoformat()
                }) \
                .eq('user_id', user_id) \
                .eq('status', 'active') \
                .execute()

            supabase.table('user_subscriptions').insert({
                'user_id': user_id,
                'plan_id': plan_id,
                'billing_cycle': billing_cycle,
                'status': 'active',
                'started_at': started_at.isoformat(),
                'expires_at': expires_at.isoformat(),
                'next_billing_date': next_billing_date.isoformat(),
                'stripe_customer_id': session.customer,
                'stripe_subscription_id': session.subscription,
                'payment_method': 'stripe',
                'is_trial': False,
                'updated_at': datetime.now(timezone.utc).isoformat()
            }).execute()

            logger.info(f"Subscription activated for user {user_id}, plan {plan_id} (fallback mode)")

    except Exception as e:
        logger.error(f"Error handling successful payment: {str(e)}")
        raise


def get_plan_id_from_stripe_price(stripe_price_id: str):
    """
    Map Stripe Price ID back to plan_id from database

    Helper function for handling plan upgrades/downgrades in webhooks.

    Args:
        stripe_price_id: Stripe Price ID (e.g., price_xxx...)

    Returns:
        Plan UUID if found, None otherwise
    """
    try:
        supabase = get_supabase_admin_client()

        # Search for plan with matching price ID (monthly or yearly)
        result = supabase.table('subscription_plans') \
            .select('id') \
            .or_(f"stripe_price_id_monthly.eq.{stripe_price_id},stripe_price_id_yearly.eq.{stripe_price_id}") \
            .execute()

        if result.data and len(result.data) > 0:
            return result.data[0]['id']

        return None

    except Exception as e:
        logger.error(f"Error mapping stripe price to plan: {str(e)}")
        return None


def handle_subscription_updated(subscription: stripe.Subscription) -> None:
    """
    Handle subscription update events from Stripe webhook

    CRITICAL FIX: Now handles plan upgrades/downgrades by checking if
    the Stripe price ID changed and updating plan_id accordingly.
    """
    try:
        supabase = get_supabase_admin_client()

        # Check if subscription exists in database
        existing = supabase.table('user_subscriptions') \
            .select('id, plan_id') \
            .eq('stripe_subscription_id', subscription.id) \
            .execute()

        if not existing.data or len(existing.data) == 0:
            logger.warning(f"Subscription {subscription.id} not found in database, skipping update")
            return

        from datetime import datetime, timezone

        # Build update data - only include fields that are available
        update_data = {
            'status': subscription.status,
            'updated_at': datetime.now(timezone.utc).isoformat()
        }

        # Only update period dates if they exist in the subscription
        if hasattr(subscription, 'current_period_end') and subscription.current_period_end:
            expires_at = datetime.fromtimestamp(subscription.current_period_end, tz=timezone.utc)
            update_data['expires_at'] = expires_at.isoformat()
            update_data['next_billing_date'] = expires_at.isoformat()

        # CRITICAL FIX: Check if plan changed (upgrade/downgrade)
        # Extract price ID from subscription items
        if hasattr(subscription, 'items') and subscription.items and len(subscription.items.data) > 0:
            stripe_price_id = subscription.items.data[0].price.id

            # Map Stripe price ID to our plan_id
            new_plan_id = get_plan_id_from_stripe_price(stripe_price_id)

            if new_plan_id:
                current_plan_id = existing.data[0].get('plan_id')

                if current_plan_id != new_plan_id:
                    # Plan changed! Update it
                    update_data['plan_id'] = new_plan_id
                    logger.info(f"Plan changed for subscription {subscription.id}: {current_plan_id} -> {new_plan_id}")

        # Update existing subscription
        supabase.table('user_subscriptions') \
            .update(update_data) \
            .eq('stripe_subscription_id', subscription.id) \
            .execute()

        logger.info(f"Subscription updated: {subscription.id}")

    except Exception as e:
        logger.error(f"Error handling subscription update: {str(e)}")
        raise


def handle_subscription_deleted(subscription: stripe.Subscription) -> None:
    """Handle subscription cancellation from Stripe webhook"""
    try:
        supabase = get_supabase_admin_client()

        from datetime import datetime, timezone

        # Cancel the subscription in database
        result = supabase.table('user_subscriptions') \
            .update({
                'status': 'cancelled',
                'cancelled_at': datetime.now(timezone.utc).isoformat(),
                'updated_at': datetime.now(timezone.utc).isoformat()
            }) \
            .eq('stripe_subscription_id', subscription.id) \
            .execute()

        logger.info(f"Subscription cancelled: {subscription.id}")

        # Downgrade to Free plan
        if result.data and len(result.data) > 0:
            user_id = result.data[0].get('user_id')
            if user_id:
                downgrade_to_free_plan(user_id)

    except Exception as e:
        logger.error(f"Error handling subscription deletion: {str(e)}")
        raise


def downgrade_to_free_plan(user_id: str) -> None:
    """
    Downgrade user to Free plan after subscription cancellation

    CRITICAL FIX: Uses PostgreSQL transaction to atomically:
    1. Cancel all active paid subscriptions
    2. Create Free plan subscription
    This prevents duplicate active subscriptions where user keeps paid features.

    Prerequisites: Run sql/subscription_transactions.sql in Supabase SQL Editor

    Args:
        user_id: Supabase user ID
    """
    try:
        supabase = get_supabase_admin_client()

        # CRITICAL FIX: Use atomic database transaction via PostgreSQL function
        try:
            result = supabase.rpc('downgrade_to_free_plan_transaction', {
                'p_user_id': user_id
            }).execute()

            if result.data:
                message = result.data.get('message', '')
                cancelled_count = result.data.get('cancelled_count', 0)
                logger.info(f"{message} (cancelled {cancelled_count} paid subscription(s))")
            else:
                logger.info(f"User {user_id} downgraded to Free plan")

        except Exception as rpc_error:
            # If RPC function doesn't exist, fall back to non-transactional approach
            logger.warning(f"RPC function not found, using fallback approach: {str(rpc_error)}")
            logger.warning("DEPLOY sql/subscription_transactions.sql for atomic transactions!")

            # Fallback: Non-transactional (RISKY but better than nothing)
            from datetime import datetime, timezone

            free_plan = supabase.table('subscription_plans') \
                .select('id') \
                .eq('name', 'Free') \
                .single() \
                .execute()

            if not free_plan.data:
                logger.error(f"Free plan not found in database")
                return

            existing_free = supabase.table('user_subscriptions') \
                .select('id') \
                .eq('user_id', user_id) \
                .eq('plan_id', free_plan.data['id']) \
                .eq('status', 'active') \
                .execute()

            if existing_free.data and len(existing_free.data) > 0:
                logger.info(f"User {user_id} already has active Free plan")
                return

            cancelled_subs = supabase.table('user_subscriptions') \
                .update({
                    'status': 'cancelled',
                    'cancelled_at': datetime.now(timezone.utc).isoformat(),
                    'updated_at': datetime.now(timezone.utc).isoformat()
                }) \
                .eq('user_id', user_id) \
                .eq('status', 'active') \
                .execute()

            if cancelled_subs.data and len(cancelled_subs.data) > 0:
                logger.info(f"Cancelled {len(cancelled_subs.data)} active subscription(s) for user {user_id}")

            supabase.table('user_subscriptions').insert({
                'user_id': user_id,
                'plan_id': free_plan.data['id'],
                'status': 'active',
                'billing_cycle': 'monthly',
                'started_at': datetime.now(timezone.utc).isoformat(),
                'payment_method': 'none',
                'is_trial': False
            }).execute()

            logger.info(f"User {user_id} downgraded to Free plan (fallback mode)")

    except Exception as e:
        logger.error(f"Error downgrading to Free plan: {str(e)}")
        raise


def handle_payment_failed(invoice: stripe.Invoice) -> None:
    """
    Handle failed payment from Stripe webhook

    Args:
        invoice: Stripe invoice object
    """
    try:
        supabase = get_supabase_admin_client()
        from datetime import datetime, timezone

        # Get subscription ID from invoice
        subscription_id = invoice.subscription

        if not subscription_id:
            logger.warning(f"Invoice {invoice.id} has no subscription")
            return

        # Update subscription status to past_due
        supabase.table('user_subscriptions') \
            .update({
                'status': 'past_due',
                'updated_at': datetime.now(timezone.utc).isoformat()
            }) \
            .eq('stripe_subscription_id', subscription_id) \
            .execute()

        logger.warning(f"Payment failed for subscription {subscription_id}, status set to past_due")

    except Exception as e:
        logger.error(f"Error handling payment failure: {str(e)}")
        raise
