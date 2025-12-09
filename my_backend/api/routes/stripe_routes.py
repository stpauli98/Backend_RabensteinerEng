"""Stripe payment integration endpoints"""
import os
import logging
import stripe
from flask import Blueprint, request, jsonify, g
from shared.auth.jwt import require_auth
from shared.payments.stripe import (
    get_or_create_stripe_customer,
    get_stripe_price_id,
    handle_successful_payment,
    handle_subscription_updated,
    handle_subscription_deleted,
    handle_payment_failed,
    is_webhook_processed,
    mark_webhook_processed
)

logger = logging.getLogger(__name__)

stripe_bp = Blueprint('stripe', __name__, url_prefix='/api/stripe')

@stripe_bp.route('/create-checkout-session', methods=['POST'])
@require_auth
def create_checkout_session():
    """
    Create Stripe checkout session for subscription

    Request body:
    {
        "plan_id": "uuid",
        "billing_cycle": "monthly" | "yearly"
    }

    Returns:
        200: {"session_id": "cs_...", "url": "https://checkout.stripe.com/..."}
        400: {"error": "message"}
    """
    try:
        data = request.get_json()
        plan_id = data.get('plan_id')
        billing_cycle = data.get('billing_cycle', 'monthly')

        if not plan_id:
            return jsonify({'error': 'plan_id is required'}), 400

        if billing_cycle not in ['monthly', 'yearly']:
            return jsonify({'error': 'Invalid billing_cycle'}), 400

        # Get user info from token (set by @require_auth)
        user_id = g.user_id
        user_email = g.user_email

        # Get or create Stripe customer
        customer_id = get_or_create_stripe_customer(user_id, user_email)

        # Get Stripe price ID
        price_id = get_stripe_price_id(plan_id, billing_cycle)

        # Create checkout session
        frontend_url = os.getenv('FRONTEND_URL', 'http://localhost:3000')

        session = stripe.checkout.Session.create(
            customer=customer_id,
            mode='subscription',
            line_items=[{
                'price': price_id,
                'quantity': 1,
            }],
            success_url=f'{frontend_url}/payment/success?session_id={{CHECKOUT_SESSION_ID}}',
            cancel_url=f'{frontend_url}/payment/cancel',
            metadata={
                'user_id': user_id,
                'plan_id': plan_id,
                'billing_cycle': billing_cycle
            },
            allow_promotion_codes=True,
            billing_address_collection='required',
        )

        logger.info(f"✅ Created checkout session: {session.id} for user {user_id}")

        return jsonify({
            'session_id': session.id,
            'url': session.url
        }), 200

    except Exception as e:
        logger.error(f"❌ Error creating checkout session: {str(e)}")
        # SECURITY FIX: Don't expose internal error details to client
        return jsonify({'error': 'Failed to create checkout session. Please try again or contact support.'}), 500


@stripe_bp.route('/webhook', methods=['POST'])
def stripe_webhook():
    """
    Handle Stripe webhook events with proper idempotency

    CRITICAL FIX: Marks event as processed BEFORE handling to prevent race conditions.
    If two webhooks arrive simultaneously, only one will succeed in marking it.

    Events handled:
    - checkout.session.completed: Subscription activated
    - customer.subscription.updated: Subscription modified
    - customer.subscription.deleted: Subscription cancelled
    - invoice.payment_failed: Payment failure handling
    """
    payload = request.data
    sig_header = request.headers.get('Stripe-Signature')
    webhook_secret = os.getenv('STRIPE_WEBHOOK_SECRET')

    try:
        # Verify webhook signature
        event = stripe.Webhook.construct_event(
            payload, sig_header, webhook_secret
        )

        event_id = event['id']
        event_type = event['type']
        logger.info(f"📥 Received webhook: {event_type} (ID: {event_id})")

        # CRITICAL FIX: Mark event as processed FIRST (before handling)
        # UNIQUE constraint on event_id ensures only one request can succeed
        # This prevents race conditions where two simultaneous requests both pass the check
        try:
            mark_webhook_processed(event_id, event_type)
            logger.info(f"✅ Marked webhook {event_id} as processing")
        except Exception as mark_error:
            # If marking fails, event was already processed (UNIQUE constraint violation)
            # or database error - either way, skip processing
            logger.info(f"ℹ️ Webhook {event_id} already processed or marking failed, skipping: {str(mark_error)}")
            return jsonify({'status': 'success', 'message': 'already processed'}), 200

        # Now handle the event (we know it hasn't been processed yet)
        try:
            if event_type == 'checkout.session.completed':
                session = event['data']['object']
                handle_successful_payment(session)

            elif event_type == 'customer.subscription.updated':
                subscription = event['data']['object']
                handle_subscription_updated(subscription)

            elif event_type == 'customer.subscription.deleted':
                subscription = event['data']['object']
                handle_subscription_deleted(subscription)

            elif event_type == 'invoice.payment_failed':
                invoice = event['data']['object']
                handle_payment_failed(invoice)

            else:
                logger.warning(f"⚠️ Unhandled webhook type: {event_type}")

            logger.info(f"✅ Successfully processed webhook {event_id}")
            return jsonify({'status': 'success'}), 200

        except Exception as handler_error:
            # Handler failed, but event is marked as processed
            # This prevents infinite retries from Stripe
            logger.error(f"❌ Error handling webhook {event_id}: {str(handler_error)}")
            logger.error(f"⚠️ Event {event_id} marked as processed but handling failed!")
            logger.error(f"⚠️ Manual intervention may be required for event {event_id}")

            # Return 200 to prevent Stripe from retrying
            # The event is logged for manual review
            return jsonify({'status': 'error', 'message': 'Processing failed but logged'}), 200

    except ValueError as e:
        # Invalid payload
        logger.error(f"❌ Invalid webhook payload: {str(e)}")
        return jsonify({'error': 'Invalid payload'}), 400

    except stripe.error.SignatureVerificationError as e:
        # Invalid signature
        logger.error(f"❌ Invalid webhook signature: {str(e)}")
        return jsonify({'error': 'Invalid signature'}), 400

    except Exception as e:
        # Unexpected error before processing
        logger.error(f"❌ Unexpected webhook error: {str(e)}")
        return jsonify({'error': 'Internal error'}), 500


@stripe_bp.route('/activate-free-plan', methods=['POST'])
@require_auth
def activate_free_plan():
    """
    Activate Free plan for user (no Stripe checkout needed)

    Returns:
        200: {"success": true, "message": "Free plan activated"}
        400: {"error": "message"}
    """
    try:
        from shared.payments.stripe import downgrade_to_free_plan

        user_id = g.user_id

        # Directly activate Free plan
        downgrade_to_free_plan(user_id)

        logger.info(f"✅ Free plan activated for user {user_id}")

        return jsonify({
            'success': True,
            'message': 'Free plan activated successfully'
        }), 200

    except Exception as e:
        logger.error(f"❌ Error activating free plan: {str(e)}")
        # SECURITY FIX: Don't expose internal error details to client
        return jsonify({'error': 'Failed to activate Free plan. Please try again or contact support.'}), 500


@stripe_bp.route('/customer-portal', methods=['POST'])
@require_auth
def customer_portal():
    """
    Create Stripe Customer Portal session for subscription management

    Returns:
        200: {"url": "https://billing.stripe.com/..."}
        400: {"error": "message"}
    """
    try:
        user_id = g.user_id
        user_email = g.user_email

        logger.info(f"🔍 Creating portal session for user {user_id} ({user_email})")

        # Get Stripe customer ID
        customer_id = get_or_create_stripe_customer(user_id, user_email)
        logger.info(f"✅ Got Stripe customer ID: {customer_id}")

        # Create portal session
        frontend_url = os.getenv('FRONTEND_URL', 'http://localhost:3000')
        logger.info(f"🌐 Using frontend URL: {frontend_url}")

        portal_session = stripe.billing_portal.Session.create(
            customer=customer_id,
            return_url=f'{frontend_url}/pricing',
        )

        logger.info(f"✅ Created portal session for user {user_id}: {portal_session.url}")

        return jsonify({'url': portal_session.url}), 200

    except stripe.error.InvalidRequestError as e:
        logger.error(f"❌ Stripe InvalidRequestError: {str(e)}")
        logger.error(f"⚠️ Hint: Check if Stripe Customer Portal is activated in Dashboard")
        return jsonify({'error': 'Customer portal not configured. Please contact support.'}), 500

    except Exception as e:
        logger.error(f"❌ Error creating portal session: {str(e)}")
        logger.error(f"🔍 Error type: {type(e).__name__}")
        import traceback
        logger.error(f"📊 Traceback: {traceback.format_exc()}")
        # SECURITY FIX: Don't expose internal error details to client
        return jsonify({'error': 'Failed to open customer portal. Please try again or contact support.'}), 500
