"""
Stripe Payment API Routes
Handles Stripe checkout, webhooks, and customer portal
"""
import os
import logging
import stripe
from flask import Blueprint, request, jsonify, g
from shared.auth.jwt import require_auth
from shared.database.client import get_supabase_admin_client
from shared.payments.stripe import (
    get_or_create_stripe_customer,
    get_stripe_price_id,
    handle_successful_payment,
    handle_subscription_updated,
    handle_subscription_deleted,
    handle_payment_failed,
    handle_payment_succeeded,
    handle_charge_refunded,
    is_webhook_processed,
    mark_webhook_processed,
)
from shared.responses.errors import error_response as _err
from core.rate_limits import limiter, training_limit_string

logger = logging.getLogger(__name__)

bp = Blueprint('stripe', __name__)


@bp.route('/create-checkout-session', methods=['POST'])
@limiter.limit(training_limit_string)
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

        # FIX-6: Block duplicate-subscription creation. Without this, a user
        # already on PREMIUM clicking BASIC would create a parallel paid
        # subscription instead of switching plans — double-billing in prod.
        # Plan switches MUST go through the Stripe customer portal so
        # Stripe handles proration and cancels the old line item.
        sb = get_supabase_admin_client()
        existing = sb.table('user_subscriptions') \
            .select('id, plan_id, status, expires_at') \
            .eq('user_id', user_id) \
            .in_('status', ['active', 'trialing']) \
            .execute()
        if existing.data:
            logger.info(
                f"Rejecting checkout for user {user_id}: already has active subscription"
            )
            return _err(
                'SUBSCRIPTION_ALREADY_ACTIVE',
                'You already have an active subscription. '
                'Use the billing portal to change plans.',
                409,
                suggestion='Call POST /api/stripe/customer-portal to manage your plan.',
            )

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

        logger.info(f"Created checkout session: {session.id} for user {user_id}")

        return jsonify({
            'session_id': session.id,
            'url': session.url
        }), 200

    except Exception as e:
        logger.error(f"Error creating checkout session: {str(e)}")
        return jsonify({'error': 'Failed to create checkout session. Please try again or contact support.'}), 500


@bp.route('/verify-session', methods=['POST'])
@limiter.limit(training_limit_string)
@require_auth
def verify_session():
    """
    Verify a Stripe checkout session belongs to the caller and was paid.

    FIX-6: Closes a trivial spoof on /payment/success — previously the FE
    rendered "Payment Successful!" for ANY session_id query param without
    ever asking the BE. Now the FE must POST the session_id here and
    render success only if this endpoint returns 200.

    Request body:
        {"session_id": "cs_test_..." | "cs_live_..."}

    Returns:
        200: {"verified": true, "plan_name", "amount", "customer_id"}
        400: BAD_REQUEST — missing or malformed session_id
        402: PAYMENT_NOT_COMPLETED — Stripe session not paid yet
        403: SESSION_FORBIDDEN — session does not belong to caller
        404: SESSION_NOT_FOUND — Stripe returned no such session
        500: VERIFICATION_FAILED — unexpected Stripe SDK / internal error
    """
    try:
        data = request.get_json(silent=True) or {}
        session_id = data.get('session_id')

        if not session_id or not isinstance(session_id, str):
            return _err('BAD_REQUEST', 'session_id is required', 400)

        # Format guard. Stripe checkout sessions start with cs_test_ or
        # cs_live_; reject anything else before round-tripping to Stripe.
        if not (session_id.startswith('cs_test_') or session_id.startswith('cs_live_')):
            return _err('BAD_REQUEST', 'Invalid session_id format', 400)

        # Fetch the session from Stripe, expanding line_items so we can return
        # the plan name without a second round-trip. expand=['line_items'] adds
        # one DB-equivalent hop on Stripe's side; cheap enough to be worth the
        # nicer FE UX.
        try:
            session = stripe.checkout.Session.retrieve(
                session_id, expand=['line_items']
            )
        except stripe.error.InvalidRequestError:
            # Stripe returns 404 (wrapped as InvalidRequestError) when the
            # session id doesn't exist or is from a different account.
            logger.warning(
                f"verify-session: Stripe session not found for user {g.user_id}"
            )
            return _err('SESSION_NOT_FOUND', 'Session not found', 404)
        except stripe.error.StripeError as stripe_err:
            logger.error(f"Stripe error retrieving session: {str(stripe_err)}")
            return _err(
                'VERIFICATION_FAILED',
                'Unable to verify payment right now. Please contact support if '
                'your subscription does not activate within a few minutes.',
                500,
            )

        # Ownership check. Prefer metadata.user_id (set when we created the
        # session) since it's tamper-evident from our side. Fall back to
        # matching the Stripe customer to the user's stripe_customer_id if
        # metadata is missing on legacy sessions.
        metadata = session.get('metadata') or {}
        session_user_id = metadata.get('user_id')

        owned = False
        if session_user_id and session_user_id == g.user_id:
            owned = True
        else:
            customer_id = session.get('customer')
            if customer_id:
                sb = get_supabase_admin_client()
                cust_row = sb.table('user_subscriptions') \
                    .select('user_id') \
                    .eq('stripe_customer_id', customer_id) \
                    .eq('user_id', g.user_id) \
                    .limit(1) \
                    .execute()
                if cust_row.data:
                    owned = True

        if not owned:
            logger.warning(
                f"verify-session: user {g.user_id} attempted to verify "
                f"session belonging to {session_user_id or session.get('customer')}"
            )
            return _err(
                'SESSION_FORBIDDEN',
                'This session does not belong to your account',
                403,
            )

        # Payment status check.
        payment_status = session.get('payment_status')
        if payment_status != 'paid':
            return _err(
                'PAYMENT_NOT_COMPLETED',
                'Payment has not yet been completed for this session',
                402,
            )

        # Build the safe response. Resolve plan name from the first line item
        # (we always create single-line-item checkout sessions in
        # create_checkout_session). Amounts come back in the smallest currency
        # unit (cents) from Stripe; surface in major units for FE display.
        plan_name = None
        amount = None
        currency = None
        line_items = session.get('line_items')
        if line_items and line_items.get('data'):
            first_item = line_items['data'][0]
            price_obj = first_item.get('price') or {}
            product_id = price_obj.get('product')
            amount = first_item.get('amount_total')
            currency = first_item.get('currency')
            # Plan name comes from our DB (which already maps stripe price ID
            # to plan name); avoids a second Stripe round-trip for the product.
            stripe_price_id = price_obj.get('id')
            if stripe_price_id:
                try:
                    sb = get_supabase_admin_client()
                    plan_row = sb.table('subscription_plans') \
                        .select('name') \
                        .or_(
                            f"stripe_price_id_monthly.eq.{stripe_price_id},"
                            f"stripe_price_id_yearly.eq.{stripe_price_id}"
                        ) \
                        .limit(1) \
                        .execute()
                    if plan_row.data:
                        plan_name = plan_row.data[0].get('name')
                except Exception as plan_lookup_err:
                    # Plan name is nice-to-have, not load-bearing for verification.
                    logger.warning(
                        f"verify-session: plan-name lookup failed: {str(plan_lookup_err)}"
                    )
                    plan_name = None
            if not plan_name:
                # Fallback: surface Stripe product id so FE can at least show
                # something meaningful.
                plan_name = product_id

        amount_major = None
        if amount is not None:
            amount_major = amount / 100.0

        return jsonify({
            'verified': True,
            'plan_name': plan_name,
            'amount': amount_major,
            'currency': currency,
            'customer_id': session.get('customer'),
        }), 200

    except Exception as e:
        logger.error(f"Unexpected error in verify-session: {str(e)}")
        return _err(
            'VERIFICATION_FAILED',
            'Unable to verify payment right now. Please contact support if '
            'your subscription does not activate within a few minutes.',
            500,
        )


@bp.route('/webhook', methods=['POST'])
def stripe_webhook():
    """
    Handle Stripe webhook events with proper idempotency

    Events handled:
    - checkout.session.completed: Subscription activated
    - customer.subscription.updated: Subscription modified
    - customer.subscription.deleted: Subscription cancelled
    - invoice.payment_succeeded: Payment recovered (past_due → active)
    - invoice.payment_failed: Payment failure handling
    - charge.refunded: Full refund revokes paid access
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
        logger.info(f"Received webhook: {event_type} (ID: {event_id})")

        # Idempotency short-circuit: if we already finished processing this
        # event, return 200 immediately and do nothing.
        if is_webhook_processed(event_id):
            logger.info(f"Webhook {event_id} already processed; skipping")
            return jsonify({'status': 'success', 'message': 'already processed'}), 200

        # Run the handler. We mark the event processed only AFTER the handler
        # succeeds — if it raises we return 500 so Stripe retries on its
        # exponential-backoff schedule. Handlers are idempotent (UNIQUE INDEX
        # on user_subscriptions.stripe_subscription_id + IF EXISTS guard in
        # handle_successful_payment_transaction RPC) so a retry is safe.
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

            elif event_type == 'invoice.payment_succeeded':
                invoice = event['data']['object']
                handle_payment_succeeded(invoice)

            elif event_type == 'invoice.payment_failed':
                invoice = event['data']['object']
                handle_payment_failed(invoice)

            elif event_type == 'charge.refunded':
                charge = event['data']['object']
                handle_charge_refunded(charge)

            else:
                logger.warning(f"Unhandled webhook type: {event_type}")

        except Exception as handler_error:
            logger.error(
                f"Error handling webhook {event_id}: {handler_error}; "
                f"returning 500 so Stripe will retry"
            )
            return jsonify({'status': 'error', 'message': 'Processing failed; will retry'}), 500

        # Handler succeeded → record it. The idempotency short-circuit at
        # the top of this function blocks duplicate event_ids before they
        # reach this point, so the realistic remaining failure mode here
        # is a transient DB error. We still return 200 in that case
        # because the user-visible work is done.
        try:
            mark_webhook_processed(event_id, event_type)
        except Exception as mark_error:
            logger.warning(
                f"Webhook {event_id} handled but failed to mark processed: {mark_error}"
            )

        logger.info(f"Successfully processed webhook {event_id}")
        return jsonify({'status': 'success'}), 200

    except ValueError as e:
        logger.error(f"Invalid webhook payload: {str(e)}")
        return jsonify({'error': 'Invalid payload'}), 400

    except stripe.error.SignatureVerificationError as e:
        logger.error(f"Invalid webhook signature: {str(e)}")
        return jsonify({'error': 'Invalid signature'}), 400

    except Exception as e:
        logger.error(f"Unexpected webhook error: {str(e)}")
        return jsonify({'error': 'Internal error'}), 500


@bp.route('/activate-free-plan', methods=['POST'])
@require_auth
def activate_free_plan():
    """
    Free plan retired. Endpoint kept for legacy clients but returns 410 Gone.
    Users must purchase a paid plan via the standard checkout flow.
    """
    return jsonify({
        'error': 'Free plan has been retired. Please choose a paid plan.',
        'redirect_to': '/pricing',
    }), 410


@bp.route('/customer-portal', methods=['POST'])
@limiter.limit(training_limit_string)
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

        logger.info(f"Creating portal session for user {user_id}")

        # Get Stripe customer ID
        customer_id = get_or_create_stripe_customer(user_id, user_email)
        logger.info(f"Got Stripe customer ID: {customer_id}")

        # Create portal session
        frontend_url = os.getenv('FRONTEND_URL', 'http://localhost:3000')
        logger.info(f"Using frontend URL: {frontend_url}")

        portal_session = stripe.billing_portal.Session.create(
            customer=customer_id,
            return_url=f'{frontend_url}/pricing',
        )

        logger.info(f"Created portal session for user {user_id}: {portal_session.url}")

        return jsonify({'url': portal_session.url}), 200

    except stripe.error.InvalidRequestError as e:
        logger.error(f"Stripe InvalidRequestError: {str(e)}")
        logger.error(f"Hint: Check if Stripe Customer Portal is activated in Dashboard")
        return jsonify({'error': 'Customer portal not configured. Please contact support.'}), 500

    except Exception as e:
        logger.error(f"Error creating portal session: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({'error': 'Failed to open customer portal. Please try again or contact support.'}), 500
