"""Subscription validation middleware for plan limits"""
import logging
from functools import wraps
from datetime import datetime, timezone
from typing import Optional
from flask import request, jsonify, g
from shared.database.client import get_supabase_user_client

logger = logging.getLogger(__name__)

def get_user_subscription(user_id: str, access_token: str) -> Optional[dict]:
    """
    Get active subscription for user

    Args:
        user_id: User ID from auth
        access_token: User's JWT access token

    Returns:
        dict: Subscription data with plan details, or None if not found
    """
    try:
        supabase = get_supabase_user_client(access_token)

        response = supabase.table('user_subscriptions') \
            .select('*, subscription_plans(*)') \
            .eq('user_id', user_id) \
            .eq('status', 'active') \
            .gt('expires_at', datetime.now(timezone.utc).isoformat()) \
            .order('created_at', desc=True) \
            .limit(1) \
            .execute()

        if response.data:
            return response.data[0]

        logger.warning(f"No active subscription found for user {user_id}")
        return None

    except Exception as e:
        logger.error(f"Error fetching subscription: {str(e)}")
        return None


def get_user_usage(user_id: str, access_token: str) -> dict:
    """
    Get current usage for user in current billing period

    Args:
        user_id: User ID from auth
        access_token: User's JWT access token

    Returns:
        dict: Usage data with counts for uploads, processing, storage
    """
    try:
        supabase = get_supabase_user_client(access_token)

        try:
            period_resp = supabase.rpc('get_current_period_start').execute()
            period_start_iso = str(period_resp.data) if period_resp and period_resp.data else None
        except Exception as e:
            logger.error(f"get_current_period_start RPC failed: {e}")
            period_start_iso = None
        if not period_start_iso:
            period_start_iso = datetime.now(timezone.utc).date().replace(day=1).isoformat()

        response = supabase.table('usage_tracking') \
            .select('*') \
            .eq('user_id', user_id) \
            .eq('period_start', period_start_iso) \
            .limit(1) \
            .execute()

        if response and response.data and len(response.data) > 0:
            return response.data[0]

        logger.debug(f"No usage record for user {user_id} in current period")
        return {
            'uploads_count': 0,
            'processing_jobs_count': 0,
            'training_runs_count': 0,
            'storage_used_gb': 0
        }

    except Exception as e:
        logger.error(f"Error fetching usage: {str(e)}")
        return {
            'uploads_count': 0,
            'processing_jobs_count': 0,
            'training_runs_count': 0,
            'storage_used_gb': 0
        }


def require_subscription(f):
    """
    Decorator to require active subscription

    Must be used AFTER @require_auth decorator

    Usage:
        @require_auth
        @require_subscription
        def protected_route():
            subscription = g.subscription
            plan = g.plan
            return jsonify({'plan': plan['name']})
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not hasattr(g, 'user_id'):
            logger.error("require_subscription used without require_auth")
            return jsonify({'error': 'Authentication required'}), 401

        # API key auth already checks subscription in the middleware — skip here
        if getattr(g, 'auth_method', None) == 'api_key':
            return f(*args, **kwargs)

        subscription = get_user_subscription(g.user_id, g.access_token)

        if not subscription:
            logger.warning(f"No active subscription for user {g.user_id}")
            return jsonify({
                'error': 'No active subscription',
                'message': 'Please subscribe to a plan to access this feature'
            }), 403

        g.subscription = subscription
        g.plan = subscription.get('subscription_plans', {})



        return f(*args, **kwargs)

    return decorated_function



def check_processing_limit(f):
    """
    Decorator to check processing limit before processing.

    Hybrid gating:
      - If the user's plan has `total_compute_hours > 0`, gate by aggregated
        compute seconds (RPC `get_total_compute_seconds`). On exceed, return
        403 with `error_code='compute_hours_exhausted'` so the frontend can
        route the user to /pricing.
      - Otherwise, fall back to the legacy job-count gate against
        `processing_jobs_count` / `max_processing_jobs_per_month`.

    Must be used AFTER @require_auth and @require_subscription decorators.

    Usage:
        @require_auth
        @require_subscription
        @check_processing_limit
        def process_route():
            return jsonify({'message': 'Processing successful'})
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not hasattr(g, 'user_id') or not hasattr(g, 'plan'):
            logger.error("check_processing_limit used without require_auth and require_subscription")
            return jsonify({'error': 'Authentication and subscription required'}), 401

        plan = g.plan or {}
        total_compute_hours = plan.get('total_compute_hours') or 0

        if total_compute_hours > 0:
            # Compute-hours model: aggregate seconds via RPC.
            try:
                supabase = get_supabase_user_client(g.access_token)
                rpc_result = supabase.rpc(
                    'get_total_compute_seconds',
                    {'p_user_id': g.user_id},
                ).execute()
                used_seconds = int(rpc_result.data or 0)
            except Exception as e:
                logger.error(f"Error reading compute seconds for {g.user_id}: {e}")
                # Fail-open on RPC errors so transient infra issues don't lock
                # paying users out. Operation proceeds and the user is gated
                # only on confirmed exceedance.
                return f(*args, **kwargs)

            limit_seconds = total_compute_hours * 3600
            used_hours = used_seconds / 3600

            if used_seconds >= limit_seconds:
                logger.warning(
                    f"Compute hours exhausted for user {g.user_email}: "
                    f"{used_hours:.2f}/{total_compute_hours}h"
                )
                return jsonify({
                    'error': (
                        f"Compute hours exhausted. Your {plan.get('name', 'current')} plan "
                        f"includes {total_compute_hours}h. You've used {used_hours:.2f}h. "
                        f"Upgrade your plan or wait for the next billing period."
                    ),
                    'error_code': 'compute_hours_exhausted',
                    'used_hours': round(used_hours, 2),
                    'limit_hours': total_compute_hours,
                    'plan': plan.get('name'),
                    'redirect_to': '/pricing',
                }), 403

            # Surface the same shape downstream code expected.
            g.usage = getattr(g, 'usage', {}) or {}
            g.usage['total_compute_seconds'] = used_seconds
            g.compute_seconds_remaining = limit_seconds - used_seconds
            return f(*args, **kwargs)

        # Legacy count-based gate.
        usage = g.usage or get_user_usage(g.user_id, g.access_token)
        processing_used = usage.get('processing_count', usage.get('processing_jobs_count', 0))
        processing_limit = plan.get('max_processing_jobs_per_month', 0)

        if processing_used >= processing_limit:
            logger.warning(f"Processing limit reached for user {g.user_email}: {processing_used}/{processing_limit}")
            return jsonify({
                'error': 'Processing limit reached',
                'message': f'You have reached your monthly processing limit of {processing_limit}',
                'current_usage': processing_used,
                'limit': processing_limit,
                'plan': plan.get('name'),
            }), 403

        g.usage = usage
        g.processing_remaining = processing_limit - processing_used
        return f(*args, **kwargs)

    return decorated_function


def check_storage_limit(f):
    """
    Decorator to check storage limit before upload

    Must be used AFTER @require_auth and @require_subscription decorators

    Usage:
        @require_auth
        @require_subscription
        @check_storage_limit
        def upload_large_file():
            # Upload will only proceed if under storage limit
            return jsonify({'message': 'Upload successful'})
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not hasattr(g, 'user_id') or not hasattr(g, 'plan'):
            logger.error("check_storage_limit used without require_auth and require_subscription")
            return jsonify({'error': 'Authentication and subscription required'}), 401

        usage = get_user_usage(g.user_id, g.access_token)
        storage_used_mb = usage.get('storage_used_mb', 0)
        storage_limit_gb = g.plan.get('max_storage_gb', 0)
        storage_limit_mb = storage_limit_gb * 1024

        if storage_used_mb >= storage_limit_mb:
            logger.warning(f"Storage limit reached for user {g.user_email}: {storage_used_mb}/{storage_limit_mb} MB")
            return jsonify({
                'error': 'Storage limit reached',
                'message': f'You have reached your storage limit of {storage_limit_mb} MB',
                'current_usage_mb': storage_used_mb,
                'limit_mb': storage_limit_mb,
                'plan': g.plan.get('name')
            }), 403

        g.usage = usage
        g.storage_remaining_mb = storage_limit_mb - storage_used_mb



        return f(*args, **kwargs)

    return decorated_function


def check_training_limit(f):
    """
    Decorator to check training limit before starting training

    Must be used AFTER @require_auth and @require_subscription decorators

    Usage:
        @require_auth
        @require_subscription
        @check_training_limit
        def train_models_route():
            # Training will only proceed if user has permission and quota
            return jsonify({'message': 'Training started'})
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not hasattr(g, 'user_id') or not hasattr(g, 'plan'):
            logger.error("check_training_limit used without require_auth and require_subscription")
            return jsonify({'error': 'Authentication and subscription required'}), 401

        can_use_training = g.plan.get('can_use_training', False)

        if not can_use_training:
            logger.warning(f"Training not available for user {g.user_email}'s plan")
            return jsonify({
                'error': 'Training not available',
                'message': 'Training is not available in your plan. Upgrade to STANDARD or PREMIUM to unlock model training.',
                'plan': g.plan.get('name')
            }), 403

        usage = get_user_usage(g.user_id, g.access_token)
        training_used = usage.get('training_runs_count', 0)
        training_limit = g.plan.get('max_training_runs_per_month', 0)

        if training_limit == -1:
            logger.debug(f"Unlimited training for {g.user_email}")
            g.usage = usage
            return f(*args, **kwargs)

        if training_used >= training_limit:
            logger.warning(f"Training limit reached for user {g.user_email}: {training_used}/{training_limit}")
            return jsonify({
                'error': 'Training limit reached',
                'message': f'You have reached your monthly training limit of {training_limit} runs',
                'current_usage': training_used,
                'limit': training_limit,
                'plan': g.plan.get('name')
            }), 403

        g.usage = usage
        g.training_remaining = training_limit - training_used



        return f(*args, **kwargs)

    return decorated_function
