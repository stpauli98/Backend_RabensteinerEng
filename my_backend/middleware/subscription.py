"""Subscription validation middleware for plan limits"""
import logging
from functools import wraps
from datetime import datetime, timezone
from typing import Optional
from flask import request, jsonify, g
from utils.supabase_client import get_supabase_user_client

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
            .single() \
            .execute()

        if response.data:
            return response.data

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

        now = datetime.now(timezone.utc)
        period_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

        response = supabase.table('usage_tracking') \
            .select('*') \
            .eq('user_id', user_id) \
            .gte('period_start', period_start.isoformat()) \
            .maybe_single() \
            .execute()

        if response and response.data:
            return response.data

        logger.info(f"No usage record for user {user_id} in current period")
        return {
            'uploads_count': 0,
            'processing_count': 0,
            'training_runs_count': 0,
            'storage_used_mb': 0
        }

    except Exception as e:
        logger.error(f"Error fetching usage: {str(e)}")
        return {
            'uploads_count': 0,
            'processing_count': 0,
            'training_runs_count': 0,
            'storage_used_mb': 0
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

        subscription = get_user_subscription(g.user_id, g.access_token)

        if not subscription:
            logger.warning(f"No active subscription for user {g.user_id}")
            return jsonify({
                'error': 'No active subscription',
                'message': 'Please subscribe to a plan to access this feature'
            }), 403

        g.subscription = subscription
        g.plan = subscription.get('subscription_plans', {})

        logger.info(f"User {g.user_email} has {g.plan.get('name', 'Unknown')} plan")

        return f(*args, **kwargs)

    return decorated_function


def check_upload_limit(f):
    """
    Decorator to check upload limit before processing

    Must be used AFTER @require_auth and @require_subscription decorators

    Usage:
        @require_auth
        @require_subscription
        @check_upload_limit
        def upload_route():
            # Upload will only proceed if under limit
            return jsonify({'message': 'Upload successful'})
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not hasattr(g, 'user_id') or not hasattr(g, 'plan'):
            logger.error("check_upload_limit used without require_auth and require_subscription")
            return jsonify({'error': 'Authentication and subscription required'}), 401

        usage = get_user_usage(g.user_id, g.access_token)
        uploads_used = usage.get('uploads_count', 0)
        uploads_limit = g.plan.get('max_uploads_per_month', 0)

        if uploads_used >= uploads_limit:
            logger.warning(f"Upload limit reached for user {g.user_email}: {uploads_used}/{uploads_limit}")
            return jsonify({
                'error': 'Upload limit reached',
                'message': f'You have reached your monthly upload limit of {uploads_limit}',
                'current_usage': uploads_used,
                'limit': uploads_limit,
                'plan': g.plan.get('name')
            }), 403

        g.usage = usage
        g.uploads_remaining = uploads_limit - uploads_used

        logger.info(f"Upload check passed for {g.user_email}: {uploads_used}/{uploads_limit} used")

        return f(*args, **kwargs)

    return decorated_function


def check_processing_limit(f):
    """
    Decorator to check processing limit before processing

    Must be used AFTER @require_auth and @require_subscription decorators

    Usage:
        @require_auth
        @require_subscription
        @check_processing_limit
        def process_route():
            # Processing will only proceed if under limit
            return jsonify({'message': 'Processing successful'})
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not hasattr(g, 'user_id') or not hasattr(g, 'plan'):
            logger.error("check_processing_limit used without require_auth and require_subscription")
            return jsonify({'error': 'Authentication and subscription required'}), 401

        usage = get_user_usage(g.user_id, g.access_token)
        processing_used = usage.get('processing_count', 0)
        processing_limit = g.plan.get('max_processing_jobs_per_month', 0)

        if processing_used >= processing_limit:
            logger.warning(f"Processing limit reached for user {g.user_email}: {processing_used}/{processing_limit}")
            return jsonify({
                'error': 'Processing limit reached',
                'message': f'You have reached your monthly processing limit of {processing_limit}',
                'current_usage': processing_used,
                'limit': processing_limit,
                'plan': g.plan.get('name')
            }), 403

        g.usage = usage
        g.processing_remaining = processing_limit - processing_used

        logger.info(f"Processing check passed for {g.user_email}: {processing_used}/{processing_limit} used")

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

        logger.info(f"Storage check passed for {g.user_email}: {storage_used_mb}/{storage_limit_mb} MB used")

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
                'message': 'Training is not available in your plan. Upgrade to Pro or Enterprise to unlock model training.',
                'plan': g.plan.get('name')
            }), 403

        usage = get_user_usage(g.user_id, g.access_token)
        training_used = usage.get('training_runs_count', 0)
        training_limit = g.plan.get('max_training_runs_per_month', 0)

        if training_limit == -1:
            logger.info(f"Unlimited training for {g.user_email}")
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

        logger.info(f"Training check passed for {g.user_email}: {training_used}/{training_limit} used")

        return f(*args, **kwargs)

    return decorated_function
