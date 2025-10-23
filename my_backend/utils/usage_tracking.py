"""Usage tracking utilities for Supabase"""
import logging
from datetime import datetime, timezone, timedelta
from utils.supabase_client import get_supabase_admin_client

logger = logging.getLogger(__name__)

def get_current_period_start() -> datetime:
    """
    Get the start of current billing period (first day of current month)

    Returns:
        datetime: Start of current period in UTC
    """
    now = datetime.now(timezone.utc)
    return now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)


def increment_upload_count(user_id: str) -> bool:
    """
    Increment upload count for user in current period

    Args:
        user_id: User ID

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        supabase = get_supabase_admin_client()
        period_start = get_current_period_start()

        # Try to get existing record
        response = supabase.table('usage_tracking') \
            .select('*') \
            .eq('user_id', user_id) \
            .eq('period_start', period_start.isoformat()) \
            .execute()

        if response.data and len(response.data) > 0:
            # Update existing record
            usage_id = response.data[0]['id']
            current_count = response.data[0].get('uploads_count', 0)

            supabase.table('usage_tracking') \
                .update({'uploads_count': current_count + 1}) \
                .eq('id', usage_id) \
                .execute()

            logger.info(f"Incremented upload count for user {user_id}: {current_count} -> {current_count + 1}")
        else:
            # Create new record
            period_end = period_start.replace(month=period_start.month + 1 if period_start.month < 12 else 1,
                                             year=period_start.year + 1 if period_start.month == 12 else period_start.year) \
                                      .replace(day=1) - timedelta(days=1)

            supabase.table('usage_tracking') \
                .insert({
                    'user_id': user_id,
                    'period_start': period_start.isoformat(),
                    'period_end': period_end.isoformat(),
                    'uploads_count': 1,
                    'processing_jobs_count': 0,
                    'operations_count': 0,
                    'training_runs_count': 0,
                    'storage_used_gb': 0
                }) \
                .execute()

            logger.info(f"Created new usage tracking record for user {user_id}")

        return True

    except Exception as e:
        logger.error(f"Error incrementing upload count: {str(e)}")
        return False


def increment_processing_count(user_id: str) -> bool:
    """
    Increment processing count for user in current period

    Args:
        user_id: User ID

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        supabase = get_supabase_admin_client()
        period_start = get_current_period_start()

        # Try to get existing record
        response = supabase.table('usage_tracking') \
            .select('*') \
            .eq('user_id', user_id) \
            .eq('period_start', period_start.isoformat()) \
            .execute()

        if response.data and len(response.data) > 0:
            # Update existing record
            usage_id = response.data[0]['id']
            current_count = response.data[0].get('processing_jobs_count', 0)

            supabase.table('usage_tracking') \
                .update({'processing_jobs_count': current_count + 1}) \
                .eq('id', usage_id) \
                .execute()

            logger.info(f"Incremented processing count for user {user_id}: {current_count} -> {current_count + 1}")
        else:
            # Create new record
            period_end = period_start.replace(month=period_start.month + 1 if period_start.month < 12 else 1,
                                             year=period_start.year + 1 if period_start.month == 12 else period_start.year) \
                                      .replace(day=1) - timedelta(days=1)

            supabase.table('usage_tracking') \
                .insert({
                    'user_id': user_id,
                    'period_start': period_start.isoformat(),
                    'period_end': period_end.isoformat(),
                    'uploads_count': 0,
                    'processing_jobs_count': 1,
                    'operations_count': 0,
                    'training_runs_count': 0,
                    'storage_used_gb': 0
                }) \
                .execute()

            logger.info(f"Created new usage tracking record for user {user_id}")

        return True

    except Exception as e:
        logger.error(f"Error incrementing processing count: {str(e)}")
        return False


def increment_training_count(user_id: str) -> bool:
    """
    Increment training runs count for user in current period

    Args:
        user_id: User ID

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        supabase = get_supabase_admin_client()
        period_start = get_current_period_start()

        # Try to get existing record
        response = supabase.table('usage_tracking') \
            .select('*') \
            .eq('user_id', user_id) \
            .eq('period_start', period_start.isoformat()) \
            .execute()

        if response.data and len(response.data) > 0:
            # Update existing record
            usage_id = response.data[0]['id']
            current_count = response.data[0].get('training_runs_count', 0)

            supabase.table('usage_tracking') \
                .update({'training_runs_count': current_count + 1}) \
                .eq('id', usage_id) \
                .execute()

            logger.info(f"Incremented training count for user {user_id}: {current_count} -> {current_count + 1}")
        else:
            # Create new record
            period_end = period_start.replace(month=period_start.month + 1 if period_start.month < 12 else 1,
                                             year=period_start.year + 1 if period_start.month == 12 else period_start.year) \
                                      .replace(day=1) - timedelta(days=1)

            supabase.table('usage_tracking') \
                .insert({
                    'user_id': user_id,
                    'period_start': period_start.isoformat(),
                    'period_end': period_end.isoformat(),
                    'uploads_count': 0,
                    'processing_jobs_count': 0,
                    'operations_count': 0,
                    'training_runs_count': 1,
                    'storage_used_gb': 0
                }) \
                .execute()

            logger.info(f"Created new usage tracking record for user {user_id} with training count: 1")

        return True

    except Exception as e:
        logger.error(f"Error incrementing training count: {str(e)}")
        return False


def update_storage_usage(user_id: str, storage_mb: float) -> bool:
    """
    Update storage usage for user in current period

    Args:
        user_id: User ID
        storage_mb: Storage used in MB

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        supabase = get_supabase_admin_client()
        period_start = get_current_period_start()

        # Try to get existing record
        response = supabase.table('usage_tracking') \
            .select('*') \
            .eq('user_id', user_id) \
            .eq('period_start', period_start.isoformat()) \
            .execute()

        if response.data and len(response.data) > 0:
            # Update existing record
            usage_id = response.data[0]['id']
            storage_gb = storage_mb / 1024  # Convert MB to GB

            supabase.table('usage_tracking') \
                .update({'storage_used_gb': storage_gb}) \
                .eq('id', usage_id) \
                .execute()

            logger.info(f"Updated storage usage for user {user_id}: {storage_gb:.2f} GB ({storage_mb} MB)")
        else:
            # Create new record
            storage_gb = storage_mb / 1024  # Convert MB to GB
            period_end = period_start.replace(month=period_start.month + 1 if period_start.month < 12 else 1,
                                             year=period_start.year + 1 if period_start.month == 12 else period_start.year) \
                                      .replace(day=1) - timedelta(days=1)

            supabase.table('usage_tracking') \
                .insert({
                    'user_id': user_id,
                    'period_start': period_start.isoformat(),
                    'period_end': period_end.isoformat(),
                    'uploads_count': 0,
                    'processing_jobs_count': 0,
                    'operations_count': 0,
                    'training_runs_count': 0,
                    'storage_used_gb': storage_gb
                }) \
                .execute()

            logger.info(f"Created new usage tracking record for user {user_id} with storage: {storage_gb:.2f} GB ({storage_mb} MB)")

        return True

    except Exception as e:
        logger.error(f"Error updating storage usage: {str(e)}")
        return False


def get_usage_stats(user_id: str) -> dict:
    """
    Get usage statistics for user in current period

    Args:
        user_id: User ID

    Returns:
        dict: Usage statistics with uploads, processing, storage
    """
    try:
        supabase = get_supabase_admin_client()
        period_start = get_current_period_start()

        # Get usage record
        response = supabase.table('usage_tracking') \
            .select('*') \
            .eq('user_id', user_id) \
            .eq('period_start', period_start.isoformat()) \
            .single() \
            .execute()

        if response.data:
            storage_gb = response.data.get('storage_used_gb', 0)
            return {
                'period_start': period_start.isoformat(),
                'uploads_count': response.data.get('uploads_count', 0),
                'processing_jobs_count': response.data.get('processing_jobs_count', 0),
                'training_runs_count': response.data.get('training_runs_count', 0),
                'storage_used_gb': storage_gb,
                'storage_used_mb': storage_gb * 1024  # Also provide MB for backward compatibility
            }

        # No usage record yet
        return {
            'period_start': period_start.isoformat(),
            'uploads_count': 0,
            'processing_jobs_count': 0,
            'training_runs_count': 0,
            'storage_used_gb': 0,
            'storage_used_mb': 0
        }

    except Exception as e:
        logger.error(f"Error getting usage stats: {str(e)}")
        return {
            'period_start': period_start.isoformat(),
            'uploads_count': 0,
            'processing_jobs_count': 0,
            'training_runs_count': 0,
            'storage_used_gb': 0,
            'storage_used_mb': 0
        }
