"""Usage tracking utilities for Supabase"""
import logging
from calendar import monthrange
from datetime import datetime, date, timezone, timedelta
from shared.database.client import get_supabase_admin_client

logger = logging.getLogger(__name__)


def anniversary_period_end(period_start: date) -> date:
    """Last day of the anniversary window starting at period_start.

    Matches the SQL convention used by increment_usage / update_storage_usage
    (period_start + INTERVAL '1 month' - INTERVAL '1 day'), with month-end
    clamping. e.g. 2026-06-15 -> 2026-07-14; 2026-01-31 -> 2026-02-27.
    """
    y = period_start.year + (1 if period_start.month == 12 else 0)
    m = 1 if period_start.month == 12 else period_start.month + 1
    day = min(period_start.day, monthrange(y, m)[1])
    return date(y, m, day) - timedelta(days=1)

def get_current_period_start() -> datetime:
    """DEPRECATED calendar-month fallback. Use get_period_start_for_user(user_id)."""
    now = datetime.now(timezone.utc)
    return now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)


def get_period_start_for_user(user_id: str) -> date:
    """Resolve the user's current anniversary period start via the SQL source of truth.

    Falls back to the 1st of the current UTC month if the RPC is unavailable, so
    writes never crash on a transient DB issue.
    """
    try:
        supabase = get_supabase_admin_client()
        resp = supabase.rpc('get_current_period_start', {'p_user_id': user_id}).execute()
        if resp and resp.data:
            return date.fromisoformat(str(resp.data))
    except Exception as e:
        logger.error(f"get_period_start_for_user RPC failed for {user_id[:8]}...: {e}")
    return datetime.now(timezone.utc).date().replace(day=1)



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
        period_start = get_period_start_for_user(user_id)
        supabase.rpc('increment_usage', {
            'p_user_id': user_id,
            'p_period_start': period_start.isoformat(),
            'p_field': 'processing_jobs_count',
        }).execute()
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
        period_start = get_period_start_for_user(user_id)
        supabase.rpc('increment_usage', {
            'p_user_id': user_id,
            'p_period_start': period_start.isoformat(),
            'p_field': 'training_runs_count',
        }).execute()
        return True
    except Exception as e:
        logger.error(f"Error incrementing training count: {str(e)}")
        return False


def update_storage_usage(user_id: str, storage_mb: float) -> bool:
    """
    Update storage usage for user in current period (additive).

    Args:
        user_id: User ID
        storage_mb: Storage used in MB (added atomically to current period total)

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        supabase = get_supabase_admin_client()
        period_start = get_period_start_for_user(user_id)
        supabase.rpc('increment_storage_usage', {
            'p_user_id': user_id,
            'p_period_start': period_start.isoformat(),
            'p_storage_gb': storage_mb / 1024.0,
        }).execute()
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
    period_start = get_period_start_for_user(user_id)
    try:
        supabase = get_supabase_admin_client()

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
                'storage_used_mb': storage_gb * 1024
            }

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


def atomic_increment_with_check(user_id: str, resource_type: str) -> tuple:
    """
    Atomski provjeri kvotu i inkrementiraj ako je dozvoljeno.

    Koristi Supabase RPC funkciju za atomsku operaciju koja sprječava
    race condition kod paralelnih zahtjeva.

    Args:
        user_id: UUID korisnika
        resource_type: 'processing', 'training', ili 'upload'

    Returns:
        tuple: (allowed: bool, details: dict)
            - allowed: True ako je operacija dozvoljena i inkrementirana
            - details: {'current': int, 'limit': int, 'message': str}
    """
    try:
        supabase = get_supabase_admin_client()
        period_start = get_period_start_for_user(user_id)

        result = supabase.rpc('atomic_check_and_increment_quota', {
            'p_user_id': user_id,
            'p_resource_type': resource_type,
            'p_period_start': period_start.isoformat()
        }).execute()

        if result.data and len(result.data) > 0:
            row = result.data[0]
            allowed = row.get('allowed', False)
            details = {
                'current': row.get('current_count', 0),
                'limit': row.get('max_limit', 0),
                'message': row.get('message', '')
            }

            if allowed:
                logger.debug(f"Atomic quota check passed for user {user_id}, resource {resource_type}: {details['current']}/{details['limit']}")
            else:
                logger.warning(f"Atomic quota check failed for user {user_id}, resource {resource_type}: {details['message']}")

            return allowed, details

        logger.error(f"Atomic quota RPC returned no data for user {user_id}")
        return False, {'current': 0, 'limit': 0, 'message': 'RPC returned no data'}

    except Exception as e:
        logger.error(f"Atomic quota check failed for user {user_id}: {str(e)}")
        # Graceful degradation: dozvoli ako RPC ne radi, ali logiraj upozorenje
        # Ovo osigurava da korisnici nisu blokirani ako RPC ima problem
        logger.warning(f"Falling back to allow for user {user_id} due to RPC error")
        return True, {'current': 0, 'limit': -1, 'message': f'Fallback allowed: {str(e)}'}


def log_compute_duration(user_id: str, duration_seconds: float, resource_type: str, metadata: dict = None) -> bool:
    """
    Log compute duration to usage_events for Stundenkontingent tracking.

    Args:
        user_id: User UUID
        duration_seconds: Actual backend compute time in seconds
        resource_type: Pipeline stage identifier (e.g., 'rohdaten', 'erste-bearbeitung')
        metadata: Optional extra context (upload_id, file_count, etc.)

    Returns:
        bool: True if logged successfully
    """
    try:
        supabase = get_supabase_admin_client()
        supabase.table('usage_events').insert({
            'user_id': user_id,
            'event_type': 'processing',
            'resource_type': resource_type,
            'processing_duration_sec': max(1, round(duration_seconds)),
            'metadata': metadata or {}
        }).execute()
        logger.debug(f"Compute duration: {user_id[:8]}... {resource_type} {duration_seconds:.1f}s")
        return True
    except Exception as e:
        logger.error(f"Error logging compute duration: {str(e)}")
        return False
