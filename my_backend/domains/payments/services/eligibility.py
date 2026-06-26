"""Eligibility checks for subscription-plan purchases."""
from typing import Any


def has_trained_model(supabase: Any, user_id: str) -> bool:
    """Return True if the user has at least one successfully trained model.

    A "trained model" is a ``training_results`` row with ``status='completed'``
    whose session belongs to the user. ``training_results`` has no ``user_id``
    column, so ownership is resolved through ``sessions.user_id``. Failed
    trainings (``status='failed'``) do NOT count.
    """
    sessions = (
        supabase.table('sessions')
        .select('id')
        .eq('user_id', user_id)
        .execute()
    )
    session_ids = [row['id'] for row in (sessions.data or [])]
    if not session_ids:
        return False

    completed = (
        supabase.table('training_results')
        .select('id')
        .in_('session_id', session_ids)
        .eq('status', 'completed')
        .limit(1)
        .execute()
    )
    return bool(completed.data)
