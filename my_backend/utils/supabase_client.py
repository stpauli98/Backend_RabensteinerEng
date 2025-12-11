"""
DEPRECATED: Use shared.database.client instead.

This module is kept for backward compatibility. All functions are re-exported
from the canonical shared.database.client module.
"""
import warnings
from shared.database.client import (
    get_supabase_client,
    get_supabase_user_client,
    get_supabase_admin_client,
)

# Emit deprecation warning when this module is imported
warnings.warn(
    "utils.supabase_client is deprecated. Use shared.database.client instead.",
    DeprecationWarning,
    stacklevel=2
)

__all__ = [
    'get_supabase_client',
    'get_supabase_user_client',
    'get_supabase_admin_client',
]
