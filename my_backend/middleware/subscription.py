"""
DEPRECATED: Use shared.auth.subscription instead.

This module is kept for backward compatibility. All functions are re-exported
from the canonical shared.auth.subscription module.
"""
import warnings
from shared.auth.subscription import (
    get_user_subscription,
    get_user_usage,
    require_subscription,
    check_upload_limit,
    check_processing_limit,
    check_storage_limit,
    check_training_limit,
)

# Emit deprecation warning when this module is imported
warnings.warn(
    "middleware.subscription is deprecated. Use shared.auth.subscription instead.",
    DeprecationWarning,
    stacklevel=2
)

__all__ = [
    'get_user_subscription',
    'get_user_usage',
    'require_subscription',
    'check_upload_limit',
    'check_processing_limit',
    'check_storage_limit',
    'check_training_limit',
]
