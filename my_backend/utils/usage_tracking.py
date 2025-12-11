"""
DEPRECATED: Use shared.tracking.usage instead.

This module is kept for backward compatibility. All functions are re-exported
from the canonical shared.tracking.usage module.
"""
import warnings
from shared.tracking.usage import (
    get_current_period_start,
    increment_upload_count,
    increment_processing_count,
    increment_training_count,
    update_storage_usage,
    get_usage_stats,
)

# Emit deprecation warning when this module is imported
warnings.warn(
    "utils.usage_tracking is deprecated. Use shared.tracking.usage instead.",
    DeprecationWarning,
    stacklevel=2
)

__all__ = [
    'get_current_period_start',
    'increment_upload_count',
    'increment_processing_count',
    'increment_training_count',
    'update_storage_usage',
    'get_usage_stats',
]
