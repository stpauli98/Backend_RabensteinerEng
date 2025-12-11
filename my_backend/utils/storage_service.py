"""
DEPRECATED: Use shared.storage.service instead.

This module is kept for backward compatibility. All classes and instances are re-exported
from the canonical shared.storage.service module.
"""
import warnings
from shared.storage.service import (
    StorageService,
    storage_service,
    BUCKET_NAME,
    FILE_EXPIRY_SECONDS,
)

# Emit deprecation warning when this module is imported
warnings.warn(
    "utils.storage_service is deprecated. Use shared.storage.service instead.",
    DeprecationWarning,
    stacklevel=2
)

__all__ = [
    'StorageService',
    'storage_service',
    'BUCKET_NAME',
    'FILE_EXPIRY_SECONDS',
]
