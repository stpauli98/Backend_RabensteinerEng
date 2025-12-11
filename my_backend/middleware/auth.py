"""
DEPRECATED: Use shared.auth.jwt instead.

This module is kept for backward compatibility. All functions are re-exported
from the canonical shared.auth.jwt module.
"""
import warnings
from shared.auth.jwt import (
    require_auth,
    optional_auth,
)

# Emit deprecation warning when this module is imported
warnings.warn(
    "middleware.auth is deprecated. Use shared.auth.jwt instead.",
    DeprecationWarning,
    stacklevel=2
)

__all__ = [
    'require_auth',
    'optional_auth',
]
