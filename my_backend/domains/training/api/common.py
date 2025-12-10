"""
Common utilities and imports for training API routes.

This module contains shared configurations, imports, and helper functions
used across all training sub-blueprints.
"""

import os
import json
import logging
from datetime import datetime
from flask import request, jsonify, current_app, g

# Database operations
from shared.database.operations import (
    save_session_to_supabase, get_string_id_from_uuid,
    create_or_get_session_uuid, get_supabase_client
)

# Authentication and authorization
from shared.auth.jwt import require_auth
from shared.auth.subscription import require_subscription, check_processing_limit, check_training_limit

# Usage tracking
from shared.tracking.usage import increment_processing_count, increment_training_count, atomic_increment_with_check

# Security
from shared.security.sanitization import sanitize_filename

# Validation utilities
from utils.validation import validate_session_id, create_error_response, create_success_response
from utils.session_helpers import resolve_session_id, is_uuid_format, get_string_session_id, get_uuid_session_id

# Configuration
UPLOAD_BASE_DIR = 'uploads/file_uploads'

# Logger setup
def get_logger(name: str) -> logging.Logger:
    """Get a configured logger for the given module name."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    return logger


# Re-export commonly used items for convenience
__all__ = [
    # Flask
    'request', 'jsonify', 'current_app', 'g',
    # OS and utilities
    'os', 'json', 'datetime', 'logging',
    # Database
    'save_session_to_supabase', 'get_string_id_from_uuid',
    'create_or_get_session_uuid', 'get_supabase_client',
    # Auth
    'require_auth', 'require_subscription',
    'check_processing_limit', 'check_training_limit',
    # Tracking
    'increment_processing_count', 'increment_training_count',
    'atomic_increment_with_check',
    # Security
    'sanitize_filename',
    # Validation
    'validate_session_id', 'create_error_response', 'create_success_response',
    # Session helpers
    'resolve_session_id', 'is_uuid_format',
    'get_string_session_id', 'get_uuid_session_id',
    # Config
    'UPLOAD_BASE_DIR', 'get_logger',
]
