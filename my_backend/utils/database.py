"""
DEPRECATED: This module is deprecated.
Use shared.database.operations instead.

This file re-exports all symbols from shared.database.operations for backward compatibility.
All new code should import directly from shared.database.operations.
"""

import warnings

warnings.warn(
    "utils.database is deprecated. Use shared.database.operations instead.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export everything from the canonical location
from shared.database.operations import (
    # Config
    DatabaseConfig,

    # Validation functions
    validate_session_id,
    validate_file_info,
    validate_time_info,

    # Exceptions
    DatabaseError,
    SessionNotFoundError,
    ValidationError,
    StorageError,
    ConfigurationError,

    # Client functions
    get_supabase_client,
    retry_database_operation,

    # Session functions
    create_or_get_session_uuid,
    get_string_id_from_uuid,

    # Save functions
    save_time_info,
    save_zeitschritte,
    save_file_info,
    save_csv_file_content,
    save_session_to_supabase,

    # Update functions
    update_session_name,
)

# Also expose internal helpers for any edge cases
from shared.database.operations import (
    _get_session_uuid,
    _load_session_metadata,
    _save_metadata_to_database,
    _save_files_to_database,
    _finalize_session,
    _prepare_file_batch_data,
    _batch_upsert_files,
    _batch_insert_files,
    _transform_time_info_to_jsonb,
)

__all__ = [
    'DatabaseConfig',
    'validate_session_id',
    'validate_file_info',
    'validate_time_info',
    'DatabaseError',
    'SessionNotFoundError',
    'ValidationError',
    'StorageError',
    'ConfigurationError',
    'get_supabase_client',
    'retry_database_operation',
    'create_or_get_session_uuid',
    'get_string_id_from_uuid',
    'save_time_info',
    'save_zeitschritte',
    'save_file_info',
    'save_csv_file_content',
    'save_session_to_supabase',
    'update_session_name',
]
