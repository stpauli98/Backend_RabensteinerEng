"""
Database operations module for Supabase.

BACKWARD COMPATIBILITY FACADE
=============================
This module re-exports all symbols from the new modular structure for backward compatibility.
All 18+ files that import from this module will continue to work unchanged.

For new code, consider importing from specific submodules:
  - shared.database.config      - Configuration constants
  - shared.database.exceptions  - Custom exception classes
  - shared.database.validators  - Input validation functions
  - shared.database.session     - Session UUID management
  - shared.database.persistence - Data persistence (time_info, zeitschritte, file_info)
  - shared.database.storage     - File storage operations
  - shared.database.batch       - Batch file operations
  - shared.database.lifecycle   - Session lifecycle management
"""

import logging

# Configuration
from .config import (
    DatabaseConfig,
    DomainDefaults,
    TableNames,
    BucketNames,
)

# Exceptions
from .exceptions import (
    DatabaseError,
    SessionNotFoundError,
    ValidationError,
    StorageError,
    ConfigurationError,
)

# Validators
from .validators import (
    validate_session_id,
    validate_file_info,
    validate_time_info,
    sanitize_filename,
    _sanitize_filename,  # Backward compatibility alias
)

# Session management
from .session import (
    get_supabase_client,
    retry_database_operation,
    create_or_get_session_uuid,
    get_string_id_from_uuid,
    get_session_uuid,
    _get_session_uuid,  # Backward compatibility alias
)

# Persistence
from .persistence import (
    save_time_info,
    save_zeitschritte,
    save_file_info,
    transform_time_info_to_jsonb,
    _transform_time_info_to_jsonb,  # Backward compatibility alias
)

# Storage
from .storage import (
    save_csv_file_content,
    delete_csv_file_content,
    get_csv_file_url,
)

# Batch operations
from .batch import (
    prepare_file_batch_data,
    batch_upsert_files,
    _prepare_file_batch_data,  # Backward compatibility alias
    _batch_upsert_files,  # Backward compatibility alias
)

# Lifecycle
from .lifecycle import (
    load_session_metadata,
    save_metadata_to_database,
    save_files_to_database,
    finalize_session,
    update_session_name,
    save_session_to_supabase,
    _load_session_metadata,  # Backward compatibility alias
    _save_metadata_to_database,  # Backward compatibility alias
    _save_files_to_database,  # Backward compatibility alias
    _finalize_session,  # Backward compatibility alias
)

# Logger for backward compatibility
logger = logging.getLogger(__name__)

__all__ = [
    # Configuration
    'DatabaseConfig',
    'DomainDefaults',
    'TableNames',
    'BucketNames',

    # Exceptions
    'DatabaseError',
    'SessionNotFoundError',
    'ValidationError',
    'StorageError',
    'ConfigurationError',

    # Validators
    'validate_session_id',
    'validate_file_info',
    'validate_time_info',
    'sanitize_filename',
    '_sanitize_filename',

    # Session management
    'get_supabase_client',
    'retry_database_operation',
    'create_or_get_session_uuid',
    'get_string_id_from_uuid',
    'get_session_uuid',
    '_get_session_uuid',

    # Persistence
    'save_time_info',
    'save_zeitschritte',
    'save_file_info',
    'transform_time_info_to_jsonb',
    '_transform_time_info_to_jsonb',

    # Storage
    'save_csv_file_content',
    'delete_csv_file_content',
    'get_csv_file_url',

    # Batch operations
    'prepare_file_batch_data',
    'batch_upsert_files',
    '_prepare_file_batch_data',
    '_batch_upsert_files',

    # Lifecycle
    'load_session_metadata',
    'save_metadata_to_database',
    'save_files_to_database',
    'finalize_session',
    'update_session_name',
    'save_session_to_supabase',
    '_load_session_metadata',
    '_save_metadata_to_database',
    '_save_files_to_database',
    '_finalize_session',

    # Logger
    'logger',
]
