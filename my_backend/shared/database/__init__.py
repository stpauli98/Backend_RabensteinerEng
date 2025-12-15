"""Database client and operations.

This package provides:
- Supabase client management (client.py)
- Configuration constants (config.py)
- Custom exceptions (exceptions.py)
- Input validation (validators.py)
- Session management (session.py)
- Data persistence (persistence.py)
- File storage (storage.py)
- Batch operations (batch.py)
- Session lifecycle (lifecycle.py)
- Backward compatibility facade (operations.py)
"""

from .client import (
    get_supabase_client,
    get_supabase_user_client,
    get_supabase_admin_client,
)

from .config import (
    DatabaseConfig,
    DomainDefaults,
    TableNames,
    BucketNames,
)

from .exceptions import (
    DatabaseError,
    SessionNotFoundError,
    ValidationError,
    StorageError,
    ConfigurationError,
)

__all__ = [
    # Client
    'get_supabase_client',
    'get_supabase_user_client',
    'get_supabase_admin_client',

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
]
