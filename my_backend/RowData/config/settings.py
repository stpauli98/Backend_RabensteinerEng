"""
Konfiguracija specifična za RowData modul
"""
import os

# Upload limits
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
MAX_CHUNK_SIZE = 5 * 1024 * 1024   # 5MB
CHUNK_STORAGE_PATH = os.environ.get('ROWDATA_STORAGE_PATH', '/tmp/row_data_uploads')
ALLOWED_EXTENSIONS = {'.csv', '.txt'}
UPLOAD_EXPIRY_TIME = 30 * 60  # 30 minuta u sekundama

# Date parsing
SUPPORTED_DATE_FORMATS = [
    '%Y-%m-%dT%H:%M:%S%z',      # ISO format with timezone and seconds
    '%Y-%m-%dT%H:%M%z',         # ISO format with timezone
    '%Y-%m-%dT%H:%M:%S',        # ISO format without timezone
    '%Y-%m-%d %H:%M:%S',
    '%d.%m.%Y %H:%M',
    '%Y-%m-%d %H:%M',
    '%d.%m.%Y %H:%M:%S',
    '%d.%m.%Y %H:%M:%S.%f',     # With milliseconds
    '%Y/%m/%d %H:%M:%S',
    '%d/%m/%Y %H:%M:%S',
    '%Y/%m/%d',
    '%d/%m/%Y',
    '%d-%m-%Y %H:%M:%S',
    '%d-%m-%Y %H:%M',
    '%Y/%m/%d %H:%M',
    '%d/%m/%Y %H:%M',
    '%d-%m-%Y',
    '%H:%M:%S',                  # Pure time format
    '%H:%M'                      # Pure time format
]

# Supported delimiters
ALLOWED_DELIMITERS = [',', ';', '\t']

# Redis configuration za RowData
REDIS_CONFIG = {
    'host': os.environ.get('REDIS_HOST', 'localhost'),
    'port': int(os.environ.get('REDIS_PORT', 6379)),
    'db': int(os.environ.get('REDIS_ROWDATA_DB', 1)),  # Posebna DB za RowData
    'decode_responses': True,
    'socket_timeout': 5,
    'socket_connect_timeout': 5,
    'socket_keepalive': True,
    'socket_keepalive_options': {},
    'connection_pool_kwargs': {
        'max_connections': 50
    }
}

# Redis key configuration
REDIS_KEY_PREFIX = 'rowdata:'
REDIS_TTL = {
    'upload_metadata': 3600,      # 1 sat
    'chunk_info': 1800,           # 30 minuta
    'processing_result': 7200,    # 2 sata
    'rate_limit': 60              # 1 minut
}

# Celery configuration
CELERY_CONFIG = {
    'broker_url': os.environ.get('CELERY_BROKER_URL', 'redis://localhost:6379/2'),
    'result_backend': os.environ.get('CELERY_RESULT_BACKEND', 'redis://localhost:6379/2'),
    'task_prefix': 'rowdata.',
    'queue_name': 'rowdata_queue',
    'task_serializer': 'json',
    'result_serializer': 'json',
    'accept_content': ['json'],
    'timezone': 'UTC',
    'enable_utc': True,
    'task_track_started': True,
    'task_time_limit': 30 * 60,  # 30 minuta
    'task_soft_time_limit': 25 * 60,  # 25 minuta
    'task_acks_late': True,
    'worker_prefetch_multiplier': 4
}

# Rate limiting configuration
RATE_LIMITS = {
    'upload_chunk': '100 per minute',
    'finalize_upload': '10 per minute',
    'download_file': '50 per minute'
}

# Security configuration
SECURITY_CONFIG = {
    'require_auth': os.environ.get('ROWDATA_REQUIRE_AUTH', 'false').lower() == 'true',
    'jwt_secret': os.environ.get('JWT_SECRET', 'your-secret-key-here'),
    'jwt_algorithm': 'HS256',
    'jwt_expiry': 3600,  # 1 sat
    'allowed_origins': os.environ.get('ALLOWED_ORIGINS', '*').split(',')
}

# Logging configuration
LOGGING_CONFIG = {
    'level': os.environ.get('LOG_LEVEL', 'INFO'),
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'handlers': {
        'file': {
            'filename': 'rowdata.log',
            'maxBytes': 10 * 1024 * 1024,  # 10MB
            'backupCount': 5
        }
    }
}

# Monitoring configuration
METRICS_CONFIG = {
    'enabled': os.environ.get('METRICS_ENABLED', 'true').lower() == 'true',
    'port': int(os.environ.get('METRICS_PORT', 8000)),
    'namespace': 'rowdata'
}

# Validation rules
VALIDATION_RULES = {
    'upload_id': {
        'pattern': r'^[a-zA-Z0-9\-_]+$',
        'min_length': 10,
        'max_length': 100
    },
    'chunk_index': {
        'min': 0,
        'max': 10000
    },
    'total_chunks': {
        'min': 1,
        'max': 10000
    },
    'timezone': {
        'allowed': ['UTC', 'Europe/Berlin', 'Europe/Vienna', 'America/New_York', 
                   'America/Los_Angeles', 'Asia/Tokyo', 'Australia/Sydney']
    }
}

# Storage backend configuration
STORAGE_BACKEND = os.environ.get('ROWDATA_STORAGE_BACKEND', 'auto')
# Opcije: 'auto', 'redis', 'file', 'memory'
# - 'auto': Automatski detektuje (Redis > File > Memory)
# - 'redis': Forsiraj Redis (greška ako nije dostupan)
# - 'file': Forsiraj file-based storage
# - 'memory': Forsiraj in-memory storage

# File-based storage configuration
FILE_STORAGE_CONFIG = {
    'base_path': os.environ.get('ROWDATA_FILE_STORAGE_PATH', CHUNK_STORAGE_PATH),
    'cleanup_interval': 3600,  # 1 sat
    'file_permissions': 0o600,  # Read/write samo za owner
}