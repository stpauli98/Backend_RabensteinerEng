"""
Cloud Domain Configuration
Constants and settings for cloud data processing
"""
import os
import re
import tempfile

# File size limits
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
MAX_ROWS = 1_000_000
MAX_COLUMNS = 100
MAX_ACTIVE_UPLOADS = 1000
FILE_BUFFER_SIZE = 1024 * 1024  # 1MB

# Upload validation
UPLOAD_ID_PATTERN = re.compile(r'^[\w\-]{1,64}$')

# Tolerance settings
TOLERANCE_ADJUSTMENT_FACTOR = 2
MIN_TOLERANCE_THRESHOLD = 0.01
DEFAULT_TOLERANCE_RATIO = 0.1

# Streaming settings
STREAMING_CHUNK_SIZE = 5000

# Valid file types for upload
VALID_FILE_TYPES = ['temp_file', 'load_file', 'interpolate_file']

# Chunk storage directory
CHUNK_DIR = os.path.join(tempfile.gettempdir(), 'cloud_chunks')
os.makedirs(CHUNK_DIR, exist_ok=True)
