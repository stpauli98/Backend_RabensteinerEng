"""
Adjustments Domain Configuration
Constants and settings for data adjustment operations
"""
import os

# Upload folder configuration
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'temp_uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# DataFrame processing
DATAFRAME_CHUNK_SIZE = 10000

# Timeout configurations (in seconds)
CHUNK_BUFFER_TIMEOUT = 30 * 60  # 30 minutes
ADJUSTMENT_CHUNKS_TIMEOUT = 60 * 60  # 1 hour
STORED_DATA_TIMEOUT = 60 * 60  # 1 hour
INFO_CACHE_TIMEOUT = 2 * 60 * 60  # 2 hours

# SocketIO settings
SOCKETIO_CHUNK_DELAY = 0.1  # seconds between chunk emits

# Date/time format
UTC_FORMAT = "%Y-%m-%d %H:%M:%S"

# Valid processing methods
VALID_METHODS = {'mean', 'intrpl', 'nearest', 'nearest (mean)', 'nearest (max. delta)'}
