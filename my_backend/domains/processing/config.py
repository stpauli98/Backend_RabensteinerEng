"""
Processing domain configuration constants.
"""
import os
import tempfile

# Upload folders
CHUNK_UPLOAD_FOLDER = "chunk_uploads"
DATA_PROCESSING_UPLOAD_FOLDER = os.path.join(tempfile.gettempdir(), "upload_chunks")

# File constraints
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MB
MAX_CHUNK_SIZE = 10 * 1024 * 1024  # 10 MB
ALLOWED_EXTENSIONS = {'.csv', '.txt'}

# Streaming configuration
STREAMING_CHUNK_SIZE = 50000
BACKPRESSURE_DELAY = 0.01  # 10ms between chunks

# Progress tracking
EMIT_INTERVAL = 0.5  # Emit every 500ms
MIN_CALIBRATION_ROWS = 1000  # Wait for 1000 rows for stable ETA estimate

# Cleaning steps count
TOTAL_CLEANING_STEPS = 6
