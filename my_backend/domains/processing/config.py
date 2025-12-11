"""
Processing domain configuration constants.

Note: Chunk upload folders are no longer used locally.
Chunks are now stored in Supabase Storage (temp-chunks bucket) for
multi-instance Cloud Run support. See shared/storage/chunk_service.py
"""

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
