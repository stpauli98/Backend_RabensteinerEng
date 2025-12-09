"""
Chunk upload handling service.
Manages chunked file uploads, locking, and assembly.
"""
import os
import re
import tempfile
import threading
import logging
from domains.processing.config import MAX_FILE_SIZE, MAX_CHUNK_SIZE, ALLOWED_EXTENSIONS

logger = logging.getLogger(__name__)

# Lock for preventing race conditions during chunk processing
upload_locks = {}
upload_locks_lock = threading.Lock()


def get_upload_lock(upload_id: str) -> threading.Lock:
    """Get or create a lock for a specific upload_id."""
    with upload_locks_lock:
        if upload_id not in upload_locks:
            upload_locks[upload_id] = threading.Lock()
        return upload_locks[upload_id]


def cleanup_upload_lock(upload_id: str):
    """Clean up lock after upload completion."""
    with upload_locks_lock:
        if upload_id in upload_locks:
            del upload_locks[upload_id]


def secure_path_join(base_dir, user_input):
    """Safely join paths preventing directory traversal attacks"""
    if not user_input:
        raise ValueError("Empty path component")

    if '..' in user_input or '/' in user_input or '\\' in user_input:
        raise ValueError("Path traversal attempt detected")

    if '%2e%2e' in user_input.lower() or '%2f' in user_input.lower() or '%5c' in user_input.lower():
        raise ValueError("Encoded path traversal attempt detected")

    clean_input = os.path.basename(user_input)

    dangerous_chars = ['<', '>', ':', '"', '|', '?', '*', '\x00']
    for char in dangerous_chars:
        clean_input = clean_input.replace(char, '')

    if not clean_input or clean_input in ['.', '..', ''] or len(clean_input.strip()) == 0:
        raise ValueError("Invalid path component")

    full_path = os.path.join(base_dir, clean_input)

    base_real = os.path.realpath(base_dir)
    full_real = os.path.realpath(full_path)

    if not full_real.startswith(base_real + os.sep) and full_real != base_real:
        raise ValueError("Path traversal attempt detected")

    return full_path


def validate_file_upload(file_chunk, filename):
    """Validate uploaded file security and format"""
    if not filename:
        raise ValueError("Empty filename")

    _, ext = os.path.splitext(filename.lower())
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {ext}. Only CSV and TXT files allowed.")

    chunk_data = file_chunk.read()
    if len(chunk_data) > MAX_CHUNK_SIZE:
        raise ValueError("Chunk size too large")

    file_chunk.seek(0)
    return chunk_data


def combine_chunks_efficiently(upload_dir, total_chunks):
    """Memory-efficient chunk combination using temporary file streaming"""
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    total_size = 0

    try:
        for i in range(total_chunks):
            chunk_path = os.path.join(upload_dir, f"chunk_{i:04d}.part")
            if not os.path.exists(chunk_path):
                temp_file.close()
                os.unlink(temp_file.name)
                raise FileNotFoundError(f"Missing chunk file: {i}")

            with open(chunk_path, "rb") as chunk_file:
                while True:
                    block = chunk_file.read(8192)
                    if not block:
                        break
                    temp_file.write(block)
                    total_size += len(block)

                    if total_size > MAX_FILE_SIZE:
                        temp_file.close()
                        os.unlink(temp_file.name)
                        raise ValueError(f"Combined file size exceeds {MAX_FILE_SIZE} bytes")

        temp_file.close()
        return temp_file.name, total_size

    except Exception:
        if not temp_file.closed:
            temp_file.close()
        if os.path.exists(temp_file.name):
            os.unlink(temp_file.name)
        raise


def extract_chunk_index(filename):
    """Extract chunk index from filename"""
    try:
        parts = filename.split("_")
        chunk_part = parts[-1].split(".")[0]
        return int(chunk_part)
    except (ValueError, IndexError) as e:
        logger.error(f"Error parsing chunk filename {filename}: {e}")
        return 0
