"""
CSV Data Upload and Processing Routes

This module handles chunked CSV file uploads with the following features:
- Chunked upload support for large files
- Multiple datetime format support with auto-detection
- Timezone conversion to UTC
- Real-time progress tracking via WebSocket
- Usage tracking and quota enforcement

Main endpoints:
- POST /upload-chunk: Upload individual file chunks
- POST /finalize-upload: Complete upload and process file
- POST /cancel-upload: Cancel in-progress upload
- POST /prepare-save: Prepare processed data for download
- GET /download/<file_id>: Download processed CSV file
"""

import os
import tempfile
import traceback
import json
import csv
import time
import threading
from io import StringIO
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from flask import Blueprint, request, jsonify, send_file, current_app, g, Response, redirect
import pandas as pd
from flask_socketio import join_room
from shared.auth.jwt import require_auth
from shared.auth.subscription import require_subscription, check_processing_limit
from shared.tracking.usage import increment_processing_count, update_storage_usage
from shared.storage.service import storage_service
from shared.exceptions import (
    MissingParameterError,
    InvalidParameterError,
    DelimiterMismatchError,
    CSVParsingError,
    DateTimeParsingError,
    EncodingError,
    ChunkUploadError,
    UploadNotFoundError,
    IncompleteUploadError,
    UnsupportedTimezoneError,
    TimezoneConversionError,
    LoadDataException
)

# ============================================================================
# CONSTANTS
# ============================================================================

# Upload Configuration
UPLOAD_EXPIRY_SECONDS: int = 1800  # 30 minutes
CHUNK_SIZE_MB: int = 5

# CSV Configuration
DEFAULT_DELIMITER: str = ','
SUPPORTED_DELIMITERS: List[str] = [',', ';', '	']

# Date Format Support
SUPPORTED_DATE_FORMATS: List[str] = [
    # ISO 8601 formats (most common in data exports)
    '%Y-%m-%dT%H:%M:%S%z',          # 2024-01-15T10:30:00+0100
    '%Y-%m-%dT%H:%M:%S.%f%z',       # 2024-01-15T10:30:00.123456+0100
    '%Y-%m-%dT%H:%M%z',             # 2024-01-15T10:30+0100
    '%Y-%m-%dT%H:%M:%S',            # 2024-01-15T10:30:00
    '%Y-%m-%dT%H:%M:%S.%f',         # 2024-01-15T10:30:00.123456
    '%Y-%m-%dT%H:%M',               # 2024-01-15T10:30

    # Standard datetime formats (space separated)
    '%Y-%m-%d %H:%M:%S',            # 2024-01-15 10:30:00
    '%Y-%m-%d %H:%M:%S.%f',         # 2024-01-15 10:30:00.123456
    '%Y-%m-%d %H:%M',               # 2024-01-15 10:30
    '%Y-%m-%d',                     # 2024-01-15

    # European formats (dot separator - common in AT/DE/CH)
    '%d.%m.%Y %H:%M:%S',            # 15.01.2024 10:30:00
    '%d.%m.%Y %H:%M:%S.%f',         # 15.01.2024 10:30:00.123456
    '%d.%m.%Y %H:%M',               # 15.01.2024 10:30
    '%d.%m.%Y',                     # 15.01.2024

    # European formats (slash separator)
    '%d/%m/%Y %H:%M:%S',            # 15/01/2024 10:30:00
    '%d/%m/%Y %H:%M:%S.%f',         # 15/01/2024 10:30:00.123456
    '%d/%m/%Y %H:%M',               # 15/01/2024 10:30
    '%d/%m/%Y',                     # 15/01/2024

    # US formats (month first)
    '%m/%d/%Y %H:%M:%S',            # 01/15/2024 10:30:00
    '%m/%d/%Y %H:%M',               # 01/15/2024 10:30
    '%m/%d/%Y',                     # 01/15/2024
    '%m-%d-%Y %H:%M:%S',            # 01-15-2024 10:30:00
    '%m-%d-%Y %H:%M',               # 01-15-2024 10:30
    '%m-%d-%Y',                     # 01-15-2024

    # Asian formats (year/month/day with slashes)
    '%Y/%m/%d %H:%M:%S',            # 2024/01/15 10:30:00
    '%Y/%m/%d %H:%M:%S.%f',         # 2024/01/15 10:30:00.123456
    '%Y/%m/%d %H:%M',               # 2024/01/15 10:30
    '%Y/%m/%d',                     # 2024/01/15

    # European formats (dash separator)
    '%d-%m-%Y %H:%M:%S',            # 15-01-2024 10:30:00
    '%d-%m-%Y %H:%M:%S.%f',         # 15-01-2024 10:30:00.123456
    '%d-%m-%Y %H:%M',               # 15-01-2024 10:30
    '%d-%m-%Y',                     # 15-01-2024

    # Excel/spreadsheet common formats
    '%Y%m%d %H:%M:%S',              # 20240115 10:30:00
    '%Y%m%d%H%M%S',                 # 20240115103000
    '%Y%m%d',                       # 20240115

    # Unix timestamp string (will be handled separately but included for completeness)
    # Note: Actual unix timestamps need special handling

    # Time-only formats (for separate date/time columns)
    '%H:%M:%S.%f',                  # 10:30:00.123456
    '%H:%M:%S',                     # 10:30:00
    '%H:%M',                        # 10:30

    # 12-hour formats with AM/PM
    '%Y-%m-%d %I:%M:%S %p',         # 2024-01-15 10:30:00 AM
    '%Y-%m-%d %I:%M %p',            # 2024-01-15 10:30 AM
    '%d.%m.%Y %I:%M:%S %p',         # 15.01.2024 10:30:00 AM
    '%d.%m.%Y %I:%M %p',            # 15.01.2024 10:30 AM
    '%d/%m/%Y %I:%M:%S %p',         # 15/01/2024 10:30:00 AM
    '%m/%d/%Y %I:%M:%S %p',         # 01/15/2024 10:30:00 AM
]

# Encoding Options
SUPPORTED_ENCODINGS: List[str] = ['utf-8', 'utf-16', 'utf-16le', 'utf-16be', 'latin1', 'cp1252']

# Streaming Configuration
STREAMING_CHUNK_SIZE: int = 50000


# ============================================================================
# ProgressTracker Class - Real-time progress tracking with per-step ETA
# ============================================================================

class ProgressTracker:
    """
    Progress tracking sa ETA za SVE faze procesiranja.
    
    ETA se računa na osnovu:
    - Veličine fajla (bytes) za procjenu ukupnog vremena
    - Broja redova kada postane poznat
    - Real-time mjerenja za streaming fazu
    
    PAYLOAD koji se šalje na frontend:
    {
        'uploadId': str,           # ID uploada za WebSocket room
        'step': str,               # Naziv trenutne faze (validating, parsing, datetime, utc, build, streaming, complete, error)
        'progress': int,           # Progress percentage (0-100)
        'message': str,            # Poruka za prikaz korisniku
        'status': str,             # Status (processing, completed, error)
        'currentStep': int,        # Redni broj trenutnog koraka (1-5)
        'totalSteps': int,         # Ukupan broj koraka (5)
        'eta': int,                # ETA u sekundama
        'etaFormatted': str        # ETA formatiran (npr. "2m 30s")
    }
    
    Benchmark vremena po fazi (za 100k redova / ~5MB):
    - validation: ~0.1s
    - parsing: ~0.5s
    - datetime: ~0.8s
    - utc: ~0.2s
    - build: ~1.0s
    - streaming: ~5-10s (zavisi od chunk size)
    """
    
    # Benchmark koeficijenti (sekunde po MB)
    PHASE_TIME_PER_MB = {
        'validation': 0.02,
        'parsing': 0.10,
        'datetime': 0.15,
        'utc': 0.04,
        'build': 0.20,
        'streaming': 1.0  # Ovo se računa real-time
    }
    
    def __init__(self, upload_id: str, socketio=None, file_size_bytes: int = 0):
        self.upload_id = upload_id
        self.socketio = socketio
        self.start_time = time.time()
        self.phase_start_times: Dict[str, float] = {}
        self.phase_durations: Dict[str, float] = {}
        
        # File size za procjenu ETA
        self.file_size_bytes = file_size_bytes
        self.file_size_mb = file_size_bytes / (1024 * 1024) if file_size_bytes > 0 else 1
        
        # Procijenjeno ukupno vrijeme na osnovu veličine fajla
        self.estimated_total_time = self._estimate_total_time()

        self.last_emit_time = 0
        self.emit_interval = 0.3  # Emit svakih 300ms za smooth ETA

        # Step tracking za frontend
        self.current_step = 0
        self.total_steps = 0
        
        # Broj redova (postavlja se nakon parsiranja)
        self.total_rows = 0
    
    def _estimate_total_time(self) -> float:
        """Procijeni ukupno vrijeme na osnovu veličine fajla"""
        total = 0
        for phase, time_per_mb in self.PHASE_TIME_PER_MB.items():
            total += time_per_mb * self.file_size_mb
        return max(total, 2.0)  # Minimum 2 sekunde
    
    def set_total_rows(self, rows: int) -> None:
        """Postavi broj redova i ažuriraj procjenu vremena"""
        self.total_rows = rows
        streaming_time = rows * 0.00005
        self.estimated_total_time = (
            self.PHASE_TIME_PER_MB['validation'] * self.file_size_mb +
            self.PHASE_TIME_PER_MB['parsing'] * self.file_size_mb +
            self.PHASE_TIME_PER_MB['datetime'] * self.file_size_mb +
            self.PHASE_TIME_PER_MB['utc'] * self.file_size_mb +
            self.PHASE_TIME_PER_MB['build'] * self.file_size_mb +
            streaming_time
        )

    def start_phase(self, phase_name: str) -> None:
        """Označi početak nove faze procesiranja"""
        self.phase_start_times[phase_name] = time.time()

    def end_phase(self, phase_name: str) -> None:
        """Završi fazu i snimi stvarno vrijeme trajanja"""
        if phase_name in self.phase_start_times:
            duration = time.time() - self.phase_start_times[phase_name]
            self.phase_durations[phase_name] = duration

    def calculate_eta_for_progress(self, current_progress: float) -> int:
        """
        Izračunaj ETA na osnovu trenutnog progresa i procijenjenog ukupnog vremena.
        
        Za brze faze koristi procijenjeno vrijeme bazirano na veličini fajla,
        za sporije faze (kad je proteklo >1s) koristi linearnu ekstrapolaciju.
        
        Args:
            current_progress: Trenutni progress (0-100)
            
        Returns:
            ETA u sekundama
        """
        if current_progress <= 0:
            return int(self.estimated_total_time)
        
        if current_progress >= 100:
            return 0
        
        elapsed = time.time() - self.start_time
        remaining_progress = 100 - current_progress
        
        # Ako je proteklo više od 1 sekunde, koristi linearnu ekstrapolaciju
        # (pouzdanija jer ima više podataka)
        if elapsed > 1.0:
            time_per_percent = elapsed / current_progress
            eta_seconds = int(remaining_progress * time_per_percent)
        else:
            # Za brze faze koristi procijenjeno vrijeme bazirano na file size
            eta_seconds = int(self.estimated_total_time * remaining_progress / 100)
        
        # Ograniči na razumne vrijednosti
        return max(0, min(eta_seconds, 3600))  # Max 1 sat

    def emit(self, step: str, progress: float, message_key: str, eta_seconds: Optional[int] = None, force: bool = False, processed_rows: Optional[int] = None, message_params: Optional[Dict[str, Any]] = None) -> None:
        """
        Šalje progress update preko WebSocket-a sa ključem za prevod.

        Args:
            step: Naziv faze (validating, parsing, datetime, utc, build, streaming, complete, error)
            progress: Procenat (0-100)
            message_key: Ključ za prevod na frontendu (npr. 'validating_params', 'parsing_csv')
            eta_seconds: ETA u sekundama (opciono, ako nije proslijeđeno pokušava izračunati)
            force: Ignoriši rate limiting
            processed_rows: Broj obrađenih redova (opciono)
            message_params: Dodatni parametri za poruku (npr. {'rowCount': 1000})
        """
        current_time = time.time()

        # Rate limiting - ne šalje češće od emit_interval osim ako je force=True
        if not force and (current_time - self.last_emit_time) < self.emit_interval:
            return

        payload = {
            'uploadId': self.upload_id,
            'step': step,
            'progress': int(progress),
            'messageKey': message_key,
            'status': 'processing' if step not in ['complete', 'error'] else ('completed' if step == 'complete' else 'error')
        }

        # Dodaj parametre za poruku ako postoje
        if message_params:
            payload['messageParams'] = message_params

        # Dodaj currentStep i totalSteps ako su postavljeni
        if self.total_steps > 0:
            payload['currentStep'] = self.current_step
            payload['totalSteps'] = self.total_steps

        # Dodaj totalRows i processedRows ako su dostupni
        if self.total_rows > 0:
            payload['totalRows'] = self.total_rows
        if processed_rows is not None:
            payload['processedRows'] = processed_rows

        # Dodaj ETA
        if eta_seconds is not None:
            payload['eta'] = eta_seconds
            payload['etaFormatted'] = self.format_time(eta_seconds)
        else:
            progress_eta = self.calculate_eta_for_progress(progress)
            payload['eta'] = progress_eta
            payload['etaFormatted'] = self.format_time(progress_eta)

        try:
            if self.socketio:
                self.socketio.emit('upload_progress', payload, room=self.upload_id)
            self.last_emit_time = current_time

        except Exception:
            pass

    @staticmethod
    def format_time(seconds: Optional[int]) -> str:
        """Formatira sekunde u čitljiv format"""
        if seconds is None:
            return "Procjenjujem..."
        if seconds < 60:
            return f"{seconds}s"
        elif seconds < 3600:
            minutes = seconds // 60
            secs = seconds % 60
            return f"{minutes}m {secs}s"
        else:
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            return f"{hours}h {minutes}m"


# ============================================================================
# DateTimeParser Class - Consolidated datetime parsing logic
# ============================================================================

class DateTimeParser:
    """
    Centralized datetime parsing with format detection and validation.
    
    Consolidates logic from multiple parsing functions to eliminate duplication
    and provide a single source of truth for datetime operations.
    """
    
    def __init__(self, supported_formats: Optional[List[str]] = None):
        """
        Initialize parser with supported formats.
        
        Args:
            supported_formats: List of datetime format strings. 
                             If None, uses SUPPORTED_DATE_FORMATS constant.
        """
        self.formats = supported_formats or SUPPORTED_DATE_FORMATS
        
    def detect_format(self, sample: str) -> Optional[str]:
        """
        Detect which format matches the sample datetime string.
        
        Args:
            sample: Sample datetime string to check
            
        Returns:
            Matching format string, or None if no format matches
        """
        if not isinstance(sample, str):
            sample = str(sample)
        
        sample = sample.strip()
        
        for fmt in self.formats:
            try:
                pd.to_datetime(sample, format=fmt)
                return fmt
            except (ValueError, TypeError):
                continue
        
        return None
    
    def is_supported(self, sample: str) -> bool:
        """
        Check if datetime string format is supported.
        
        Args:
            sample: Datetime string to validate
            
        Returns:
            True if format is supported, False otherwise
        """
        return self.detect_format(sample) is not None
    
    def validate_format(self, sample: str) -> Tuple[bool, Optional[Dict[str, str]]]:
        """
        Validate datetime format with error information.
        
        Args:
            sample: Sample datetime string to validate
            
        Returns:
            Tuple of (is_valid, error_dict_if_invalid)
        """
        if self.is_supported(sample):
            return True, None
        
        return False, {
            "error": "UNSUPPORTED_DATE_FORMAT",
            "message": "Format nicht unterstützt. Beispielwert: %d.%m.%Y %H:%M:%S"
        }
    
    def parse_series(
        self, 
        series: pd.Series,
        custom_format: Optional[str] = None
    ) -> Tuple[bool, Optional[pd.Series], Optional[str]]:
        """
        Parse datetime series using custom or auto-detected format.
        
        Args:
            series: Pandas Series containing datetime strings
            custom_format: Optional custom datetime format string
            
        Returns:
            Tuple of (success, parsed_series, error_message)
        """
        try:
            # Clean and prepare series
            clean_series = series.astype(str).str.strip()
            sample_value = clean_series.iloc[0]
            
            # Try custom format first if provided
            if custom_format:
                try:
                    parsed = pd.to_datetime(clean_series, format=custom_format, errors='coerce')
                    if not parsed.isna().all():
                        return True, parsed, None
                except Exception as e:
                    return False, None, f"Fehler mit custom Format: {str(e)}. Beispielwert: {sample_value}"
            
            # Try all supported formats
            for fmt in self.formats:
                try:
                    parsed = pd.to_datetime(clean_series, format=fmt, errors='coerce')
                    if not parsed.isna().all():
                        return True, parsed, None
                except Exception:
                    continue
            
            return False, None, "Format nicht unterstützt. Beispielwert: %d.%m.%Y %H:%M:%S"
            
        except Exception as e:
            return False, None, f"Fehler beim Parsen: {str(e)}"
    
    def parse_combined_columns(
        self,
        df: pd.DataFrame,
        date_column: str,
        time_column: str,
        custom_format: Optional[str] = None
    ) -> Tuple[bool, Optional[pd.Series], Optional[str]]:
        """
        Combine separate date and time columns and parse to datetime.
        
        Args:
            df: DataFrame containing date and time columns
            date_column: Name of the date column
            time_column: Name of the time column
            custom_format: Optional custom datetime format string
            
        Returns:
            Tuple of (success, parsed_series, error_message)
        """
        try:
            # Combine date + time columns
            combined = (
                df[date_column].astype(str).str.strip() + ' ' +
                df[time_column].astype(str).str.strip()
            )
            
            # Parse combined series
            return self.parse_series(combined, custom_format)
            
        except Exception as e:
            return False, None, f"Fehler beim Kombinieren von Datum/Zeit: {str(e)}"
    
    def convert_to_utc(
        self, 
        series: pd.Series, 
        source_timezone: str = 'UTC'
    ) -> pd.Series:
        """
        Convert datetime series to UTC timezone.
        
        If series has no timezone info, localizes to source_timezone first,
        then converts to UTC.
        
        Args:
            series: Pandas Series with datetime values
            source_timezone: Source timezone (default: 'UTC')
            
        Returns:
            Series with UTC-converted datetime values
            
        Raises:
            ValueError: If timezone is not supported
        """
        try:
            # Ensure series is datetime type
            if not pd.api.types.is_datetime64_any_dtype(series):
                series = pd.to_datetime(series, errors='coerce')
            
            # Localize if no timezone info
            if series.dt.tz is None:
                try:
                    series = series.dt.tz_localize(
                        source_timezone,
                        ambiguous='NaT',
                        nonexistent='NaT'
                    )
                except Exception as e:
                    raise UnsupportedTimezoneError(
                        timezone=source_timezone,
                        original_exception=e
                    )
                
                # Convert to UTC if not already
                if source_timezone.upper() != 'UTC':
                    series = series.dt.tz_convert('UTC')
            else:
                # Already has timezone, convert to UTC if needed
                if str(series.dt.tz) != 'UTC':
                    series = series.dt.tz_convert('UTC')
            
            return series
            
        except Exception as e:
            raise


# Global parser instance
_datetime_parser = DateTimeParser()

# ============================================================================
# BLUEPRINT SETUP
# ============================================================================

def get_socketio():
    """Get the SocketIO instance from the Flask app extensions."""
    return current_app.extensions['socketio']


bp = Blueprint('load_row_data', __name__)


# ============================================================================
# MODULE STATE (TODO: Consider moving to proper state management)
# ============================================================================



# ============================================================================
# UploadStateManager Class - Thread-safe state management
# ============================================================================

class UploadStateManager:
    """
    Thread-safe state manager for upload chunks and temporary files.
    
    Replaces global dict-based state with proper encapsulation and thread safety.
    Handles both chunked uploads and temporary file storage.
    """
    
    def __init__(self):
        """Initialize state manager with thread locks."""
        self._chunk_storage: Dict[str, Dict[str, Any]] = {}
        self._temp_files: Dict[str, Dict[str, Any]] = {}
        self._chunk_lock = threading.Lock()
        self._temp_lock = threading.Lock()
    
    # ========================================================================
    # Chunk Storage Operations
    # ========================================================================
    
    def create_upload(
        self, 
        upload_id: str, 
        total_chunks: int,
        parameters: Dict[str, Any]
    ) -> None:
        """
        Create new upload session.
        
        Args:
            upload_id: Unique upload identifier
            total_chunks: Total number of chunks expected
            parameters: Upload parameters (delimiter, timezone, etc.)
        """
        with self._chunk_lock:
            self._chunk_storage[upload_id] = {
                'chunks': {},
                'total_chunks': total_chunks,
                'received_chunks': 0,
                'last_activity': time.time(),
                'parameters': parameters
            }
    
    def upload_exists(self, upload_id: str) -> bool:
        """Check if upload session exists."""
        with self._chunk_lock:
            return upload_id in self._chunk_storage
    
    def get_upload(self, upload_id: str) -> Optional[Dict[str, Any]]:
        """
        Get upload session data.
        
        Args:
            upload_id: Upload identifier
            
        Returns:
            Upload data dict or None if not found
        """
        with self._chunk_lock:
            return self._chunk_storage.get(upload_id)
    
    def store_chunk(
        self, 
        upload_id: str, 
        chunk_index: int, 
        chunk_data: bytes
    ) -> None:
        """
        Store uploaded chunk.
        
        Args:
            upload_id: Upload identifier
            chunk_index: Index of the chunk
            chunk_data: Chunk binary data
        """
        with self._chunk_lock:
            if upload_id in self._chunk_storage:
                self._chunk_storage[upload_id]['chunks'][chunk_index] = chunk_data
                self._chunk_storage[upload_id]['received_chunks'] += 1
                self._chunk_storage[upload_id]['last_activity'] = time.time()
    
    def update_total_chunks(self, upload_id: str, total_chunks: int) -> None:
        """Update total chunks count (handles retries)."""
        with self._chunk_lock:
            if upload_id in self._chunk_storage:
                current = self._chunk_storage[upload_id]['total_chunks']
                self._chunk_storage[upload_id]['total_chunks'] = max(current, total_chunks)
    
    def get_chunk_progress(self, upload_id: str) -> tuple[int, int]:
        """
        Get upload progress.
        
        Returns:
            Tuple of (received_chunks, total_chunks). Returns (0, 0) if upload not found.
        """
        with self._chunk_lock:
            if upload_id in self._chunk_storage:
                data = self._chunk_storage[upload_id]
                return (data['received_chunks'], data['total_chunks'])
        return (0, 0)
    
    def is_upload_complete(self, upload_id: str) -> bool:
        """Check if all chunks have been received."""
        with self._chunk_lock:
            if upload_id in self._chunk_storage:
                data = self._chunk_storage[upload_id]
                return data['received_chunks'] == data['total_chunks']
        return False
    
    def get_chunks(self, upload_id: str) -> Optional[Dict[int, bytes]]:
        """
        Get chunks dictionary.
        
        Returns:
            Dictionary mapping chunk indices to chunk data, or None if upload not found
        """
        with self._chunk_lock:
            if upload_id not in self._chunk_storage:
                return None
            
            return self._chunk_storage[upload_id]['chunks']
    
    def get_parameters(self, upload_id: str) -> Optional[Dict[str, Any]]:
        """Get upload parameters."""
        with self._chunk_lock:
            if upload_id in self._chunk_storage:
                return self._chunk_storage[upload_id]['parameters']
        return None
    
    def delete_upload(self, upload_id: str) -> bool:
        """
        Delete upload session.
        
        Returns:
            True if deleted, False if not found
        """
        with self._chunk_lock:
            if upload_id in self._chunk_storage:
                del self._chunk_storage[upload_id]
                return True
        return False
    
    def cleanup_expired_uploads(self, expiry_seconds: int = UPLOAD_EXPIRY_SECONDS) -> int:
        """
        Clean up expired uploads.
        
        Args:
            expiry_seconds: Expiration time in seconds
            
        Returns:
            Number of uploads cleaned up
        """
        current_time = time.time()
        expired_ids = []
        
        with self._chunk_lock:
            for upload_id, data in list(self._chunk_storage.items()):
                last_activity = data.get('last_activity', 0)
                if current_time - last_activity > expiry_seconds:
                    expired_ids.append(upload_id)
            
            for upload_id in expired_ids:
                del self._chunk_storage[upload_id]
        
        return len(expired_ids)
    
    # ========================================================================
    # Temporary File Operations
    # ========================================================================
    
    def store_temp_file(
        self, 
        file_id: str, 
        file_path: str, 
        file_name: str
    ) -> None:
        """
        Store temporary file information.
        
        Args:
            file_id: Unique file identifier
            file_path: Path to temporary file
            file_name: Original file name
        """
        with self._temp_lock:
            self._temp_files[file_id] = {
                'path': file_path,
                'fileName': file_name,
                'timestamp': time.time()
            }
    
    def get_temp_file(self, file_id: str) -> Optional[Dict[str, Any]]:
        """
        Get temporary file information.
        
        Returns:
            File info dict or None if not found
        """
        with self._temp_lock:
            return self._temp_files.get(file_id)
    
    def delete_temp_file(self, file_id: str) -> bool:
        """
        Delete temporary file record.
        
        Returns:
            True if deleted, False if not found
        """
        with self._temp_lock:
            if file_id in self._temp_files:
                del self._temp_files[file_id]
                return True
        return False
    
    # ========================================================================
    # Utility Methods
    # ========================================================================
    
    def get_all_upload_ids(self) -> List[str]:
        """Get list of all active upload IDs."""
        with self._chunk_lock:
            return list(self._chunk_storage.keys())
    
    def clear_all(self, storage_type: str = None) -> None:
        """Clear all state (for testing purposes).
        
        Args:
            storage_type: Optional. If 'chunk', clears only chunk storage.
                         If 'temp', clears only temp files.
                         If None, clears both.
        """
        if storage_type is None or storage_type == 'chunk':
            with self._chunk_lock:
                self._chunk_storage.clear()
        
        if storage_type is None or storage_type == 'temp':
            with self._temp_lock:
                self._temp_files.clear()


# Global state manager instance
_upload_state = UploadStateManager()

# Legacy global dictionaries - now delegating to UploadStateManager for thread-safety
# These maintain backward compatibility while providing thread-safe operations
class _StateProxy:
    """Proxy to maintain backward compatibility while using UploadStateManager."""
    def __init__(self, state_manager: UploadStateManager, storage_type: str):
        self._state = state_manager
        self._type = storage_type
    
    def __getitem__(self, key: str) -> Dict[str, Any]:
        if self._type == 'chunk':
            result = self._state.get_upload(key)
            if result is None:
                raise KeyError(key)
            return result
        else:  # temp
            result = self._state.get_temp_file(key)
            if result is None:
                raise KeyError(key)
            return result
    
    def __setitem__(self, key: str, value: Dict[str, Any]) -> None:
        if self._type == 'chunk':
            params = value.get('parameters', {})
            total_chunks = value.get('total_chunks', 0)
            self._state.create_upload(key, total_chunks, params)
            
            # If chunks are provided (for testing), store them
            if 'chunks' in value:
                for chunk_idx, chunk_data in value['chunks'].items():
                    self._state.store_chunk(key, chunk_idx, chunk_data)
            
            # Update fields that may be provided (for testing)
            with self._state._chunk_lock:
                if key in self._state._chunk_storage:
                    if 'received_chunks' in value:
                        self._state._chunk_storage[key]['received_chunks'] = value['received_chunks']
                    if 'last_activity' in value:
                        self._state._chunk_storage[key]['last_activity'] = value['last_activity']
        else:  # temp
            path = value.get('path', '')
            fileName = value.get('fileName', '')
            self._state.store_temp_file(key, path, fileName)
    
    def __delitem__(self, key: str) -> None:
        if self._type == 'chunk':
            self._state.delete_upload(key)
        else:  # temp
            self._state.delete_temp_file(key)
    
    def __contains__(self, key: str) -> bool:
        if self._type == 'chunk':
            return self._state.upload_exists(key)
        else:  # temp
            return self._state.get_temp_file(key) is not None
    
    def get(self, key: str, default=None):
        try:
            return self[key]
        except KeyError:
            return default
    
    def items(self):
        if self._type == 'chunk':
            return [(uid, self._state.get_upload(uid)) for uid in self._state.get_all_upload_ids()]
        return []

    def clear(self) -> None:
        """Clear all items from storage."""
        self._state.clear_all(self._type)

    def __len__(self) -> int:
        """Return number of items in storage."""
        if self._type == 'chunk':
            return len(self._state.get_all_upload_ids())
        else:  # temp
            with self._state._temp_lock:
                return len(self._state._temp_files)

chunk_storage = _StateProxy(_upload_state, 'chunk')
temp_files = _StateProxy(_upload_state, 'temp')

def cleanup_old_uploads() -> None:
    """
    Clean up old incomplete uploads and expired temp files.
    
    Removes upload chunks that haven't been active within UPLOAD_EXPIRY_SECONDS.
    Also removes temp files older than 30 minutes.
    Should be called periodically to prevent memory leaks.
    """
    _upload_state.cleanup_expired_uploads()
    
    # Cleanup expired temp files (older than 30 minutes)
    current_time = time.time()
    expired_files = []
    
    for file_id, file_info in list(temp_files.items()):
        if current_time - file_info.get('timestamp', 0) > 1800:  # 30 minutes
            expired_files.append(file_id)
    
    for file_id in expired_files:
        try:
            file_path = temp_files[file_id].get('path')
            if file_path and os.path.exists(file_path):
                os.unlink(file_path)
            del temp_files[file_id]
        except Exception:
            pass


def _error_response(error_code: str, message: str, status_code: int = 400) -> Tuple[Response, int]:
    """
    Create standardized error response.
    
    Args:
        error_code: Machine-readable error code
        message: Human-readable error message
        status_code: HTTP status code (default: 400)
        
    Returns:
        Tuple of (JSON response, status code)
    """
    return jsonify({"error": error_code, "message": message}), status_code


def check_date_format(sample_date: Any) -> Tuple[bool, Optional[Dict[str, str]]]:
    """
    Check if date format is supported.
    
    Args:
        sample_date: Sample date value to check
        
    Returns:
        Tuple of (is_supported, error_dict_if_not_supported)
    """
    if not isinstance(sample_date, str):
        sample_date = str(sample_date)
    
    return _datetime_parser.validate_format(sample_date)

def detect_delimiter(file_content: str, sample_lines: int = 5) -> str:
    """
    Detect CSV delimiter from file content using consistency check.
    
    A proper delimiter should produce the same number of columns across all lines.
    
    Args:
        file_content: CSV file content as string
        sample_lines: Number of lines to sample for detection
        
    Returns:
        Detected delimiter character
    """
    lines = [l for l in file_content.splitlines()[:sample_lines] if l.strip()]
    
    if not lines:
        return DEFAULT_DELIMITER
    
    best_delimiter = DEFAULT_DELIMITER
    best_score = -1
    
    for delimiter in SUPPORTED_DELIMITERS:
        counts = [line.count(delimiter) + 1 for line in lines]
        
        if not counts or counts[0] < 2:
            continue
            
        if len(set(counts)) == 1:
            score = counts[0]
            if score > best_score:
                best_score = score
                best_delimiter = delimiter
    
    return best_delimiter

def clean_time(time_str: Any) -> Any:
    """
    Clean time string by removing invalid characters.

    Keeps only numbers and time separators (: - + . T / and space).
    Example: '00:00:00.000Kdd' -> '00:00:00.000'

    Args:
        time_str: Time string to clean

    Returns:
        Cleaned time string
    """
    if not isinstance(time_str, str):
        return time_str

    cleaned = ''.join(c for c in str(time_str) if c.isdigit() or c in ':-+.T/ ')
    return cleaned

def clean_file_content(file_content: str, delimiter: str) -> str:
    """
    Remove excess delimiters and whitespace from file content.
    
    Args:
        file_content: Raw CSV file content
        delimiter: CSV delimiter character
        
    Returns:
        Cleaned file content
    """
    cleaned_lines = [line.rstrip(f"{delimiter};,") for line in file_content.splitlines()]
    return "\n".join(cleaned_lines)

def parse_datetime_column(
    df: pd.DataFrame, 
    datetime_col: str, 
    custom_format: Optional[str] = None
) -> Tuple[bool, Optional[pd.Series], Optional[str]]:
    """
    Parse datetime column using custom or supported formats.
    
    Args:
        df: DataFrame containing the datetime column
        datetime_col: Name of the datetime column
        custom_format: Optional custom datetime format string
        
    Returns:
        Tuple of (success, parsed_dates_series, error_message)
    """
    try:
        return _datetime_parser.parse_series(df[datetime_col], custom_format)
    except Exception as e:
        return False, None, f"Fehler beim Parsen: {str(e)}"

def is_format_supported(value: Any, formats: List[str]) -> Tuple[bool, Optional[str]]:
    """
    Check if value matches any of the provided formats.
    
    Args:
        value: Value to check
        formats: List of datetime format strings
        
    Returns:
        Tuple of (is_supported, matching_format)
    """
    if not isinstance(value, str):
        value = str(value)
    
    # Use custom parser with provided formats
    parser = DateTimeParser(supported_formats=formats)
    detected_format = parser.detect_format(value)
    
    if detected_format:
        return True, detected_format
    return False, None

def parse_datetime(
    df: pd.DataFrame, 
    date_column: str, 
    time_column: str, 
    custom_format: Optional[str] = None
) -> pd.DataFrame:
    """
    Combine separate date and time columns into single datetime column.
    
    Args:
        df: DataFrame with date and time columns
        date_column: Name of date column
        time_column: Name of time column
        custom_format: Optional custom datetime format
        
    Returns:
        DataFrame with added 'datetime' column
        
    Raises:
        ValueError: If datetime parsing fails
    """
    success, parsed_series, error_msg = _datetime_parser.parse_combined_columns(
        df, date_column, time_column, custom_format
    )
    
    if not success:
        raise DateTimeParsingError(
            column=f"{date_column}+{time_column}",
            format_info=error_msg or "Fehler beim Parsen von Datum/Zeit"
        )
    
    df = df.copy()
    df['datetime'] = parsed_series
    return df

def validate_datetime_format(datetime_str: Any) -> bool:
    """
    Validate if datetime string format is supported.
    
    Args:
        datetime_str: Datetime string to validate
        
    Returns:
        True if format is supported, False otherwise
    """
    if not isinstance(datetime_str, str):
        datetime_str = str(datetime_str)
    
    return _datetime_parser.is_supported(datetime_str)

def convert_to_utc(df: pd.DataFrame, date_column: str, timezone: str = 'UTC') -> pd.DataFrame:
    """
    Convert datetime column to UTC timezone.
    
    If datetime has no timezone, localizes it to the specified timezone first.
    
    Args:
        df: DataFrame with datetime column
        date_column: Name of the datetime column
        timezone: Source timezone (default: 'UTC')
        
    Returns:
        DataFrame with UTC-converted datetime column
        
    Raises:
        ValueError: If timezone is not supported
    """
    try:
        df = df.copy()
        df[date_column] = _datetime_parser.convert_to_utc(df[date_column], timezone)
        return df
    except UnsupportedTimezoneError:
        # Re-raise timezone errors
        raise
    except Exception as e:
        # Wrap other errors in TimezoneConversionError
        raise TimezoneConversionError(
            from_tz=timezone,
            to_tz='UTC',
            reason=str(e),
            original_exception=e
        )

@bp.route('/upload-chunk', methods=['POST'])
@require_auth
def upload_chunk() -> Tuple[Response, int]:
    """
    Handle chunked file upload.
    
    Receives and stores individual file chunks. Tracks upload progress via WebSocket.
    
    Expected form parameters:
        - fileChunk: File chunk data
        - uploadId: Unique upload identifier
        - chunkIndex: Index of this chunk
        - totalChunks: Total number of chunks
        - delimiter: CSV delimiter
        - selected_columns: JSON string of selected columns
        - timezone: Timezone for date conversion
        - dropdown_count: Number of column dropdowns (2 or 3)
        - hasHeader: Whether CSV has header row
        
    Returns:
        JSON response with upload status and remaining chunks
    """
    try:
        if 'fileChunk' not in request.files:
            return jsonify({"error": "Chunk file not found"}), 400
        
        required_params = ['uploadId', 'chunkIndex', 'totalChunks', 'delimiter', 'selected_columns', 'timezone', 'dropdown_count', 'hasHeader']
        missing_params = [param for param in required_params if param not in request.form]
        if missing_params:
            return jsonify({"error": f"Missing required parameters: {', '.join(missing_params)}"}), 400
        
        upload_id = request.form['uploadId']
        chunk_index = int(request.form['chunkIndex'])
        total_chunks = int(request.form['totalChunks'])
        
        if upload_id not in chunk_storage:
            try:
                selected_columns_str = request.form.get('selected_columns')
                if not selected_columns_str:
                    return jsonify({"error": "selected_columns parameter is required"}), 400
                selected_columns = json.loads(selected_columns_str)
                if not isinstance(selected_columns, dict):
                    return jsonify({"error": "selected_columns must be a JSON object"}), 400
                
                chunk_storage[upload_id] = {
                    'chunks': {},
                    'total_chunks': total_chunks,
                    'received_chunks': 0,
                    'last_activity': time.time(),
                    'parameters': {
                        'delimiter': request.form.get('delimiter'),
                        'timezone': request.form.get('timezone', 'UTC'),
                        'has_header': request.form.get('hasHeader', 'nein'),
                        'selected_columns': selected_columns,
                        'custom_date_format': request.form.get('custom_date_format'),
                        'value_column_name': request.form.get('valueColumnName', '').strip(),
                        'dropdown_count': int(request.form.get('dropdown_count', '2'))
                    }
                }
            except json.JSONDecodeError as e:
                return jsonify({"error": "Invalid JSON format for selected_columns"}), 400
        else:
            chunk_storage[upload_id]['total_chunks'] = max(chunk_storage[upload_id]['total_chunks'], total_chunks)

        file_chunk = request.files['fileChunk']
        chunk_content = file_chunk.read()
        chunk_storage[upload_id]['chunks'][chunk_index] = chunk_content
        chunk_storage[upload_id]['received_chunks'] += 1
        chunk_storage[upload_id]['last_activity'] = time.time()
        
        # Note: Frontend handles its own upload progress (0-95%)
        # Backend ProgressTracker handles processing progress (5-100%) with status='processing'
        # We don't emit progress here to avoid conflicts with frontend's chunk tracking
        
        return jsonify({
            "message": f"Chunk {chunk_index + 1}/{chunk_storage[upload_id]['total_chunks']} received",
            "uploadId": upload_id,
            "remainingChunks": chunk_storage[upload_id]['total_chunks'] - chunk_storage[upload_id]['received_chunks']
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@bp.route('/finalize-upload', methods=['POST'])
@require_auth
@require_subscription
@check_processing_limit
def finalize_upload() -> Tuple[Response, int]:
    """
    Finalize chunked upload and process the complete file.

    Verifies all chunks are received and triggers processing.

    Expected JSON body:
        - uploadId: Unique upload identifier

    Returns:
        JSON response with processed data or error
    """
    try:
        data = request.get_json(force=True, silent=True)

        if not data or 'uploadId' not in data:
            return jsonify({"error": "uploadId is required"}), 400

        upload_id = data['uploadId']

        if upload_id not in chunk_storage:
            return jsonify({"error": "Upload not found or already processed"}), 404

        if chunk_storage[upload_id]['received_chunks'] != chunk_storage[upload_id]['total_chunks']:
            remaining = chunk_storage[upload_id]['total_chunks'] - chunk_storage[upload_id]['received_chunks']
            return jsonify({
                "error": f"Not all chunks received. Missing {remaining} chunks.",
                "received": chunk_storage[upload_id]['received_chunks'],
                "total": chunk_storage[upload_id]['total_chunks']
            }), 400

        return process_chunks(upload_id)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@bp.route('/cancel-upload', methods=['POST'])
@require_auth
def cancel_upload() -> Tuple[Response, int]:
    """
    Cancel an in-progress upload.
    
    Removes upload data and notifies via WebSocket.
    
    Expected JSON body:
        - uploadId: Upload identifier to cancel
        
    Returns:
        JSON response confirming cancellation
    """
    try:
        data = request.get_json(force=True, silent=True)
        
        if not data or 'uploadId' not in data:
            return jsonify({"error": "uploadId is required"}), 400
            
        upload_id = data['uploadId']
        
        if upload_id in chunk_storage:
            del chunk_storage[upload_id]
            
            socketio = get_socketio()
            socketio.emit('upload_progress', {
                'uploadId': upload_id,
                'progress': 0,
                'status': 'error',
                'message': 'Upload canceled by user'
            }, room=upload_id)
        
        return jsonify({
            "success": True,
            "message": "Upload canceled successfully"
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400

def process_chunks(upload_id: str) -> Tuple[Response, int]:
    """
    Process and decode uploaded file chunks.

    Combines chunks, attempts decoding with multiple encodings, and triggers file processing.

    Args:
        upload_id: Unique identifier for the upload

    Returns:
        JSON response from upload_files() processing
    """
    try:
        socketio = get_socketio()
        
        # Note: Don't emit progress here - upload_files has its own ProgressTracker
        # that handles all processing progress (5-100%)

        upload_data = chunk_storage[upload_id]
        chunks = [upload_data['chunks'][i] for i in range(upload_data['total_chunks'])]
        
        combined_bytes = b"".join(chunks)
        
        encodings = ['utf-8', 'latin1', 'cp1252', 'utf-16', 'utf-16le', 'utf-16be']
        full_content = None

        for encoding in encodings:
            try:
                decoded = combined_bytes.decode(encoding)
                
                first_line = decoded.split('\n')[0] if decoded else ''
                
                has_delimiter = any(d in first_line for d in [',', ';', '\t'])
                printable_ratio = sum(1 for c in first_line[:200] if ord(c) < 256 and (c.isprintable() or c in '\n\r\t')) / max(len(first_line[:200]), 1)
                
                if has_delimiter and printable_ratio > 0.9:
                    full_content = decoded
                    break
                    
            except UnicodeDecodeError:
                continue

        if full_content is None:
            error = EncodingError(
                reason="Could not decode file content with any supported encoding",
                details={'tried_encodings': encodings}
            )
            socketio.emit('upload_progress', {
                'uploadId': upload_id,
                'progress': 0,
                'status': 'error',
                'message': error.message
            }, room=upload_id)
            return jsonify(error.to_dict()), 400

        params = upload_data['parameters']
        params['uploadId'] = upload_id
        del chunk_storage[upload_id]

        # Don't emit 'completed' here - upload_files will handle progress from here
        return upload_files(full_content, params)
    except LoadDataException as e:
        from flask import has_app_context
        if has_app_context():
            return jsonify(e.to_dict()), 400
        else:
            raise
    except KeyError:
        error = UploadNotFoundError(upload_id)
        from flask import has_app_context
        if has_app_context():
            return jsonify(error.to_dict()), 404
        else:
            raise error
    except Exception as e:
        from flask import has_app_context
        if has_app_context():
            return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500
        else:
            raise

def check_upload_status(upload_id: str) -> Tuple[Response, int]:
    """
    Check the status of an upload.
    
    Args:
        upload_id: Upload identifier to check
        
    Returns:
        JSON response with upload progress information
    """
    try:
        if upload_id not in chunk_storage:
            return jsonify({
                "error": "Upload not found or already completed"
            }), 404
            
        upload_info = chunk_storage[upload_id]
        return jsonify({
            "success": True,
            "totalChunks": upload_info['total_chunks'],
            "receivedChunks": upload_info['received_chunks'],
            "isComplete": upload_info['received_chunks'] == upload_info['total_chunks']
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 400

def _validate_and_extract_params(
    params: Dict[str, Any], 
    file_content: str
) -> Dict[str, Any]:
    """
    Validate and extract parameters from upload request.
    
    Args:
        params: Raw parameters from request
        file_content: CSV file content for delimiter validation
        
    Returns:
        Dictionary with validated parameters
        
    Raises:
        ValueError: If validation fails
    """
    delimiter = params.get('delimiter')
    if not delimiter:
        raise MissingParameterError('delimiter')
    
    # Validate delimiter against detected
    detected_delimiter = detect_delimiter(file_content)
    if delimiter != detected_delimiter:
        raise DelimiterMismatchError(
            provided=delimiter,
            detected=detected_delimiter
        )
    
    timezone = params.get('timezone', 'UTC')
    selected_columns = params.get('selected_columns', {})
    custom_date_format = params.get('custom_date_format')
    value_column_name = params.get('value_column_name', '').strip()
    dropdown_count = int(params.get('dropdown_count', '2'))
    has_separate_date_time = dropdown_count == 3
    has_header = params.get('has_header', False)
    upload_id = params.get('uploadId')
    
    date_column = selected_columns.get('column1')
    time_column = selected_columns.get('column2') if has_separate_date_time else None
    value_column = (
        selected_columns.get('column3') if has_separate_date_time 
        else selected_columns.get('column2')
    )
    
    return {
        'upload_id': upload_id,
        'delimiter': delimiter,
        'timezone': timezone,
        'custom_date_format': custom_date_format,
        'value_column_name': value_column_name,
        'has_separate_date_time': has_separate_date_time,
        'has_header': has_header,
        'date_column': date_column,
        'time_column': time_column,
        'value_column': value_column,
    }


def _parse_csv_to_dataframe(
    file_content: str,
    validated_params: Dict[str, Any]
) -> pd.DataFrame:
    """
    Parse CSV content to pandas DataFrame.
    
    Args:
        file_content: Raw CSV file content
        validated_params: Validated parameters dict
        
    Returns:
        Parsed and cleaned DataFrame
        
    Raises:
        ValueError: If CSV parsing fails or data is empty
    """
    delimiter = validated_params['delimiter']
    has_header = validated_params['has_header']
    value_column = validated_params['value_column']
    
    cleaned_content = clean_file_content(file_content, delimiter)
    
    try:
        df = pd.read_csv(
            StringIO(cleaned_content),
            delimiter=delimiter,
            header=0 if has_header == 'ja' else None
        )
        
        if has_header == 'nein':
            df.columns = [str(i) for i in range(len(df.columns))]
        else:
            df.columns = [col.strip() for col in df.columns]
        
        df = df.dropna(axis=1, how='all')
        df.columns = df.columns.astype(str)
        df.columns = [col.strip() for col in df.columns]
        
    except Exception as e:
        raise CSVParsingError(
            reason=str(e),
            original_exception=e
        )
    
    if df.empty:
        raise CSVParsingError(reason="No data loaded from file")
    
    # Convert value column to numeric
    if value_column and value_column in df.columns:
        df[value_column] = pd.to_numeric(df[value_column], errors='coerce')
    
    return df


def _process_datetime_columns(
    df: pd.DataFrame,
    validated_params: Dict[str, Any]
) -> pd.DataFrame:
    """
    Process and parse datetime columns in DataFrame.
    
    Handles both separate date/time columns and combined datetime column.
    
    Args:
        df: DataFrame with date/time columns
        validated_params: Validated parameters dict
        
    Returns:
        DataFrame with added 'datetime' column (datetime64 type)
        
    Raises:
        ValueError: If datetime parsing fails
    """
    has_separate_date_time = validated_params['has_separate_date_time']
    date_column = validated_params['date_column']
    time_column = validated_params['time_column']
    custom_date_format = validated_params['custom_date_format']
    
    try:
        datetime_col = date_column or df.columns[0]
        
        if has_separate_date_time and date_column and time_column:
            # Clean time columns
            df[time_column] = df[time_column].apply(clean_time)
            df[date_column] = df[date_column].apply(clean_time)

            # Extract date-only part if date_column contains datetime with dummy time
            # Check if date column contains time separator (space or 'T')
            sample_date = str(df[date_column].iloc[0])
            if ' ' in sample_date or 'T' in sample_date:
                # Split by space or 'T' to get only date part
                df['date_only'] = df[date_column].astype(str).str.split(' ').str[0].str.split('T').str[0]
            else:
                df['date_only'] = df[date_column].astype(str)

            # Combine date + time
            df['datetime'] = (
                df['date_only'] + ' ' +
                df[time_column].astype(str)
            )

            # Try parsing
            success, parsed_dates, err = parse_datetime_column(df, 'datetime')
            
            # Retry with custom format if needed
            if not success and custom_date_format:
                success, parsed_dates, err = parse_datetime_column(
                    df, 'datetime', custom_format=custom_date_format
                )
            
            if not success:
                raise DateTimeParsingError(
                    column='datetime',
                    format_info="Format nicht unterstützt. Beispielwert: %d.%m.%Y %H:%M:%S"
                )
        else:
            # Single datetime column
            success, parsed_dates, err = parse_datetime_column(
                df, datetime_col, custom_format=custom_date_format
            )
            
            if not success:
                raise DateTimeParsingError(
                    column=datetime_col,
                    format_info=err or "Unsupported datetime format"
                )
        
        df['datetime'] = parsed_dates
        return df
        
    except DateTimeParsingError:
        # Re-raise custom exceptions
        raise
    except Exception as e:
        # Wrap other exceptions in DateTimeParsingError
        raise DateTimeParsingError(
            format_info=f"Error parsing date/time: {str(e)}",
            original_exception=e
        )


def _build_result_dataframe(
    df: pd.DataFrame,
    validated_params: Dict[str, Any]
) -> List[List[str]]:
    """
    Build final result data structure from processed DataFrame.
    
    Args:
        df: Processed DataFrame with 'datetime' and value columns
        validated_params: Validated parameters dict
        
    Returns:
        List of lists: [headers_row, data_row1, data_row2, ...]
        
    Raises:
        ValueError: If required columns are missing
    """
    value_column = validated_params['value_column']
    value_column_name = validated_params['value_column_name']
    
    if not value_column or value_column not in df.columns:
        raise ValueError("Datum, Wert 1 oder Wert 2 nicht ausgewählt")
    
    result_df = pd.DataFrame()
    result_df['UTC'] = df['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    final_value_column = value_column_name if value_column_name else value_column
    result_df[final_value_column] = df[value_column].apply(
        lambda x: str(x) if pd.notnull(x) else ""
    )
    
    result_df.dropna(subset=['UTC'], inplace=True)
    result_df.sort_values('UTC', inplace=True)
    
    headers = result_df.columns.tolist()
    data_list = [headers] + result_df.values.tolist()
    
    return data_list


def upload_files(file_content: str, params: Dict[str, Any]) -> Tuple[Response, int]:
    """
    Process uploaded CSV file and convert to standardized format.
    
    Main processing pipeline:
    1. Validate parameters and detect delimiter
    2. Parse CSV to DataFrame
    3. Process datetime columns (separate or combined)
    4. Convert timezone to UTC
    5. Build output dataframe with UTC timestamps and values
    6. Save to temp file and return metadata + preview
    7. Track usage metrics
    
    Args:
        file_content: Raw CSV file content as string
        params: Dictionary containing upload parameters:
            - uploadId: Unique upload identifier
            - delimiter: CSV delimiter character
            - timezone: Source timezone
            - selected_columns: Dict mapping column roles to column names
            - custom_date_format: Optional custom datetime format
            - value_column_name: Optional name for value column
            - dropdown_count: Number of dropdowns (2 or 3)
            - has_header: Whether CSV has header row
            
    Returns:
        JSON response with fileId, metadata, and preview (first 100 rows)
    """
    try:
        socketio = get_socketio()
        upload_id = params.get('uploadId')
        
        file_size_bytes = len(file_content.encode('utf-8'))
        
        # Initialize ProgressTracker for granular progress with file size for ETA
        tracker = ProgressTracker(upload_id, socketio, file_size_bytes=file_size_bytes)
        tracker.total_steps = 5  # Validation, Parsing, DateTime, UTC, Saving

        # Step 1: Validate and extract parameters (5-15%)
        tracker.current_step = 1
        tracker.start_phase('validation')
        tracker.emit('validating', 5, 'validating_params', force=True)
        
        try:
            validated_params = _validate_and_extract_params(params, file_content)
        except LoadDataException as e:
            return jsonify(e.to_dict()), 400
        
        tracker.end_phase('validation')
        tracker.emit('validating', 15, 'params_validated', force=True)

        # Step 2: Parse CSV to DataFrame (15-40%)
        tracker.current_step = 2
        tracker.start_phase('parsing')
        tracker.emit('parsing', 15, 'parsing_csv', force=True)
        
        try:
            df = _parse_csv_to_dataframe(file_content, validated_params)
            total_rows = len(df)
            tracker.set_total_rows(total_rows)
            tracker.emit('parsing', 40, 'csv_parsed', force=True, message_params={'rowCount': total_rows})
        except LoadDataException as e:
            return jsonify(e.to_dict()), 400
        
        tracker.end_phase('parsing')

        # Step 3: Process datetime columns (40-60%)
        tracker.current_step = 3
        tracker.start_phase('datetime')
        tracker.emit('datetime', 40, 'processing_datetime', force=True)
        
        try:
            df = _process_datetime_columns(df, validated_params)
            tracker.emit('datetime', 60, 'datetime_processed', force=True)
        except LoadDataException as e:
            return jsonify(e.to_dict()), 400
        
        tracker.end_phase('datetime')
        
        # Step 4: Convert to UTC (60-75%)
        tracker.current_step = 4
        tracker.start_phase('utc')
        tracker.emit('utc', 60, 'converting_to_utc', force=True, message_params={'timezone': validated_params["timezone"]})
        
        try:
            df = convert_to_utc(df, 'datetime', validated_params['timezone'])
            tracker.emit('utc', 75, 'utc_conversion_complete', force=True)
        except LoadDataException as e:
            tracker.emit('error', 0, 'error_occurred', force=True, message_params={'error': e.message})
            return jsonify(e.to_dict()), 400
        
        tracker.end_phase('utc')
        
        # Step 5: Build result DataFrame and save to temp file (75-100%)
        tracker.current_step = 5
        tracker.start_phase('saving')
        tracker.emit('saving', 75, 'creating_result_dataframe', force=True)
        
        try:
            value_column = validated_params['value_column']
            value_column_name = validated_params['value_column_name']
            
            if not value_column or value_column not in df.columns:
                raise ValueError("Datum, Wert 1 oder Wert 2 nicht ausgewählt")
            
            result_df = pd.DataFrame()
            result_df['UTC'] = df['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
            
            final_value_column = value_column_name if value_column_name else value_column
            result_df[final_value_column] = df[value_column].apply(
                lambda x: str(x) if pd.notnull(x) else ""
            )
            
            result_df.dropna(subset=['UTC'], inplace=True)
            result_df.sort_values('UTC', inplace=True)
            result_df.reset_index(drop=True, inplace=True)
            
            total_rows = len(result_df)
            tracker.emit('saving', 85, 'result_created', force=True, message_params={'rowCount': total_rows})
            
            # Save to Supabase Storage for persistent access on Cloud Run
            tracker.emit('saving', 90, 'saving_to_cloud_storage', force=True)

            # Convert DataFrame to CSV string
            csv_content = result_df.to_csv(sep=';', index=False)

            # Upload to Supabase Storage
            user_id = g.user_id
            file_id = storage_service.upload_csv(
                user_id=user_id,
                csv_content=csv_content,
                original_filename=f"processed_{upload_id}.csv",
                metadata={
                    'totalRows': total_rows,
                    'headers': result_df.columns.tolist(),
                    'uploadId': upload_id
                }
            )

            if not file_id:
                raise ValueError("Failed to upload file to storage")

            tracker.emit('saving', 95, 'cloud_storage_saved', force=True)
            
            # Generate preview (first 100 rows)
            preview_rows = min(100, total_rows)
            preview_data = []
            
            # Add header row
            preview_data.append(result_df.columns.tolist())
            
            # Add data rows
            for _, row in result_df.head(preview_rows).iterrows():
                preview_data.append([row['UTC'], row[final_value_column]])
            
            tracker.emit('complete', 100, 'processing_complete', force=True, message_params={'rowCount': total_rows})
            
        except Exception as e:
            tracker.emit('error', 0, 'error_occurred', force=True, message_params={'error': str(e)})
            return jsonify({"error": str(e)}), 400
        
        tracker.end_phase('saving')
        
        # Track usage metrics
        try:
            increment_processing_count(g.user_id)
            
            file_size_bytes = len(file_content.encode('utf-8'))
            file_size_mb = file_size_bytes / (1024 * 1024)
            
            update_storage_usage(g.user_id, file_size_mb)
        except Exception as e:
            pass  # Don't fail upload if tracking fails
        
        # Return metadata + preview instead of full data
        return jsonify({
            "success": True,
            "message": "File processed successfully",
            "fileId": file_id,
            "totalRows": total_rows,
            "headers": result_df.columns.tolist(),
            "previewRowCount": preview_rows,
            "preview": preview_data
        }), 200
        
    except LoadDataException as e:
        socketio = get_socketio()
        socketio.emit('upload_progress', {
            'uploadId': params.get('uploadId'),
            'progress': 0,
            'status': 'error',
            'message': f'Error: {e.message}'
        }, room=params.get('uploadId'))
        
        return jsonify(e.to_dict()), 400
    except Exception as e:
        # Unexpected errors - log and return generic error
        socketio = get_socketio()
        socketio.emit('upload_progress', {
            'uploadId': params.get('uploadId'),
            'progress': 0,
            'status': 'error',
            'message': f'Unexpected error: {str(e)}'
        }, room=params.get('uploadId'))
        
        return jsonify({"error": str(e)}), 500

@bp.route('/prepare-save', methods=['POST'])
@require_auth
def prepare_save() -> Tuple[Response, int]:
    """
    Prepare merged/processed data for download.

    Saves merged CSV data to Supabase Storage for persistent access on Cloud Run.

    Expected JSON body:
        - data: Dict containing:
            - data: Array of rows to save (merged from multiple files)
            - fileName: Optional filename

    Returns:
        JSON response with file ID for download
    """
    try:
        data = request.json

        if not data or 'data' not in data:
            return jsonify({"error": "No data received"}), 400

        data_wrapper = data['data']
        save_data = data_wrapper.get('data', [])
        file_name = data_wrapper.get('fileName', '')


        if not save_data:
            return jsonify({"error": "Empty data"}), 400

        # Convert data array to CSV string
        output = StringIO()
        writer = csv.writer(output, delimiter=';')
        for row in save_data:
            writer.writerow(row)
        csv_content = output.getvalue()


        # Upload to Supabase Storage
        user_id = g.user_id

        file_id = storage_service.upload_csv(
            user_id=user_id,
            csv_content=csv_content,
            original_filename=file_name or "merged_data.csv",
            metadata={
                'totalRows': len(save_data) - 1,  # Exclude header
                'source': 'prepare-save',
                'merged': True
            }
        )

        if not file_id:
            return jsonify({"error": "Failed to save file to storage"}), 500


        return jsonify({
            "message": "File prepared for download",
            "fileId": file_id,
            "totalRows": len(save_data) - 1
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.route('/merge-and-prepare', methods=['POST'])
@require_auth
def merge_and_prepare() -> Tuple[Response, int]:
    """
    Merge multiple processed files from Supabase Storage into one file.

    Downloads individual CSV files, merges them by date, and uploads
    the merged result back to Supabase Storage.

    Expected JSON body:
        - fileIds: Array of file IDs to merge (format: user_id/timestamp_uuid)
        - fileName: Desired filename for merged result

    Returns:
        JSON response with merged file ID for download
    """
    try:
        data = request.json

        if not data:
            return jsonify({"error": "No data received"}), 400

        file_ids = data.get('fileIds', [])
        file_name = data.get('fileName', 'merged_data.csv')


        if not file_ids:
            return jsonify({"error": "No file IDs provided"}), 400

        if len(file_ids) == 1:
            # Single file - no merge needed, return same ID
            return jsonify({
                "message": "Single file, no merge needed",
                "fileId": file_ids[0],
                "downloadFileId": file_ids[0]
            }), 200

        # Download and merge multiple files

        all_dataframes = []
        headers = None

        for i, file_id in enumerate(file_ids):

            csv_content = storage_service.download_csv(file_id)

            if not csv_content:
                return jsonify({"error": f"Failed to download file: {file_id}"}), 404


            # Parse CSV content
            df = pd.read_csv(StringIO(csv_content), sep=';')

            if headers is None:
                headers = list(df.columns)
            all_dataframes.append(df)

        # Concatenate all dataframes
        merged_df = pd.concat(all_dataframes, ignore_index=True)

        # Sort by UTC column if present
        if 'UTC' in merged_df.columns:
            merged_df['UTC'] = pd.to_datetime(merged_df['UTC'], errors='coerce')
            merged_df = merged_df.sort_values('UTC')
            merged_df['UTC'] = merged_df['UTC'].dt.strftime('%Y-%m-%d %H:%M:%S')

        # Remove duplicates if any
        merged_df = merged_df.drop_duplicates()

        # Convert to CSV
        csv_content = merged_df.to_csv(sep=';', index=False)

        # Upload merged file to Supabase Storage
        user_id = g.user_id

        merged_file_id = storage_service.upload_csv(
            user_id=user_id,
            csv_content=csv_content,
            original_filename=file_name,
            metadata={
                'totalRows': len(merged_df),
                'source': 'merge-and-prepare',
                'merged': True,
                'sourceFiles': len(file_ids)
            }
        )

        if not merged_file_id:
            return jsonify({"error": "Failed to save merged file"}), 500


        # Clean up: Delete individual source files after successful merge
        deleted_count = 0
        for file_id in file_ids:
            try:
                if storage_service.delete_file(file_id):
                    deleted_count += 1
            except Exception:
                pass

        return jsonify({
            "message": "Files merged successfully",
            "fileId": merged_file_id,
            "downloadFileId": merged_file_id,
            "totalRows": len(merged_df),
            "sourceFilesCount": len(file_ids),
            "deletedSourceFiles": deleted_count
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.route('/download/<path:file_id>', methods=['GET'])
@require_auth
def download_file(file_id: str) -> Response:
    """
    Download prepared CSV file from Supabase Storage.

    Returns either a redirect to signed URL or the file content directly.

    Args:
        file_id: File identifier from upload_files (format: user_id/file_id)

    Returns:
        CSV file download or redirect to signed URL
    """
    try:

        # Get signed URL from Supabase Storage (valid for 1 hour)
        signed_url = storage_service.get_download_url(file_id, expires_in=3600)

        if signed_url:
            # Redirect to signed URL for direct download
            return redirect(signed_url)


        # Fallback: try to download content directly and serve it
        csv_content = storage_service.download_csv(file_id)

        if csv_content:
            # Create response with CSV content
            response = Response(
                csv_content,
                mimetype='text/csv',
                headers={
                    'Content-Disposition': f'attachment; filename="processed_{file_id.split("/")[-1]}.csv"'
                }
            )
            return response

        return jsonify({"error": "File not found"}), 404

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.route('/cleanup-files', methods=['POST'])
@require_auth
def cleanup_files() -> Tuple[Response, int]:
    """
    Delete files from Supabase Storage after successful download.

    This endpoint should be called after the user has successfully
    downloaded their file(s) to clean up storage.

    Expected JSON body:
        - fileIds: Array of file IDs to delete (format: user_id/timestamp_uuid)

    Returns:
        JSON response with deletion results
    """
    try:
        data = request.json

        if not data:
            return jsonify({"error": "No data received"}), 400

        file_ids = data.get('fileIds', [])


        if not file_ids:
            return jsonify({"message": "No files to delete", "deletedCount": 0}), 200

        deleted_count = 0
        failed_ids = []

        for file_id in file_ids:
            try:
                if storage_service.delete_file(file_id):
                    deleted_count += 1
                else:
                    failed_ids.append(file_id)
            except Exception as del_error:
                failed_ids.append(file_id)


        return jsonify({
            "message": "Cleanup complete",
            "deletedCount": deleted_count,
            "totalRequested": len(file_ids),
            "failedIds": failed_ids
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
