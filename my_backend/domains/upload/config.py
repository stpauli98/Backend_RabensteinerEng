"""
Upload Domain Configuration Constants
"""
from typing import List

# Upload Configuration
UPLOAD_EXPIRY_SECONDS: int = 1800  # 30 minutes
CHUNK_SIZE_MB: int = 5

# CSV Configuration
DEFAULT_DELIMITER: str = ','
SUPPORTED_DELIMITERS: List[str] = [',', ';', '\t']

# Date Format Support - comprehensive list for auto-detection
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

# Progress tracking
EMIT_INTERVAL: float = 0.3  # Emit every 300ms for smooth ETA

# Phase time benchmarks (seconds per MB) for ETA calculation
PHASE_TIME_PER_MB = {
    'validation': 0.02,
    'parsing': 0.10,
    'datetime': 0.15,
    'utc': 0.04,
    'build': 0.20,
    'streaming': 1.0
}
