"""
RowData modul za procesiranje CSV/TXT fajlova sa vremenskim serijama

Ovaj modul pruža funkcionalnosti za:
- Chunked upload velikih fajlova
- Parsiranje različitih formata datuma i vremena
- Konverziju u UTC
- Streaming procesiranje bez opterećenja memorije
- Validaciju i error handling
"""

from .load_row_data import bp as rowdata_blueprint

# Servisi
from .services.file_upload_service import FileUploadService
from .services.date_parsing_service import DateParsingService
from .services.data_processing_service import DataProcessingService

# Repository (Redis je opciono)
try:
    from .repositories.upload_repository import UploadRepository
except ImportError:
    UploadRepository = None  # Redis nije instaliran

# Utilities
from .utils.validators import UploadValidator, DataValidator, SecurityValidator
from .utils.exceptions import (
    RowDataException,
    ValidationError,
    UploadError,
    ProcessingError,
    DateParsingError,
    handle_exception
)

__version__ = '1.0.0'
__author__ = 'Rabensteiner Engineering'

__all__ = [
    # Blueprint
    'rowdata_blueprint',
    
    # Services
    'FileUploadService',
    'DateParsingService', 
    'DataProcessingService',
    
    # Repository
    'UploadRepository',
    
    # Validators
    'UploadValidator',
    'DataValidator',
    'SecurityValidator',
    
    # Exceptions
    'RowDataException',
    'ValidationError',
    'UploadError',
    'ProcessingError',
    'DateParsingError',
    'handle_exception'
]