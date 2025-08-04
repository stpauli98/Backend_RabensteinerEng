"""
Custom exceptions za RowData modul
"""


class RowDataException(Exception):
    """Bazna exception klasa za RowData modul"""
    def __init__(self, message: str, code: str = None, status_code: int = 400):
        self.message = message
        self.code = code
        self.status_code = status_code
        super().__init__(self.message)


class ValidationError(RowDataException):
    """Exception za validacione greške"""
    def __init__(self, message: str, field: str = None):
        self.field = field
        code = f"VALIDATION_ERROR_{field.upper()}" if field else "VALIDATION_ERROR"
        super().__init__(message, code, 400)


class AuthenticationError(RowDataException):
    """Exception za autentifikacione greške"""
    def __init__(self, message: str = "Authentication required"):
        super().__init__(message, "AUTHENTICATION_ERROR", 401)


class AuthorizationError(RowDataException):
    """Exception za autorizacione greške"""
    def __init__(self, message: str = "Insufficient permissions"):
        super().__init__(message, "AUTHORIZATION_ERROR", 403)


class UploadError(RowDataException):
    """Exception za greške pri upload-u"""
    def __init__(self, message: str, upload_id: str = None):
        self.upload_id = upload_id
        super().__init__(message, "UPLOAD_ERROR", 400)


class ChunkError(UploadError):
    """Exception za greške sa chunk-ovima"""
    def __init__(self, message: str, upload_id: str = None, chunk_index: int = None):
        self.chunk_index = chunk_index
        super().__init__(message, upload_id)
        self.code = "CHUNK_ERROR"


class ProcessingError(RowDataException):
    """Exception za greške pri procesiranju podataka"""
    def __init__(self, message: str, stage: str = None):
        self.stage = stage
        code = f"PROCESSING_ERROR_{stage.upper()}" if stage else "PROCESSING_ERROR"
        super().__init__(message, code, 500)


class DateParsingError(ProcessingError):
    """Exception za greške pri parsiranju datuma"""
    def __init__(self, message: str, sample_value: str = None, expected_format: str = None):
        self.sample_value = sample_value
        self.expected_format = expected_format
        super().__init__(message, "DATE_PARSING")
        self.code = "DATE_PARSING_ERROR"


class StorageError(RowDataException):
    """Exception za greške sa skladištenjem"""
    def __init__(self, message: str, storage_type: str = None):
        self.storage_type = storage_type
        code = f"STORAGE_ERROR_{storage_type.upper()}" if storage_type else "STORAGE_ERROR"
        super().__init__(message, code, 500)


class RedisError(StorageError):
    """Exception za Redis greške"""
    def __init__(self, message: str):
        super().__init__(message, "REDIS")


class FileSystemError(StorageError):
    """Exception za file system greške"""
    def __init__(self, message: str, file_path: str = None):
        self.file_path = file_path
        super().__init__(message, "FILESYSTEM")


class RateLimitError(RowDataException):
    """Exception za rate limit greške"""
    def __init__(self, message: str = "Rate limit exceeded", retry_after: int = None):
        self.retry_after = retry_after
        super().__init__(message, "RATE_LIMIT_ERROR", 429)


class TimeoutError(RowDataException):
    """Exception za timeout greške"""
    def __init__(self, message: str = "Operation timed out", operation: str = None):
        self.operation = operation
        code = f"TIMEOUT_ERROR_{operation.upper()}" if operation else "TIMEOUT_ERROR"
        super().__init__(message, code, 408)


class ConfigurationError(RowDataException):
    """Exception za konfiguracione greške"""
    def __init__(self, message: str, config_key: str = None):
        self.config_key = config_key
        super().__init__(message, "CONFIGURATION_ERROR", 500)


def handle_exception(e: Exception) -> tuple:
    """
    Konvertuje exception u HTTP response format
    
    Returns:
        tuple: (response_dict, status_code)
    """
    if isinstance(e, RowDataException):
        response = {
            "error": e.code or e.__class__.__name__,
            "message": e.message
        }
        
        # Dodaj dodatne informacije ako postoje
        if isinstance(e, ValidationError) and e.field:
            response["field"] = e.field
        elif isinstance(e, UploadError) and e.upload_id:
            response["uploadId"] = e.upload_id
        elif isinstance(e, ChunkError) and e.chunk_index is not None:
            response["chunkIndex"] = e.chunk_index
        elif isinstance(e, DateParsingError):
            if e.sample_value:
                response["sampleValue"] = e.sample_value
            if e.expected_format:
                response["expectedFormat"] = e.expected_format
        elif isinstance(e, RateLimitError) and e.retry_after:
            response["retryAfter"] = e.retry_after
            
        return response, e.status_code
    else:
        # Za sve ostale exception-e, vrati generičku grešku
        return {
            "error": "INTERNAL_ERROR",
            "message": "An unexpected error occurred"
        }, 500