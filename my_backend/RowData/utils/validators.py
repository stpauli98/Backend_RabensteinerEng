"""
Validatori za RowData modul
"""
import re
import os
from typing import Dict, Any, List, Optional, Tuple
from werkzeug.datastructures import FileStorage
from ..config.settings import (
    MAX_FILE_SIZE, MAX_CHUNK_SIZE, ALLOWED_EXTENSIONS,
    ALLOWED_DELIMITERS, VALIDATION_RULES
)
from .exceptions import ValidationError


class UploadValidator:
    """Validator za upload zahteve"""
    
    @staticmethod
    def validate_file_extension(filename: str) -> None:
        """Validira ekstenziju fajla"""
        if not filename:
            raise ValidationError("Filename is required")
            
        ext = os.path.splitext(filename)[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise ValidationError(
                f"File type {ext} not allowed. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}"
            )
    
    @staticmethod
    def validate_file_size(file_size: int) -> None:
        """Validira veličinu fajla"""
        if file_size > MAX_FILE_SIZE:
            size_mb = MAX_FILE_SIZE / (1024 * 1024)
            raise ValidationError(f"File size exceeds maximum allowed size of {size_mb}MB")
    
    @staticmethod
    def validate_chunk_size(chunk: FileStorage) -> None:
        """Validira veličinu chunk-a"""
        chunk.seek(0, 2)  # Idi na kraj fajla
        size = chunk.tell()
        chunk.seek(0)  # Vrati na početak
        
        if size > MAX_CHUNK_SIZE:
            size_mb = MAX_CHUNK_SIZE / (1024 * 1024)
            raise ValidationError(f"Chunk size exceeds maximum allowed size of {size_mb}MB")
    
    @staticmethod
    def validate_upload_id(upload_id: str) -> None:
        """Validira upload ID"""
        if not upload_id:
            raise ValidationError("Upload ID is required")
            
        rules = VALIDATION_RULES['upload_id']
        
        if len(upload_id) < rules['min_length'] or len(upload_id) > rules['max_length']:
            raise ValidationError(
                f"Upload ID must be between {rules['min_length']} and {rules['max_length']} characters"
            )
            
        if not re.match(rules['pattern'], upload_id):
            raise ValidationError("Upload ID contains invalid characters")
    
    @staticmethod
    def validate_chunk_index(chunk_index: int, total_chunks: int) -> None:
        """Validira chunk index"""
        rules = VALIDATION_RULES['chunk_index']
        
        if chunk_index < rules['min'] or chunk_index >= total_chunks:
            raise ValidationError(f"Invalid chunk index: {chunk_index}")
            
        if chunk_index > rules['max']:
            raise ValidationError(f"Chunk index exceeds maximum: {rules['max']}")
    
    @staticmethod
    def validate_total_chunks(total_chunks: int) -> None:
        """Validira ukupan broj chunk-ova"""
        rules = VALIDATION_RULES['total_chunks']
        
        if total_chunks < rules['min'] or total_chunks > rules['max']:
            raise ValidationError(
                f"Total chunks must be between {rules['min']} and {rules['max']}"
            )
    
    @staticmethod
    def validate_delimiter(delimiter: str) -> None:
        """Validira delimiter"""
        if delimiter not in ALLOWED_DELIMITERS:
            raise ValidationError(
                f"Invalid delimiter. Allowed delimiters: {', '.join(repr(d) for d in ALLOWED_DELIMITERS)}"
            )
    
    @staticmethod
    def validate_timezone(timezone: str) -> None:
        """Validira timezone"""
        allowed_timezones = VALIDATION_RULES['timezone']['allowed']
        
        if timezone not in allowed_timezones:
            raise ValidationError(
                f"Invalid timezone. Allowed timezones: {', '.join(allowed_timezones)}"
            )
    
    @staticmethod
    def validate_selected_columns(selected_columns: Dict[str, str], dropdown_count: int) -> None:
        """Validira selektovane kolone"""
        required_columns = ['column1', 'column2']
        if dropdown_count == 3:
            required_columns.append('column3')
            
        for col in required_columns:
            if col not in selected_columns:
                raise ValidationError(f"Missing required column: {col}")
                
            if not selected_columns[col]:
                raise ValidationError(f"Column {col} cannot be empty")
    
    @staticmethod
    def validate_upload_request(request_data: Dict[str, Any], files: Dict[str, Any]) -> Dict[str, Any]:
        """Validira kompletan upload zahtev"""
        # Validacija fajla
        if 'fileChunk' not in files:
            raise ValidationError("File chunk is required")
            
        file_chunk = files['fileChunk']
        UploadValidator.validate_chunk_size(file_chunk)
        
        # Validacija parametara
        required_params = [
            'uploadId', 'chunkIndex', 'totalChunks', 'delimiter',
            'selected_columns', 'timezone', 'dropdown_count', 'hasHeader'
        ]
        
        missing_params = [param for param in required_params if param not in request_data]
        if missing_params:
            raise ValidationError(f"Missing required parameters: {', '.join(missing_params)}")
        
        # Validacija individual parametara
        UploadValidator.validate_upload_id(request_data['uploadId'])
        
        try:
            chunk_index = int(request_data['chunkIndex'])
            total_chunks = int(request_data['totalChunks'])
            dropdown_count = int(request_data['dropdown_count'])
        except (ValueError, TypeError) as e:
            raise ValidationError(f"Invalid numeric parameter: {str(e)}")
            
        UploadValidator.validate_total_chunks(total_chunks)
        UploadValidator.validate_chunk_index(chunk_index, total_chunks)
        UploadValidator.validate_delimiter(request_data['delimiter'])
        UploadValidator.validate_timezone(request_data['timezone'])
        
        # Validacija selected_columns
        import json
        try:
            selected_columns = json.loads(request_data['selected_columns'])
            if not isinstance(selected_columns, dict):
                raise ValidationError("selected_columns must be a JSON object")
        except json.JSONDecodeError:
            raise ValidationError("Invalid JSON format for selected_columns")
            
        UploadValidator.validate_selected_columns(selected_columns, dropdown_count)
        
        # Validacija has_header
        if request_data['hasHeader'] not in ['ja', 'nein']:
            raise ValidationError("hasHeader must be 'ja' or 'nein'")
        
        return {
            'upload_id': request_data['uploadId'],
            'chunk_index': chunk_index,
            'total_chunks': total_chunks,
            'delimiter': request_data['delimiter'],
            'timezone': request_data['timezone'],
            'selected_columns': selected_columns,
            'dropdown_count': dropdown_count,
            'has_header': request_data['hasHeader'] == 'ja',
            'custom_date_format': request_data.get('custom_date_format'),
            'value_column_name': request_data.get('valueColumnName', '').strip()
        }


class DataValidator:
    """Validator za podatke"""
    
    @staticmethod
    def validate_csv_content(content: str, delimiter: str) -> Tuple[bool, Optional[str]]:
        """Validira CSV sadržaj"""
        if not content:
            return False, "Empty content"
            
        lines = content.strip().split('\n')
        if not lines:
            return False, "No data lines found"
            
        # Proveri da li sve linije imaju isti broj kolona
        first_line_cols = len(lines[0].split(delimiter))
        
        for i, line in enumerate(lines[1:], 1):
            if len(line.split(delimiter)) != first_line_cols:
                return False, f"Inconsistent number of columns at line {i+1}"
                
        return True, None
    
    @staticmethod
    def validate_date_format(date_string: str, custom_format: Optional[str] = None) -> Tuple[bool, Optional[str]]:
        """Validira format datuma"""
        if not date_string:
            return False, "Empty date string"
            
        # Osnovna validacija - da li sadrži brojeve
        if not any(c.isdigit() for c in date_string):
            return False, "Date string must contain numbers"
            
        return True, None


class SecurityValidator:
    """Validator za sigurnosne aspekte"""
    
    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """Sanitizuje ime fajla"""
        # Ukloni path separatore
        filename = os.path.basename(filename)
        
        # Ukloni opasne karaktere
        sanitized = re.sub(r'[^a-zA-Z0-9._\-]', '_', filename)
        
        # Ograniči dužinu
        max_length = 255
        if len(sanitized) > max_length:
            name, ext = os.path.splitext(sanitized)
            sanitized = name[:max_length - len(ext)] + ext
            
        return sanitized
    
    @staticmethod
    def validate_path_traversal(path: str) -> None:
        """Proverava path traversal pokušaje"""
        if '..' in path or path.startswith('/'):
            raise ValidationError("Invalid path: potential path traversal detected")
    
    @staticmethod
    def validate_json_input(json_string: str, max_size: int = 1024 * 1024) -> Dict[str, Any]:
        """Validira JSON input sa ograničenjem veličine"""
        if len(json_string) > max_size:
            raise ValidationError("JSON input too large")
            
        try:
            import json
            data = json.loads(json_string)
            if not isinstance(data, dict):
                raise ValidationError("JSON must be an object")
            return data
        except json.JSONDecodeError as e:
            raise ValidationError(f"Invalid JSON: {str(e)}")