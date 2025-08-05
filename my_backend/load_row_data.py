import os
import tempfile
import logging
import json
import csv
import time
import secrets
from io import StringIO
from datetime import datetime
from typing import Dict, Tuple, Optional, Any
from threading import Lock
from flask import Blueprint, request, jsonify, send_file, current_app
import pandas as pd
from werkzeug.utils import secure_filename

# Helper function to get socketio instance
def get_socketio():
    return current_app.extensions['socketio']

# Socket.IO event handlers will be registered in app.py

# Create Blueprint
bp = Blueprint('load_row_data', __name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Thread-safe storage for temporary files and chunks
temp_files: Dict[str, Dict[str, Any]] = {}
chunk_storage: Dict[str, Dict[str, Any]] = {}
storage_lock = Lock()

# Configuration constants
UPLOAD_EXPIRY_TIME = 30 * 60  # 30 minutes in seconds
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
ALLOWED_EXTENSIONS = {'.csv', '.txt'}

def cleanup_old_uploads() -> None:
    """Remove expired uploads from storage"""
    current_time = time.time()
    with storage_lock:
        expired_uploads = [
            upload_id for upload_id, info in chunk_storage.items()
            if current_time - info.get('last_activity', 0) > UPLOAD_EXPIRY_TIME
        ]
        for upload_id in expired_uploads:
            del chunk_storage[upload_id]

# Supported date/time formats based on frontend patterns
SUPPORTED_DATE_FORMATS = [
    '%Y-%m-%d %H:%M:%S',      # ISO format
    '%d.%m.%Y %H:%M:%S',      # German format (primary)
    '%d.%m.%Y %H:%M:%S.%f',   # German with milliseconds
    '%Y-%m-%dT%H:%M:%S',      # ISO with T separator
    '%Y-%m-%dT%H:%M:%S%z',    # ISO with timezone
    '%Y-%m-%dT%H:%M%z',       # ISO with timezone without seconds
    '%d.%m.%Y %H:%M',         # German without seconds
    '%Y-%m-%d %H:%M',         # ISO without seconds
    '%Y/%m/%d %H:%M:%S',      # Alternative format
    '%d/%m/%Y %H:%M:%S',      # European format
    '%d-%m-%Y %H:%M:%S',      # Alternative European format
    '%d-%m-%Y %H:%M',         # Alternative European without seconds
    '%Y/%m/%d',               # Date only formats
    '%d/%m/%Y',
    '%d.%m.%Y',
    '%Y-%m-%d',
    '%H:%M:%S',               # Time only formats
    '%H:%M'
]

# Security and validation helpers
def validate_upload_id(upload_id: str) -> bool:
    """Validate upload ID format for security"""
    if not upload_id or len(upload_id) > 50:
        return False
    # Allow alphanumeric, hyphen, and underscore
    return all(c.isalnum() or c in '-_' for c in upload_id)

def generate_secure_file_id() -> str:
    """Generate cryptographically secure file ID"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    random_part = secrets.token_urlsafe(8)
    return f"{timestamp}_{random_part}"

def validate_file_size(content_length: Optional[int]) -> bool:
    """Check if file size is within allowed limits"""
    if content_length is None:
        return True
    return content_length <= MAX_FILE_SIZE


def detect_delimiter(file_content: str, sample_lines: int = 3) -> str:
    """Detect CSV delimiter from file content"""
    delimiters = [',', ';', '\t']
    lines = file_content.splitlines()[:sample_lines]
    counts = {d: sum(line.count(d) for line in lines) for d in delimiters}
    if counts and max(counts.values()) > 0:
        return max(counts, key=counts.get)
    return ','

def clean_datetime_string(datetime_str: Any) -> str:
    """Clean datetime string by keeping only valid characters"""
    if not isinstance(datetime_str, str):
        datetime_str = str(datetime_str)
    
    # Keep only valid datetime characters
    valid_chars = set('0123456789:-+.T /')
    return ''.join(c for c in datetime_str if c in valid_chars)

def clean_file_content(file_content: str, delimiter: str) -> str:
    """Remove trailing delimiters and whitespace from each line"""
    cleaned_lines = [line.rstrip(f"{delimiter};,") for line in file_content.splitlines()]
    return "\n".join(cleaned_lines)

def parse_datetime_column(df: pd.DataFrame, datetime_col: str, custom_format: Optional[str] = None) -> Tuple[bool, Optional[pd.Series], Optional[str]]:
    """
    Parse datetime column using custom format or supported formats.
    Returns: (success, parsed_dates, error_message)
    """
    try:
        # Clean data before parsing
        df = df.copy()
        df[datetime_col] = df[datetime_col].astype(str).str.strip()

        # Try custom format first if provided
        if custom_format:
            try:
                parsed_dates = pd.to_datetime(df[datetime_col], format=custom_format, errors='coerce')
                if not parsed_dates.isna().all():
                    return True, parsed_dates, None
            except Exception:
                pass

        # Try supported formats with caching for performance
        detected_format = None
        for fmt in SUPPORTED_DATE_FORMATS:
            try:
                # Test with first non-null value for performance
                test_value = df[datetime_col].dropna().iloc[0] if not df[datetime_col].dropna().empty else None
                if test_value:
                    pd.to_datetime(test_value, format=fmt)
                    detected_format = fmt
                    break
            except (ValueError, IndexError):
                continue

        if detected_format:
            parsed_dates = pd.to_datetime(df[datetime_col], format=detected_format, errors='coerce')
            if not parsed_dates.isna().all():
                return True, parsed_dates, None
        
        # If no format worked, try pandas auto-detection as last resort
        try:
            parsed_dates = pd.to_datetime(df[datetime_col], errors='coerce')
            if not parsed_dates.isna().all():
                return True, parsed_dates, None
        except Exception:
            pass

        return False, None, "Unsupported date format. Please use custom format or one of the supported formats: DD.MM.YYYY HH:MM:SS"

    except Exception as e:
        return False, None, f"Date parsing error: {str(e)}"




def convert_to_utc(df: pd.DataFrame, date_column: str, timezone: str = 'UTC') -> pd.DataFrame:
    """
    Convert datetime column to UTC timezone.
    If datetime has no timezone, localize it to the given timezone first.
    """
    try:
        if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
            df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
        
        if df[date_column].dt.tz is None:
            try:
                df[date_column] = df[date_column].dt.tz_localize(timezone, ambiguous='NaT', nonexistent='NaT')
            except Exception as e:
                raise ValueError(f"Unsupported timezone: {timezone}")
            
            if timezone.upper() != 'UTC':
                df[date_column] = df[date_column].dt.tz_convert('UTC')
        else:
            if str(df[date_column].dt.tz) != 'UTC':
                df[date_column] = df[date_column].dt.tz_convert('UTC')
        
        return df
    except Exception as e:
        raise ValueError(f"Error converting to UTC: {str(e)}")

@bp.route('/upload-chunk', methods=['POST'])
def upload_chunk():
    try:
        # Validate file chunk
        if 'fileChunk' not in request.files:
            return jsonify({"error": "Chunk file not found"}), 400
        
        file_chunk = request.files['fileChunk']
        
        # Validate file size
        if file_chunk.content_length and not validate_file_size(file_chunk.content_length):
            return jsonify({"error": f"File too large. Maximum size: {MAX_FILE_SIZE // (1024*1024)}MB"}), 400
        
        # Validate required parameters
        required_params = ['uploadId', 'chunkIndex', 'totalChunks', 'delimiter', 'selected_columns', 'timezone', 'dropdown_count', 'hasHeader']
        missing_params = [param for param in required_params if param not in request.form]
        if missing_params:
            return jsonify({"error": f"Missing required parameters: {', '.join(missing_params)}"}), 400
        
        upload_id = request.form['uploadId']
        
        # Validate upload ID for security
        if not validate_upload_id(upload_id):
            return jsonify({"error": "Invalid upload ID format"}), 400
        
        chunk_index = int(request.form['chunkIndex'])
        total_chunks = int(request.form['totalChunks'])
        
        with storage_lock:
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
                            'has_header': request.form.get('hasHeader', 'no') == 'ja',  # Convert to boolean
                            'selected_columns': selected_columns,
                            'custom_date_format': request.form.get('custom_date_format'),
                            'value_column_name': request.form.get('valueColumnName', '').strip(),
                            'dropdown_count': int(request.form.get('dropdown_count', '2'))
                        }
                    }
                except json.JSONDecodeError:
                    return jsonify({"error": "Invalid JSON format for selected_columns"}), 400
            else:
                chunk_storage[upload_id]['total_chunks'] = max(chunk_storage[upload_id]['total_chunks'], total_chunks)

            chunk_content = file_chunk.read()
            chunk_storage[upload_id]['chunks'][chunk_index] = chunk_content
            chunk_storage[upload_id]['received_chunks'] += 1
            chunk_storage[upload_id]['last_activity'] = time.time()
        
        progress_percentage = int((chunk_index + 1) / chunk_storage[upload_id]['total_chunks'] * 100)
        socketio = get_socketio()
        socketio.emit('upload_progress', {
            'uploadId': upload_id,
            'fileName': file_chunk.filename or 'unknown',
            'progress': progress_percentage,
            'status': 'uploading',
            'message': f'Processing chunk {chunk_index + 1}/{chunk_storage[upload_id]["total_chunks"]}'
        }, room=upload_id)
        
        if chunk_storage[upload_id]['received_chunks'] == chunk_storage[upload_id]['total_chunks']:
            socketio.emit('upload_progress', {
                'uploadId': upload_id,
                'progress': 100,
                'status': 'uploading',
                'message': 'All chunks received, waiting for finalization...'
            }, room=upload_id)
        
        return jsonify({
            "message": f"Chunk {chunk_index + 1}/{chunk_storage[upload_id]['total_chunks']} received",
            "uploadId": upload_id,
            "remainingChunks": chunk_storage[upload_id]['total_chunks'] - chunk_storage[upload_id]['received_chunks']
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@bp.route('/finalize-upload', methods=['POST'])
def finalize_upload():
    """Finalize chunked upload and process the complete file"""
    try:
        data = request.get_json(force=True, silent=True)
        
        if not data or 'uploadId' not in data:
            return jsonify({"error": "uploadId is required"}), 400
        
        upload_id = data['uploadId']
        
        # Validate upload ID
        if not validate_upload_id(upload_id):
            return jsonify({"error": "Invalid upload ID format"}), 400
        
        with storage_lock:
            if upload_id not in chunk_storage:
                return jsonify({"error": "Upload not found or already processed"}), 404
            
            upload_info = chunk_storage[upload_id]
            if upload_info['received_chunks'] != upload_info['total_chunks']:
                remaining = upload_info['total_chunks'] - upload_info['received_chunks']
                return jsonify({
                    "error": f"Not all chunks received. Missing {remaining} chunks.",
                    "received": upload_info['received_chunks'],
                    "total": upload_info['total_chunks']
                }), 400
        
        return process_chunks(upload_id)
        
    except Exception as e:
        return jsonify({"error": "Error finalizing upload"}), 400

@bp.route('/cancel-upload', methods=['POST'])
def cancel_upload():
    """Cancel an ongoing upload and clean up resources"""
    try:
        data = request.get_json(force=True, silent=True)
        
        if not data or 'uploadId' not in data:
            return jsonify({"error": "uploadId is required"}), 400
        
        upload_id = data['uploadId']
        
        # Validate upload ID
        if not validate_upload_id(upload_id):
            return jsonify({"error": "Invalid upload ID format"}), 400
        
        with storage_lock:
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
        return jsonify({"error": "Error canceling upload"}), 400

def process_chunks(upload_id: str):
    """Process uploaded chunks and combine them into complete file"""
    try:
        socketio = get_socketio()
        socketio.emit('upload_progress', {
            'uploadId': upload_id,
            'progress': 50,
            'status': 'processing',
            'message': 'Processing file data...'
        }, room=upload_id)
        
        with storage_lock:
            if upload_id not in chunk_storage:
                return jsonify({"error": "Upload not found"}), 404
            
            upload_info = chunk_storage[upload_id]
            chunks = [upload_info['chunks'][i] for i in range(upload_info['total_chunks'])]
            params = upload_info['parameters'].copy()
            params['uploadId'] = upload_id
            
            # Clean up storage
            del chunk_storage[upload_id]
        
        # Try different encodings
        encodings = ['utf-8', 'utf-16', 'utf-16le', 'utf-16be', 'latin1', 'cp1252']
        full_content = None
        
        for encoding in encodings:
            try:
                decoded_chunks = [chunk.decode(encoding) for chunk in chunks]
                full_content = "".join(decoded_chunks)
                break
            except UnicodeDecodeError:
                continue
        
        if full_content is None:
            socketio.emit('upload_progress', {
                'uploadId': upload_id,
                'progress': 0,
                'status': 'error',
                'message': 'Could not decode file content with any supported encoding'
            }, room=upload_id)
            return jsonify({"error": "Could not decode file content with any supported encoding"}), 400
        
        # Check file size after decoding
        content_size = len(full_content.encode('utf-8'))
        if content_size > MAX_FILE_SIZE:
            return jsonify({"error": f"File too large: {content_size // (1024*1024)}MB, max: {MAX_FILE_SIZE // (1024*1024)}MB"}), 400
        
        return upload_files(full_content, params)
        
    except Exception as e:
        return jsonify({"error": "Error processing file data"}), 400

@bp.route('/upload-status/<upload_id>', methods=['GET'])
def check_upload_status(upload_id: str):
    """Check the status of an ongoing upload"""
    try:
        # Validate upload ID
        if not validate_upload_id(upload_id):
            return jsonify({"error": "Invalid upload ID format"}), 400
        
        with storage_lock:
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
        return jsonify({"error": "Error checking upload status"}), 400

def process_csv_data(df: pd.DataFrame, params: Dict[str, Any]) -> Tuple[bool, Any, Optional[str]]:
    """Process CSV data with date parsing and timezone conversion"""
    try:
        selected_columns = params.get('selected_columns', {})
        custom_date_format = params.get('custom_date_format')
        dropdown_count = int(params.get('dropdown_count', '2'))
        has_separate_date_time = dropdown_count == 3
        timezone = params.get('timezone', 'UTC')
        value_column_name = params.get('value_column_name', '').strip()
        has_header = params.get('has_header', False)
        
        # Get column names/indices
        date_column = selected_columns.get('column1')
        time_column = selected_columns.get('column2') if has_separate_date_time else None
        value_column = selected_columns.get('column3') if has_separate_date_time else selected_columns.get('column2')
        
        # When no header, frontend sends indices as strings like "0", "1", etc.
        # We need to convert them to actual column names that pandas uses
        if not has_header:
            try:
                # Convert string indices to integers and then to column names
                if date_column is not None:
                    date_idx = int(date_column)
                    if date_idx == -1:
                        # -1 means column not selected, use first column as default
                        date_column = df.columns[0] if len(df.columns) > 0 else None
                    elif date_idx >= 0 and date_idx < len(df.columns):
                        date_column = df.columns[date_idx]
                    else:
                        date_column = None
                    
                if time_column is not None:
                    time_idx = int(time_column)
                    if time_idx == -1:
                        time_column = None  # Not selected
                    elif time_idx >= 0 and time_idx < len(df.columns):
                        time_column = df.columns[time_idx]
                    else:
                        time_column = None
                        
                if value_column is not None:
                    value_idx = int(value_column)
                    if value_idx == -1:
                        # -1 means column not selected by user
                        # For files without headers, try to auto-select a value column
                        # that is not the date column
                        if date_column and len(df.columns) > 1:
                            for col in df.columns:
                                if col != date_column and col != time_column:
                                    value_column = col
                                    break
                            else:
                                value_column = None
                        else:
                            value_column = None
                    elif value_idx >= 0 and value_idx < len(df.columns):
                        value_column = df.columns[value_idx]
                    else:
                        value_column = None
            except (ValueError, IndexError):
                return False, None, f"Invalid column selection"
        
        # Validate columns exist
        if not value_column or value_column not in df.columns:
            # For files without headers, provide more helpful message
            if not has_header:
                available_cols = [f"Column {i+1}" for i in range(len(df.columns))]
                return False, None, f"No value column selected. The file has {len(df.columns)} columns. Please ensure the frontend allows column selection for files without headers."
            else:
                return False, None, f"Please select a value column from the dropdown menu before processing."
        
        # Handle datetime parsing
        if has_separate_date_time and date_column and time_column:
            # Clean datetime strings
            df[time_column] = df[time_column].apply(clean_datetime_string)
            df[date_column] = df[date_column].apply(clean_datetime_string)
            df['datetime'] = df[date_column].astype(str) + ' ' + df[time_column].astype(str)
            datetime_col = 'datetime'
        else:
            datetime_col = date_column or df.columns[0]
        
        # Parse datetime
        success, parsed_dates, err = parse_datetime_column(df, datetime_col, custom_date_format)
        if not success:
            return False, None, err
        
        df['datetime'] = parsed_dates
        
        # Convert to UTC
        df = convert_to_utc(df, 'datetime', timezone)
        
        # Prepare result
        result_df = pd.DataFrame()
        result_df['UTC'] = df['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
        final_value_column = value_column_name if value_column_name else value_column
        
        # Handle multiple value columns if user wants to process more than one
        # For now, we'll process just the selected column
        # Convert value column to numeric if possible, keep as string if not
        if value_column in df.columns:
            # Try to convert to numeric, but keep original if it fails
            numeric_values = pd.to_numeric(df[value_column], errors='coerce')
            # Only use numeric values if at least some conversions succeeded
            if not numeric_values.isna().all():
                df[value_column] = numeric_values
            else:
                # Keep as string if all numeric conversions failed
                pass
        
        result_df[final_value_column] = df[value_column].apply(lambda x: str(x) if pd.notnull(x) else "")
        result_df.dropna(subset=['UTC'], inplace=True)
        result_df.sort_values('UTC', inplace=True)
        
        headers = result_df.columns.tolist()
        data_list = [headers] + result_df.values.tolist()
        
        return True, data_list, None
        
    except Exception as e:
        return False, None, str(e)

def upload_files(file_content: str, params: Dict[str, Any]):
    """Process uploaded file content and return formatted data"""
    try:
        upload_id = params.get('uploadId')
        socketio = get_socketio()
        
        # Emit progress update
        socketio.emit('upload_progress', {
            'uploadId': upload_id,
            'progress': 60,
            'status': 'processing',
            'message': 'Parsing CSV data...'
        }, room=upload_id)
        
        delimiter = params.get('delimiter')
        if not delimiter:
            return jsonify({"error": "Delimiter not provided"}), 400
        
        has_header = params.get('has_header', False) 
        
        # Auto-detect delimiter but use user's choice
        detected_delimiter = detect_delimiter(file_content)
        # Continue with user's delimiter choice
        
        # Parse CSV
        cleaned_content = clean_file_content(file_content, delimiter)
        try:
            df = pd.read_csv(
                StringIO(cleaned_content),
                delimiter=delimiter,
                header=0 if has_header else None,
                dtype=str,  # Treat all columns as strings initially
                low_memory=False
            )
            
            # Clean column names
            if not has_header:
                df.columns = [str(i) for i in range(len(df.columns))]
            else:
                df.columns = [col.strip() for col in df.columns]
            
            # Remove empty columns
            df = df.dropna(axis=1, how='all')
            df.columns = df.columns.astype(str)
            
        except Exception as e:
            return jsonify({"error": f"Error processing CSV: {str(e)}"}), 400
        
        if df.empty:
            return jsonify({"error": "No data loaded from file"}), 400

        
        # Process the data
        socketio.emit('upload_progress', {
            'uploadId': upload_id,
            'progress': 80,
            'status': 'processing',
            'message': 'Processing data...'
        }, room=upload_id)
        
        success, data_list, error = process_csv_data(df, params)
        
        if not success:
            socketio.emit('upload_progress', {
                'uploadId': upload_id,
                'progress': 0,
                'status': 'error',
                'message': f'Error: {error}'
            }, room=upload_id)
            return jsonify({"error": error}), 400
        
        # Success
        socketio.emit('upload_progress', {
            'uploadId': upload_id,
            'progress': 100,
            'status': 'completed',
            'message': 'Data processing completed'
        }, room=upload_id)
        
        return jsonify({"data": data_list, "fullData": data_list})
    except Exception as e:
        if 'uploadId' in params:
            socketio = get_socketio()
            socketio.emit('upload_progress', {
                'uploadId': params['uploadId'],
                'progress': 0,
                'status': 'error',
                'message': f'Error: {str(e)}'
            }, room=params['uploadId'])
        return jsonify({"error": str(e)}), 400

@bp.route('/prepare-save', methods=['POST'])
def prepare_save():
    """Prepare CSV file for download with security checks"""
    try:
        data = request.json
        if not data or 'data' not in data:
            return jsonify({"error": "No data received"}), 400
        
        data_wrapper = data['data']
        save_data = data_wrapper.get('data', [])
        file_name = data_wrapper.get('fileName', '')
        
        # Sanitize filename
        if file_name:
            file_name = secure_filename(file_name)
        
        
        if not save_data:
            return jsonify({"error": "Empty data"}), 400
        
        # Create temporary file securely
        temp_dir = tempfile.gettempdir()
        temp_file = tempfile.NamedTemporaryFile(
            mode='w+', 
            delete=False, 
            suffix='.csv',
            dir=temp_dir,
            prefix='export_'
        )
        
        # Write CSV data
        writer = csv.writer(temp_file, delimiter=';')
        for row in save_data:
            writer.writerow(row)
        temp_file.close()
        
        # Generate secure file ID
        file_id = generate_secure_file_id()
        
        with storage_lock:
            temp_files[file_id] = {
                'path': temp_file.name,
                'fileName': file_name or f"data_{file_id}.csv",
                'timestamp': time.time()
            }
        
        return jsonify({"message": "File prepared for download", "fileId": file_id}), 200
    except Exception as e:
        return jsonify({"error": "Error preparing file for download"}), 500

@bp.route('/download/<file_id>', methods=['GET'])
def download_file(file_id: str):
    """Download prepared file with security validation"""
    try:
        # Validate file ID format
        if not validate_upload_id(file_id):
            return jsonify({"error": "Invalid file ID"}), 400
        
        with storage_lock:
            if file_id not in temp_files:
                return jsonify({"error": "File not found"}), 404
            
            file_info = temp_files[file_id].copy()
        
        file_path = file_info['path']
        
        # Security check: ensure file is in temp directory
        temp_dir = tempfile.gettempdir()
        real_path = os.path.realpath(file_path)
        real_temp_dir = os.path.realpath(temp_dir)
        
        # Check if the file is in the temp directory (handles symlinks)
        if not real_path.startswith(real_temp_dir):
            return jsonify({"error": "Access denied"}), 403
        
        if not os.path.exists(file_path):
            return jsonify({"error": "File not found"}), 404
        
        download_name = file_info['fileName']
        
        # Send file
        response = send_file(
            file_path,
            as_attachment=True,
            download_name=download_name,
            mimetype='text/csv'
        )
        
        # Schedule cleanup after response
        @response.call_on_close
        def cleanup():
            try:
                if os.path.exists(file_path):
                    os.unlink(file_path)
                with storage_lock:
                    if file_id in temp_files:
                        del temp_files[file_id]
            except Exception as e:
                pass
        
        return response
        
    except Exception as e:
        return jsonify({"error": "Error downloading file"}), 500