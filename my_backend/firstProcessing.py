"""
First Processing Module - Time Series Data Processing Service

This module handles chunked CSV file uploads and processes time series data
with various resampling methods (mean, interpolation, nearest neighbor).
Includes automatic cleanup of temporary files and real-time progress tracking
via Socket.IO.

API Endpoints:
    POST /api/firstProcessing/upload_chunk - Upload and process CSV chunks
    POST /api/firstProcessing/prepare-save - Prepare processed data for download
    GET  /api/firstProcessing/download/<file_id> - Download processed CSV file
"""

# Standard library imports
import csv
import gzip
import json
import logging
import os
import re
import statistics
import tempfile
import threading
import time
import traceback
from datetime import datetime, timedelta
from io import BytesIO, StringIO

# Third-party imports
import numpy as np
import pandas as pd
from flask import Blueprint, Response, jsonify, request, send_file
from flask_socketio import emit

# ============================================================================
# CONFIGURATION
# ============================================================================

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# File size limits
MAX_CHUNK_SIZE = 10 * 1024 * 1024  # 10MB per chunk
MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB total
TEMP_FILE_LIFETIME = 3600  # 1 hour in seconds
CLEANUP_INTERVAL = 600  # 10 minutes

# Upload configuration
UPLOAD_FOLDER = "chunk_uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global storage for temporary files
temp_files = {}
temp_files_lock = threading.Lock()

# Create Flask blueprint
bp = Blueprint('first_processing', __name__)

# ============================================================================
# ERROR MESSAGES
# ============================================================================

ERROR_MESSAGES = {
    'no_chunk': 'No file chunk found',
    'invalid_params': 'Invalid parameter format',
    'missing_upload_id': 'Upload ID is required',
    'missing_mode': 'Processing mode is required',
    'invalid_tss': 'Time step size must be positive',
    'empty_chunk': 'Empty chunk received',
    'chunk_too_large': 'Chunk too large: {size} bytes (max: {max_size})',
    'no_data': 'No data received',
    'empty_data': 'Empty data',
    'file_not_found': 'File not found or expired',
    'no_timestamps': 'No valid timestamps found',
    'csv_parse_error': 'CSV parsing failed: {error}',
    'internal_error': 'Internal server error',
    'prepare_failed': 'Failed to prepare file',
    'download_failed': 'Failed to download file',
}

# ============================================================================
# CLEANUP FUNCTIONS
# ============================================================================

def cleanup_old_files():
    """Clean up temporary files older than TEMP_FILE_LIFETIME.
    
    This function runs periodically to remove old temporary files
    and prevent disk space exhaustion.
    """
    with temp_files_lock:
        current_time = time.time()
        files_to_delete = [
            file_id for file_id, file_info in temp_files.items()
            if current_time - file_info.get('timestamp', 0) > TEMP_FILE_LIFETIME
        ]
        
        for file_id in files_to_delete:
            _remove_temp_file(file_id)

def _remove_temp_file(file_id):
    """Remove a temporary file and its registry entry.
    
    Thread-safe removal of temporary files with proper locking.
    
    Args:
        file_id: Unique identifier for the temporary file
    """
    try:
        with temp_files_lock:
            if file_id in temp_files:
                file_path = temp_files[file_id].get('path')
                if file_path and os.path.exists(file_path):
                    os.unlink(file_path)
                del temp_files[file_id]
                logger.info(f"Removed temp file: {file_id}")
    except Exception as e:
        logger.error(f"Error removing temp file {file_id}: {e}")

def schedule_cleanup():
    """Schedule periodic cleanup of old temporary files"""
    cleanup_old_files()
    timer = threading.Timer(CLEANUP_INTERVAL, schedule_cleanup)
    timer.daemon = True
    timer.start()

# Initialize cleanup scheduler
schedule_cleanup()

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def clean_for_json(obj):
    """Convert numpy and pandas types to Python native types for JSON serialization.
    
    Args:
        obj: Object to convert (numpy/pandas type or native Python type)
        
    Returns:
        JSON-serializable Python native type
    """
    # Check for NaN/None first
    if pd.isna(obj) or obj is None or (isinstance(obj, float) and np.isnan(obj)):
        return None
    elif isinstance(obj, (np.integer, np.int_)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        # Additional check for NaN in numpy floats
        val = float(obj)
        return None if np.isnan(val) else val
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    return obj


# ============================================================================
# MAIN PROCESSING FUNCTIONS
# ============================================================================

def process_csv(file_content, tss, offset, mode_input, intrpl_max, upload_id=None):
    """
    Process CSV content and return result as gzip-compressed JSON response.
    
    Implements exact processing logic from data_prep2.py reference implementation.
    
    Args:
        file_content: CSV content as string
        tss: Time step size in minutes (float)
        offset: Offset in minutes (float)
        mode_input: Processing mode ('mean', 'intrpl', 'nearest', 'nearest (mean)')
        intrpl_max: Maximum interpolation time in minutes (float)
        upload_id: Optional upload ID for Socket.IO progress tracking (str)
        
    Returns:
        Flask Response with gzip-compressed JSON data
    """
    def emit_progress(step, progress, message):
        """Emit progress update via Socket.IO"""
        if upload_id:
            try:
                from flask import current_app
                socketio = current_app.extensions.get('socketio')
                if socketio:
                    socketio.emit('processing_progress', {
                        'uploadId': upload_id,
                        'step': step,
                        'progress': progress,
                        'message': message
                    }, room=upload_id)
            except Exception as e:
                logger.error(f"Error emitting progress: {e}")
    
    try:
        emit_progress('parsing', 10, 'Parsing CSV data...')
        
        try:
            # Parse CSV data
            emit_progress('parsing', 20, 'Loading CSV data...')
            emit_progress('parsing', 30, 'Parsing CSV with pandas...')
            try:
                df = pd.read_csv(StringIO(file_content), delimiter=';', skipinitialspace=True, on_bad_lines='skip')
                # Successfully parsed CSV
                emit_progress('parsing', 40, f'Successfully parsed {len(df)} rows')
            except Exception as pandas_error:
                logger.error(f"CSV parsing failed: {str(pandas_error)}")
                # Try with quoting=csv.QUOTE_NONE
                import csv
                try:
                    df = pd.read_csv(StringIO(file_content), delimiter=';', skipinitialspace=True, 
                                   quoting=csv.QUOTE_NONE, on_bad_lines='skip')
                    # Successfully parsed with QUOTE_NONE
                except Exception as final_error:
                    logger.error(f"All parsing attempts failed: {str(final_error)}")
                    raise pandas_error
            
            df.columns = df.columns.str.strip()
            
            if len(df.columns) < 2:
                raise ValueError(f"CSV must have at least 2 columns, found {len(df.columns)}: {list(df.columns)}")
            
            # Get column names
            utc_col_name = df.columns[0]
            value_col_name = df.columns[1]
            # Using identified columns
            
            # Rename only UTC column to standard name, keep value column name flexible
            df = df.rename(columns={utc_col_name: 'UTC'})
            
            emit_progress('preprocessing', 50, 'Converting data types...')
            # Convert UTC column to datetime and values to numeric format
            df['UTC'] = pd.to_datetime(df['UTC'], format='%Y-%m-%d %H:%M:%S')
            df[value_col_name] = pd.to_numeric(df[value_col_name], errors='coerce')
            
            # Remove rows with NaN values
            initial_count = len(df)
            df = df.dropna(subset=['UTC', value_col_name])
            final_count = len(df)
            
            if initial_count != final_count:
                logger.info(f"Removed {initial_count - final_count} invalid rows")
                
        except Exception as e:
            logger.error(f"Error parsing CSV data: {str(e)}")
            return jsonify({"error": f"CSV parsing failed: {str(e)}"}), 400
        
        # Clean and sort data (matching data_prep2.py logic)
        df = df.drop_duplicates(subset=['UTC']).sort_values('UTC').reset_index(drop=True)
        
        if df.empty:
            return jsonify({"error": "No valid timestamps found"}), 400
            
        # Get time boundaries from raw data
        time_min_raw = df['UTC'].iloc[0]
        time_max_raw = df['UTC'].iloc[-1]
        
        # Calculate offset following data_prep2.py logic
        # Offset der unteren Zeitgrenze in der Rohdaten
        offset_strt = pd.Timedelta(minutes=time_min_raw.minute,
                                  seconds=time_min_raw.second,
                                  microseconds=time_min_raw.microsecond)
        
        # Adjust offset to be within time step size
        while offset >= tss:
            offset -= tss
        
        # Find lower time boundary in processed data
        i = 0
        loop = True
        while loop:
            a = pd.Timedelta(minutes=i*tss+offset)
            if a >= offset_strt:
                loop = False
            else:
                i += 1
        time_min = time_min_raw - offset_strt + a
        
        logger.info(f"Calculated start time: {time_min} (offset: {offset} minutes)")
        
        # Create continuous timestamp list
        time_list = []
        i = 0
        loop = True
        while loop:
            a = pd.Timedelta(minutes=i*tss)
            time = time_min + a
            if time <= time_max_raw:
                time_list.append(time)
            else:
                loop = False
            i += 1
        
        emit_progress('processing', 60, f'Starting {mode_input} processing...')
        
        # Initialize value list for results
        value_list = []
        
        # Counter for iterating through raw data
        i_raw = 0
        
        if mode_input == "mean":
            emit_progress('processing', 65, 'Calculating mean values...')
            
            # Loop through all time steps in continuous timestamp
            for i in range(len(time_list)):
                if i % 100 == 0:  # Update progress periodically
                    progress = 65 + (20 * i / len(time_list))
                    emit_progress('processing', progress, f'Processing {i}/{len(time_list)} time points...')
                
                # Time boundaries for mean calculation (investigation window)
                time_int_min = time_list[i] - pd.Timedelta(minutes=tss/2)
                time_int_max = time_list[i] + pd.Timedelta(minutes=tss/2)
                
                # Consider adjacent investigation windows
                if i > 0:
                    i_raw -= 1
                if i > 0 and df['UTC'].iloc[i_raw] < time_int_min:
                    i_raw += 1
                
                # Initialize list for values in investigation window
                value_int_list = []
                
                # List numeric values in investigation window
                while (i_raw < len(df) and 
                       df['UTC'].iloc[i_raw] <= time_int_max and 
                       df['UTC'].iloc[i_raw] >= time_int_min):
                    
                    try:
                        value_int_list.append(float(df[value_col_name].iloc[i_raw]))
                    except:
                        pass
                    i_raw += 1
                
                # Calculate mean over numeric values in investigation window
                if len(value_int_list) > 0:
                    value_list.append(statistics.mean(value_int_list))
                else:
                    value_list.append(np.nan)
            
        elif mode_input == "intrpl":
            emit_progress('processing', 65, 'Starting interpolation...')
            
            # Reset counter for raw data iteration
            i_raw = 0
            
            # Direction of loop execution (1: forward, -1: backward)
            direct = 1
            
            # Loop through all time steps in continuous timestamp
            for i in range(len(time_list)):
                if i % 100 == 0:  # Update progress periodically
                    progress = 65 + (20 * i / len(time_list))
                    emit_progress('processing', progress, f'Processing interpolation {i}/{len(time_list)}...')
                
                # Find next value (forward search)
                if direct == 1:
                    loop = True
                    while i_raw < len(df) and loop:
                        # Current time in raw data is after or equal to current time in continuous timestamp
                        if df['UTC'].iloc[i_raw] >= time_list[i]:
                            try:
                                # Get UTC and value from next point
                                time_next = df['UTC'].iloc[i_raw]
                                value_next = float(df[value_col_name].iloc[i_raw])
                                loop = False
                            except:
                                # Increment counter if value is not numeric
                                i_raw += 1
                        else:
                            # Increment counter if current time in raw data is before current time
                            i_raw += 1
                    
                    # Entire raw data traversed without finding valid value
                    if i_raw >= len(df):
                        value_list.append(np.nan)
                        i_raw = 0
                        direct = 1
                    else:
                        # Switch direction
                        direct = -1
                
                # Find prior value (backward search)
                if direct == -1:
                    loop = True
                    while i_raw >= 0 and loop:
                        # Current time in raw data is before or equal to current time in continuous timestamp
                        if df['UTC'].iloc[i_raw] <= time_list[i]:
                            try:
                                # Get UTC and value from prior point
                                time_prior = df['UTC'].iloc[i_raw]
                                value_prior = float(df[value_col_name].iloc[i_raw])
                                loop = False
                            except:
                                # Decrement counter if value is not numeric
                                i_raw -= 1
                        else:
                            # Decrement counter if current time in raw data is after current time
                            i_raw -= 1
                    
                    # Entire raw data traversed without finding valid value
                    if i_raw < 0:
                        value_list.append(np.nan)
                        i_raw = 0
                        direct = 1
                    else:
                        # Valid values found before and after current time
                        delta_time = time_next - time_prior
                        delta_time_sec = delta_time.total_seconds()
                        delta_value = value_prior - value_next
                        
                        # Times coincide or constant value - no interpolation needed
                        if delta_time_sec == 0 or (delta_value == 0 and delta_time_sec <= intrpl_max * 60):
                            value_list.append(value_prior)
                        # Time gap too large - no interpolation possible
                        elif delta_time_sec > intrpl_max * 60:
                            value_list.append(np.nan)
                        # Linear interpolation
                        else:
                            delta_time_prior_sec = (time_list[i] - time_prior).total_seconds()
                            value_list.append(value_prior - delta_value / delta_time_sec * delta_time_prior_sec)
                        
                        direct = 1
            
        elif mode_input in ["nearest", "nearest (mean)"]:
            emit_progress('processing', 65, f'Processing {mode_input}...')
            
            # Reset counter for raw data iteration
            i_raw = 0
            
            # Loop through all time steps in continuous timestamp
            for i in range(len(time_list)):
                if i % 100 == 0:  # Update progress periodically
                    progress = 65 + (20 * i / len(time_list))
                    emit_progress('processing', progress, f'Processing {i}/{len(time_list)} time points...')
                
                # Time boundaries for investigation (investigation window)
                time_int_min = time_list[i] - pd.Timedelta(minutes=tss/2)
                time_int_max = time_list[i] + pd.Timedelta(minutes=tss/2)
                
                # Consider adjacent investigation windows
                if i > 0:
                    i_raw -= 1
                if i > 0 and df['UTC'].iloc[i_raw] < time_int_min:
                    i_raw += 1
                
                # Initialize lists for investigation window
                value_int_list = []
                delta_time_int_list = []
                
                # List numeric values with time differences in investigation window
                while (i_raw < len(df) and 
                       df['UTC'].iloc[i_raw] <= time_int_max and 
                       df['UTC'].iloc[i_raw] >= time_int_min):
                    
                    try:
                        value_int_list.append(float(df[value_col_name].iloc[i_raw]))
                        delta_time_int_list.append(abs(time_list[i] - df['UTC'].iloc[i_raw]))
                    except:
                        pass
                    i_raw += 1
                
                # Find temporally nearest value
                if len(value_int_list) > 0:
                    if mode_input == "nearest":
                        # If two values have same minimum distance, take first one
                        delta_time_int_array = np.array(delta_time_int_list)
                        item_index = np.where(delta_time_int_array == min(delta_time_int_list))
                        value_list.append(value_int_list[item_index[0][0]])
                    
                    elif mode_input == "nearest (mean)":
                        # If two values have same minimum distance, take mean of all such values
                        delta_time_int_array = np.array(delta_time_int_list)
                        item_index = np.where(delta_time_int_array == min(delta_time_int_list))
                        value_int_mean_list = []
                        for i1 in range(len(item_index[0])):
                            value_int_mean_list.append(value_int_list[item_index[0][i1]])
                        value_list.append(statistics.mean(value_int_mean_list))
                else:
                    value_list.append(np.nan)
        
        # Create DataFrame with processed data (matching data_prep2.py format)
        df_processed = pd.DataFrame({"UTC": time_list, value_col_name: value_list})
        
        emit_progress('finalizing', 85, 'Converting results to JSON...')
        # Convert results to JSON format with specific time format
        result = []
        for i in range(len(df_processed)):
            result.append({
                "UTC": df_processed["UTC"].iloc[i].strftime("%Y-%m-%d %H:%M:%S"),
                value_col_name: clean_for_json(df_processed[value_col_name].iloc[i])
            })
        
        emit_progress('finalizing', 95, 'Compressing data...')
        # Compress and return result
        result_json = json.dumps(result)
        compressed_data = gzip.compress(result_json.encode('utf-8'))
        
        emit_progress('complete', 100, f'Processing complete! Generated {len(result)} data points.')
        
        response = Response(compressed_data)
        response.headers['Content-Encoding'] = 'gzip'
        response.headers['Content-Type'] = 'application/json'
        return response
    except Exception as e:
        error_msg = f"Error processing CSV: {str(e)}"
        logger.error(error_msg)
        return jsonify({"error": str(e)}), 400

# ============================================================================
# API ENDPOINTS
# ============================================================================

@bp.route('/upload_chunk', methods=['POST'])
def upload_chunk():
    """
    Handle chunked CSV file uploads and processing.
    
    Expected form data parameters:
        uploadId: Unique upload identifier (string)
        chunkIndex: Zero-based chunk index (int)
        totalChunks: Total number of chunks (int)
        fileChunk: Blob/File containing CSV data chunk
        tss: Time step size in minutes (float)
        offset: Offset in minutes (float)
        mode: Processing mode ('mean', 'intrpl', 'nearest', 'nearest (mean)')
        intrplMax: Max interpolation time in minutes (float, default: 60)
        
    Returns:
        JSON response with processing result or status update
    """
    try:
        # Check if we have all required parameters
        if 'fileChunk' not in request.files:
            return jsonify({"error": "No file chunk found"}), 400

        # Load parameters from form data
        try:
            # Sanitize and validate parameters
            raw_upload_id = request.form.get('uploadId')
            upload_id = re.sub(r'[^a-zA-Z0-9_-]', '', raw_upload_id) if raw_upload_id else None
            chunk_index = int(request.form.get('chunkIndex', 0))
            total_chunks = int(request.form.get('totalChunks', 0))
            tss = float(request.form.get('tss', 0))
            offset = float(request.form.get('offset', 0))
            mode = request.form.get('mode', '')
            intrpl_max = float(request.form.get('intrplMax', 60))
        except (ValueError, TypeError) as e:
            logger.error(f"Invalid parameters: {e}")
            return jsonify({"error": "Invalid parameter format"}), 400

        # Validate required parameters
        if not upload_id:
            return jsonify({"error": "Upload ID is required"}), 400
        if not mode:
            return jsonify({"error": "Processing mode is required"}), 400
        if tss <= 0:
            return jsonify({"error": "Time step size must be positive"}), 400

        # Get chunk file
        chunk = request.files['fileChunk']
        if not chunk:
            return jsonify({"error": "Empty chunk received"}), 400

        # Create folder if doesn't exist
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)

        # Check chunk size before saving
        chunk.stream.seek(0, 2)  # Seek to end
        chunk_size = chunk.stream.tell()
        chunk.stream.seek(0)  # Reset to beginning
        
        if chunk_size > MAX_CHUNK_SIZE:
            return jsonify({"error": f"Chunk too large: {chunk_size} bytes (max: {MAX_CHUNK_SIZE})"}), 413
        
        # Save chunk
        chunk_filename = os.path.join(UPLOAD_FOLDER, f"{upload_id}_{chunk_index}.chunk")
        chunk.save(chunk_filename)

        # Saved chunk

        # Check if all chunks are received
        received_chunks = [f for f in os.listdir(UPLOAD_FOLDER) 
                         if f.startswith(upload_id + "_")]

        if len(received_chunks) == total_chunks:
            # All chunks received, processing
            
            # Sort chunks by index - properly extract chunk index from filename
            def extract_chunk_index(filename):
                try:
                    # Format: {uploadId}_{chunkIndex}.chunk
                    # Look for last part before .chunk
                    parts = filename.split("_")
                    # Last part is chunkIndex.chunk, get numeric value only
                    chunk_part = parts[-1].split(".")[0]
                    return int(chunk_part)
                except (ValueError, IndexError) as e:
                    logger.error(f"Error parsing chunk filename {filename}: {e}")
                    return 0  # Fallback value
            
            chunks_sorted = sorted(received_chunks, key=extract_chunk_index)
            
            try:
                
                # Join all chunks using BytesIO for better performance
                full_content_buffer = BytesIO()
                
                for i, chunk_file in enumerate(chunks_sorted):
                    chunk_path = os.path.join(UPLOAD_FOLDER, chunk_file)
                    
                    with open(chunk_path, 'rb') as f:
                        chunk_bytes = f.read()
                        full_content_buffer.write(chunk_bytes)
                    
                    # Delete chunk file after reading
                    os.remove(chunk_path)
                
                # Decode the complete content
                full_content_buffer.seek(0)
                try:
                    full_content = full_content_buffer.read().decode('utf-8')
                except UnicodeDecodeError:
                    # Try alternative encodings
                    full_content_buffer.seek(0)
                    full_content = full_content_buffer.read().decode('latin-1')
                
                # Process joined content with upload_id for progress tracking
                return process_csv(full_content, tss, offset, mode, intrpl_max, upload_id)
                
            except Exception as e:
                # In case of error, delete all chunks
                for chunk_file in chunks_sorted:
                    try:
                        os.remove(os.path.join(UPLOAD_FOLDER, chunk_file))
                    except:
                        pass
                raise

        # Return status about received chunk
        return jsonify({
            "message": f"Chunk {chunk_index + 1}/{total_chunks} received",
            "uploadId": upload_id,
            "chunkIndex": chunk_index,
            "totalChunks": total_chunks,
            "remainingChunks": total_chunks - len(received_chunks)
        }), 200

    except Exception as e:
        logger.error(f"Unexpected error in upload_chunk: {str(e)}", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500

@bp.route('/prepare-save', methods=['POST'])
def prepare_save():
    """Prepare processed data for download.
    
    Expects JSON with 'data' containing CSV rows and optional 'fileName'.
    
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

        # Create temporary file and write CSV data
        temp_file = tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.csv')
        writer = csv.writer(temp_file, delimiter=';')
        for row in save_data:
            writer.writerow(row)
        temp_file.close()

        # Generate unique ID based on current time
        file_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        with temp_files_lock:
            temp_files[file_id] = {
                'path': temp_file.name,
                'fileName': file_name or f"data_{file_id}.csv",  # Use sent name or default
                'timestamp': time.time()
            }

        return jsonify({"message": "File prepared for download", "fileId": file_id}), 200
    except Exception as e:
        logger.error(f"Error in prepare_save: {str(e)}", exc_info=True)
        return jsonify({"error": "Failed to prepare file"}), 500

@bp.route('/cancel-upload', methods=['POST'])
def cancel_upload():
    """Cancel an ongoing upload and clean up resources"""
    try:
        data = request.get_json(force=True, silent=True)
        
        if not data or 'uploadId' not in data:
            return jsonify({"error": "uploadId is required"}), 400
        
        upload_id = data['uploadId']
        
        # Validate upload ID format (basic security)
        if not re.match(r'^fp_\d+_[a-zA-Z0-9]+$', upload_id):
            return jsonify({"error": "Invalid upload ID format"}), 400
        
        # Clean up any temporary chunk files
        try:
            chunk_pattern = os.path.join(UPLOAD_FOLDER, f"{upload_id}_*.chunk")
            import glob
            for chunk_file in glob.glob(chunk_pattern):
                if os.path.exists(chunk_file):
                    os.remove(chunk_file)
                    logger.info(f"Removed chunk file: {chunk_file}")
        except Exception as cleanup_error:
            logger.warning(f"Error cleaning up chunks for {upload_id}: {cleanup_error}")
        
        # Send cancellation message via Socket.IO
        try:
            from flask import current_app
            socketio = current_app.extensions.get('socketio')
            if socketio:
                socketio.emit('processing_progress', {
                    'uploadId': upload_id,
                    'progress': 0,
                    'status': 'error',
                    'message': 'Upload canceled by user',
                    'step': 'canceled'
                }, room=upload_id)
        except Exception as socket_error:
            logger.warning(f"Error sending cancel message via socket: {socket_error}")
        
        return jsonify({
            "message": "Upload canceled successfully",
            "uploadId": upload_id
        }), 200
        
    except Exception as e:
        logger.error(f"Error in cancel_upload: {str(e)}", exc_info=True)
        return jsonify({"error": "Failed to cancel upload"}), 500

@bp.route('/download/<file_id>', methods=['GET'])
def download_file(file_id):
    """Download a prepared CSV file.
    
    Args:
        file_id: Unique identifier for the file
        
    Returns:
        CSV file download or error response
    """
    try:
        with temp_files_lock:
            if file_id not in temp_files:
                return jsonify({"error": "File not found or expired"}), 404
            
            file_info = temp_files[file_id].copy()  # Make a copy to use outside the lock
            file_path = file_info['path']
        
        if not os.path.exists(file_path):
            return jsonify({"error": "File not found"}), 404

        # Use saved file name
        download_name = file_info['fileName']
        
        response = send_file(
            file_path,
            as_attachment=True,
            download_name=download_name,
            mimetype='text/csv; charset=utf-8'
        )
        
        # Cleanup after sending
        _remove_temp_file(file_id)
        return response

    except Exception as e:
        logger.error(f"Error in download_file: {str(e)}", exc_info=True)
        return jsonify({"error": "Failed to download file"}), 500