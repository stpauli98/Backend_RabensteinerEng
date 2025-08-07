"""
Data Pipeline API - Consolidated Data Processing Services

Consolidates functionality from:
- firstProcessing.py - Initial CSV processing with chunking
- load_row_data.py - Raw data loading operations  
- data_processing_main.py - Advanced data cleaning and filtering
- adjustmentsOfData.py - Data adjustment operations

API Endpoints:
    POST /api/data/upload-chunk - File chunk uploads
    POST /api/data/finalize-upload - Complete file assembly
    POST /api/data/process - Data processing with resampling
    POST /api/data/clean - Advanced cleaning and filtering
    POST /api/data/adjust - Data adjustments
    POST /api/data/prepare-save - Prepare for download
    GET  /api/data/download/<file_id> - Download processed files
    POST /api/data/cancel-upload - Cancel ongoing uploads
"""

import os
import sys
import tempfile
import logging
import json
import csv
import time
import secrets
import gzip
import re
import statistics
import threading
import traceback
from datetime import datetime, timedelta
from io import StringIO, BytesIO
from typing import Dict, Tuple, Optional, Any
from threading import Lock
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Blueprint, request, jsonify, send_file, Response, current_app
from werkzeug.utils import secure_filename
from flask_socketio import emit

# Import centralized storage config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.storage_config import storage_config

# ============================================================================
# CONFIGURATION
# ============================================================================

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create Blueprint
bp = Blueprint('data_pipeline', __name__)

# Helper function to get socketio instance
def get_socketio():
    return current_app.extensions['socketio']

# File size limits
MAX_CHUNK_SIZE = 10 * 1024 * 1024  # 10MB per chunk
MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB total
TEMP_FILE_LIFETIME = 3600  # 1 hour in seconds

# Thread-safe storage for temporary files and chunks
temp_files: Dict[str, Dict[str, Any]] = {}
chunk_storage: Dict[str, Dict[str, Any]] = {}
storage_lock = Lock()

# Configuration constants
UPLOAD_EXPIRY_TIME = 30 * 60  # 30 minutes in seconds
ALLOWED_EXTENSIONS = {'.csv', '.txt'}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

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

def detect_datetime_format(datetime_str: str) -> Optional[str]:
    """Detect datetime format from string"""
    formats = [
        "%Y-%m-%d %H:%M:%S",
        "%m/%d/%Y %H:%M",
        "%d.%m.%Y %H:%M",
        "%Y-%m-%d",
        "%m/%d/%Y",
        "%d.%m.%Y",
        "%d-%m-%Y %H:%M:%S",
        "%d-%m-%Y",
        "%Y/%m/%d %H:%M:%S",
        "%Y/%m/%d"
    ]
    
    for fmt in formats:
        try:
            datetime.strptime(datetime_str, fmt)
            return fmt
        except ValueError:
            continue
    return None

def emit_progress(room, stage, progress, message="", error=None):
    """Emit progress update via SocketIO"""
    try:
        socketio = get_socketio()
        data = {
            'stage': stage,
            'progress': progress,
            'message': message,
            'timestamp': datetime.now().isoformat()
        }
        if error:
            data['error'] = str(error)
        
        socketio.emit('progress_update', data, room=room)
        logger.info(f"Emitted progress to room {room}: {stage} - {progress}%")
    except Exception as e:
        logger.error(f"Failed to emit progress: {e}")

# ============================================================================
# FILE UPLOAD ENDPOINTS
# ============================================================================

@bp.route('/upload-chunk', methods=['POST'])
def upload_chunk():
    """
    Upload file chunks for processing
    Consolidated from firstProcessing and load_row_data modules
    """
    try:
        # Get request data
        upload_id = request.form.get('uploadId')
        chunk_index = int(request.form.get('chunkIndex', 0))
        total_chunks = int(request.form.get('totalChunks', 1))
        filename = request.form.get('filename', '')
        file_type = request.form.get('fileType', 'data_file')
        
        if not upload_id:
            return jsonify({'error': 'Upload ID is required'}), 400
            
        if 'chunk' not in request.files:
            return jsonify({'error': 'No chunk file provided'}), 400
            
        chunk_file = request.files['chunk']
        if chunk_file.filename == '':
            return jsonify({'error': 'No chunk selected'}), 400

        # Validate file size
        chunk_data = chunk_file.read()
        if len(chunk_data) > MAX_CHUNK_SIZE:
            return jsonify({'error': f'Chunk size exceeds limit of {MAX_CHUNK_SIZE} bytes'}), 413

        # Initialize upload tracking
        with storage_lock:
            if upload_id not in chunk_storage:
                chunk_storage[upload_id] = {
                    'chunks': {},
                    'filename': filename,
                    'file_type': file_type,
                    'total_chunks': total_chunks,
                    'last_activity': time.time(),
                    'session_dir': str(storage_config.get_session_dir(upload_id))
                }
                os.makedirs(chunk_storage[upload_id]['session_dir'], exist_ok=True)
            
            # Store chunk
            chunk_storage[upload_id]['chunks'][chunk_index] = {
                'data': chunk_data,
                'received_at': time.time()
            }
            chunk_storage[upload_id]['last_activity'] = time.time()
            
            received_chunks = len(chunk_storage[upload_id]['chunks'])
            progress = int((received_chunks / total_chunks) * 100)

        # Emit progress
        emit_progress(upload_id, 'upload', progress, f'Uploaded chunk {chunk_index + 1}/{total_chunks}')

        # Check if all chunks received
        if received_chunks == total_chunks:
            try:
                # Reassemble file
                full_data = b''.join([
                    chunk_storage[upload_id]['chunks'][i]['data'] 
                    for i in range(total_chunks)
                ])
                
                # Save assembled file
                file_path = os.path.join(chunk_storage[upload_id]['session_dir'], filename)
                with open(file_path, 'wb') as f:
                    f.write(full_data)
                
                emit_progress(upload_id, 'upload', 100, f'File {filename} assembled successfully')
                
                return jsonify({
                    'status': 'complete',
                    'message': f'All chunks received and file assembled',
                    'filename': filename,
                    'file_path': file_path,
                    'file_size': len(full_data)
                })
                
            except Exception as e:
                logger.error(f"Error assembling file: {e}")
                emit_progress(upload_id, 'upload', -1, error=f'Failed to assemble file: {str(e)}')
                return jsonify({'error': f'Failed to assemble file: {str(e)}'}), 500
        
        return jsonify({
            'status': 'chunk_received',
            'chunk_index': chunk_index,
            'received_chunks': received_chunks,
            'total_chunks': total_chunks,
            'progress': progress
        })
        
    except Exception as e:
        logger.error(f"Error in upload_chunk: {e}")
        if upload_id:
            emit_progress(upload_id, 'upload', -1, error=str(e))
        return jsonify({'error': str(e)}), 500

@bp.route('/finalize-upload', methods=['POST'])
def finalize_upload():
    """
    Finalize chunked file upload and prepare for processing
    """
    try:
        data = request.get_json()
        upload_id = data.get('uploadId')
        
        if not upload_id or upload_id not in chunk_storage:
            return jsonify({'error': 'Invalid upload ID'}), 400
            
        upload_info = chunk_storage[upload_id]
        
        return jsonify({
            'status': 'success',
            'upload_id': upload_id,
            'filename': upload_info['filename'],
            'session_dir': upload_info['session_dir'],
            'message': 'Upload finalized successfully'
        })
        
    except Exception as e:
        logger.error(f"Error finalizing upload: {e}")
        return jsonify({'error': str(e)}), 500

@bp.route('/cancel-upload', methods=['POST'])
def cancel_upload():
    """Cancel ongoing upload and cleanup"""
    try:
        data = request.get_json()
        upload_id = data.get('uploadId')
        
        if not upload_id:
            return jsonify({'error': 'Upload ID is required'}), 400
            
        with storage_lock:
            if upload_id in chunk_storage:
                # Cleanup session directory if exists
                session_dir = chunk_storage[upload_id].get('session_dir')
                if session_dir and os.path.exists(session_dir):
                    import shutil
                    shutil.rmtree(session_dir)
                
                del chunk_storage[upload_id]
        
        emit_progress(upload_id, 'upload', -1, 'Upload cancelled')
        
        return jsonify({
            'status': 'success',
            'message': 'Upload cancelled and cleaned up'
        })
        
    except Exception as e:
        logger.error(f"Error cancelling upload: {e}")
        return jsonify({'error': str(e)}), 500

# ============================================================================
# DATA PROCESSING ENDPOINTS
# ============================================================================

@bp.route('/process', methods=['POST'])
def process_data():
    """
    Process uploaded data with resampling methods
    Consolidated from firstProcessing functionality
    """
    try:
        data = request.get_json()
        upload_id = data.get('uploadId')
        resample_method = data.get('resampleMethod', 'mean')
        target_interval = data.get('targetInterval', '1H')
        
        if not upload_id or upload_id not in chunk_storage:
            return jsonify({'error': 'Invalid upload ID'}), 400
            
        upload_info = chunk_storage[upload_id]
        file_path = os.path.join(upload_info['session_dir'], upload_info['filename'])
        
        if not os.path.exists(file_path):
            return jsonify({'error': 'File not found'}), 404
            
        emit_progress(upload_id, 'processing', 10, 'Starting data processing')
        
        # Read and process CSV
        df = pd.read_csv(file_path)
        emit_progress(upload_id, 'processing', 30, 'File loaded, applying processing')
        
        # Apply resampling if datetime column detected
        if len(df.columns) > 0:
            first_col = df.iloc[:, 0]
            datetime_format = detect_datetime_format(str(first_col.iloc[0]))
            
            if datetime_format:
                try:
                    df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0], format=datetime_format)
                    df = df.set_index(df.columns[0])
                    
                    # Apply resampling
                    if resample_method == 'mean':
                        df_resampled = df.resample(target_interval).mean()
                    elif resample_method == 'interpolate':
                        df_resampled = df.resample(target_interval).interpolate()
                    else:  # nearest
                        df_resampled = df.resample(target_interval).nearest()
                    
                    df = df_resampled.reset_index()
                    emit_progress(upload_id, 'processing', 70, f'Applied {resample_method} resampling')
                    
                except Exception as e:
                    logger.warning(f"Resampling failed: {e}")
                    emit_progress(upload_id, 'processing', 50, 'Resampling skipped, processing raw data')
        
        # Save processed file
        processed_filename = f"processed_{upload_info['filename']}"
        processed_path = os.path.join(upload_info['session_dir'], processed_filename)
        df.to_csv(processed_path, index=False)
        
        emit_progress(upload_id, 'processing', 100, 'Processing completed')
        
        return jsonify({
            'status': 'success',
            'processed_file': processed_filename,
            'processed_path': processed_path,
            'rows': len(df),
            'columns': list(df.columns),
            'message': 'Data processing completed successfully'
        })
        
    except Exception as e:
        logger.error(f"Error processing data: {e}")
        if upload_id:
            emit_progress(upload_id, 'processing', -1, error=str(e))
        return jsonify({'error': str(e)}), 500

@bp.route('/clean', methods=['POST'])
def clean_data():
    """
    Advanced data cleaning and filtering
    Consolidated from data_processing_main functionality
    """
    try:
        data = request.get_json()
        upload_id = data.get('uploadId')
        cleaning_options = data.get('cleaningOptions', {})
        
        if not upload_id or upload_id not in chunk_storage:
            return jsonify({'error': 'Invalid upload ID'}), 400
            
        upload_info = chunk_storage[upload_id]
        file_path = os.path.join(upload_info['session_dir'], upload_info['filename'])
        
        if not os.path.exists(file_path):
            return jsonify({'error': 'File not found'}), 404
            
        emit_progress(upload_id, 'cleaning', 10, 'Starting data cleaning')
        
        df = pd.read_csv(file_path)
        original_rows = len(df)
        
        # Remove outliers if requested
        if cleaning_options.get('remove_outliers', False):
            emit_progress(upload_id, 'cleaning', 30, 'Removing outliers')
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            for col in numeric_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        
        # Fill missing values if requested
        if cleaning_options.get('fill_missing', False):
            emit_progress(upload_id, 'cleaning', 60, 'Filling missing values')
            method = cleaning_options.get('fill_method', 'interpolate')
            
            if method == 'interpolate':
                df = df.interpolate()
            elif method == 'forward_fill':
                df = df.fillna(method='ffill')
            elif method == 'backward_fill':
                df = df.fillna(method='bfill')
        
        # Remove duplicates if requested
        if cleaning_options.get('remove_duplicates', False):
            emit_progress(upload_id, 'cleaning', 80, 'Removing duplicates')
            df = df.drop_duplicates()
        
        # Save cleaned file
        cleaned_filename = f"cleaned_{upload_info['filename']}"
        cleaned_path = os.path.join(upload_info['session_dir'], cleaned_filename)
        df.to_csv(cleaned_path, index=False)
        
        emit_progress(upload_id, 'cleaning', 100, 'Data cleaning completed')
        
        return jsonify({
            'status': 'success',
            'cleaned_file': cleaned_filename,
            'cleaned_path': cleaned_path,
            'original_rows': original_rows,
            'cleaned_rows': len(df),
            'rows_removed': original_rows - len(df),
            'message': 'Data cleaning completed successfully'
        })
        
    except Exception as e:
        logger.error(f"Error cleaning data: {e}")
        if upload_id:
            emit_progress(upload_id, 'cleaning', -1, error=str(e))
        return jsonify({'error': str(e)}), 500

@bp.route('/adjust', methods=['POST'])
def adjust_data():
    """
    Data adjustment operations
    Consolidated from adjustmentsOfData functionality
    """
    try:
        data = request.get_json()
        upload_id = data.get('uploadId')
        adjustments = data.get('adjustments', {})
        
        if not upload_id or upload_id not in chunk_storage:
            return jsonify({'error': 'Invalid upload ID'}), 400
            
        upload_info = chunk_storage[upload_id]
        file_path = os.path.join(upload_info['session_dir'], upload_info['filename'])
        
        if not os.path.exists(file_path):
            return jsonify({'error': 'File not found'}), 404
            
        emit_progress(upload_id, 'adjusting', 10, 'Starting data adjustments')
        
        df = pd.read_csv(file_path)
        
        # Apply column scaling if requested
        if 'scaling' in adjustments:
            emit_progress(upload_id, 'adjusting', 40, 'Applying scaling adjustments')
            scaling_config = adjustments['scaling']
            
            for col_name, scale_factor in scaling_config.items():
                if col_name in df.columns and pd.api.types.is_numeric_dtype(df[col_name]):
                    df[col_name] = df[col_name] * scale_factor
        
        # Apply offset adjustments if requested
        if 'offsets' in adjustments:
            emit_progress(upload_id, 'adjusting', 70, 'Applying offset adjustments')
            offset_config = adjustments['offsets']
            
            for col_name, offset_value in offset_config.items():
                if col_name in df.columns and pd.api.types.is_numeric_dtype(df[col_name]):
                    df[col_name] = df[col_name] + offset_value
        
        # Save adjusted file
        adjusted_filename = f"adjusted_{upload_info['filename']}"
        adjusted_path = os.path.join(upload_info['session_dir'], adjusted_filename)
        df.to_csv(adjusted_path, index=False)
        
        emit_progress(upload_id, 'adjusting', 100, 'Data adjustments completed')
        
        return jsonify({
            'status': 'success',
            'adjusted_file': adjusted_filename,
            'adjusted_path': adjusted_path,
            'rows': len(df),
            'columns': list(df.columns),
            'message': 'Data adjustments completed successfully'
        })
        
    except Exception as e:
        logger.error(f"Error adjusting data: {e}")
        if upload_id:
            emit_progress(upload_id, 'adjusting', -1, error=str(e))
        return jsonify({'error': str(e)}), 500

# ============================================================================
# FILE DOWNLOAD ENDPOINTS
# ============================================================================

@bp.route('/prepare-save', methods=['POST'])
def prepare_save():
    """Prepare processed file for download"""
    try:
        data = request.get_json()
        upload_id = data.get('uploadId')
        filename = data.get('filename')
        
        if not upload_id or upload_id not in chunk_storage:
            return jsonify({'error': 'Invalid upload ID'}), 400
            
        upload_info = chunk_storage[upload_id]
        file_path = os.path.join(upload_info['session_dir'], filename)
        
        if not os.path.exists(file_path):
            return jsonify({'error': 'File not found'}), 404
            
        # Generate download ID
        download_id = secrets.token_urlsafe(16)
        
        # Store file info for download
        with storage_lock:
            temp_files[download_id] = {
                'path': file_path,
                'filename': filename,
                'created_at': time.time()
            }
        
        return jsonify({
            'status': 'success',
            'download_id': download_id,
            'filename': filename,
            'message': 'File prepared for download'
        })
        
    except Exception as e:
        logger.error(f"Error preparing save: {e}")
        return jsonify({'error': str(e)}), 500

@bp.route('/download/<file_id>', methods=['GET'])
def download_file(file_id):
    """Download processed file"""
    try:
        if file_id not in temp_files:
            return jsonify({'error': 'Invalid file ID or file expired'}), 404
            
        file_info = temp_files[file_id]
        file_path = file_info['path']
        filename = file_info['filename']
        
        if not os.path.exists(file_path):
            return jsonify({'error': 'File not found'}), 404
            
        def cleanup_after_download():
            # Cleanup temp file reference after short delay
            time.sleep(5)
            with storage_lock:
                if file_id in temp_files:
                    del temp_files[file_id]
        
        # Start cleanup in background
        threading.Thread(target=cleanup_after_download, daemon=True).start()
        
        return send_file(
            file_path,
            as_attachment=True,
            download_name=filename,
            mimetype='text/csv'
        )
        
    except Exception as e:
        logger.error(f"Error downloading file: {e}")
        return jsonify({'error': str(e)}), 500

# ============================================================================
# UTILITY ENDPOINTS
# ============================================================================

@bp.route('/upload-status/<upload_id>', methods=['GET'])
def get_upload_status(upload_id):
    """Get current upload status"""
    try:
        if upload_id not in chunk_storage:
            return jsonify({'error': 'Upload ID not found'}), 404
            
        upload_info = chunk_storage[upload_id]
        received_chunks = len(upload_info['chunks'])
        total_chunks = upload_info['total_chunks']
        progress = int((received_chunks / total_chunks) * 100) if total_chunks > 0 else 0
        
        return jsonify({
            'upload_id': upload_id,
            'filename': upload_info['filename'],
            'received_chunks': received_chunks,
            'total_chunks': total_chunks,
            'progress': progress,
            'status': 'complete' if received_chunks == total_chunks else 'in_progress'
        })
        
    except Exception as e:
        logger.error(f"Error getting upload status: {e}")
        return jsonify({'error': str(e)}), 500

# Cleanup old files on module load
cleanup_old_uploads()