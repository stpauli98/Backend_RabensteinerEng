import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from datetime import datetime, timedelta
import tempfile
import os
import csv
import logging
import traceback
import re
from io import StringIO
from flask import request, jsonify, send_file, Blueprint, Response
import json
import shutil
import base64
from collections import OrderedDict
from threading import Lock
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Polygon


# Create blueprint
bp = Blueprint('cloud', __name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Security and validation constants
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB per file
MAX_ROWS = 1_000_000  # Maximum rows in CSV
MAX_COLUMNS = 100  # Maximum columns in CSV
MAX_ACTIVE_UPLOADS = 1000  # Maximum concurrent uploads
UPLOAD_ID_PATTERN = re.compile(r'^[\w\-]{1,64}$')  # Alphanumeric, underscore, hyphen only

# Performance constants
TOLERANCE_ADJUSTMENT_FACTOR = 2
MIN_TOLERANCE_THRESHOLD = 0.01  # 1% of data range
DEFAULT_TOLERANCE_RATIO = 0.1   # 10% of data range
STREAMING_CHUNK_SIZE = 5000
FILE_BUFFER_SIZE = 1024 * 1024  # 1MB

# Upload session management with TTL
class UploadManager:
    """Thread-safe upload session manager with TTL cleanup."""

    def __init__(self, max_size=MAX_ACTIVE_UPLOADS, ttl_hours=1):
        self.uploads = OrderedDict()
        self.max_size = max_size
        self.ttl = timedelta(hours=ttl_hours)
        self.lock = Lock()

    def __getitem__(self, upload_id: str):
        """Dictionary-style access for backward compatibility."""
        return self.get(upload_id)

    def __setitem__(self, upload_id: str, data: dict):
        """Dictionary-style assignment for backward compatibility."""
        self.add(upload_id, data)

    def __contains__(self, upload_id: str):
        """Check if upload_id exists."""
        return self.contains(upload_id)

    def __delitem__(self, upload_id: str):
        """Dictionary-style deletion for backward compatibility."""
        self.remove(upload_id)

    def add(self, upload_id: str, data: dict):
        """Add or update upload session with timestamp."""
        with self.lock:
            self.cleanup_expired()

            # Remove oldest if at capacity
            if len(self.uploads) >= self.max_size and upload_id not in self.uploads:
                oldest_id, oldest_data = self.uploads.popitem(last=False)
                logger.warning(f"Upload capacity reached. Removed oldest upload: {oldest_id}")
                # Clean up files for removed upload
                try:
                    chunk_dir = os.path.join(CHUNK_DIR, oldest_id)
                    if os.path.exists(chunk_dir):
                        shutil.rmtree(chunk_dir)
                except Exception as e:
                    logger.error(f"Error cleaning up removed upload {oldest_id}: {e}")

            self.uploads[upload_id] = {
                'data': data,
                'created_at': datetime.now()
            }
            # Move to end (most recently used)
            self.uploads.move_to_end(upload_id)

    def get(self, upload_id: str) -> dict:
        """Get upload session data."""
        with self.lock:
            self.cleanup_expired()
            if upload_id in self.uploads:
                # Move to end (most recently used)
                self.uploads.move_to_end(upload_id)
                return self.uploads[upload_id]['data']
            return None

    def remove(self, upload_id: str):
        """Remove upload session."""
        with self.lock:
            if upload_id in self.uploads:
                del self.uploads[upload_id]

    def clear(self):
        """Clear all upload sessions."""
        with self.lock:
            self.uploads.clear()

    def contains(self, upload_id: str) -> bool:
        """Check if upload session exists."""
        with self.lock:
            self.cleanup_expired()
            return upload_id in self.uploads

    def cleanup_expired(self):
        """Remove expired upload sessions."""
        now = datetime.now()
        expired = [
            uid for uid, data in self.uploads.items()
            if now - data['created_at'] > self.ttl
        ]
        for uid in expired:
            logger.info(f"Removing expired upload session: {uid}")
            del self.uploads[uid]
            # Clean up files
            try:
                chunk_dir = os.path.join(CHUNK_DIR, uid)
                if os.path.exists(chunk_dir):
                    shutil.rmtree(chunk_dir)
            except Exception as e:
                logger.error(f"Error cleaning up expired upload {uid}: {e}")

# Dictionary to store temporary files
temp_files = {}

# Initialize upload manager (replaces chunk_uploads dictionary)
upload_manager = UploadManager()
chunk_uploads = upload_manager  # Backward compatibility alias

# Directory for storing chunks
CHUNK_DIR = os.path.join(tempfile.gettempdir(), 'cloud_chunks')
os.makedirs(CHUNK_DIR, exist_ok=True)

# Valid file types for chunked uploads
VALID_FILE_TYPES = ['temp_file', 'load_file', 'interpolate_file']

def sanitize_upload_id(upload_id: str) -> str:
    """
    Sanitize upload ID to prevent path traversal attacks.

    Args:
        upload_id: The upload ID to sanitize

    Returns:
        Sanitized upload ID

    Raises:
        ValueError: If upload_id contains invalid characters or is empty
    """
    if not upload_id:
        raise ValueError("Upload ID cannot be empty")

    if not UPLOAD_ID_PATTERN.match(upload_id):
        logger.error(f"Invalid upload ID format: {upload_id}")
        raise ValueError(f"Invalid upload ID format. Only alphanumeric characters, hyphens, and underscores allowed (max 64 chars)")

    # Additional check: ensure no path separators
    if os.path.sep in upload_id or '/' in upload_id or '\\' in upload_id:
        logger.error(f"Upload ID contains path separators: {upload_id}")
        raise ValueError("Upload ID cannot contain path separators")

    return upload_id

def validate_csv_size(file_path: str):
    """
    Validate CSV file size to prevent resource exhaustion.

    Args:
        file_path: Path to the CSV file

    Raises:
        ValueError: If file exceeds size limits
    """
    if not os.path.exists(file_path):
        raise ValueError(f"File not found: {file_path}")

    size = os.path.getsize(file_path)
    if size > MAX_FILE_SIZE:
        raise ValueError(f"File too large: {size / FILE_BUFFER_SIZE:.2f}MB (max {MAX_FILE_SIZE / FILE_BUFFER_SIZE:.0f}MB)")

def validate_dataframe(df: pd.DataFrame, name: str = "DataFrame"):
    """
    Validate DataFrame dimensions to prevent resource exhaustion.

    Args:
        df: DataFrame to validate
        name: Name of the DataFrame for error messages

    Raises:
        ValueError: If DataFrame exceeds size limits
    """
    if len(df) > MAX_ROWS:
        raise ValueError(f"{name} has too many rows: {len(df):,} (max {MAX_ROWS:,})")

    if len(df.columns) > MAX_COLUMNS:
        raise ValueError(f"{name} has too many columns: {len(df.columns)} (max {MAX_COLUMNS})")

def get_chunk_dir(upload_id: str) -> str:
    """
    Create and return a directory path for storing chunks of a specific upload.

    Args:
        upload_id: Sanitized upload ID

    Returns:
        Path to chunk directory
    """
    # upload_id should already be sanitized by caller, but validate anyway
    sanitized_id = sanitize_upload_id(upload_id)
    chunk_dir = os.path.join(CHUNK_DIR, sanitized_id)
    os.makedirs(chunk_dir, exist_ok=True)
    return chunk_dir

@bp.route('/upload-chunk', methods=['POST'])
def upload_chunk():
    """
    Handle chunk upload for large files (5MB chunks). 
    Frontend expects: { success: bool, data: { uploadId, progress, ... } }
    Ako dođe do greške, data sadrži opis greške.
    """
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'data': {'error': 'No file part in the request'}}), 400

        file_chunk = request.files['file']
        upload_id = request.form.get('uploadId')
        file_type = request.form.get('fileType')
        # chunkIndex i totalChunks moraju biti int, validacija
        try:
            chunk_index = int(request.form.get('chunkIndex', 0))
            total_chunks = int(request.form.get('totalChunks', 1))
        except Exception:
            return jsonify({'success': False, 'data': {'error': 'Invalid chunk index or total chunks'}}), 400

        if not upload_id:
            return jsonify({'success': False, 'data': {'error': 'No upload ID provided'}}), 400
        if not file_type or file_type not in VALID_FILE_TYPES:
            return jsonify({'success': False, 'data': {'error': 'Invalid file type'}}), 400

        # Sanitize upload_id to prevent path traversal
        try:
            upload_id = sanitize_upload_id(upload_id)
        except ValueError as e:
            return jsonify({'success': False, 'data': {'error': str(e)}}), 400

        # Create directory for this upload if it doesn't exist
        chunk_dir = get_chunk_dir(upload_id)

        # Save chunk information
        if upload_id not in chunk_uploads:
            chunk_uploads[upload_id] = {
                'temp_file': {'total_chunks': 0, 'received_chunks': set(), 'filename': None},
                'load_file': {'total_chunks': 0, 'received_chunks': set(), 'filename': None},
                'interpolate_file': {'total_chunks': 0, 'received_chunks': set(), 'filename': None}
            }

        # Update chunk tracking
        chunk_uploads[upload_id][file_type]['total_chunks'] = total_chunks
        chunk_uploads[upload_id][file_type]['received_chunks'].add(chunk_index)
        chunk_uploads[upload_id][file_type]['filename'] = file_chunk.filename

        # Save the chunk to disk
        chunk_path = os.path.join(chunk_dir, f"{file_type}_{chunk_index}")
        file_chunk.save(chunk_path)

        return jsonify({
            'success': True,
            'data': {
                'uploadId': upload_id,
                'progress': len(chunk_uploads[upload_id][file_type]['received_chunks']) / total_chunks,
                'chunkIndex': chunk_index,
                'totalChunks': total_chunks,
                'fileType': file_type
            }
        })
    except Exception as e:
        logger.error(f"Error in chunk upload: {str(e)}")
        traceback.print_exc()
        return jsonify({'success': False, 'data': {'error': str(e)}}), 500

def calculate_bounds(predictions, tolerance_type, tol_cnt, tol_dep):
    """Calculate upper and lower bounds based on tolerance type."""
    if tolerance_type == 'cnt':
        upper_bound = predictions + tol_cnt
        lower_bound = predictions - tol_cnt
    else:  # tolerance_type == 'dep'
        upper_bound = predictions * (1 + tol_dep) + tol_cnt
        lower_bound = predictions * (1 - tol_dep) - tol_cnt

    # Do not force lower bound to be >= 0; allow negative values if regression/tolerance allows
    # lower_bound = np.maximum(lower_bound, 0)

    return upper_bound, lower_bound

# Route for handling chunked upload completion
@bp.route('/complete', methods=['POST', 'OPTIONS'])
def complete_redirect():
    """Handle chunked upload completion directly instead of redirecting."""
    try:
        if request.method == 'OPTIONS':
            # Handle CORS preflight request, response format uvek sa 'data'
            response = jsonify({
                'success': True,
                'data': {'message': 'CORS preflight request successful'}
            })
            # Set CORS headers
            response.headers.add('Access-Control-Allow-Origin', '*')
            response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
            response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
            return response
            
        logger.info("=== HANDLING COMPLETE UPLOAD REQUEST ===")
        logger.info(f"Request method: {request.method}")
        logger.info(f"Request content type: {request.content_type}")
        
        # Handle both FormData and JSON
        if request.content_type and 'multipart/form-data' in request.content_type:
            logger.info("Processing FormData request")
            data = request.form.to_dict()
            logger.info(f"Form data: {data}")
        else:
            logger.info("Processing JSON request")
            try:
                data = request.get_json(force=True)
                logger.info(f"JSON data: {data}")
            except Exception as e:
                logger.error(f"Error parsing JSON: {str(e)}")
                # Error: loš format zahteva
                return jsonify({
                    'success': False,
                    'data': {'error': 'Invalid request format. Expected JSON or FormData.'}
                }), 400

        upload_id = data.get('uploadId')
        logger.info(f"Completing upload for ID: {upload_id}")

        if not upload_id:
            logger.error("No upload ID provided")
            return jsonify({'success': False, 'data': {'error': 'No upload ID provided'}}), 400

        # Sanitize upload_id to prevent path traversal
        try:
            upload_id = sanitize_upload_id(upload_id)
        except ValueError as e:
            return jsonify({'success': False, 'data': {'error': str(e)}}), 400

        if upload_id not in chunk_uploads:
            logger.error(f"Invalid upload ID: {upload_id}")
            # Error: uploadId nije validan
            return jsonify({'success': False, 'data': {'error': 'Invalid upload ID'}}), 400

        upload_info = chunk_uploads[upload_id]
        chunk_dir = get_chunk_dir(upload_id)

        # Check if all chunks have been received for both files
        temp_info = upload_info['temp_file']
        load_info = upload_info['load_file']

        logger.info(f"Temp file: {temp_info['received_chunks']}/{temp_info['total_chunks']} chunks")
        logger.info(f"Load file: {load_info['received_chunks']}/{load_info['total_chunks']} chunks")

        # Validate that both files have chunks uploaded
        if temp_info['total_chunks'] == 0 or load_info['total_chunks'] == 0:
            logger.error(f"Missing file uploads. Temp chunks: {temp_info['total_chunks']}, Load chunks: {load_info['total_chunks']}")
            return jsonify({
                'success': False,
                'data': {
                    'error': 'Not all chunks received',
                    'temp_progress': len(temp_info['received_chunks']) / max(temp_info['total_chunks'], 1),
                    'load_progress': len(load_info['received_chunks']) / max(load_info['total_chunks'], 1)
                }
            }), 400

        if (len(temp_info['received_chunks']) != temp_info['total_chunks'] or
            len(load_info['received_chunks']) != load_info['total_chunks']):
            logger.error(f"Not all chunks received. Temp: {len(temp_info['received_chunks'])}/{temp_info['total_chunks']}, Load: {len(load_info['received_chunks'])}/{load_info['total_chunks']}")
            # Error: nisu svi chunkovi primljeni
            return jsonify({
                'success': False,
                'data': {
                    'error': 'Not all chunks received',
                    'temp_progress': len(temp_info['received_chunks']) / max(temp_info['total_chunks'], 1),
                    'load_progress': len(load_info['received_chunks']) / max(load_info['total_chunks'], 1)
                }
            }), 400

        logger.info("All chunks received, reassembling files")
        
        # Reassemble the files
        temp_file_path = os.path.join(chunk_dir, 'temp_out.csv')
        load_file_path = os.path.join(chunk_dir, 'load.csv')
        
        with open(temp_file_path, 'wb') as temp_file:
            for i in range(temp_info['total_chunks']):
                chunk_path = os.path.join(chunk_dir, f"temp_file_{i}")
                with open(chunk_path, 'rb') as chunk:
                    temp_file.write(chunk.read())
        
        with open(load_file_path, 'wb') as load_file:
            for i in range(load_info['total_chunks']):
                chunk_path = os.path.join(chunk_dir, f"load_file_{i}")
                with open(chunk_path, 'rb') as chunk:
                    load_file.write(chunk.read())
        
        logger.info("Files reassembled, processing data")

        # Validate file sizes before processing
        try:
            validate_csv_size(temp_file_path)
            validate_csv_size(load_file_path)
        except ValueError as e:
            logger.error(f"File size validation failed: {str(e)}")
            return jsonify({'success': False, 'data': {'error': str(e)}}), 400

        # Read the reassembled files
        try:
            df1 = pd.read_csv(temp_file_path, sep=';')
            df2 = pd.read_csv(load_file_path, sep=';')

            # Validate DataFrame dimensions
            validate_dataframe(df1, "Temperature file")
            validate_dataframe(df2, "Load file")

            # Extract parameters from request
            processing_params = {
                'REG': data.get('REG', 'lin'),
                'TR': data.get('TR', 'cnt'),
                'TOL_CNT': data.get('TOL_CNT', '0'),
                'TOL_DEP': data.get('TOL_DEP', '0'),
                'TOL_DEP_EXTRA': data.get('TOL_DEP_EXTRA', '0')
            }
            
            logger.info(f"Processing data with parameters: {processing_params}")
            
            # Process the data
            result = _process_data_frames(df1, df2, processing_params)
            
            # Clean up chunks
            try:
                chunk_dir = get_chunk_dir(upload_id)
                if os.path.exists(chunk_dir):
                    shutil.rmtree(chunk_dir)
                if upload_id in chunk_uploads:
                    del chunk_uploads[upload_id]
                logger.info(f"Successfully cleaned up chunks for upload ID: {upload_id}")
            except Exception as e:
                logger.warning(f"Error cleaning up chunks: {str(e)}")

            # Očekuje se da _process_data_frames vraća jsonify sa {'success': True, 'data': ...}
            return result
        except Exception as e:
            logger.error(f"Error processing uploaded files: {str(e)}")
            # Error: problem sa obradom fajlova
            return jsonify({'success': False, 'data': {'error': f'Error processing uploaded files: {str(e)}'}}), 500
    except Exception as e:
        logger.error(f"Error in complete_redirect: {str(e)}")
        # Error: generalni exception
        return jsonify({'success': False, 'data': {'error': str(e)}}), 500

def interpolate_data(df1, df2, x_col, y_col, max_time_span):
    # Create a copy of the input data
    df = pd.DataFrame()
    df['UTC'] = pd.to_datetime(df1['UTC'])
    df['value'] = pd.to_numeric(df2[y_col], errors='coerce')
    df = df.sort_values('UTC').reset_index(drop=True)
    
    # Initialize final dataframe
    df_final = df.copy()
    
    # Find first non-NaN value
    i = 0
    while i < len(df_final):
        if pd.isna(df_final.at[i, 'value']):
            i += 1
        else:
            break
    
    if i == len(df_final):
        logger.warning("No numeric data found for interpolation")
        return df_final, 0
    
    # Initialize variables
    frame = "non"
    i_start = 0
    
    # Main processing loop
    i = 0
    while i < len(df_final):
        # No frame open - look for NaN to start a frame
        if frame == "non":
            if pd.isna(df_final.at[i, 'value']):
                i_start = i
                frame = "open"
            i += 1
        # Frame is open - look for a number to close it
        elif frame == "open":
            if not pd.isna(df_final.at[i, 'value']):
                # Calculate frame width in minutes
                frame_width = (df_final.at[i, 'UTC'] - df_final.at[i_start-1, 'UTC']).total_seconds() / 60
                
                # Only interpolate if the gap is within max_time_span
                if frame_width <= max_time_span:
                    # Get the values for interpolation
                    y0 = df_final.at[i_start-1, 'value']
                    y1 = df_final.at[i, 'value']
                    t0 = df_final.at[i_start-1, 'UTC']
                    t1 = df_final.at[i, 'UTC']
                    
                    # Calculate the difference and rate of change
                    y_diff = y1 - y0
                    diff_per_min = y_diff / frame_width
                    
                    # Perform linear interpolation for each point in the gap
                    for j in range(i_start, i):
                        gap_min = (df_final.at[j, 'UTC'] - t0).total_seconds() / 60
                        df_final.at[j, 'value'] = y0 + (gap_min * diff_per_min)
                
                frame = "non"
                i += 1
            else:
                i += 1
        else:
            i += 1
        
        # Safety check to prevent infinite loops
        if i >= len(df_final):
            break
    
    # Calculate how many points were added (interpolated)
    original_nans = df['value'].isna().sum()
    final_nans = df_final['value'].isna().sum()
    added_points = original_nans - final_nans
    
    return df_final, added_points

def _validate_and_prepare_data(df1, df2):
    """Validate and prepare CSV data for regression analysis."""
    logger.info(f"Processing dataframes with shapes: {df1.shape}, {df2.shape}")
    logger.info(f"Columns in temperature file: {df1.columns.tolist()}")
    logger.info(f"Columns in load file: {df2.columns.tolist()}")

    # Find temperature and load columns
    temp_cols = [col for col in df1.columns if col != 'UTC']
    load_cols = [col for col in df2.columns if col != 'UTC']

    if not temp_cols:
        raise ValueError(f'No valid temperature column found. Available columns: {df1.columns.tolist()}')
    if not load_cols:
        raise ValueError(f'No valid load column found. Available columns: {df2.columns.tolist()}')

    x = temp_cols[0]
    y = load_cols[0]
    logger.info(f"Using temperature column: {x}")
    logger.info(f"Using load column: {y}")

    # Convert timestamps and sort
    df1['UTC'] = pd.to_datetime(df1['UTC'], format="%Y-%m-%d %H:%M:%S")
    df2['UTC'] = pd.to_datetime(df2['UTC'], format="%Y-%m-%d %H:%M:%S")

    # Convert to numeric
    df1[x] = pd.to_numeric(df1[x], errors='coerce')
    df2[y] = pd.to_numeric(df2[y], errors='coerce')

    # Sort by time
    df1 = df1.sort_values('UTC')
    df2 = df2.sort_values('UTC')

    logger.info(f"Data ranges after sorting:")
    logger.info(f"Temperature range: {df1[x].min():.2f} to {df1[x].max():.2f} °C")
    logger.info(f"Load range before conversion: {df2[y].min():.2f} to {df2[y].max():.2f} kW")

    # Merge dataframes
    df_merged = pd.merge(df1[['UTC', x]], df2[['UTC', y]], on='UTC', how='inner')
    if df_merged.empty:
        raise ValueError('No matching timestamps found between files. Please ensure both files have matching timestamps.')

    logger.info(f"First few timestamps in first file: {df1['UTC'].head().tolist()}")
    logger.info(f"First few timestamps in second file: {df2['UTC'].head().tolist()}")

    # Check for duplicates
    df1_duplicates = df1['UTC'].duplicated().sum()
    df2_duplicates = df2['UTC'].duplicated().sum()
    if df1_duplicates > 0 or df2_duplicates > 0:
        raise ValueError('Duplicate timestamps found in data')

    # Clean data
    df1 = df1.dropna()
    df2 = df2.dropna()
    if df1.empty or df2.empty:
        raise ValueError('No valid numeric data found after cleaning')

    logger.info(f"Data cleaned. New shapes: {df1.shape}, {df2.shape}")

    # Combine data
    cld = pd.DataFrame()
    cld[x] = df1[x]
    cld[y] = df2[y]
    cld = cld.dropna()
    if cld.empty:
        raise ValueError('No valid data points after combining datasets')

    logger.info(f"Combined data shape: {cld.shape}")

    # Sort by x and validate
    cld_srt = cld.sort_values(by=x).copy()
    if cld_srt[x].isna().any() or cld_srt[y].isna().any():
        raise ValueError('NaN values found in processed data')

    return cld_srt, x, y, df2

def _calculate_tolerance_params(data, y_range):
    """Calculate and validate tolerance parameters."""
    default_tol = y_range * DEFAULT_TOLERANCE_RATIO

    try:
        TOL_CNT = float(data.get('TOL_CNT', str(default_tol)))
        TOL_DEP = float(data.get('TOL_DEP', '0.1'))

        TOL_CNT = TOL_CNT / TOLERANCE_ADJUSTMENT_FACTOR
        logger.info(f"Received tolerance values - TOL_CNT: {TOL_CNT}, TOL_DEP: {TOL_DEP}")

        if TOL_CNT <= 0:
            logger.warning(f"Invalid TOL_CNT value: {TOL_CNT}, using default")
            TOL_CNT = default_tol
        if TOL_DEP <= 0:
            logger.warning(f"Invalid TOL_DEP value: {TOL_DEP}, using default")
            TOL_DEP = 0.1
    except (ValueError, TypeError) as e:
        logger.warning(f"Error parsing tolerance values: {str(e)}, using defaults")
        TOL_CNT = default_tol
        TOL_DEP = 0.1

    if TOL_CNT < y_range * MIN_TOLERANCE_THRESHOLD:
        TOL_CNT = y_range * DEFAULT_TOLERANCE_RATIO
        logger.info(f"Adjusted tolerance to {TOL_CNT:.2f} ({DEFAULT_TOLERANCE_RATIO*100}% of data range)")

    return TOL_CNT, TOL_DEP

def _perform_linear_regression(cld_srt, x, y, TR, TOL_CNT, TOL_DEP):
    """Perform linear regression and apply tolerance filtering."""
    lin_mdl = LinearRegression()
    lin_mdl.fit(cld_srt[[x]], cld_srt[y])
    lin_prd = lin_mdl.predict(cld_srt[[x]])
    lin_fcn = f"y = {lin_mdl.coef_[0]:.2f}x + {lin_mdl.intercept_:.2f}"
    logger.info(f"Linearna regresija: {lin_fcn}")

    upper_bound, lower_bound = calculate_bounds(lin_prd, TR, TOL_CNT, TOL_DEP)

    if TR == "cnt":
        mask = np.abs(cld_srt[y] - lin_prd) <= TOL_CNT
        logger.info(f"Using constant tolerance: {TOL_CNT}")
    elif TR == "dep":
        mask = np.abs(cld_srt[y] - lin_prd) <= (np.abs(lin_prd) * TOL_DEP + TOL_CNT)
        logger.info(f"Using dependent tolerance: {TOL_DEP} + {TOL_CNT}")
    else:
        raise ValueError(f'Unknown tolerance type: {TR}')

    cld_srt_flt = cld_srt[mask]

    if len(cld_srt_flt) == 0:
        logger.warning(f"No points within tolerance bounds. Min y: {cld_srt[y].min()}, Max y: {cld_srt[y].max()}")
        logger.warning(f"Bounds: lower={lower_bound.min()}, upper={upper_bound.max()}")
        raise ValueError('No points within tolerance bounds. Try increasing the tolerance values.')

    return {
        'x_values': cld_srt[x].tolist(),
        'y_values': cld_srt[y].tolist(),
        'predicted_y': lin_prd.tolist(),
        'upper_bound': upper_bound.tolist(),
        'lower_bound': lower_bound.tolist(),
        'filtered_x': cld_srt_flt[x].tolist(),
        'filtered_y': cld_srt_flt[y].tolist(),
        'equation': lin_fcn,
        'removed_points': len(cld_srt) - len(cld_srt_flt)
    }

def _perform_polynomial_regression(cld_srt, x, y, TR, TOL_CNT, TOL_DEP):
    """Perform polynomial regression and apply tolerance filtering."""
    poly_features = PolynomialFeatures(degree=2)
    X_poly = poly_features.fit_transform(cld_srt[[x]])
    poly_mdl = LinearRegression()
    poly_mdl.fit(X_poly, cld_srt[y])
    poly_prd = poly_mdl.predict(X_poly)
    coeffs = poly_mdl.coef_
    intercept = poly_mdl.intercept_
    poly_fcn = f"y = {coeffs[2]:.2f}x² + {coeffs[1]:.2f}x + {intercept:.2f}"
    logger.info(f"Polynom-Regression (Grad 2): {poly_fcn}")

    upper_bound, lower_bound = calculate_bounds(poly_prd, TR, TOL_CNT, TOL_DEP)

    if TR == "cnt":
        mask = np.abs(cld_srt[y] - poly_prd) <= TOL_CNT
    elif TR == "dep":
        mask = np.abs(cld_srt[y] - poly_prd) <= (np.abs(poly_prd) * TOL_DEP + TOL_CNT)
    else:
        raise ValueError(f'Unknown tolerance type: {TR}')

    cld_srt_flt = cld_srt[mask]

    if len(cld_srt_flt) == 0:
        logger.warning(f"No points within tolerance bounds. Min y: {cld_srt[y].min()}, Max y: {cld_srt[y].max()}")
        logger.warning(f"Bounds: lower={lower_bound.min()}, upper={upper_bound.max()}")
        raise ValueError('No points within tolerance bounds. Try increasing the tolerance values.')

    return {
        'x_values': cld_srt[x].tolist(),
        'y_values': cld_srt[y].tolist(),
        'predicted_y': poly_prd.tolist(),
        'upper_bound': upper_bound.tolist(),
        'lower_bound': lower_bound.tolist(),
        'filtered_x': cld_srt_flt[x].tolist(),
        'filtered_y': cld_srt_flt[y].tolist(),
        'equation': poly_fcn,
        'removed_points': len(cld_srt) - len(cld_srt_flt)
    }

def _process_data_frames(df1, df2, data):
    """
    Process data from dataframes for both direct and chunked uploads.
    Refactored to use helper functions for better maintainability.
    """
    try:
        # Validate and prepare data
        cld_srt, x, y, df2_original = _validate_and_prepare_data(df1, df2)

        # Get regression and tolerance parameters
        REG = data.get('REG', 'lin')
        TR = data.get('TR', 'cnt')
        y_range = df2_original[y].max() - df2_original[y].min()
        TOL_CNT, TOL_DEP = _calculate_tolerance_params(data, y_range)

        logger.info(f"Final parameters: REG={REG}, TR={TR}, TOL_CNT={TOL_CNT}, TOL_DEP={TOL_DEP}")

        # Perform regression
        if REG == "lin":
            result_data = _perform_linear_regression(cld_srt, x, y, TR, TOL_CNT, TOL_DEP)
        else:  # REG == "poly"
            result_data = _perform_polynomial_regression(cld_srt, x, y, TR, TOL_CNT, TOL_DEP)

        logger.info("Sending response:")
        return jsonify({'success': True, 'data': result_data})

    except ValueError as ve:
        logger.error(f"Validation error: {str(ve)}")
        return jsonify({'success': False, 'data': {'error': str(ve)}}), 400
    except Exception as e:
        logger.error(f"Error in data processing: {str(e)}")
        traceback.print_exc()
        return jsonify({'success': False, 'data': {'error': str(e)}}), 500

# Route for handling chunked upload completion
@bp.route('/clouddata', methods=['POST'])
def clouddata():
    if request.method == 'OPTIONS':
        response = app.make_default_options_response()
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
        return response

    try:
        print("\nReceived request to /clouddata")
        return _process_data()
    except Exception as e:
        print(f"Error in clouddata endpoint: {str(e)}")
        import traceback
        traceback.print_exc()


def _process_data():
    try:
        logger.info("\nReceived request to /clouddata")
        data = request.json
        logger.info(f"Received data: {data}")

        if data is None:
            logger.error("No data received")
            return jsonify({'success': False, 'data': {'error': 'No data received'}}), 400

        # Preuzmi fajlove iz zahteva
        temp_data = data['files'].get('temp_out.csv')
        load_data = data['files'].get('load.csv')

        if not temp_data or not load_data:
            logger.error("One or both files are empty")
            return jsonify({'success': False, 'data': {'error': 'One or both files are empty'}}), 400

        try:
            # Dekodiraj base64 podatke i učitaj u DataFrame
            logger.info("Attempting to decode and read temperature data...")
            temp_decoded = base64.b64decode(temp_data).decode('utf-8')
            logger.debug(f"Temperature data preview: {temp_decoded[:200]}")
            df1 = pd.read_csv(StringIO(temp_decoded), sep=';')

            logger.info("Attempting to decode and read load data...")
            load_decoded = base64.b64decode(load_data).decode('utf-8')
            logger.debug(f"Load data preview: {load_decoded[:200]}")
            df2 = pd.read_csv(StringIO(load_decoded), sep=';')

            logger.info(f"Successfully read data. Shapes: {df1.shape}, {df2.shape}")
            logger.info(f"Columns in temperature file: {df1.columns.tolist()}")
            logger.info(f"Columns in load file: {df2.columns.tolist()}")
            logger.debug("First few rows of temperature file:")
            logger.debug(f"{df1.head()}")
            logger.debug(f"{df2.head()}")

            # Validate DataFrame dimensions
            validate_dataframe(df1, "Temperature file")
            validate_dataframe(df2, "Load file")

            # Obradi podatke koristeći zajedničku funkciju
            return _process_data_frames(df1, df2, data)
        except ValueError as e:
            # Validation errors
            logger.error(f"Validation error: {str(e)}")
            return jsonify({'success': False, 'data': {'error': str(e)}}), 400
        except Exception as e:
            logger.error(f"Error reading CSV files: {str(e)}")
            return jsonify({'success': False, 'data': {'error': f'Error reading CSV files: {str(e)}'}}), 400

    except Exception as e:
        logger.error(f"Error processing data: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'success': False, 'data': {'error': str(e)}}), 500


# Route for handling chunked upload for interpolation
@bp.route('/interpolate-chunked', methods=['POST'])
def interpolate_chunked():
    """
    Process a chunked file upload for interpolation.
    Svi odgovori su u formatu: {'success': bool, 'data': ...}
    U slučaju greške, data je objekat sa 'error' poljem.
    """
    try:
        logger.info("Received request to /interpolate-chunked")

        # Get upload ID and parameters from request
        data = request.json
        if not data or 'uploadId' not in data:
            logger.error("Missing uploadId in request")
            # Error: uploadId nedostaje
            return jsonify({'success': False, 'data': {'error': 'Upload ID is required'}}), 400

        upload_id = data['uploadId']
        logger.info(f"Processing upload ID: {upload_id}")

        # Sanitize upload_id to prevent path traversal
        try:
            upload_id = sanitize_upload_id(upload_id)
        except ValueError as e:
            return jsonify({'success': False, 'data': {'error': str(e)}}), 400

        # Validate max_time_span parameter
        try:
            max_time_span = float(data.get('max_time_span', '60'))
            logger.info(f"Using max_time_span: {max_time_span}")
        except ValueError as e:
            logger.error(f"Invalid max_time_span value: {data.get('max_time_span')}")
            # Error: loš max_time_span
            return jsonify({'success': False, 'data': {'error': 'Invalid max_time_span parameter'}}), 400

        # Check if upload exists
        if upload_id not in chunk_uploads:
            logger.error(f"Upload ID not found: {upload_id}")
            # Error: upload ne postoji
            return jsonify({'success': False, 'data': {'error': 'Upload ID not found'}}), 404

        # Check if all chunks have been received
        upload_info = chunk_uploads[upload_id]['interpolate_file']
        if len(upload_info['received_chunks']) < upload_info['total_chunks']:
            logger.error(f"Not all chunks received for upload {upload_id}")
            # Error: nisu svi chunkovi primljeni
            return jsonify({'success': False, 'data': {'error': f"Incomplete upload: Only {len(upload_info['received_chunks'])}/{upload_info['total_chunks']} chunks received"}}), 400

        # Combine chunks into a single file
        chunk_dir = get_chunk_dir(upload_id)
        combined_file_path = os.path.join(chunk_dir, 'combined_interpolate_file.csv')

        # Optimizacija: Koristimo binary mode i veći buffer za brže kombinovanje fajlova
        with open(combined_file_path, 'wb') as outfile:
            for i in range(upload_info['total_chunks']):
                chunk_path = os.path.join(chunk_dir, f"interpolate_file_{i}")
                if os.path.exists(chunk_path):
                    with open(chunk_path, 'rb') as infile:
                        # Kopiranje u većim blokovima za bolje performanse
                        shutil.copyfileobj(infile, outfile, FILE_BUFFER_SIZE)

        logger.info(f"Combined file created at: {combined_file_path}")

        # Validate file size to prevent resource exhaustion
        try:
            validate_csv_size(combined_file_path)
        except ValueError as e:
            logger.error(f"File size validation failed: {str(e)}")
            return jsonify({'success': False, 'data': {'error': str(e)}}), 400

        # Optimizacija: Koristimo engine='c' za brže parsiranje CSV-a
        try:
            # Detektujemo separator bez učitavanja celog fajla
            with open(combined_file_path, 'r', encoding='utf-8') as f:
                first_line = f.readline()

            # Try different separators if needed
            if ';' in first_line:
                sep = ';'
            elif ',' in first_line:
                sep = ','
            else:
                sep = None  # Let pandas detect

            logger.info(f"Using separator: {sep}")

            # Optimizacija: Učitavamo samo potrebne kolone i koristimo engine='c'
            df2 = pd.read_csv(combined_file_path,
                             sep=sep,
                             decimal=',',
                             engine='c')   # Brži C engine

            # Validate DataFrame dimensions
            validate_dataframe(df2, "Interpolation file")

        except ValueError as e:
            # Validation errors (file size, DataFrame dimensions)
            logger.error(f"Validation error: {str(e)}")
            return jsonify({'success': False, 'data': {'error': str(e)}}), 400
        except Exception as e:
            logger.error(f"Error reading CSV file: {str(e)}")
            # Error: CSV parsiranje nije uspelo
            return jsonify({'success': False, 'data': {'error': f'Error reading CSV file: {str(e)}'}}), 400

        # Check if UTC column exists
        if 'UTC' not in df2.columns:
            logger.error("UTC column not found in the file")
            # Error: nema UTC kolone
            return jsonify({'success': False, 'data': {'error': 'UTC column not found. The file must contain a UTC column with timestamps'}}), 400

        # Optimizacija: Brža pretraga load kolone
        load_terms = set(['last', 'load', 'leistung', 'kw', 'w'])
        load_cols = [col for col in df2.columns if 
                    any(term in str(col).lower() for term in load_terms)]

        if not load_cols:
            # If no specific load column found, use the first non-UTC column
            non_utc_cols = [col for col in df2.columns if col != 'UTC']
            if non_utc_cols:
                y_col = non_utc_cols[0]
                logger.info(f"No specific load column found, using first non-UTC column: {y_col}")
            else:
                logger.error("No suitable load column found")
                return jsonify({
                    'success': False, 
                    'error': 'Load column not found', 
                    'message': 'The file must contain a column with load data'
                }), 400
        else:
            y_col = load_cols[0]
            logger.info(f"Found load column: {y_col}")
        
        # Optimizacija: Zadržavamo samo potrebne kolone za smanjenje memorije
        df2 = df2[['UTC', y_col]].copy()
        
        # Optimizacija: Brža konverzija u numeričke vrednosti
        if not pd.api.types.is_numeric_dtype(df2[y_col]):
            # Direktna konverzija u numeričke vrednosti
            df2[y_col] = pd.to_numeric(df2[y_col].astype(str).str.replace(',', '.').str.replace(r'[^\d\-\.]', '', regex=True), errors='coerce')
        
        # Keep NaN values as NaN for interpolation (don't convert to string yet)
        # This ensures the column remains numeric for interpolation
        
        # Optimizacija: Brža konverzija vremena
        try:
            # Koristimo cache=True za brže parsiranje datuma
            df2['UTC'] = pd.to_datetime(df2['UTC'], errors='coerce', cache=True)
            # Drop rows with invalid datetime
            df2.dropna(subset=['UTC'], inplace=True)
                
        except Exception as e:
            logger.error(f"Error converting UTC to datetime: {str(e)}")
            return jsonify({
                'success': False, 
                'error': f'Error processing timestamps: {str(e)}', 
                'message': 'Please check the timestamp format in the file'
            }), 400
        
        # Sort by time
        df2.sort_values('UTC', inplace=True)
        
        # Optimizacija: Direktno koristimo postojeće kolone umesto kreiranja novog DataFrame-a
        df2.rename(columns={y_col: 'load'}, inplace=True)
        df_load = df2.set_index('UTC')
        
        # Check if we have enough data points
        if len(df_load) < 2:
            logger.error("Not enough valid data points for interpolation")
            return jsonify({
                'success': False, 
                'error': 'Not enough valid data points', 
                'message': 'The file must contain at least 2 valid data points for interpolation'
            }), 400
        
        # Optimizacija: Brže računanje vremenskih razlika
        time_diffs = (df_load.index[1:] - df_load.index[:-1]).total_seconds() / 60  # in minutes
        max_gap = time_diffs.max() if len(time_diffs) > 0 else 0
        logger.info(f"Maximum time gap in data: {max_gap} minutes")
        
        # Optimizacija: Prilagodljivi interval resample-a za velike skupove podataka
        # Koristimo veći interval za velike skupove podataka da smanjimo broj tačaka
        total_minutes = (df_load.index[-1] - df_load.index[0]).total_seconds() / 60
        
        # Izaberemo interval na osnovu ukupnog vremenskog raspona
        if total_minutes > 10000:  # Ako je vremenski raspon veći od ~7 dana
            resample_interval = '5min'  # Koristimo 5-minutni interval
            logger.info(f"Large time span detected ({total_minutes} minutes), using 5-minute intervals")
        else:
            resample_interval = '1min'  # Standardni 1-minutni interval
            logger.info(f"Using standard 1-minute intervals")
        
        # Replace the resampling section with this:
        if not pd.api.types.is_numeric_dtype(df_load['load']):
            logger.info("Converting load column to numeric before interpolation")
            df_load['load'] = pd.to_numeric(df_load['load'], errors='coerce')
        
        # Define limit for interpolation
        limit = int(max_time_span)  # Convert to integer number of minutes
        logger.info(f"Using interpolation limit of {limit} minutes")
            
        # Instead of resampling, just interpolate at existing points
        df2_resampled = df_load.copy()
        df2_resampled['load'] = df_load['load'].interpolate(method='linear', limit=limit)

        # Reset index to get UTC back as a column
        df2_resampled.reset_index(inplace=True)
        
        # Calculate added points
        original_points = len(df2)
        total_points = len(df2_resampled)
        added_points = total_points - original_points
        
        logger.info(f"Original points: {original_points}")
        logger.info(f"Interpolated points: {total_points}")
        logger.info(f"Added points: {added_points}")
        
        # Prepare data for frontend chart
        chart_data = []
        for _, row in df2_resampled.iterrows():
            # Convert NaN values to string 'NaN' instead of skipping them
            # Only skip rows with invalid timestamps (NaT)
            if pd.isna(row['UTC']):
                logger.warning(f"Skipping row with NaT timestamp: {row}")
                continue
            
            # Convert NaN load values to string 'NaN'
            load_value = 'NaN' if pd.isna(row['load']) else float(row['load'])
                
            try:
                chart_data.append({
                    'UTC': row['UTC'].strftime("%Y-%m-%d %H:%M:%S"),
                    'value': load_value
                })
            except (ValueError, TypeError) as e:
                logger.warning(f"Error converting row to chart data: {e}. Row: {row}")
                continue

        # Clean up the chunks after processing
        try:
            # Only remove the specific upload's directory, not the entire chunk directory
            shutil.rmtree(chunk_dir)
            logger.info(f"Cleaned up chunk directory for upload {upload_id}")
            # Remove from memory
            del chunk_uploads[upload_id]
        except Exception as e:
            logger.warning(f"Error cleaning up chunks: {str(e)}")
        
        logger.info(f"Sample of chart data being sent: {chart_data[:5] if chart_data else 'No data'}")
        logger.info(f"Total points in chart data: {len(chart_data)}")
        
        # Ako nema validnih tačaka, vrati grešku u novom formatu
        if not chart_data:
            logger.error("No valid data points after processing")
            return jsonify({
                'success': False,
                'data': {
                    'error': 'No valid data points after processing',
                    'message': 'The file contains no valid data points for interpolation'
                }
            }), 400

        # Define chunk size for streaming (number of data points per chunk)
        CHUNK_SIZE = STREAMING_CHUNK_SIZE
        
        # Calculate total number of chunks needed
        total_rows = len(df2_resampled)
        total_chunks = (total_rows + CHUNK_SIZE - 1) // CHUNK_SIZE  # Ceiling division
        
        logger.info(f"Total rows: {total_rows}, will be sent in {total_chunks} chunks")
        
        # Prepare data for streaming
        # Optimizacija: Efikasnija konverzija DataFrame-a u JSON format
        # Only filter out rows with invalid timestamps, keep NaN load values
        valid_mask = ~df2_resampled['UTC'].isna()
        valid_df = df2_resampled[valid_mask].copy()
        
        # Convert NaN load values to string 'NaN' after interpolation
        valid_df['load'] = valid_df['load'].apply(lambda x: 'NaN' if pd.isna(x) else x)
        
        # Formatiramo UTC kolonu
        valid_df['UTC'] = valid_df['UTC'].dt.strftime("%Y-%m-%d %H:%M:%S")
        
        # Konvertujemo u DataFrame sa potrebnim kolonama
        chart_df = valid_df.rename(columns={'load': 'value'})[['UTC', 'value']]
        
        # Određujemo broj redova i chunk-ova
        total_rows = len(chart_df)
        CHUNK_SIZE = STREAMING_CHUNK_SIZE  # Optimizovana veličina chunk-a za bolje performanse
        total_chunks = (total_rows + CHUNK_SIZE - 1) // CHUNK_SIZE  # Ceiling division
        
        # Sačuvamo broj originalnih tačaka za meta podatke
        original_points_count = original_points  # Koristimo već definisanu promenljivu
        
        logger.info(f"Total rows: {total_rows}, will be sent in {total_chunks} chunks")
        
        # Funkcija za generisanje chunk-ova
        def generate_chunks():
            # First, send metadata about the dataset
            meta_data = {
                'type': 'meta',
                'total_rows': total_rows,
                'total_chunks': total_chunks,
                'added_points': added_points,
                'original_points': original_points_count,
                'success': True
            }
            yield json.dumps(meta_data, separators=(',', ':')) + '\n'
            
            # Optimizacija: Procesiramo chunk-ove efikasnije
            for chunk_idx in range(total_chunks):
                start_idx = chunk_idx * CHUNK_SIZE
                end_idx = min(start_idx + CHUNK_SIZE, total_rows)
                
                # Konvertujemo chunk direktno u listu rečnika
                chunk_data_list = chart_df.iloc[start_idx:end_idx].to_dict('records')
                
                chunk_data = {
                    'type': 'data',
                    'chunk_index': chunk_idx,
                    'data': chunk_data_list
                }
                
                yield json.dumps(chunk_data, separators=(',', ':')) + '\n'
            
            # Finally, send completion message
            yield json.dumps({
                'type': 'complete',
                'message': 'Data streaming completed',
                'success': True
            }, separators=(',', ':')) + '\n'
            
            # Clean up the chunks after processing
            try:
                chunk_dir = get_chunk_dir(upload_id)
                if os.path.exists(chunk_dir):
                    shutil.rmtree(chunk_dir)
                if upload_id in chunk_uploads:
                    del chunk_uploads[upload_id]
                logger.info(f"Successfully cleaned up chunks for upload ID: {upload_id}")
            except Exception as e:
                logger.warning(f"Error cleaning up chunks: {str(e)}")
        
        # Return a streaming response
        return Response(generate_chunks(), mimetype='application/x-ndjson')
    except Exception as e:
        logger.error(f"Error in interpolation-chunked endpoint: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False, 
            'error': str(e), 
            'message': 'An error occurred during interpolation'
        }), 500


# Route for handling prepare save
@bp.route('/prepare-save', methods=['POST'])
def prepare_save():
    """
    Save data to a CSV file for download.
    
    Expected payload: {
        'data': array of arrays (CSV data with headers and values),
        'filename': string (optional, default is 'interpolated_data')
    }
    
    Returns: {
        'success': bool,
        'fileId': string (ID for download endpoint),
        'filename': string
    }
    """
    try:
        data = request.json
        if not data:
            logger.error("No data provided in request.")
            return jsonify({"success": False, "data": {"error": "No data provided in request."}}), 400
            
        # Handle data saving request
        if 'data' in data:
            csv_data = data['data']
            filename = data.get('filename', 'interpolated_data')
            
            if not isinstance(csv_data, list) or len(csv_data) == 0:
                logger.error("Invalid data format for CSV")
                return jsonify({"success": False, "data": {"error": "Invalid data format"}}), 400
                
            logger.info(f"Preparing CSV file with name: {filename}")
            
            # Create a temporary file to store the CSV data
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
            
            try:
                # Write the data to the CSV file
                with open(temp_file.name, 'w', newline='') as f:
                    writer = csv.writer(f, delimiter=';')
                    for row in csv_data:
                        writer.writerow(row)
                        
                # Generate a unique file ID
                file_id = f"csv_{datetime.now().strftime('%Y%m%d%H%M%S')}_{os.getpid()}"
                
                # Store the file info for later retrieval
                temp_files[file_id] = {
                    'path': temp_file.name,
                    'filename': filename,
                    'created_at': datetime.now().isoformat()
                }
                
                logger.info(f"CSV file prepared with ID: {file_id}")
                
                return jsonify({
                    "success": True,
                    "fileId": file_id,
                    "filename": filename
                })
                
            except Exception as e:
                # Clean up the temporary file if an error occurs
                if os.path.exists(temp_file.name):
                    os.unlink(temp_file.name)
                raise e
        else:
            logger.error("No 'data' field provided in request")
            return jsonify({"success": False, "data": {"error": "Missing 'data' field in request"}}), 400
    except Exception as e:
        logger.error(f"Error in prepare_save: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"success": False, "data": {"error": str(e)}}), 500

# Route for handling file download
@bp.route('/download/<file_id>', methods=['GET'])
def download_file(file_id):
    """Download a previously prepared file.
    
    Args:
        file_id (str): The unique identifier for the file
        request: The Flask request object
    
    Returns:
        Flask response with the file or error message
    """
    try:
        logger.info(f"Received download request for file ID: {file_id}")
        if file_id not in temp_files:
            logger.error(f"File ID not found: {file_id}")
            return jsonify({"success": False, "error": "File not found"}), 404
            
        file_info = temp_files[file_id]
        file_path = file_info['path']
        custom_filename = file_info['filename']
        
        if not os.path.exists(file_path):
            logger.error(f"File path does not exist: {file_path}")
            return jsonify({"success": False, "error": "File not found"}), 404

        # Use the custom filename provided by the user, or fall back to a default
        download_name = f"{custom_filename}.csv"
        logger.info(f"Sending file with name: {download_name}")
        
        return send_file(
            file_path,
            as_attachment=True,
            download_name=download_name,
            mimetype='text/csv'
        )
    except Exception as e:
        logger.error(f"Error in download_file: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"success": False, "error": str(e)}), 500
    finally:
        # Clean up the temporary file
        if file_id in temp_files:
            try:
                os.unlink(temp_files[file_id]['path'])
                del temp_files[file_id]
                logger.info(f"Cleaned up temporary file for ID: {file_id}")
            except Exception as ex:
                logger.error(f"Error cleaning up temp file: {ex}")