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
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Polygon


bp = Blueprint('cloud', __name__)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_FILE_SIZE = 100 * 1024 * 1024
MAX_ROWS = 1_000_000
MAX_COLUMNS = 100
MAX_ACTIVE_UPLOADS = 1000
UPLOAD_ID_PATTERN = re.compile(r'^[\w\-]{1,64}$')

TOLERANCE_ADJUSTMENT_FACTOR = 2
MIN_TOLERANCE_THRESHOLD = 0.01
DEFAULT_TOLERANCE_RATIO = 0.1
STREAMING_CHUNK_SIZE = 5000
FILE_BUFFER_SIZE = 1024 * 1024

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

            if len(self.uploads) >= self.max_size and upload_id not in self.uploads:
                oldest_id, oldest_data = self.uploads.popitem(last=False)
                logger.warning(f"Upload capacity reached. Removed oldest upload: {oldest_id}")
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
            self.uploads.move_to_end(upload_id)

    def get(self, upload_id: str) -> dict:
        """Get upload session data."""
        with self.lock:
            self.cleanup_expired()
            if upload_id in self.uploads:
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
            try:
                chunk_dir = os.path.join(CHUNK_DIR, uid)
                if os.path.exists(chunk_dir):
                    shutil.rmtree(chunk_dir)
            except Exception as e:
                logger.error(f"Error cleaning up expired upload {uid}: {e}")

temp_files = {}

upload_manager = UploadManager()
chunk_uploads = upload_manager

CHUNK_DIR = os.path.join(tempfile.gettempdir(), 'cloud_chunks')
os.makedirs(CHUNK_DIR, exist_ok=True)

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
        try:
            chunk_index = int(request.form.get('chunkIndex', 0))
            total_chunks = int(request.form.get('totalChunks', 1))
        except Exception:
            return jsonify({'success': False, 'data': {'error': 'Invalid chunk index or total chunks'}}), 400

        if not upload_id:
            return jsonify({'success': False, 'data': {'error': 'No upload ID provided'}}), 400
        if not file_type or file_type not in VALID_FILE_TYPES:
            return jsonify({'success': False, 'data': {'error': 'Invalid file type'}}), 400

        try:
            upload_id = sanitize_upload_id(upload_id)
        except ValueError as e:
            return jsonify({'success': False, 'data': {'error': str(e)}}), 400

        chunk_dir = get_chunk_dir(upload_id)

        if upload_id not in chunk_uploads:
            chunk_uploads[upload_id] = {
                'temp_file': {'total_chunks': 0, 'received_chunks': set(), 'filename': None},
                'load_file': {'total_chunks': 0, 'received_chunks': set(), 'filename': None},
                'interpolate_file': {'total_chunks': 0, 'received_chunks': set(), 'filename': None}
            }

        chunk_uploads[upload_id][file_type]['total_chunks'] = total_chunks
        chunk_uploads[upload_id][file_type]['received_chunks'].add(chunk_index)
        chunk_uploads[upload_id][file_type]['filename'] = file_chunk.filename

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
    else:
        upper_bound = predictions * (1 + tol_dep) + tol_cnt
        lower_bound = predictions * (1 - tol_dep) - tol_cnt


    return upper_bound, lower_bound

@bp.route('/complete', methods=['POST', 'OPTIONS'])
def complete_redirect():
    """Handle chunked upload completion directly instead of redirecting."""
    try:
        if request.method == 'OPTIONS':
            response = jsonify({
                'success': True,
                'data': {'message': 'CORS preflight request successful'}
            })
            response.headers.add('Access-Control-Allow-Origin', '*')
            response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
            response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
            return response
            
        logger.info("=== HANDLING COMPLETE UPLOAD REQUEST ===")
        logger.info(f"Request method: {request.method}")
        logger.info(f"Request content type: {request.content_type}")
        
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
                return jsonify({
                    'success': False,
                    'data': {'error': 'Invalid request format. Expected JSON or FormData.'}
                }), 400

        upload_id = data.get('uploadId')
        logger.info(f"Completing upload for ID: {upload_id}")

        if not upload_id:
            logger.error("No upload ID provided")
            return jsonify({'success': False, 'data': {'error': 'No upload ID provided'}}), 400

        try:
            upload_id = sanitize_upload_id(upload_id)
        except ValueError as e:
            return jsonify({'success': False, 'data': {'error': str(e)}}), 400

        if upload_id not in chunk_uploads:
            logger.error(f"Invalid upload ID: {upload_id}")
            return jsonify({'success': False, 'data': {'error': 'Invalid upload ID'}}), 400

        upload_info = chunk_uploads[upload_id]
        chunk_dir = get_chunk_dir(upload_id)

        temp_info = upload_info['temp_file']
        load_info = upload_info['load_file']

        logger.info(f"Temp file: {temp_info['received_chunks']}/{temp_info['total_chunks']} chunks")
        logger.info(f"Load file: {load_info['received_chunks']}/{load_info['total_chunks']} chunks")

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
            return jsonify({
                'success': False,
                'data': {
                    'error': 'Not all chunks received',
                    'temp_progress': len(temp_info['received_chunks']) / max(temp_info['total_chunks'], 1),
                    'load_progress': len(load_info['received_chunks']) / max(load_info['total_chunks'], 1)
                }
            }), 400

        logger.info("All chunks received, reassembling files")
        
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

        try:
            validate_csv_size(temp_file_path)
            validate_csv_size(load_file_path)
        except ValueError as e:
            logger.error(f"File size validation failed: {str(e)}")
            return jsonify({'success': False, 'data': {'error': str(e)}}), 400

        try:
            df1 = pd.read_csv(temp_file_path, sep=';')
            df2 = pd.read_csv(load_file_path, sep=';')

            validate_dataframe(df1, "Temperature file")
            validate_dataframe(df2, "Load file")

            processing_params = {
                'REG': data.get('REG', 'lin'),
                'TR': data.get('TR', 'cnt'),
                'TOL_CNT': data.get('TOL_CNT', '0'),
                'TOL_DEP': data.get('TOL_DEP', '0'),
                'TOL_DEP_EXTRA': data.get('TOL_DEP_EXTRA', '0')
            }
            
            logger.info(f"Processing data with parameters: {processing_params}")
            
            result = _process_data_frames(df1, df2, processing_params)
            
            try:
                chunk_dir = get_chunk_dir(upload_id)
                if os.path.exists(chunk_dir):
                    shutil.rmtree(chunk_dir)
                if upload_id in chunk_uploads:
                    del chunk_uploads[upload_id]
                logger.info(f"Successfully cleaned up chunks for upload ID: {upload_id}")
            except Exception as e:
                logger.warning(f"Error cleaning up chunks: {str(e)}")

            return result
        except Exception as e:
            logger.error(f"Error processing uploaded files: {str(e)}")
            return jsonify({'success': False, 'data': {'error': f'Error processing uploaded files: {str(e)}'}}), 500
    except Exception as e:
        logger.error(f"Error in complete_redirect: {str(e)}")
        return jsonify({'success': False, 'data': {'error': str(e)}}), 500

def interpolate_data(df1, df2, x_col, y_col, max_time_span):
    df = pd.DataFrame()
    df['UTC'] = pd.to_datetime(df1['UTC'])
    df['value'] = pd.to_numeric(df2[y_col], errors='coerce')
    df = df.sort_values('UTC').reset_index(drop=True)
    
    df_final = df.copy()
    
    i = 0
    while i < len(df_final):
        if pd.isna(df_final.at[i, 'value']):
            i += 1
        else:
            break
    
    if i == len(df_final):
        logger.warning("No numeric data found for interpolation")
        return df_final, 0
    
    frame = "non"
    i_start = 0
    
    i = 0
    while i < len(df_final):
        if frame == "non":
            if pd.isna(df_final.at[i, 'value']):
                i_start = i
                frame = "open"
            i += 1
        elif frame == "open":
            if not pd.isna(df_final.at[i, 'value']):
                frame_width = (df_final.at[i, 'UTC'] - df_final.at[i_start-1, 'UTC']).total_seconds() / 60
                
                if frame_width <= max_time_span:
                    y0 = df_final.at[i_start-1, 'value']
                    y1 = df_final.at[i, 'value']
                    t0 = df_final.at[i_start-1, 'UTC']
                    t1 = df_final.at[i, 'UTC']
                    
                    y_diff = y1 - y0
                    diff_per_min = y_diff / frame_width
                    
                    for j in range(i_start, i):
                        gap_min = (df_final.at[j, 'UTC'] - t0).total_seconds() / 60
                        df_final.at[j, 'value'] = y0 + (gap_min * diff_per_min)
                
                frame = "non"
                i += 1
            else:
                i += 1
        else:
            i += 1
        
        if i >= len(df_final):
            break
    
    original_nans = df['value'].isna().sum()
    final_nans = df_final['value'].isna().sum()
    added_points = original_nans - final_nans
    
    return df_final, added_points

def _validate_and_prepare_data(df1, df2):
    """Validate and prepare CSV data for regression analysis."""
    logger.info(f"Processing dataframes with shapes: {df1.shape}, {df2.shape}")
    logger.info(f"Columns in temperature file: {df1.columns.tolist()}")
    logger.info(f"Columns in load file: {df2.columns.tolist()}")

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

    df1['UTC'] = pd.to_datetime(df1['UTC'], format="%Y-%m-%d %H:%M:%S")
    df2['UTC'] = pd.to_datetime(df2['UTC'], format="%Y-%m-%d %H:%M:%S")

    df1[x] = pd.to_numeric(df1[x], errors='coerce')
    df2[y] = pd.to_numeric(df2[y], errors='coerce')

    df1 = df1.sort_values('UTC')
    df2 = df2.sort_values('UTC')

    logger.info(f"Data ranges after sorting:")
    logger.info(f"Temperature range: {df1[x].min():.2f} to {df1[x].max():.2f} °C")
    logger.info(f"Load range before conversion: {df2[y].min():.2f} to {df2[y].max():.2f} kW")

    df_merged = pd.merge(df1[['UTC', x]], df2[['UTC', y]], on='UTC', how='inner')
    if df_merged.empty:
        raise ValueError('No matching timestamps found between files. Please ensure both files have matching timestamps.')

    logger.info(f"First few timestamps in first file: {df1['UTC'].head().tolist()}")
    logger.info(f"First few timestamps in second file: {df2['UTC'].head().tolist()}")

    df1_duplicates = df1['UTC'].duplicated().sum()
    df2_duplicates = df2['UTC'].duplicated().sum()
    if df1_duplicates > 0 or df2_duplicates > 0:
        raise ValueError('Duplicate timestamps found in data')

    df1 = df1.dropna()
    df2 = df2.dropna()
    if df1.empty or df2.empty:
        raise ValueError('No valid numeric data found after cleaning')

    logger.info(f"Data cleaned. New shapes: {df1.shape}, {df2.shape}")

    cld = pd.DataFrame()
    cld[x] = df1[x]
    cld[y] = df2[y]
    cld = cld.dropna()
    if cld.empty:
        raise ValueError('No valid data points after combining datasets')

    logger.info(f"Combined data shape: {cld.shape}")

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
        cld_srt, x, y, df2_original = _validate_and_prepare_data(df1, df2)

        REG = data.get('REG', 'lin')
        TR = data.get('TR', 'cnt')
        y_range = df2_original[y].max() - df2_original[y].min()
        TOL_CNT, TOL_DEP = _calculate_tolerance_params(data, y_range)

        logger.info(f"Final parameters: REG={REG}, TR={TR}, TOL_CNT={TOL_CNT}, TOL_DEP={TOL_DEP}")

        if REG == "lin":
            result_data = _perform_linear_regression(cld_srt, x, y, TR, TOL_CNT, TOL_DEP)
        else:
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

        temp_data = data['files'].get('temp_out.csv')
        load_data = data['files'].get('load.csv')

        if not temp_data or not load_data:
            logger.error("One or both files are empty")
            return jsonify({'success': False, 'data': {'error': 'One or both files are empty'}}), 400

        try:
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

            validate_dataframe(df1, "Temperature file")
            validate_dataframe(df2, "Load file")

            return _process_data_frames(df1, df2, data)
        except ValueError as e:
            logger.error(f"Validation error: {str(e)}")
            return jsonify({'success': False, 'data': {'error': str(e)}}), 400
        except Exception as e:
            logger.error(f"Error reading CSV files: {str(e)}")
            return jsonify({'success': False, 'data': {'error': f'Error reading CSV files: {str(e)}'}}), 400

    except Exception as e:
        logger.error(f"Error processing data: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'success': False, 'data': {'error': str(e)}}), 500


@bp.route('/interpolate-chunked', methods=['POST'])
def interpolate_chunked():
    """
    Process a chunked file upload for interpolation.
    Svi odgovori su u formatu: {'success': bool, 'data': ...}
    U slučaju greške, data je objekat sa 'error' poljem.
    """
    try:
        logger.info("Received request to /interpolate-chunked")

        data = request.json
        if not data or 'uploadId' not in data:
            logger.error("Missing uploadId in request")
            return jsonify({'success': False, 'data': {'error': 'Upload ID is required'}}), 400

        upload_id = data['uploadId']
        logger.info(f"Processing upload ID: {upload_id}")

        try:
            upload_id = sanitize_upload_id(upload_id)
        except ValueError as e:
            return jsonify({'success': False, 'data': {'error': str(e)}}), 400

        try:
            max_time_span = float(data.get('max_time_span', '60'))
            logger.info(f"Using max_time_span: {max_time_span}")
        except ValueError as e:
            logger.error(f"Invalid max_time_span value: {data.get('max_time_span')}")
            return jsonify({'success': False, 'data': {'error': 'Invalid max_time_span parameter'}}), 400

        if upload_id not in chunk_uploads:
            logger.error(f"Upload ID not found: {upload_id}")
            return jsonify({'success': False, 'data': {'error': 'Upload ID not found'}}), 404

        upload_info = chunk_uploads[upload_id]['interpolate_file']
        if len(upload_info['received_chunks']) < upload_info['total_chunks']:
            logger.error(f"Not all chunks received for upload {upload_id}")
            return jsonify({'success': False, 'data': {'error': f"Incomplete upload: Only {len(upload_info['received_chunks'])}/{upload_info['total_chunks']} chunks received"}}), 400

        chunk_dir = get_chunk_dir(upload_id)
        combined_file_path = os.path.join(chunk_dir, 'combined_interpolate_file.csv')

        with open(combined_file_path, 'wb') as outfile:
            for i in range(upload_info['total_chunks']):
                chunk_path = os.path.join(chunk_dir, f"interpolate_file_{i}")
                if os.path.exists(chunk_path):
                    with open(chunk_path, 'rb') as infile:
                        shutil.copyfileobj(infile, outfile, FILE_BUFFER_SIZE)

        logger.info(f"Combined file created at: {combined_file_path}")

        try:
            validate_csv_size(combined_file_path)
        except ValueError as e:
            logger.error(f"File size validation failed: {str(e)}")
            return jsonify({'success': False, 'data': {'error': str(e)}}), 400

        try:
            with open(combined_file_path, 'r', encoding='utf-8') as f:
                first_line = f.readline()

            if ';' in first_line:
                sep = ';'
            elif ',' in first_line:
                sep = ','
            else:
                sep = None

            logger.info(f"Using separator: {sep}")

            df2 = pd.read_csv(combined_file_path,
                             sep=sep,
                             decimal=',',
                             engine='c')

            validate_dataframe(df2, "Interpolation file")

        except ValueError as e:
            logger.error(f"Validation error: {str(e)}")
            return jsonify({'success': False, 'data': {'error': str(e)}}), 400
        except Exception as e:
            logger.error(f"Error reading CSV file: {str(e)}")
            return jsonify({'success': False, 'data': {'error': f'Error reading CSV file: {str(e)}'}}), 400

        if 'UTC' not in df2.columns:
            logger.error("UTC column not found in the file")
            return jsonify({'success': False, 'data': {'error': 'UTC column not found. The file must contain a UTC column with timestamps'}}), 400

        load_terms = set(['last', 'load', 'leistung', 'kw', 'w'])
        load_cols = [col for col in df2.columns if 
                    any(term in str(col).lower() for term in load_terms)]

        if not load_cols:
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
        
        df2 = df2[['UTC', y_col]].copy()
        
        if not pd.api.types.is_numeric_dtype(df2[y_col]):
            df2[y_col] = pd.to_numeric(df2[y_col].astype(str).str.replace(',', '.').str.replace(r'[^\d\-\.]', '', regex=True), errors='coerce')
        
        
        try:
            df2['UTC'] = pd.to_datetime(df2['UTC'], errors='coerce', cache=True)
            df2.dropna(subset=['UTC'], inplace=True)
                
        except Exception as e:
            logger.error(f"Error converting UTC to datetime: {str(e)}")
            return jsonify({
                'success': False, 
                'error': f'Error processing timestamps: {str(e)}', 
                'message': 'Please check the timestamp format in the file'
            }), 400
        
        df2.sort_values('UTC', inplace=True)
        
        df2.rename(columns={y_col: 'load'}, inplace=True)
        df_load = df2.set_index('UTC')
        
        if len(df_load) < 2:
            logger.error("Not enough valid data points for interpolation")
            return jsonify({
                'success': False, 
                'error': 'Not enough valid data points', 
                'message': 'The file must contain at least 2 valid data points for interpolation'
            }), 400
        
        time_diffs = (df_load.index[1:] - df_load.index[:-1]).total_seconds() / 60
        max_gap = time_diffs.max() if len(time_diffs) > 0 else 0
        logger.info(f"Maximum time gap in data: {max_gap} minutes")
        
        total_minutes = (df_load.index[-1] - df_load.index[0]).total_seconds() / 60
        
        if total_minutes > 10000:
            resample_interval = '5min'
            logger.info(f"Large time span detected ({total_minutes} minutes), using 5-minute intervals")
        else:
            resample_interval = '1min'
            logger.info(f"Using standard 1-minute intervals")
        
        if not pd.api.types.is_numeric_dtype(df_load['load']):
            logger.info("Converting load column to numeric before interpolation")
            df_load['load'] = pd.to_numeric(df_load['load'], errors='coerce')
        
        limit = int(max_time_span)
        logger.info(f"Using interpolation limit of {limit} minutes")
            
        df2_resampled = df_load.copy()
        df2_resampled['load'] = df_load['load'].interpolate(method='linear', limit=limit)

        df2_resampled.reset_index(inplace=True)
        
        original_points = len(df2)
        total_points = len(df2_resampled)
        added_points = total_points - original_points
        
        logger.info(f"Original points: {original_points}")
        logger.info(f"Interpolated points: {total_points}")
        logger.info(f"Added points: {added_points}")
        
        chart_data = []
        for _, row in df2_resampled.iterrows():
            if pd.isna(row['UTC']):
                logger.warning(f"Skipping row with NaT timestamp: {row}")
                continue
            
            load_value = 'NaN' if pd.isna(row['load']) else float(row['load'])
                
            try:
                chart_data.append({
                    'UTC': row['UTC'].strftime("%Y-%m-%d %H:%M:%S"),
                    'value': load_value
                })
            except (ValueError, TypeError) as e:
                logger.warning(f"Error converting row to chart data: {e}. Row: {row}")
                continue

        try:
            shutil.rmtree(chunk_dir)
            logger.info(f"Cleaned up chunk directory for upload {upload_id}")
            del chunk_uploads[upload_id]
        except Exception as e:
            logger.warning(f"Error cleaning up chunks: {str(e)}")
        
        logger.info(f"Sample of chart data being sent: {chart_data[:5] if chart_data else 'No data'}")
        logger.info(f"Total points in chart data: {len(chart_data)}")
        
        if not chart_data:
            logger.error("No valid data points after processing")
            return jsonify({
                'success': False,
                'data': {
                    'error': 'No valid data points after processing',
                    'message': 'The file contains no valid data points for interpolation'
                }
            }), 400

        CHUNK_SIZE = STREAMING_CHUNK_SIZE
        
        total_rows = len(df2_resampled)
        total_chunks = (total_rows + CHUNK_SIZE - 1) // CHUNK_SIZE
        
        logger.info(f"Total rows: {total_rows}, will be sent in {total_chunks} chunks")
        
        valid_mask = ~df2_resampled['UTC'].isna()
        valid_df = df2_resampled[valid_mask].copy()
        
        valid_df['load'] = valid_df['load'].apply(lambda x: 'NaN' if pd.isna(x) else x)
        
        valid_df['UTC'] = valid_df['UTC'].dt.strftime("%Y-%m-%d %H:%M:%S")
        
        chart_df = valid_df.rename(columns={'load': 'value'})[['UTC', 'value']]
        
        total_rows = len(chart_df)
        CHUNK_SIZE = STREAMING_CHUNK_SIZE
        total_chunks = (total_rows + CHUNK_SIZE - 1) // CHUNK_SIZE
        
        original_points_count = original_points
        
        logger.info(f"Total rows: {total_rows}, will be sent in {total_chunks} chunks")
        
        def generate_chunks():
            meta_data = {
                'type': 'meta',
                'total_rows': total_rows,
                'total_chunks': total_chunks,
                'added_points': added_points,
                'original_points': original_points_count,
                'success': True
            }
            yield json.dumps(meta_data, separators=(',', ':')) + '\n'
            
            for chunk_idx in range(total_chunks):
                start_idx = chunk_idx * CHUNK_SIZE
                end_idx = min(start_idx + CHUNK_SIZE, total_rows)
                
                chunk_data_list = chart_df.iloc[start_idx:end_idx].to_dict('records')
                
                chunk_data = {
                    'type': 'data',
                    'chunk_index': chunk_idx,
                    'data': chunk_data_list
                }
                
                yield json.dumps(chunk_data, separators=(',', ':')) + '\n'
            
            yield json.dumps({
                'type': 'complete',
                'message': 'Data streaming completed',
                'success': True
            }, separators=(',', ':')) + '\n'
            
            try:
                chunk_dir = get_chunk_dir(upload_id)
                if os.path.exists(chunk_dir):
                    shutil.rmtree(chunk_dir)
                if upload_id in chunk_uploads:
                    del chunk_uploads[upload_id]
                logger.info(f"Successfully cleaned up chunks for upload ID: {upload_id}")
            except Exception as e:
                logger.warning(f"Error cleaning up chunks: {str(e)}")
        
        return Response(generate_chunks(), mimetype='application/x-ndjson')
    except Exception as e:
        logger.error(f"Error in interpolation-chunked endpoint: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False, 
            'error': str(e), 
            'message': 'An error occurred during interpolation'
        }), 500


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
            
        if 'data' in data:
            csv_data = data['data']
            filename = data.get('filename', 'interpolated_data')
            
            if not isinstance(csv_data, list) or len(csv_data) == 0:
                logger.error("Invalid data format for CSV")
                return jsonify({"success": False, "data": {"error": "Invalid data format"}}), 400
                
            logger.info(f"Preparing CSV file with name: {filename}")
            
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
            
            try:
                with open(temp_file.name, 'w', newline='') as f:
                    writer = csv.writer(f, delimiter=';')
                    for row in csv_data:
                        writer.writerow(row)
                        
                file_id = f"csv_{datetime.now().strftime('%Y%m%d%H%M%S')}_{os.getpid()}"
                
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
        if file_id in temp_files:
            try:
                os.unlink(temp_files[file_id]['path'])
                del temp_files[file_id]
                logger.info(f"Cleaned up temporary file for ID: {file_id}")
            except Exception as ex:
                logger.error(f"Error cleaning up temp file: {ex}")
