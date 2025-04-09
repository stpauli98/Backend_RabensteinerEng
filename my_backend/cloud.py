import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from datetime import datetime
import tempfile
import os
import time
import csv
import logging
import traceback
import time
from io import StringIO
from flask import request, jsonify, send_file, Blueprint, Response
import json
import uuid
import shutil
import base64
from io import StringIO, BytesIO
import os
import sys

# Create blueprint
bp = Blueprint('cloud', __name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Dictionary to store temporary files
temp_files = {}

# Dictionary to store chunked file uploads
chunk_uploads = {}

# Directory for storing chunks
CHUNK_DIR = os.path.join(tempfile.gettempdir(), 'cloud_chunks')
os.makedirs(CHUNK_DIR, exist_ok=True)

# Valid file types for chunked uploads
VALID_FILE_TYPES = ['temp_file', 'load_file', 'interpolate_file']

def get_chunk_dir(upload_id):
    """Create and return a directory path for storing chunks of a specific upload."""
    chunk_dir = os.path.join(CHUNK_DIR, upload_id)
    os.makedirs(chunk_dir, exist_ok=True)
    return chunk_dir

@bp.route('/upload-chunk', methods=['POST'])
def upload_chunk():
    """Handle chunk upload for large files."""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file part in the request'}), 400
            
        file_chunk = request.files['file']
        upload_id = request.form.get('uploadId')
        chunk_index = int(request.form.get('chunkIndex', 0))
        total_chunks = int(request.form.get('totalChunks', 1))
        file_type = request.form.get('fileType')
        
        if not upload_id:
            return jsonify({'success': False, 'error': 'No upload ID provided'}), 400
            
        if not file_type or file_type not in VALID_FILE_TYPES:
            return jsonify({'success': False, 'error': 'Invalid file type'}), 400
        
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
        
        logger.info(f"Received chunk {chunk_index+1}/{total_chunks} for {file_type} in upload {upload_id}")
        
        return jsonify({
            'success': True,
            'message': f'Chunk {chunk_index+1}/{total_chunks} received',
            'uploadId': upload_id,
            'progress': len(chunk_uploads[upload_id][file_type]['received_chunks']) / total_chunks
        })
        
    except Exception as e:
        logger.error(f"Error in chunk upload: {str(e)}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

def calculate_bounds(predictions, tolerance_type, tol_cnt, tol_dep): 
    """Calculate upper and lower bounds based on tolerance type."""
    if tolerance_type == 'cnt':
        upper_bound = predictions + tol_cnt
        lower_bound = predictions - tol_cnt
    else:  # tolerance_type == 'dep'
        upper_bound = predictions * (1 + tol_dep) + tol_cnt
        lower_bound = predictions * (1 - tol_dep) - tol_cnt
    
    # Ensure lower bound is not negative
    lower_bound = np.maximum(lower_bound, 0)
    
    return upper_bound, lower_bound

# Route for handling chunked upload completion
@bp.route('/complete', methods=['POST', 'OPTIONS'])
def complete_redirect():
    """Handle chunked upload completion directly instead of redirecting."""
    try:
        if request.method == 'OPTIONS':
            # Handle preflight request
            response = jsonify({
                'success': True,
                'message': 'CORS preflight request successful'
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
                return jsonify({
                    'success': False,
                    'error': 'Invalid request format. Expected JSON or FormData.'
                }), 400
        
        upload_id = data.get('uploadId')
        logger.info(f"Completing upload for ID: {upload_id}")
        
        if not upload_id or upload_id not in chunk_uploads:
            logger.error(f"Invalid upload ID: {upload_id}")
            return jsonify({
                'success': False,
                'error': 'Invalid upload ID'
            }), 400
        
        upload_info = chunk_uploads[upload_id]
        chunk_dir = get_chunk_dir(upload_id)
        
        # Check if all chunks have been received for both files
        temp_info = upload_info['temp_file']
        load_info = upload_info['load_file']
        
        logger.info(f"Temp file: {temp_info['received_chunks']}/{temp_info['total_chunks']} chunks")
        logger.info(f"Load file: {load_info['received_chunks']}/{load_info['total_chunks']} chunks")
        
        if (len(temp_info['received_chunks']) != temp_info['total_chunks'] or 
            len(load_info['received_chunks']) != load_info['total_chunks']):
            logger.error(f"Not all chunks received. Temp: {len(temp_info['received_chunks'])}/{temp_info['total_chunks']}, Load: {len(load_info['received_chunks'])}/{load_info['total_chunks']}")
            return jsonify({
                'success': False,
                'error': 'Not all chunks received',
                'temp_progress': len(temp_info['received_chunks']) / max(temp_info['total_chunks'], 1),
                'load_progress': len(load_info['received_chunks']) / max(load_info['total_chunks'], 1)
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
        
        # Read the reassembled files
        try:
            df1 = pd.read_csv(temp_file_path, sep=';')
            df2 = pd.read_csv(load_file_path, sep=';')
            
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
            
            # Clean up the chunks
            try:
                import shutil
                shutil.rmtree(chunk_dir)
                del chunk_uploads[upload_id]
                logger.info(f"Cleaned up chunks for upload ID: {upload_id}")
            except Exception as e:
                logger.error(f"Error cleaning up chunks: {str(e)}")
            
            logger.info("Sending response with processed data")
            return result
            
        except Exception as e:
            logger.error(f"Error processing reassembled files: {str(e)}")
            traceback.print_exc()
            return jsonify({
                'success': False,
                'error': f'Error processing files: {str(e)}'
            }), 500
    except Exception as e:
        logger.error(f"Error in complete_redirect: {str(e)}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
        
def interpolate_data(df1, df2, x_col, y_col, max_time_span):
    """Perform linear interpolation on the data within the specified time span."""
    try:
        # Create combined dataframe
        cld = pd.DataFrame()
        cld['UTC'] = df1['UTC']
        cld[x_col] = df1[x_col]
        cld[y_col] = df2[y_col]
        
        # Store original number of points
        original_points = len(cld)
        
        # Drop any rows where either x or y is NaN
        cld = cld.dropna()
        
        if cld.empty:
            raise ValueError('No valid data points after combining datasets')
        
        print(f"Combined data shape: {cld.shape}")
        
        # Calculate time differences between consecutive points
        cld['time_diff'] = cld['UTC'].diff().dt.total_seconds() / 60  # Convert to minutes
        
        # Find gaps larger than max_time_span
        large_gaps = cld[cld['time_diff'] > max_time_span].index
        
        interpolated_points = 0
        
        if len(large_gaps) > 0:
            print(f"Found {len(large_gaps)} gaps larger than {max_time_span} minutes")
            
            # Split data into chunks where gaps are too large
            chunks = []
            start_idx = 0
            
            for gap_idx in large_gaps:
                if gap_idx > start_idx:
                    chunk = cld.loc[start_idx:gap_idx-1].copy()
                    # Perform linear interpolation within the chunk
                    interpolated_chunk = chunk.set_index('UTC').resample('1min').interpolate(method='linear')
                    chunks.append(interpolated_chunk)
                start_idx = gap_idx
            
            # Add the last chunk
            if start_idx < len(cld):
                chunk = cld.loc[start_idx:].copy()
                chunk = chunk.set_index('UTC').resample('1min').interpolate(method='linear')
                chunks.append(chunk)
            
            # Combine all interpolated chunks
            cld_interpolated = pd.concat(chunks)
            
            # Update the working dataframe
            cld = cld_interpolated.reset_index()
            
            # Calculate number of interpolated points
            interpolated_points = len(cld) - original_points
            
            print(f"After interpolation: {len(cld)} points (Added {interpolated_points} points)")
        
        return cld, interpolated_points
    except Exception as e:
        print(f"Error in interpolation: {str(e)}")
        raise

def _process_data_frames(df1, df2, data):
    """Process data from dataframes.
    This function contains the core data processing logic extracted from _process_data
    to be reusable for both direct uploads and chunked uploads.
    
    Args:
        df1: Temperature dataframe
        df2: Load dataframe
        data: Dictionary containing processing parameters
        
    Returns:
        JSON response with processed data or error
    """
    try:
        print(f"Processing dataframes with shapes: {df1.shape}, {df2.shape}")
        print(f"Columns in temperature file: {df1.columns.tolist()}")
        print(f"Columns in load file: {df2.columns.tolist()}")
        
        # Use the first column that's not UTC as the temperature column
        temp_cols = [col for col in df1.columns if col != 'UTC']
        load_cols = [col for col in df2.columns if col != 'UTC']
        
        if temp_cols:
            x = temp_cols[0]
            print(f"Using temperature column: {x}")
        else:
            print("No temperature column found. Available columns:", df1.columns.tolist())
            return jsonify({'success': False, 'error': f'No valid temperature column found. Available columns: {df1.columns.tolist()}'}), 400
            
        if load_cols:
            y = load_cols[0]
            print(f"Using load column: {y}")
        else:
            print("No load column found. Available columns:", df2.columns.tolist())
            return jsonify({'success': False, 'error': f'No valid load column found. Available columns: {df2.columns.tolist()}'}), 400
            
        # Convert time column to datetime with specific format and sort
        df1['UTC'] = pd.to_datetime(df1['UTC'], format="%Y-%m-%d %H:%M:%S")
        df2['UTC'] = pd.to_datetime(df2['UTC'], format="%Y-%m-%d %H:%M:%S")
        
        # Convert data columns to numeric
        df1[x] = pd.to_numeric(df1[x], errors='coerce')
        df2[y] = pd.to_numeric(df2[y], errors='coerce')
        
        # Sort both dataframes by time
        df1 = df1.sort_values('UTC')
        df2 = df2.sort_values('UTC')
        
        print(f"Data ranges after sorting:")
        print(f"Temperature range: {df1[x].min():.2f} to {df1[x].max():.2f} °C")
        print(f"Load range before conversion: {df2[y].min():.2f} to {df2[y].max():.2f} kW")
        
        # Convert kW to W if needed
        if 'kw' in y.lower():
            print("Converting kW to W")
            df2[y] = df2[y] * 1000
            print(f"Load range after conversion: {df2[y].min():.2f} to {df2[y].max():.2f} W")
        
        # Verify data alignment
        if not df1['UTC'].equals(df2['UTC']):
            print("Warning: Time stamps don't match exactly")
            print(f"Temperature times: {df1['UTC'].tolist()}")
            print(f"Load times: {df2['UTC'].tolist()}")
            return jsonify({'success': False, 'error': 'Time stamps in files do not match'}), 400
        
        # Clean and validate data
        try:
            # Drop any rows with NaN values
            df1 = df1.dropna()
            df2 = df2.dropna()
            
            if df1.empty or df2.empty:
                return jsonify({'success': False, 'error': 'No valid numeric data found after cleaning'}), 400
                
            print(f"Data cleaned. New shapes: {df1.shape}, {df2.shape}")
            
            # Create combined dataframe
            cld = pd.DataFrame()
            cld[x] = df1[x]
            cld[y] = df2[y]
            
            # Drop any rows where either x or y is NaN
            cld = cld.dropna()
            
            if cld.empty:
                return jsonify({'success': False, 'error': 'No valid data points after combining datasets'}), 400
            
            print(f"Combined data shape: {cld.shape}")
            
            # Sort by x values
            cld_srt = cld.sort_values(by=x).copy()
            
            # Verify no NaN values remain
            if cld_srt[x].isna().any() or cld_srt[y].isna().any():
                return jsonify({'success': False, 'error': 'NaN values found in processed data'}), 400
                
        except Exception as e:
            print(f"Error cleaning data: {str(e)}")
            return jsonify({'success': False, 'error': f'Error cleaning data: {str(e)}'}), 400
        
        # Get parameters with reasonable defaults
        REG = data.get('REG', 'lin')  # Default to linear regression
        TR = data.get('TR', 'cnt')    # Default to constant tolerance
        
        # Set default tolerances based on data range
        y_range = df2[y].max() - df2[y].min()
        default_tol = y_range * 0.1  # 10% of data range
        
        # Get tolerance parameters
        try:
            TOL_CNT = float(data.get('TOL_CNT', default_tol))
            TOL_DEP = float(data.get('TOL_DEP', 0.1))  # 10% default for dependent tolerance
        except ValueError:
            print("Using default tolerances due to invalid input")
            TOL_CNT = default_tol
            TOL_DEP = 0.1
        
        # If tolerance is too small compared to data range, adjust it
        if TOL_CNT < y_range * 0.01:  # If tolerance is less than 1% of range
            TOL_CNT = y_range * 0.1  # Set to 10% of range
            print(f"Adjusted tolerance to {TOL_CNT:.2f} (10% of data range)")
        
        print(f"Parameters: REG={REG}, TR={TR}, TOL_CNT={TOL_CNT}, TOL_DEP={TOL_DEP}")
        
        # Perform regression
        if REG == "lin":
            try:
                # Get min and max values for constraints
                min_y = cld_srt[y].min()
                max_y = cld_srt[y].max()
                x_min = cld_srt[x].min()
                x_max = cld_srt[x].max()
                
                # Print data ranges for debugging
                print(f"Data ranges - X: [{x_min}, {x_max}], Y: [{min_y}, {max_y}]")
                
                # Fit linear regression
                lin_mdl = LinearRegression()
                lin_mdl.fit(cld_srt[[x]], cld_srt[y])
                
                # Calculate predictions
                lin_prd = lin_mdl.predict(cld_srt[[x]])
                
                # Calculate average y value for high temperatures
                high_temp_threshold = x_max * 0.8  # Last 20% of temperature range
                high_temp_avg = cld_srt[cld_srt[x] >= high_temp_threshold][y].mean()
                
                # Replace predictions for high temperatures with average value
                for i in range(len(lin_prd)):
                    curr_x = cld_srt[x].iloc[i]
                    if curr_x >= high_temp_threshold:
                        lin_prd[i] = high_temp_avg
                
                # Print last few predictions for debugging
                print("Last 5 points:")
                for i in range(-5, 0):
                    print(f"X: {cld_srt[x].iloc[i]}, Y: {cld_srt[y].iloc[i]}, Pred: {lin_prd[i]}")
                
                # Calculate bounds with validation
                upper_bound, lower_bound = calculate_bounds(lin_prd, TR, TOL_CNT, TOL_DEP)
                
                # Ensure bounds are within reasonable limits and not zero
                upper_bound = np.maximum(upper_bound, high_temp_avg * 1.2)  # At least 20% above average
                lower_bound = np.maximum(lower_bound, high_temp_avg * 0.8)  # At least 20% below average
                
                # Get coefficients for equation
                lin_fcn = f"y = {lin_mdl.coef_[0]:.2f}x + {lin_mdl.intercept_:.2f}"
                print("Linear regression:", lin_fcn)
                
                # Calculate tolerance bounds
                upper_bound, lower_bound = calculate_bounds(lin_prd, TR, TOL_CNT, TOL_DEP)
                
                # Filter points within tolerance
                mask = (cld_srt[y] >= lower_bound) & (cld_srt[y] <= upper_bound)
                cld_srt_flt = cld_srt[mask]
                
                if len(cld_srt_flt) == 0:
                    print(f"No points within tolerance bounds. Min y: {cld_srt[y].min()}, Max y: {cld_srt[y].max()}")
                    print(f"Bounds: lower={lower_bound.min()}, upper={upper_bound.max()}")
                    return jsonify({
                        'error': 'No points within tolerance bounds. Try increasing the tolerance values.',
                        'min_y': float(cld_srt[y].min()),
                        'max_y': float(cld_srt[y].max()),
                        'min_bound': float(lower_bound.min()),
                        'max_bound': float(upper_bound.max())
                    }), 400
                
                # Prepare response data
                result_data = {
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
            except ValueError as ve:
                return jsonify({'error': str(ve)}), 400
            except Exception as e:
                print(f"Error in linear regression: {str(e)}")
                return jsonify({'error': f'Error in linear regression: {str(e)}'}), 500
        else:  # REG == "poly"
            try:
                # Polynomial regression
                poly_features = PolynomialFeatures(degree=2)
                X_poly = poly_features.fit_transform(cld_srt[[x]])
                
                poly_mdl = LinearRegression()
                poly_mdl.fit(X_poly, cld_srt[y])
                
                # Generate predictions
                poly_prd = poly_mdl.predict(X_poly)
                
                # Get coefficients for equation
                coeffs = poly_mdl.coef_
                intercept = poly_mdl.intercept_
                poly_fcn = f"y = {coeffs[2]:.2f}x² + {coeffs[1]:.2f}x + {intercept:.2f}"
                print("Polynomial regression:", poly_fcn)
                
                # Calculate tolerance bounds
                upper_bound, lower_bound = calculate_bounds(poly_prd, TR, TOL_CNT, TOL_DEP)
                
                # Filter points within tolerance
                mask = (cld_srt[y] >= lower_bound) & (cld_srt[y] <= upper_bound)
                cld_srt_flt = cld_srt[mask]
                
                if len(cld_srt_flt) == 0:
                    print(f"No points within tolerance bounds. Min y: {cld_srt[y].min()}, Max y: {cld_srt[y].max()}")
                    print(f"Bounds: lower={lower_bound.min()}, upper={upper_bound.max()}")
                    return jsonify({
                        'error': 'No points within tolerance bounds. Try increasing the tolerance values.',
                        'min_y': float(cld_srt[y].min()),
                        'max_y': float(cld_srt[y].max()),
                        'min_bound': float(lower_bound.min()),
                        'max_bound': float(upper_bound.max())
                    }), 400
                
                # Prepare response data
                result_data = {
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
                
            except ValueError as ve:
                return jsonify({'error': str(ve)}), 400
            except Exception as e:
                print(f"Error in polynomial regression: {str(e)}")
                return jsonify({'error': f'Error in polynomial regression: {str(e)}'}), 500

        # Send response
        print("Sending response:")
        return jsonify({'success': True, 'data': result_data})
        
    except Exception as e:
        print(f"Error in data processing: {str(e)}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

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
        return jsonify({'success': False, 'error': str(e)}), 500


def _process_data():
    try:
        print("\nReceived request to /clouddata")
        data = request.json
        print("Received data:", data)
        
        if data is None:
            print("No data received")
            return jsonify({'error': 'No data received'}), 400
            
        # Get files from request
        temp_data = data['files']['temp_out.csv']
        load_data = data['files']['load.csv']
        
        if not temp_data or not load_data:
            print("One or both files are empty")
            return jsonify({'error': 'One or both files are empty'}), 400
        
        try:
            # Convert base64 data to DataFrame
            print("Attempting to decode and read temperature data...")
            temp_decoded = base64.b64decode(temp_data).decode('utf-8')
            print("Temperature data preview:", temp_decoded[:200])
            df1 = pd.read_csv(StringIO(temp_decoded), sep=';')
            
            print("Attempting to decode and read load data...")
            load_decoded = base64.b64decode(load_data).decode('utf-8')
            print("Load data preview:", load_decoded[:200])
            df2 = pd.read_csv(StringIO(load_decoded), sep=';')
            
            print(f"Successfully read data. Shapes: {df1.shape}, {df2.shape}")
            print(f"Columns in temperature file: {df1.columns.tolist()}")
            print(f"Columns in load file: {df2.columns.tolist()}")
            
            print("First few rows of temperature file:")
            print(df1.head())
            print("First few rows of load file:")
            print(df2.head())
            

            
            # Process the data using the common function
            return _process_data_frames(df1, df2, data)
            
        except UnicodeDecodeError as e:
            print(f"Error decoding CSV files: {str(e)}")
            return jsonify({'error': 'Error decoding CSV files. Please ensure files are UTF-8 encoded.'}), 400
        except pd.errors.EmptyDataError:
            print("Error: One or both CSV files are empty")
            return jsonify({'error': 'One or both CSV files are empty'}), 400
        except Exception as e:
            print(f"Error reading CSV files: {str(e)}")
            return jsonify({'error': f'Error reading CSV files: {str(e)}'}), 400

    except Exception as e:
        print("Error processing data:", str(e))
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

# Route for handling prepare save
@bp.route('/prepare-save', methods=['POST'])
def prepare_save():
    try:
        logger.info("Received prepare_save request")
        data = request.json
        if not data or 'data' not in data:
            logger.error("No data received in request")
            return jsonify({"success": False, "error": "No data received"}), 400
            
        save_data = data['data']
        if not save_data:
            logger.error("Empty data received")
            return jsonify({"success": False, "error": "Empty data"}), 400

        # Get filename if provided, otherwise use default
        filename = data.get('filename', 'interpolated_data')
        logger.info(f"Using filename: {filename}")
        logger.info(f"Processing {len(save_data)} rows of data")

        # Create temporary file and write CSV data
        temp_file = tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.csv')
        writer = csv.writer(temp_file, delimiter=';')
        
        try:
            for row in save_data:
                writer.writerow(row)
        except Exception as e:
            logger.error(f"Error writing to CSV: {str(e)}")
            os.unlink(temp_file.name)  # Clean up the file
            raise
            
        temp_file.close()
        logger.info(f"Successfully wrote data to temporary file: {temp_file.name}")

        # Generate unique ID based on current time
        file_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        # Store both the file path and the custom filename
        temp_files[file_id] = {
            'path': temp_file.name,
            'filename': filename
        }
        logger.info(f"Generated file ID: {file_id} for filename: {filename}")

        return jsonify({
            "success": True,
            "message": "File prepared for download", 
            "fileId": file_id
        }), 200
    except Exception as e:
        logger.error(f"Error in prepare_save: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"success": False, "error": str(e)}), 500

# Route for handling chunked upload for interpolation
@bp.route('/interpolate-chunked', methods=['POST'])
def interpolate_chunked():
    """Process a chunked file upload for interpolation."""
    try:
        logger.info("Received request to /interpolate-chunked")
        logger.info(f"Request method: {request.method}")
        logger.info(f"Request content type: {request.content_type}")
        logger.info(f"Request data: {request.json}")
        
        # Get upload ID and parameters from request
        data = request.json
        if not data or 'uploadId' not in data:
            logger.error("Missing uploadId in request")
            return jsonify({
                'success': False, 
                'error': 'Upload ID is required', 
                'message': 'Please provide a valid upload ID'
            }), 400
            
        upload_id = data['uploadId']
        logger.info(f"Processing upload ID: {upload_id}")
        
        # Validate max_time_span parameter
        try:
            max_time_span = float(data.get('max_time_span', '60'))
            logger.info(f"Using max_time_span: {max_time_span}")
        except ValueError as e:
            logger.error(f"Invalid max_time_span value: {data.get('max_time_span')}")
            return jsonify({
                'success': False, 
                'error': 'Invalid max_time_span parameter', 
                'message': 'Please provide a valid number for max_time_span'
            }), 400
        
        # Check if upload exists
        if upload_id not in chunk_uploads:
            logger.error(f"Upload ID not found: {upload_id}")
            return jsonify({
                'success': False, 
                'error': 'Upload ID not found', 
                'message': 'The specified upload ID does not exist'
            }), 404
        
        # Check if all chunks have been received
        upload_info = chunk_uploads[upload_id]['interpolate_file']
        if len(upload_info['received_chunks']) < upload_info['total_chunks']:
            logger.error(f"Not all chunks received for upload {upload_id}")
            return jsonify({
                'success': False, 
                'error': 'Incomplete upload', 
                'message': f"Only {len(upload_info['received_chunks'])}/{upload_info['total_chunks']} chunks received"
            }), 400
        
        # Combine chunks into a single file
        chunk_dir = get_chunk_dir(upload_id)
        combined_file_path = os.path.join(chunk_dir, 'combined_interpolate_file.csv')
        
        with open(combined_file_path, 'wb') as outfile:
            for i in range(upload_info['total_chunks']):
                chunk_path = os.path.join(chunk_dir, f"interpolate_file_{i}")
                if os.path.exists(chunk_path):
                    with open(chunk_path, 'rb') as infile:
                        outfile.write(infile.read())
        
        logger.info(f"Combined file created at: {combined_file_path}")
        
        # Read the combined CSV file
        try:
            with open(combined_file_path, 'r', encoding='utf-8') as f:
                file_content = f.read()
            
            # Try different separators if needed
            if ';' in file_content:
                sep = ';'
            elif ',' in file_content:
                sep = ','
            else:
                sep = None  # Let pandas detect
                
            logger.info(f"Using separator: {sep}")
            
            df2 = pd.read_csv(StringIO(file_content), 
                             sep=sep,
                             decimal=',')   # Use comma as decimal separator
        except Exception as e:
            logger.error(f"Error reading CSV file: {str(e)}")
            return jsonify({
                'success': False, 
                'error': f'Error reading CSV file: {str(e)}', 
                'message': 'Please check the file format'
            }), 400
        
        logger.info(f"CSV columns: {df2.columns.tolist()}")
        
        # Check if UTC column exists
        if 'UTC' not in df2.columns:
            logger.error("UTC column not found in the file")
            return jsonify({
                'success': False, 
                'error': 'UTC column not found', 
                'message': 'The file must contain a UTC column with timestamps'
            }), 400
        
        # Find load column
        load_cols = [col for col in df2.columns if any(term in str(col).lower() 
                                                     for term in ['last', 'load', 'leistung', 'kw', 'w'])]
        
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
        
        # Print first few rows to debug
        logger.info("First few rows of data:")
        logger.info(df2.head().to_string())
        
        # Clean the data - ensure y_col is string before applying regex
        if not pd.api.types.is_numeric_dtype(df2[y_col]):
            # Convert to string first if not already
            df2[y_col] = df2[y_col].astype(str)
            # Remove any non-numeric characters and convert to float
            df2[y_col] = df2[y_col].replace(r'[^\d\-\.,]', '', regex=True)
            # Replace comma with dot for decimal point
            df2[y_col] = df2[y_col].str.replace(',', '.')
        
        # Convert to float, replacing any invalid values with NaN
        df2[y_col] = pd.to_numeric(df2[y_col], errors='coerce')
        
        # Remove rows with NaN values
        original_len = len(df2)
        df2 = df2.dropna(subset=[y_col])
        dropped_rows = original_len - len(df2)
        if dropped_rows > 0:
            logger.info(f"Dropped {dropped_rows} rows with invalid values")
        
        # Convert time column to datetime
        try:
            # Try common datetime formats
            df2['UTC'] = pd.to_datetime(df2['UTC'], errors='coerce')
            
            # Check if we have NaT values and try different formats if needed
            if df2['UTC'].isna().any():
                logger.warning("Some datetime values could not be parsed with default format")
                # Try with specific format if needed
                df2['UTC'] = pd.to_datetime(df2['UTC'], format="%Y-%m-%d %H:%M:%S", errors='coerce')
            
            # Drop rows with invalid datetime
            original_len = len(df2)
            df2 = df2.dropna(subset=['UTC'])
            dropped_rows = original_len - len(df2)
            if dropped_rows > 0:
                logger.info(f"Dropped {dropped_rows} rows with invalid datetime values")
                
        except Exception as e:
            logger.error(f"Error converting UTC to datetime: {str(e)}")
            return jsonify({
                'success': False, 
                'error': f'Error processing timestamps: {str(e)}', 
                'message': 'Please check the timestamp format in the file'
            }), 400
        
        # Sort by time
        df2 = df2.sort_values('UTC')
        
        logger.info(f"Data types after cleaning:")
        logger.info(df2.dtypes)
        logger.info("\nSample of cleaned data:")
        logger.info(df2[[y_col, 'UTC']].head().to_string())
        
        # Create a DataFrame with just UTC and load data
        df_load = pd.DataFrame({
            'UTC': df2['UTC'],
            'load': df2[y_col]  # Already converted to float
        })
        
        # Check if we have enough data points
        if len(df_load) < 2:
            logger.error("Not enough valid data points for interpolation")
            return jsonify({
                'success': False, 
                'error': 'Not enough valid data points', 
                'message': 'The file must contain at least 2 valid data points for interpolation'
            }), 400
        
        # Set UTC as index
        df_load.set_index('UTC', inplace=True)
        
        # Calculate time difference between consecutive points
        time_diffs = df_load.index.to_series().diff().dt.total_seconds() / 60  # in minutes
        max_gap = time_diffs.max()
        logger.info(f"Maximum time gap in data: {max_gap} minutes")
        
        # Only interpolate gaps smaller than max_time_span
        if max_gap > max_time_span:
            logger.info(f"Limiting interpolation to gaps smaller than {max_time_span} minutes")
        
        # Resample to 1-minute intervals and interpolate
        # Use limit to restrict interpolation to max_time_span minutes
        limit = int(max_time_span)  # Convert to integer number of minutes
        df2_resampled = df_load.resample('1min').interpolate(method='linear', limit=limit)
        
        # Reset index to get UTC back as a column
        df2_resampled.reset_index(inplace=True)
        
        # Calculate added points
        added_points = len(df2_resampled) - len(df2)
        
        logger.info(f"Original points: {len(df2)}")
        logger.info(f"Interpolated points: {len(df2_resampled)}")
        logger.info(f"Added points: {added_points}")
        
        # Prepare data for frontend chart
        chart_data = []
        for _, row in df2_resampled.iterrows():
            # Skip NaT values
            if pd.isna(row['UTC']) or pd.isna(row['load']):
                logger.warning(f"Skipping row with NaT/NaN values: {row}")
                continue
                
            try:
                chart_data.append({
                    'UTC': row['UTC'].strftime("%Y-%m-%d %H:%M:%S"),
                    'value': float(row['load'])
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
        
        # If we have no valid points, return an error
        if not chart_data:
            logger.error("No valid data points after processing")
            return jsonify({
                'success': False, 
                'error': 'No valid data points after processing', 
                'message': 'The file contains no valid data points for interpolation'
            }), 400

        # Define chunk size for streaming (number of data points per chunk)
        CHUNK_SIZE = 1000  # Adjust this based on your needs
        
        # Calculate total number of chunks needed
        total_rows = len(df2_resampled)
        total_chunks = (total_rows + CHUNK_SIZE - 1) // CHUNK_SIZE  # Ceiling division
        
        logger.info(f"Total rows: {total_rows}, will be sent in {total_chunks} chunks")
        
        # Define a generator function to stream the data in chunks
        def generate_chunks():
            # First, send metadata about the dataset
            meta_data = {
                'type': 'meta',
                'total_rows': total_rows,
                'total_chunks': total_chunks,
                'added_points': added_points,
                'original_points': len(df2),
                'success': True
            }
            yield json.dumps(meta_data, separators=(',', ':')) + '\n'
            
            # Then send the data chunks
            for chunk_idx in range(total_chunks):
                start_idx = chunk_idx * CHUNK_SIZE
                end_idx = min(start_idx + CHUNK_SIZE, total_rows)
                
                # Convert this chunk of data to the format expected by frontend
                chunk_data_list = []
                for _, row in df2_resampled.iloc[start_idx:end_idx].iterrows():
                    # Skip NaT values
                    if pd.isna(row['UTC']) or pd.isna(row['load']):
                        continue
                        
                    try:
                        chunk_data_list.append({
                            'UTC': row['UTC'].strftime("%Y-%m-%d %H:%M:%S"),
                            'value': float(row['load'])
                        })
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Error converting row to chart data: {e}. Row: {row}")
                        continue
                
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

        # Return a streaming response
        logger.info("Starting to stream response using generator")
        return Response(
            generate_chunks(),
            mimetype='application/x-ndjson'
        )
        
    except Exception as e:
        logger.error(f"Error in interpolation-chunked endpoint: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False, 
            'error': str(e), 
            'message': 'An error occurred during interpolation'
        }), 500

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