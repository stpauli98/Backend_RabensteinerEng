import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from datetime import datetime
import tempfile
import os
import csv
import logging
import traceback
from io import StringIO
from flask import request, jsonify, send_file, Blueprint, Response
import json
import shutil
import base64
import os
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

        if not upload_id or upload_id not in chunk_uploads:
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
                logger.info(f"Cleaned up chunk directory for upload {upload_id}")
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
    """
    Perform linear interpolation on the data within the specified time span.
    Vraća interpolirani DataFrame i broj interpoliranih tačaka.
    Svi logovi idu kroz logger, ne koristi se print.
    """
    try:
        # Kombinuj podatke iz oba DataFrame-a
        cld = pd.DataFrame()
        cld['UTC'] = df1['UTC']
        cld[x_col] = df1[x_col]
        cld[y_col] = df2[y_col]

        # Originalan broj tačaka
        original_points = len(cld)

        # Izbaci redove sa NaN vrednostima
        cld = cld.dropna()
        if cld.empty:
            logger.error('No valid data points after combining datasets')
            raise ValueError('No valid data points after combining datasets')

        logger.info(f"Combined data shape: {cld.shape}")

        # Izračunaj vremenske razlike između tačaka
        cld['time_diff'] = cld['UTC'].diff().dt.total_seconds() / 60  # u minutima

        # Pronađi velike praznine (gaps)
        large_gaps = cld[cld['time_diff'] > max_time_span].index
        interpolated_points = 0

        if len(large_gaps) > 0:
            logger.info(f"Found {len(large_gaps)} gaps larger than {max_time_span} minutes")
            # Podeli podatke na segmente gde su praznine prevelike
            chunks = []
            start_idx = 0
            for gap_idx in large_gaps:
                if gap_idx > start_idx:
                    chunk = cld.loc[start_idx:gap_idx-1].copy()
                    # Linearna interpolacija unutar segmenta
                    interpolated_chunk = chunk.set_index('UTC').resample('1min').interpolate(method='linear')
                    chunks.append(interpolated_chunk)
                start_idx = gap_idx
            # Dodaj poslednji segment
            if start_idx < len(cld):
                chunk = cld.loc[start_idx:].copy()
                chunk = chunk.set_index('UTC').resample('1min').interpolate(method='linear')
                chunks.append(chunk)
            # Kombinuj sve segmente
            cld_interpolated = pd.concat(chunks)
            cld = cld_interpolated.reset_index()
            interpolated_points = len(cld) - original_points
            logger.info(f"After interpolation: {len(cld)} points (Added {interpolated_points} points)")

        return cld, interpolated_points
    except Exception as e:
        logger.error(f"Error in interpolation: {str(e)}")
        raise


def _process_data_frames(df1, df2, data):
    """
    Process data from dataframes for both direct and chunked uploads.
    Svi odgovori koriste jsonify({'success': True/False, 'data': ...})
    Svi print pozivi su zamenjeni logger-om.
    """
    try:
        logger.info(f"Processing dataframes with shapes: {df1.shape}, {df2.shape}")
        logger.info(f"Columns in temperature file: {df1.columns.tolist()}")
        logger.info(f"Columns in load file: {df2.columns.tolist()}")

        # Pronađi kolone za temperaturu i opterećenje
        temp_cols = [col for col in df1.columns if col != 'UTC']
        load_cols = [col for col in df2.columns if col != 'UTC']

        if temp_cols:
            x = temp_cols[0]
            logger.info(f"Using temperature column: {x}")
        else:
            logger.error(f"No temperature column found. Available columns: {df1.columns.tolist()}")
            return jsonify({'success': False, 'data': {'error': f'No valid temperature column found. Available columns: {df1.columns.tolist()}'}}), 400

        if load_cols:
            y = load_cols[0]
            logger.info(f"Using load column: {y}")
        else:
            logger.error(f"No load column found. Available columns: {df2.columns.tolist()}")
            return jsonify({'success': False, 'data': {'error': f'No valid load column found. Available columns: {df2.columns.tolist()}'}}), 400

        # Pretvori vreme u datetime i sortiraj
        df1['UTC'] = pd.to_datetime(df1['UTC'], format="%Y-%m-%d %H:%M:%S")
        df2['UTC'] = pd.to_datetime(df2['UTC'], format="%Y-%m-%d %H:%M:%S")

        # Pretvori podatke u numeričke vrednosti
        df1[x] = pd.to_numeric(df1[x], errors='coerce')
        df2[y] = pd.to_numeric(df2[y], errors='coerce')

        # Sortiraj po vremenu
        df1 = df1.sort_values('UTC')
        df2 = df2.sort_values('UTC')

        logger.info(f"Data ranges after sorting:")
        logger.info(f"Temperature range: {df1[x].min():.2f} to {df1[x].max():.2f} °C")
        logger.info(f"Load range before conversion: {df2[y].min():.2f} to {df2[y].max():.2f} kW")

        # Ako je potrebno, konvertuj kW u W
        if 'kw' in y.lower():
            logger.info("Converting kW to W")
            df2[y] = df2[y] * 1000
            logger.info(f"Load range after conversion: {df2[y].min():.2f} to {df2[y].max():.2f} W")

        # Proveri da li se vremena poklapaju
        if not df1['UTC'].equals(df2['UTC']):
            logger.warning("Time stamps don't match exactly")
            logger.warning(f"Temperature times: {df1['UTC'].tolist()}")
            logger.warning(f"Load times: {df2['UTC'].tolist()}")
            return jsonify({'success': False, 'data': {'error': 'Time stamps in files do not match'}}), 400

        # Čišćenje i validacija podataka
        try:
            # Izbaci redove sa NaN vrednostima
            df1 = df1.dropna()
            df2 = df2.dropna()
            if df1.empty or df2.empty:
                logger.error('No valid numeric data found after cleaning')
                return jsonify({'success': False, 'data': {'error': 'No valid numeric data found after cleaning'}}), 400

            logger.info(f"Data cleaned. New shapes: {df1.shape}, {df2.shape}")

            # Kombinuj podatke
            cld = pd.DataFrame()
            cld[x] = df1[x]
            cld[y] = df2[y]
            cld = cld.dropna()
            if cld.empty:
                logger.error('No valid data points after combining datasets')
                return jsonify({'success': False, 'data': {'error': 'No valid data points after combining datasets'}}), 400

            logger.info(f"Combined data shape: {cld.shape}")

            # Sortiraj po x
            cld_srt = cld.sort_values(by=x).copy()

            # Proveri da li ima NaN vrednosti
            if cld_srt[x].isna().any() or cld_srt[y].isna().any():
                logger.error('NaN values found in processed data')
                return jsonify({'success': False, 'data': {'error': 'NaN values found in processed data'}}), 400

        except Exception as e:
            logger.error(f"Error cleaning data: {str(e)}")
            return jsonify({'success': False, 'data': {'error': f'Error cleaning data: {str(e)}'}}), 400

        # Parametri za regresiju i tolerancije
        REG = data.get('REG', 'lin')  # Default linear regression
        TR = data.get('TR', 'cnt')    # Default constant tolerance

        # Podrazumevane tolerancije na osnovu opsega
        y_range = df2[y].max() - df2[y].min()
        default_tol = y_range * 0.1  # 10% opsega

        # Dohvati tolerancije
        try:
            TOL_CNT = float(data.get('TOL_CNT', default_tol))
            TOL_DEP = float(data.get('TOL_DEP', 0.1))  # 10% default za zavisnu toleranciju
        except ValueError:
            logger.warning("Using default tolerances due to invalid input")
            TOL_CNT = default_tol
            TOL_DEP = 0.1
        
        # Ako je tolerancija premala u odnosu na opseg podataka, povećaj je
        if TOL_CNT < y_range * 0.01:  # Manje od 1% opsega
            TOL_CNT = y_range * 0.1  # Postavi na 10% opsega
            logger.info(f"Adjusted tolerance to {TOL_CNT:.2f} (10% of data range)")
        
        logger.info(f"Parameters: REG={REG}, TR={TR}, TOL_CNT={TOL_CNT}, TOL_DEP={TOL_DEP}")
        
        # Izvrši regresiju
        if REG == "lin":
            try:
                # Fit linear regression
                lin_mdl = LinearRegression()
                lin_mdl.fit(cld_srt[[x]], cld_srt[y])
                lin_prd = lin_mdl.predict(cld_srt[[x]])
                lin_fcn = f"y = {lin_mdl.coef_[0]:.2f}x + {lin_mdl.intercept_:.2f}"
                logger.info(f"Linearna regresija: {lin_fcn}")

                # Tolerancije i maskiranje kao u cloudOG.py
                if TR == "cnt":
                    upper_bound = lin_prd + TOL_CNT
                    lower_bound = lin_prd - TOL_CNT
                    mask = np.abs(cld_srt[y] - lin_prd) <= TOL_CNT
                elif TR == "dep":
                    upper_bound = lin_prd + TOL_DEP * np.abs(lin_prd) + TOL_CNT
                    lower_bound = lin_prd - TOL_DEP * np.abs(lin_prd) - TOL_CNT
                    mask = np.abs(cld_srt[y] - lin_prd) <= (np.abs(lin_prd) * TOL_DEP + TOL_CNT)
                else:
                    logger.error(f"Unknown tolerance type: {TR}")
                    return jsonify({'success': False, 'data': {'error': f'Unknown tolerance type: {TR}'}}), 400

                cld_srt_flt = cld_srt[mask]

                if len(cld_srt_flt) == 0:
                    logger.warning(f"No points within tolerance bounds. Min y: {cld_srt[y].min()}, Max y: {cld_srt[y].max()}")
                    logger.warning(f"Bounds: lower={lower_bound.min()}, upper={upper_bound.max()}")
                    return jsonify({
                        'error': 'No points within tolerance bounds. Try increasing the tolerance values.',
                        'min_y': float(cld_srt[y].min()),
                        'max_y': float(cld_srt[y].max()),
                        'min_bound': float(lower_bound.min()),
                        'max_bound': float(upper_bound.max())
                    }), 400

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
                logger.error(f"Error in linear regression: {str(e)}")
                return jsonify({'error': f'Error in linear regression: {str(e)}'}), 500
        else:  # REG == "poly"
            try:
                # Polynomial regression (degree 2)
                poly_features = PolynomialFeatures(degree=2)
                X_poly = poly_features.fit_transform(cld_srt[[x]])
                poly_mdl = LinearRegression()
                poly_mdl.fit(X_poly, cld_srt[y])
                poly_prd = poly_mdl.predict(X_poly)
                coeffs = poly_mdl.coef_
                intercept = poly_mdl.intercept_
                poly_fcn = f"y = {coeffs[2]:.2f}x² + {coeffs[1]:.2f}x + {intercept:.2f}"
                logger.info(f"Polynom-Regression (Grad 2): {poly_fcn}")

                if TR == "cnt":
                    upper_bound = poly_prd + TOL_CNT
                    lower_bound = poly_prd - TOL_CNT
                    mask = np.abs(cld_srt[y] - poly_prd) <= TOL_CNT
                elif TR == "dep":
                    upper_bound = poly_prd + TOL_DEP * np.abs(poly_prd) + TOL_CNT
                    lower_bound = poly_prd - TOL_DEP * np.abs(poly_prd) - TOL_CNT
                    mask = np.abs(cld_srt[y] - poly_prd) <= (np.abs(poly_prd) * TOL_DEP + TOL_CNT)
                else:
                    logger.error(f"Unknown tolerance type: {TR}")
                    return jsonify({'success': False, 'data': {'error': f'Unknown tolerance type: {TR}'}}), 400

                cld_srt_flt = cld_srt[mask]

                if len(cld_srt_flt) == 0:
                    logger.warning(f"No points within tolerance bounds. Min y: {cld_srt[y].min()}, Max y: {cld_srt[y].max()}")
                    logger.warning(f"Bounds: lower={lower_bound.min()}, upper={upper_bound.max()}")
                    return jsonify({
                        'error': 'No points within tolerance bounds. Try increasing the tolerance values.',
                        'min_y': float(cld_srt[y].min()),
                        'max_y': float(cld_srt[y].max()),
                        'min_bound': float(lower_bound.min()),
                        'max_bound': float(upper_bound.max())
                    }), 400

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
                logger.error(f"Error in polynomial regression: {str(e)}")
                return jsonify({'error': f'Error in polynomial regression: {str(e)}'}), 500

        # Send response
        logger.info("Sending response:")
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

            # Obradi podatke koristeći zajedničku funkciju
            return _process_data_frames(df1, df2, data)
        except Exception as e:
            logger.error(f"Error reading CSV files: {str(e)}")
            return jsonify({'success': False, 'data': {'error': f'Error reading CSV files: {str(e)}'}}), 400

    except Exception as e:
        logger.error(f"Error processing data: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'success': False, 'data': {'error': str(e)}}), 500


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
                        shutil.copyfileobj(infile, outfile, 1024*1024)  # 1MB buffer

        logger.info(f"Combined file created at: {combined_file_path}")

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
        
        # Ensure load column is numeric before interpolation
        if not pd.api.types.is_numeric_dtype(df_load['load']):
            logger.info("Converting load column to numeric before interpolation")
            df_load['load'] = pd.to_numeric(df_load['load'], errors='coerce')
            
        # Resample i interpolacija
        limit = int(max_time_span)  # Convert to integer number of minutes
        df2_resampled = df_load.resample(resample_interval).interpolate(method='linear', limit=limit)
        
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
        CHUNK_SIZE = 5000  # Adjust this based on your needs
        
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
        CHUNK_SIZE = 5000  # Optimizovana veličina chunk-a za bolje performanse
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
                # Only remove the specific upload's directory, not the entire chunk directory
                shutil.rmtree(chunk_dir)
                logger.info(f"Cleaned up chunk directory for upload {upload_id}")
                # Remove from memory
                del chunk_uploads[upload_id]
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
    Retrieve a previously generated chart image.
    Očekuje se da image_id dođe iz request.json['image_id'].
    Svi error odgovori su u formatu {'success': False, 'data': {'error': ...}}
    """
    try:
        data = request.json
        if not data or 'image_id' not in data:
            logger.error("No image_id provided in request.")
            return jsonify({"success": False, "data": {"error": "No image_id provided in request."}}), 400
        image_id = data['image_id']
        logger.info(f"Received request for chart image with ID: {image_id}")
        logger.info(f"Available temp files: {list(temp_files.keys())}")

        if image_id not in temp_files:
            logger.error(f"Image ID not found in temp_files: {image_id}")
            return jsonify({"success": False, "data": {"error": "Image not found"}}), 404

        file_info = temp_files[image_id]
        file_path = file_info['path']
        logger.info(f"Found image at path: {file_path}")

        if not os.path.exists(file_path):
            logger.error(f"Image file not found at path: {file_path}")
            return jsonify({"success": False, "data": {"error": "Image file not found"}}), 404

        return send_file(
            file_path,
            mimetype='image/png'
        )
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