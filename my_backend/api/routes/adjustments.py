import pandas as pd
import numpy as np
from datetime import datetime
import tempfile
import os
import time
import csv
import logging
import traceback
import time
from io import StringIO
from flask import request, jsonify, send_file, Blueprint
from flask_socketio import emit
import json
from services.adjustments.cleanup import cleanup_old_files

# Create Blueprint
bp = Blueprint('adjustmentsOfData_bp', __name__)

# Configure temp upload folder
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'temp_uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global dictionaries to store data
adjustment_chunks = {}  # Store chunks during adjustment
temp_files = {}  # Store temporary files for download
# Global variables to store data
stored_data = {}

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# UTC format
UTC_fmt = "%Y-%m-%d %H:%M:%S"

# Constants
UPLOAD_EXPIRY_TIME = 60 * 60  # 10 minuta u sekundama


# Dictionary to store DataFrames
info_df = pd.DataFrame(columns=['Name der Datei', 'Name der Messreihe', 'Startzeit (UTC)', 'Endzeit (UTC)',
                                'Zeitschrittweite [min]', 'Offset [min]', 'Anzahl der Datenpunkte',
                                'Anzahl der numerischen Datenpunkte', 'Anteil an numerischen Datenpunkten'])  # DataFrame for file info

# Function to check if file is a CSV

def allowed_file(filename):
    """Check if file has .csv extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() == 'csv'

# Function to detect delimiter
def detect_delimiter(file_content):
    """
    Detect the delimiter used in a CSV file contentt
    """
    # Common delimiters
    delimiters = [';', ',', '\t']
    
    # Get first line
    first_line = file_content.split('\n')[0]
    
    # Count occurrences of each delimiter in the first line
    counts = {d: first_line.count(d) for d in delimiters}
    
    # Return the delimiter with highest count
    max_count = max(counts.values())
    if max_count > 0:
        return max(counts.items(), key=lambda x: x[1])[0]
    return ';'  # Default to semicolon if no delimiter found

# Function to get time column
def get_time_column(df):
    """
    Check for common time column names and return the first one found
    """
    time_columns = ['UTC', 'Timestamp', 'Time', 'DateTime', 'Date', 'Zeit']
    for col in df.columns:
        for time_col in time_columns:
            if time_col.lower() in col.lower():
                return col
    return None

# First Step
@bp.route('/upload-chunk', methods=['POST'])
def upload_chunk():
    """
    Endpoint za prihvat pojedinačnih chunkova.
    Očekivani parametri (form data):
      - uploadId: jedinstveni ID za upload (string)
      - chunkIndex: redni broj chanka (int, počinje od 0)
      - totalChunks: ukupan broj chunkova (int)
      - filename: originalno ime fajla
      - tss, offset, mode, intrplMax: dodatni parametri za obradu
      - files[]: sadržaj fajla kao file
    Ako su svi chunkovi primljeni, oni se spajaju i obrađuju.
    """
    try:
        # Get upload metadata from form
        upload_id = request.form.get('uploadId')
        chunk_index = int(request.form.get('chunkIndex'))
        total_chunks = int(request.form.get('totalChunks'))
        filename = request.form.get('filename')
        
        # Validacija parametara
        if not all([upload_id, isinstance(chunk_index, int), isinstance(total_chunks, int)]):
            return jsonify({"error": "Missing or invalid required parameters"}), 400
            
        # Get the file from request
        if 'files[]' not in request.files:
            return jsonify({'error': 'No file part'}), 400
            
        file = request.files['files[]']
        if not file:
            return jsonify({'error': 'No selected file'}), 400
            
        # Read file content
        file_content = file.read().decode('utf-8')
        if not file_content:
            return jsonify({'error': 'Empty file content'}), 400
            
        # Create upload directory if it doesn't exist
        upload_dir = os.path.join(UPLOAD_FOLDER, upload_id)
        os.makedirs(upload_dir, exist_ok=True)
        
        # Save the chunk
        chunk_path = os.path.join(upload_dir, f'chunk_{chunk_index}')
        with open(chunk_path, 'w', encoding='utf-8') as f:
            f.write(file_content)
        
        # Check if all chunks have been received
        received_chunks = [f for f in os.listdir(upload_dir) if f.startswith('chunk_')]
        
        # If this was the last chunk and we have all chunks, combine them
        if len(received_chunks) == total_chunks:
            # Emit progress update for file combination
            try:
                emit('processing_progress', {
                    'uploadId': upload_id,
                    'progress': 25,
                    'message': f'Combining {total_chunks} chunks for {filename}',
                    'step': 'file_combination',
                    'phase': 'file_upload'
                }, room=upload_id)
            except Exception:
                pass  # Don't fail if socket emit fails
            
            # Combine all chunks into final file with original filename
            final_path = os.path.join(upload_dir, filename)
            with open(final_path, 'w', encoding='utf-8') as outfile:
                for i in range(total_chunks):
                    chunk_path = os.path.join(upload_dir, f'chunk_{i}')
                    try:
                        with open(chunk_path, 'r', encoding='utf-8') as infile:
                            outfile.write(infile.read())
                        # Delete chunk after combining
                        os.remove(chunk_path)
                    except Exception as e:
                        return jsonify({"error": f"Error processing chunk {i}"}), 500
            
            # Emit progress update for file analysis
            try:
                emit('processing_progress', {
                    'uploadId': upload_id,
                    'progress': 28,
                    'message': f'Analyzing file {filename}',
                    'step': 'file_analysis',
                    'phase': 'file_upload'
                }, room=upload_id)
            except Exception:
                pass
            
            # Process the complete file
            try:
                # Analyze the complete file
                result = analyse_data(final_path, upload_id)
                
                # Emit completion progress
                try:
                    emit('processing_progress', {
                        'uploadId': upload_id,
                        'progress': 30,
                        'message': f'File {filename} upload and analysis complete',
                        'step': 'file_complete',
                        'phase': 'file_upload'
                    }, room=upload_id)
                except Exception:
                    pass
                
                response_data = {
                    'status': 'complete',
                    'message': 'File upload and analysis complete',
                    'success': True,
                    'data': result
                }
                return jsonify(response_data)
            except Exception as e:
                logger.error(f"Error analyzing file {final_path}: {str(e)}")
                return jsonify({"error": str(e)}), 500
            
        return jsonify({
            'status': 'chunk_received',
            'message': f'Received chunk {chunk_index + 1} of {total_chunks}',
            'chunksReceived': len(received_chunks)
        })
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 400
        

# First Step Analyse
def analyse_data(file_path, upload_id=None):
    """
    Analyze CSV file and extract relevant information
    
    Args:
        file_path (str): Path to the CSV file to analyze
        upload_id (str, optional): ID of the upload if this is part of a chunked upload
    """
    try:
        global stored_data, info_df
        
        # Clear stored data for new analysis
        stored_data.clear()
        
        all_file_info = []
        processed_data = []
        
        try:
            # Read file content
            with open(file_path, 'r', encoding='utf-8') as file:
                file_content = file.read()
        except UnicodeDecodeError as e:
            logger.error(f"UnicodeDecodeError reading {file_path}: {str(e)}")
            raise ValueError(f"Could not decode file {file_path}. Make sure it's a valid UTF-8 encoded CSV file.")
        
        # Detect delimiter from content
        delimiter = detect_delimiter(file_content)
        
        # Read CSV with detected delimiter
        df = pd.read_csv(StringIO(file_content), delimiter=delimiter)
        
        # Find time column
        time_col = get_time_column(df)
        if time_col is None:
            raise ValueError(f"No time column found in file {os.path.basename(file_path)}. Expected one of: UTC, Timestamp, Time, DateTime, Date, Zeit")
        
        # If time column is not 'UTC', rename it
        if time_col != 'UTC':
            df = df.rename(columns={time_col: 'UTC'})
        
        # Convert UTC column to datetime
        df['UTC'] = pd.to_datetime(df['UTC'])
                    
        # Store the DataFrame for later use
        filename = os.path.basename(file_path)
        stored_data[filename] = df
        
        # Store DataFrame in adjustment_chunks if upload_id provided
        if upload_id:
            if upload_id not in adjustment_chunks:
                adjustment_chunks[upload_id] = {'chunks': {}, 'params': {}, 'dataframes': {}}
            adjustment_chunks[upload_id]['dataframes'][filename] = df
                    
        # Calculate time step
        time_step = None
        try:
            # Calculate time differences and convert to minutes
            time_diffs = pd.to_datetime(df['UTC']).diff().dropna()
            time_diffs_minutes = time_diffs.dt.total_seconds() / 60
            
            # Get the most common time difference (mode)
            time_step = int(time_diffs_minutes.mode()[0])
        except Exception as e:
            logger.error(f"Error calculating time step: {str(e)}")
            traceback.print_exc()
        
        # Get the measurement column (second column or first non-time column)
        measurement_col = None
        for col in df.columns:
            if col != 'UTC':
                measurement_col = col
                break

        if measurement_col:
            # Izračunaj offset iz podataka
            first_time = df['UTC'].iloc[0]
            offset = first_time.minute % time_step if time_step else 0.0
            
            file_info = {
                'Name der Datei': os.path.basename(file_path),
                'Name der Messreihe': str(measurement_col),
                'Startzeit (UTC)': df['UTC'].iloc[0].strftime(UTC_fmt) if 'UTC' in df.columns else None,
                'Endzeit (UTC)': df['UTC'].iloc[-1].strftime(UTC_fmt) if 'UTC' in df.columns else None,
                'Zeitschrittweite [min]': float(time_step) if time_step is not None else None,
                'Offset [min]': float(offset),
                'Anzahl der Datenpunkte': int(len(df)),
                'Anzahl der numerischen Datenpunkte': int(df[measurement_col].count()),
                'Anteil an numerischen Datenpunkten': float(df[measurement_col].count() / len(df) * 100)
            }
            all_file_info.append(file_info)
                    
        # Convert DataFrame to records with explicit type conversion
        df_records = []
        filename = os.path.basename(file_path)
        for record in df.to_dict('records'):
            converted_record = {
                'Name der Datei': filename  # Add filename to each record
            }
            for key, value in record.items():
                if pd.isna(value):
                    converted_record[key] = None
                elif isinstance(value, (pd.Timestamp, np.datetime64)):
                    converted_record[key] = value.strftime(UTC_fmt)
                elif isinstance(value, np.number):
                    converted_record[key] = float(value) if not pd.isna(value) else None
                else:
                    converted_record[key] = value
            df_records.append(converted_record)
        
        processed_data.append(df_records)
        
        # Update global info_df - append new info to existing
        if all_file_info:
            new_info_df = pd.DataFrame(all_file_info)
            if info_df.empty:
                info_df = new_info_df
            else:
                # Remove any existing entries for these files
                existing_files = new_info_df['Name der Datei'].tolist()
                info_df = info_df[~info_df['Name der Datei'].isin(existing_files)]
                # Append new info
                info_df = pd.concat([info_df, new_info_df], ignore_index=True)
        # Return the upload_id
        return {
            'info_df': all_file_info,
            'upload_id': upload_id
        }
        
    except Exception as e:
        logger.error(f"Error in analyse_data: {str(e)}\n{traceback.format_exc()}")
        raise

# Second Step
@bp.route('/adjust-data-chunk', methods=['POST'])
def adjust_data():
    try:
        global adjustment_chunks
        
        data = request.json
        if not data:
            return jsonify({"error": "No JSON data received"}), 400

        # Get required parameters
        upload_id = data.get('upload_id')
        if not upload_id:
            return jsonify({"error": "upload_id is required"}), 400
            
        # Check if upload_id exists in adjustment_chunks
        if upload_id not in adjustment_chunks:
            return jsonify({"error": f"No data found for upload ID: {upload_id}"}), 404
            
        # Get dataframes from adjustment_chunks
        dataframes = adjustment_chunks[upload_id]['dataframes']
        if not dataframes:
            return jsonify({"error": "No dataframes found for this upload"}), 404
        
        # Get processing parameters
        start_time = data.get('startTime')
        end_time = data.get('endTime')
        time_step_size = data.get('timeStepSize')
        offset = data.get('offset')

        # Get methods from request or existing params
        methods = data.get('methods', {})
        if not methods:
            # If methods not in request, use existing methods
            methods = adjustment_chunks[upload_id]['params'].get('methods', {})

        # Extract intrplMax values from methods
        intrpl_max_values = {}
        for filename, method_info in methods.items():
            if isinstance(method_info, dict) and 'intrpl_max' in method_info:
                try:
                    intrpl_max_values[filename] = float(method_info['intrpl_max'])
                except (TypeError, ValueError) as e:
                    logger.warning(f"Could not convert intrplMax for {filename}: {e}")
                    intrpl_max_values[filename] = None
        
        # Update or initialize chunk storage
        if upload_id not in adjustment_chunks:
            adjustment_chunks[upload_id] = {
                'params': {
            'startTime': start_time,
            'endTime': end_time,
            'timeStepSize': time_step_size,
                    'offset': offset,
                    'methods': methods,
                    'intrplMaxValues': intrpl_max_values
                }
            }
        else:
            # Update parameters if they changed
            params = adjustment_chunks[upload_id]['params']
            if start_time is not None: params['startTime'] = start_time
            if end_time is not None: params['endTime'] = end_time
            if time_step_size is not None: params['timeStepSize'] = time_step_size
            if offset is not None: params['offset'] = offset
            
            # Initialize methods if it doesn't exist
            if 'methods' not in params:
                params['methods'] = {}
            if methods:
                params['methods'].update(methods)
            
            # Update intrplMax values
            if 'intrplMaxValues' not in params:
                params['intrplMaxValues'] = {}
            params['intrplMaxValues'].update(intrpl_max_values)
        # Get list of files being processed
        filenames = list(dataframes.keys())
        
        # Emit progress update for data processing
        try:
            emit('processing_progress', {
                'uploadId': upload_id,
                'progress': 50,
                'message': f'Processing parameters for {len(filenames)} files',
                'step': 'parameter_processing',
                'phase': 'data_processing'
            }, room=upload_id)
        except Exception:
            pass
        
        return jsonify({
            "message": "Parameters updated successfully",
            "files": filenames,
            "upload_id": upload_id
        }), 200

    except Exception as e:
        logger.error(f"Error in receive_adjustment_chunk: {str(e)}\n{traceback.format_exc()}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 400

# Route to complete adjustment
@bp.route('/adjustdata/complete', methods=['POST'])
def complete_adjustment():
    """
    Ovaj endpoint se poziva kada su svi chunkovi poslani.
    Očekuje JSON payload s:
      - uploadId: jedinstveni ID za upload (string)
      - totalChunks: ukupan broj chunkova (int)
      - startTime: početno vrijeme (opciono)
      - endTime: završno vrijeme (opciono)
      - timeStepSize: veličina vremenskog koraka (opciono)
      - offset: pomak u minutama (opciono, default 0)
      - methods: metode za obradu podataka (opciono)
      - files: lista imena fajlova
    Nakon toga, backend kombinira sve primljene chunkove,
    obrađuje ih i vraća konačni rezultat.
    """
    try:
        global adjustment_chunks
        
        data = request.json
        if not data:
            return jsonify({"error": "No JSON data received"}), 400

        # Obavezni parametri
        upload_id = data.get('uploadId')
        
        if not upload_id:
            return jsonify({"error": "Missing uploadId"}), 400
            
        # Get stored parameters
        if upload_id not in adjustment_chunks:
            return jsonify({"error": "Upload ID not found"}), 
        
        # Update methods if provided in the request
        if 'methods' in data and data['methods']:
            adjustment_chunks[upload_id]['params']['methods'] = data['methods']
        
        params = adjustment_chunks[upload_id]['params']
        
        # Check required parameters
        required_params = ['timeStepSize', 'offset']
        missing_params = [param for param in required_params if param not in params]
        
        if missing_params:
            return jsonify({
                "error": f"Missing required parameters: {', '.join(missing_params)}"
            }), 400
        
        # Get required parameters
        requested_time_step = params['timeStepSize']
        requested_offset = params['offset']
        
        # Get optional parameters   OVO CE BITI BITNO ZA KASNIJE
        methods = params.get('methods', {})
        start_time = params.get('startTime')
        end_time = params.get('endTime')
        time_step = params.get('timeStepSize')
        offset = params.get('offset')
        
        # Initialize result lists
        all_results = []
        all_info_records = []
        
        # Get intrplMax values for each file
        intrpl_max_values = params.get('intrplMaxValues', {})
        
        # Get DataFrames and their filenames
        dataframes = adjustment_chunks[upload_id]['dataframes']
        if not dataframes:
            return jsonify({"error": "No data found for this upload ID"}), 404
            
        # Get list of filenames from dataframes
        filenames = list(dataframes.keys())
        
        # Emit progress update for data processing start
        try:
            emit('processing_progress', {
                'uploadId': upload_id,
                'progress': 60,
                'message': f'Starting data processing for {len(filenames)} files',
                'step': 'data_processing_start',
                'phase': 'data_processing'
            }, room=upload_id)
        except Exception:
            pass
        
        # Clean up methods by stripping whitespace
        if methods:
            methods = {k: v.strip() if isinstance(v, str) else v for k, v in methods.items()}
            
        # Define valid methods
        VALID_METHODS = {'mean', 'intrpl', 'nearest', 'nearest (mean)', 'nearest (max. delta)'}
        
        # Process each file separately
        for file_index, filename in enumerate(filenames):
            # Emit progress for current file processing
            try:
                file_progress = 60 + (file_index / len(filenames)) * 25  # 60-85% range
                emit('processing_progress', {
                    'uploadId': upload_id,
                    'progress': file_progress,
                    'message': f'Analyzing file {file_index + 1}/{len(filenames)}: {filename}',
                    'step': 'file_analysis',
                    'phase': 'data_processing',
                    'detail': f'Checking time step configuration and processing requirements'
                }, room=upload_id)
            except Exception:
                pass
            
            # Get DataFrame for this file
            df = dataframes[filename]
            
            # Ensure we have the required columns
            if 'UTC' not in df.columns:
                logger.error(f"No UTC column found in file {filename}")
                continue
                
            df['UTC'] = pd.to_datetime(df['UTC'])
            
            # Get time step and offset from info_df---------------------------
            file_info = info_df[info_df['Name der Datei'] == filename].iloc[0]
            file_time_step = file_info['Zeitschrittweite [min]']
            file_offset = file_info['Offset [min]']
            # Check if this file needs processingComplete adjustment
            # Convert requested_offset to minutes from midnight if needed
            if requested_offset >= file_time_step:
                requested_offset = requested_offset % file_time_step
                
            # Check if parameters match
            needs_processing = file_time_step != time_step or file_offset != offset
            
            # Detaljno logovanje za dijagnostiku
            logger.info(f"File: {filename}, needs_processing: {needs_processing}, file_time_step: {file_time_step}, requested_time_step: {time_step}, file_offset: {file_offset}, requested_offset: {offset}")
            
            # Ako file treba obradu, provjerimo ima li metodu
            if needs_processing:
                method_info = methods.get(filename, {})
                method = method_info.get('method', '').strip() if isinstance(method_info, dict) else ''
                has_valid_method = method and method in VALID_METHODS
                
                logger.info(f"File: {filename}, method: {method}, has_valid_method: {has_valid_method}")
                
                if not has_valid_method:
                    # Ako nemamo validnu metodu, tražimo je od korisnika
                    return jsonify({
                        "success": True,
                        "methodsRequired": True,
                        "hasValidMethod": False,
                        "message": f"Die Datei {filename} benötigt eine Verarbeitungsmethode (Zeitschrittweite: {file_time_step}->{time_step}, Offset: {file_offset}->{offset}).",
                        "data": {
                            "info_df": [{
                                "filename": filename,
                                "current_timestep": file_time_step,
                                "requested_timestep": time_step,
                                "current_offset": file_offset,
                                "requested_offset": offset,
                                "valid_methods": list(VALID_METHODS)
                            }],
                            "dataframe": []
                        }
                    }), 200
            
            # Get intrplMax for this file if available
            intrpl_max = intrpl_max_values.get(filename)
            
            # Ako nema potrebe za obradom, direktno vrati podatke bez obrade
            if not needs_processing:
                try:
                    emit('processing_progress', {
                        'uploadId': upload_id,
                        'progress': 65 + (file_index / len(filenames)) * 20,
                        'message': f'No processing needed for {filename}',
                        'step': 'data_conversion',
                        'phase': 'data_processing',
                        'detail': f'Time step and offset match requirements - converting data directly'
                    }, room=upload_id)
                except Exception:
                    pass
                    
                logger.info(f"Skipping processing for {filename} as parameters match (timestep: {file_time_step}, offset: {file_offset})")
                result_data, info_record = convert_data_without_processing(
                    dataframes[filename],
                    filename,
                    file_time_step,
                    file_offset
                )
            else:
                try:
                    method_name = methods.get(filename, {}).get('method', 'default') if isinstance(methods.get(filename), dict) else 'default'
                    emit('processing_progress', {
                        'uploadId': upload_id,
                        'progress': 65 + (file_index / len(filenames)) * 20,
                        'message': f'Processing {filename} with {method_name} method',
                        'step': 'data_adjustment',
                        'phase': 'data_processing',
                        'detail': f'Adjusting time step from {file_time_step}min to {time_step}min, offset from {file_offset}min to {offset}min'
                    }, room=upload_id)
                except Exception:
                    pass
                
                # Process the data - koristimo originalne parametre ako ne treba obradu
                process_time_step = time_step if needs_processing else file_time_step
                process_offset = offset if needs_processing else file_offset
                
                result_data, info_record = process_data_detailed(
                    dataframes[filename],
                    filename,
                    start_time,
                    end_time,
                    process_time_step,
                    process_offset,
                    methods,
                    intrpl_max
                )
            
            if result_data is not None:
                all_results.extend(result_data)
                # Emit completion progress for this file
                try:
                    emit('processing_progress', {
                        'uploadId': upload_id,
                        'progress': 70 + ((file_index + 1) / len(filenames)) * 15,
                        'message': f'Completed processing {filename}',
                        'step': 'file_complete',
                        'phase': 'data_processing',
                        'detail': f'Generated {len(result_data)} data points for {filename}'
                    }, room=upload_id)
                except Exception:
                    pass
            if info_record is not None:
                all_info_records.append(info_record)
                
        # Emit final completion progress
        try:
            emit('processing_progress', {
                'uploadId': upload_id,
                'progress': 100,
                'message': f'Data processing completed for all {len(filenames)} files',
                'step': 'completion',
                'phase': 'finalization'
            }, room=upload_id)
        except Exception:
            pass
            
        # Return processed results
        return jsonify({
            "success": True,
            "data": {
                "info_df": all_info_records,
                "dataframe": all_results
            }
        }), 200

    except Exception as e:
        logger.error(f"Error in complete_adjustment: {str(e)}\n{traceback.format_exc()}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 400

# Pomoćne funkcije za obradu podataka
def prepare_data(data, filename):
    """Priprema podataka za obradu"""
    # Kopiranje DataFrame-a
    df = data.copy()
    
    if len(df) == 0:
        raise ValueError(f"No data found for file {filename}")
    
    # Konverzija UTC kolone u datetime
    df['UTC'] = pd.to_datetime(df['UTC'])
    
    # Identifikacija kolona merenja
    measurement_cols = [col for col in df.columns if col != 'UTC']
    if not measurement_cols:
        raise ValueError(f"No measurement columns found for file {filename}")
    
    # Konverzija vrednosti merenja u float, ali sačuvaj originalne vrednosti
    for col in measurement_cols:
        # Sačuvaj originalne vrednosti pre konverzije
        df[f"{col}_original"] = df[col].copy()
        # Konvertuj u numeričke vrednosti, NaN za ne-numeričke
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df, measurement_cols

def filter_by_time_range(df, start_time, end_time):
    """Filtriranje podataka po vremenskom rasponu"""
    if start_time and end_time:
        start_time = pd.to_datetime(start_time)
        end_time = pd.to_datetime(end_time)
        return df[(df['UTC'] >= start_time) & (df['UTC'] <= end_time)]
    return df

def get_method_for_file(methods, filename):
    """Dobijanje metode obrade za fajl"""
    method_info = methods.get(filename, {})
    if isinstance(method_info, dict):
        return method_info.get('method', '').strip()
    return None

def apply_processing_method(df, col, method, time_step, offset, start_time, end_time, intrpl_max):
    """
    Refaktorisana verzija funkcije za primenu metoda obrade koja daje identične rezultate kao data_adapt4.py
    """
    import math
    import statistics
    import datetime
    import numpy as np
    import pandas as pd

    df['UTC'] = pd.to_datetime(df['UTC'])
    df = df.sort_values('UTC').reset_index(drop=True)

    t_strt = pd.to_datetime(start_time) if start_time else df['UTC'].min()
    t_end = pd.to_datetime(end_time) if end_time else df['UTC'].max()

    tss = float(time_step)
    ofst = float(offset)
    t_ref = t_strt.replace(minute=0, second=0, microsecond=0) + datetime.timedelta(minutes=ofst)
    while t_ref < t_strt:
        t_ref += datetime.timedelta(minutes=tss)

    t_list = []
    while t_ref <= t_end:
        t_list.append(t_ref)
        t_ref += datetime.timedelta(minutes=tss)

    value_list = []
    i2 = 0
    direct = 1

    for t_curr in t_list:
        if method == 'mean':
            t_min = t_curr - datetime.timedelta(minutes=tss / 2)
            t_max = t_curr + datetime.timedelta(minutes=tss / 2)
            values = []
            while i2 < len(df) and df['UTC'][i2] < t_min:
                i2 += 1
            idx_start = i2
            while i2 < len(df) and df['UTC'][i2] <= t_max:
                val = df.iloc[i2][col]
                if pd.notna(val):
                    values.append(val)
                i2 += 1
            value_list.append(statistics.mean(values) if values else float('nan'))

        elif method in ['nearest', 'nearest (mean)']:
            t_min = t_curr - datetime.timedelta(minutes=tss / 2)
            t_max = t_curr + datetime.timedelta(minutes=tss / 2)
            timestamps, values, deltas = [], [], []
            while i2 < len(df) and df['UTC'][i2] < t_min:
                i2 += 1
            idx_start = i2
            while i2 < len(df) and df['UTC'][i2] <= t_max:
                val = df.iloc[i2][col]
                if pd.notna(val):
                    ts = df.iloc[i2]['UTC']
                    delta = abs((t_curr - ts).total_seconds())
                    timestamps.append(ts)
                    values.append(val)
                    deltas.append(delta)
                i2 += 1
            if values:
                deltas_np = np.array(deltas)
                min_delta = deltas_np.min()
                idx_all = np.where(deltas_np == min_delta)[0]
                if method == 'nearest':
                    value_list.append(values[idx_all[0]])
                else:
                    grouped = [values[idx] for idx in idx_all]
                    value_list.append(statistics.mean(grouped))
            else:
                value_list.append(float('nan'))

        elif method in ['intrpl', 'nearest (max. delta)']:
            if direct == 1:
                while i2 < len(df):
                    if df['UTC'][i2] >= t_curr:
                        if pd.notna(df.iloc[i2][col]):
                            time_next = df.iloc[i2]['UTC']
                            value_next = df.iloc[i2][col]
                            break
                    i2 += 1
                else:
                    value_list.append(float('nan'))
                    i2 = 0
                    direct = 1
                    continue
                direct = -1
            if direct == -1:
                j = i2
                while j >= 0:
                    if df['UTC'][j] <= t_curr:
                        if pd.notna(df.iloc[j][col]):
                            time_prior = df.iloc[j]['UTC']
                            value_prior = df.iloc[j][col]
                            break
                    j -= 1
                else:
                    value_list.append(float('nan'))
                    i2 = 0
                    direct = 1
                    continue
                delta_t = (time_next - time_prior).total_seconds()
                if delta_t == 0 or (value_prior == value_next and delta_t <= intrpl_max * 60):
                    value_list.append(value_prior)
                elif method == 'intrpl':
                    if intrpl_max is not None and delta_t > intrpl_max * 60:
                        value_list.append(float('nan'))
                    else:
                        delta_val = value_prior - value_next
                        delta_prior = (t_curr - time_prior).total_seconds()
                        value_list.append(value_prior - (delta_val / delta_t) * delta_prior)
                elif method == 'nearest (max. delta)':
                    if intrpl_max is not None and delta_t > intrpl_max * 60:
                        value_list.append(float('nan'))
                    else:
                        d_prior = (t_curr - time_prior).total_seconds()
                        d_next = (time_next - t_curr).total_seconds()
                        value_list.append(value_prior if d_prior < d_next else value_next)
                direct = 1

        else:
            match = df[df['UTC'] == t_curr]
            if not match.empty:
                val = match.iloc[0][col]
                value_list.append(val if pd.notna(val) else float('nan'))
            else:
                value_list.append(float('nan'))

    result_df = pd.DataFrame({'UTC': t_list, col: value_list})
    return result_df
# Kreiranje info zapisa za rezultate
def create_info_record(df, col, filename, time_step, offset):
    """Kreiranje info zapisa za rezultate"""
    total_points = len(df)
    numeric_points = df[col].count()
    numeric_ratio = (numeric_points / total_points * 100) if total_points > 0 else 0
    
    def format_utc(val):
        if pd.isnull(val):
            return None
        if hasattr(val, 'strftime'):
            return val.strftime(UTC_fmt)
        try:
            # Ako je string, pokušaj parsirati
            dt = pd.to_datetime(val)
            return dt.strftime(UTC_fmt)
        except Exception:
            return str(val)

    return {
        'Name der Datei': filename,
        'Name der Messreihe': col,
        'Startzeit (UTC)': format_utc(df['UTC'].iloc[0]) if len(df) > 0 else None,
        'Endzeit (UTC)': format_utc(df['UTC'].iloc[-1]) if len(df) > 0 else None,
        'Zeitschrittweite [min]': time_step,
        'Offset [min]': offset,
        'Anzahl der Datenpunkte': int(total_points),
        'Anzahl der numerischen Datenpunkte': int(numeric_points),
        'Anteil an numerischen Datenpunkten': float(numeric_ratio)
    }
# Kreiranje zapisa za rezultate
def create_records(df, col, filename):
    """Konverzija DataFrame-a u zapise"""
    records = []
    original_col = f"{col}_original"  # Kolona sa originalnim vrednostima
    
    for _, row in df.iterrows():
        utc_timestamp = int(pd.to_datetime(row['UTC']).timestamp() * 1000)  # Konverzija u milisekunde
        
        # Provjera tipa podatka
        if pd.notnull(row[col]):
            # Ako je numerička vrijednost, koristimo je
            value = float(row[col])
        else:
            # Ako je NaN, provjeravamo originalnu vrijednost
            if original_col in df.columns and pd.notnull(row[original_col]):
                # Koristimo originalnu ne-numeričku vrijednost
                value = str(row[original_col])
            else:
                # Ako nemamo originalnu vrijednost, koristimo "None"
                value = "None"
        
        record = {
            'UTC': utc_timestamp, 
            col: value,
            'filename': filename
        }
        records.append(record)
    
    return records

def convert_data_without_processing(df, filename, time_step, offset):
    """
    Direktna konverzija podataka bez obrade kada su parametri isti.
    Ova funkcija preskače kompletan proces obrade i samo konvertuje podatke u format
    koji frontend očekuje, što značajno ubrzava proces kada nema potrebe za transformacijom.
    """
    try:
        logger.info(f"Direct conversion without processing for {filename}")
        
        # Kopiranje DataFrame-a
        df = df.copy()
        
        # Konverzija UTC kolone u datetime ako već nije
        df['UTC'] = pd.to_datetime(df['UTC'])
        
        # Identifikacija kolona merenja
        measurement_cols = [col for col in df.columns if col != 'UTC']
        
        # Ako nemamo kolone merenja, vratimo prazne rezultate
        if not measurement_cols:
            logger.warning(f"No measurement columns found for {filename}")
            return [], None
        
        all_records = []
        
        # Obrada za svaku kolonu merenja
        for col in measurement_cols:
            # Direktno kreiranje zapisa bez transformacije podataka
            records = create_records(df, col, filename)
            all_records.extend(records)
            
            # Kreiranje info zapisa za prvu kolonu
            if len(all_records) > 0 and not any(r.get('info_created') for r in all_records):
                info_record = create_info_record(df, col, filename, time_step, offset)
                return all_records, info_record
        
        # Ako nemamo zapise, vratimo prazne rezultate
        if not all_records:
            return [], None
            
        # Ako imamo više kolona, vratimo samo prvi info_record
        info_record = create_info_record(df, measurement_cols[0], filename, time_step, offset)
        return all_records, info_record
        
    except Exception as e:
        logger.error(f"Error in convert_data_without_processing: {str(e)}")
        traceback.print_exc()
        return [], None

# Glavna funkcija za obradu podataka sa detaljnim logovanjem
def process_data_detailed(data, filename, start_time=None, end_time=None, time_step=None, offset=None, methods={}, intrpl_max=None):
    try:
        # 1. Priprema podataka
        df, measurement_cols = prepare_data(data, filename)
        
        # 2. Filtriranje po vremenskom rasponu
        df = filter_by_time_range(df, start_time, end_time)
        
        # 3. Dobijanje metode obrade za ovaj fajl
        method = get_method_for_file(methods, filename)
        
        # Ako nemamo metodu, to je greška jer ako fajl treba obradu mora imati metodu
        if not method:
            logger.warning(f"No processing method specified for {filename} but processing is required")
            # Vraćamo prazne rezultate umesto da koristimo praznu metodu
            return [], None
        
        # 4. Obrada podataka za svaku kolonu merenja
        all_info_records = []
        
        # Ako imamo samo jednu kolonu merenja, obrađujemo je direktno
        if len(measurement_cols) == 1:
            measurement_col = measurement_cols[0]
            
            # Primena metode obrade
            processed_df = apply_processing_method(
                df, measurement_col, method, time_step, offset, start_time, end_time, intrpl_max
            )
            
            # Kreiranje zapisa i statistika
            records = create_records(processed_df, measurement_col, filename)
            info_record = create_info_record(processed_df, measurement_col, filename, time_step, offset)
            
            return records, info_record
        
        # Ako imamo više kolona merenja, obrađujemo svaku pojedinačno
        combined_records = []
        
        for col in measurement_cols:
            # Primena metode obrade
            processed_df = apply_processing_method(
                df, col, method, time_step, offset, start_time, end_time, intrpl_max
            )
            
            # Kreiranje zapisa i statistika
            records = create_records(processed_df, col, filename)
            info_record = create_info_record(processed_df, col, filename, time_step, offset)
            
            combined_records.extend(records)
            all_info_records.append(info_record)
        
        # Ako imamo više kolona, vraćamo samo prvi info_record za kompatibilnost sa postojećim kodom
        return combined_records, all_info_records[0] if all_info_records else None
        
    except Exception as e:
        logger.error(f"Error in process_data_detailed: {str(e)}")
        traceback.print_exc()
        raise

# Route to prepare data for saving
@bp.route('/prepare-save', methods=['POST'])
def prepare_save():
    try:
        # Try to get data from JSON
        try:
            data = request.get_json(force=True)
        except:
            # If JSON fails, try form data
            data = request.form.to_dict()
            
        if not data:
            return jsonify({"error": "No data received"}), 400
            
        # Handle both direct data and nested data format
        save_data = data.get('data', data)
        if not save_data:
            return jsonify({"error": "Empty data"}), 400

        # Convert save_data to list if it's a string (from form data)
        if isinstance(save_data, str):
            try:
                save_data = json.loads(save_data)
            except json.JSONDecodeError:
                return jsonify({"error": "Invalid data format"}), 400

        # Kreiraj privremeni fajl i zapiši CSV podatke
        temp_file = tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.csv')
        writer = csv.writer(temp_file, delimiter=';')
        for row in save_data:
            writer.writerow(row)
        temp_file.close()

        # Generiši jedinstveni ID na osnovu trenutnog vremena
        file_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        temp_files[file_id] = {
            'path': temp_file.name,
            'timestamp': time.time()
        }

        return jsonify({"message": "File prepared for download", "fileId": file_id}), 200
    except Exception as e:
        logger.error(f"Error in prepare_save: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

# The complete_upload functionality has been moved to cloud.py
# to avoid circular imports and infinite recursion
# Route to download file
@bp.route('/download/<file_id>', methods=['GET'])
def download_file(file_id):
    try:
        if file_id not in temp_files:
            return jsonify({"error": "File not found"}), 404
        file_info = temp_files[file_id]
        file_path = file_info['path']
        if not os.path.exists(file_path):
            return jsonify({"error": "File not found"}), 404

        download_name = f"data_{file_id}.csv"
        return send_file(
            file_path,
            as_attachment=True,
            download_name=download_name,
            mimetype='text/csv'
        )
    except Exception as e:
        logger.error(f"Error in download_file: {str(e)}")
        return jsonify({"error": str(e)}), 500
