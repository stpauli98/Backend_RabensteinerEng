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
import json

# Create Blueprint
bp = Blueprint('adjustmentsOfData_bp', __name__)

# API prefix
#API_PREFIX_ADJUSTMENTS_OF_DATA = '/api/adjustmentsOfData'

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
UPLOAD_EXPIRY_TIME = 5 * 60  # 10 minuta u sekundama


# Dictionary to store DataFrames
info_df = pd.DataFrame(columns=['Name der Datei', 'Name der Messreihe', 'Startzeit (UTC)', 'Endzeit (UTC)',
                                'Zeitschrittweite [min]', 'Offset [min]', 'Anzahl der Datenpunkte',
                                'Anzahl der numerischen Datenpunkte', 'Anteil an numerischen Datenpunkten'])  # DataFrame for file info

# Function to check if file is a CSV
def cleanup_old_files():
    """Clean up files older than 5 minutes from temp_uploads directory"""
    success = True
    errors = []
    deleted_count = 0
    current_time = time.time()
    EXPIRY_TIME = 5 * 60  # 5 minutes in seconds
    
    # Get temp_uploads directory path
    temp_dir = os.path.join(os.path.dirname(__file__), 'temp_uploads')
    
    try:
        # Prolazi kroz sve poddirektorijume
        for root, dirs, files in os.walk(temp_dir, topdown=False):
            for name in files:
                file_path = os.path.join(root, name)
                try:
                    # Proveri starost fajla
                    file_age = current_time - os.path.getmtime(file_path)
                    if file_age > EXPIRY_TIME:
                        os.remove(file_path)
                        deleted_count += 1
                except Exception as e:
                    success = False
                    errors.append(f"Error with {name}: {str(e)}")
                    logger.error(f"Error cleaning up file {name}: {str(e)}")
            
            # Pokušaj obrisati prazne direktorijume
            for dir_name in dirs:
                dir_path = os.path.join(root, dir_name)
                try:
                    os.rmdir(dir_path)  # Ovo će uspeti samo ako je direktorijum prazan
                except OSError:
                    pass
        
        # Očisti temp_files dictionary za obrisane fajlove
        for file_id, file_info in list(temp_files.items()):
            if not os.path.exists(file_info['path']):
                del temp_files[file_id]
    
        return jsonify({
            "success": success,
            "message": f"Cleaned up {deleted_count} files older than 5 minutes",
            "deleted_count": deleted_count,
            "errors": errors if errors else None
        }), 200 if success else 500
                
    except Exception as e:
        logger.error(f"Error in cleanup_old_files: {str(e)}")
        return jsonify({"error": str(e)}), 500

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
            
            # Process the complete file
            try:
                # Analyze the complete file
                result = analyse_data(final_path, upload_id)
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
        
        # Clean up methods by stripping whitespace
        if methods:
            methods = {k: v.strip() if isinstance(v, str) else v for k, v in methods.items()}
            
        # Define valid methods
        VALID_METHODS = {'mean', 'intrpl', 'nearest', 'nearest (mean)', 'nearest (max. delta)'}
        
        # Process each file separately
        for filename in filenames:
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
            
            # Ako file treba obradu, provjerimo ima li metodu
            if needs_processing:
                method = methods.get(filename, {})
                method = method.get('method', '').strip() if isinstance(method, dict) else ''
                has_valid_method = method and method in VALID_METHODS
                
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
            if info_record is not None:
                all_info_records.append(info_record)
                
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
    """Primena metode obrade na podatke"""
    non_numeric_mask = ~pd.to_numeric(df[col], errors='coerce').notna() & ~df[col].isna()
    if non_numeric_mask.any():
        non_numeric_values = df.loc[non_numeric_mask, col].unique()
    
    # Ako nemamo time_step, vraćamo originalne podatke
    if not time_step:
        return df
        
    # Ako je metoda prazna, generiramo sve vremenske korake ali zadržavamo originalne podatke
    if not method:
        
        # Određivanje vremenskog raspona
        if start_time is None:
            start_time = df_indexed.index.min()
        else:
            start_time = pd.to_datetime(start_time)
            
        if end_time is None:
            end_time = df_indexed.index.max()
        else:
            end_time = pd.to_datetime(end_time)
        
        # Primena offseta ako je obezbeđen
        if offset:
            start_time = start_time + pd.Timedelta(minutes=offset)
        
        # Kreiranje novog indeksa sa zadatim time step-om
        new_index = pd.date_range(start=start_time, end=end_time, freq=f'{time_step}min')
    
    # Izdvajamo UTC, relevantnu kolonu merenja i originalnu vrednost ako postoji
    original_col = f"{col}_original"
    columns_to_extract = ['UTC', col]
    if original_col in df.columns:
        columns_to_extract.append(original_col)
    
    df_single_col = df[columns_to_extract].copy()
    
    # Postavljamo UTC kao indeks
    df_indexed = df_single_col.set_index('UTC')
    
    # Rešavanje duplikata u indeksu
    if df_indexed.index.duplicated().any():
        if method in ['mean', 'nearest (mean)']:
            # Za metode koje koriste mean, grupišemo po UTC
            df_indexed = df_indexed.groupby(level=0).mean()
        else:
            # Za ostale metode, uzimamo prvi zapis za svaki timestamp
            df_indexed = df_indexed.loc[~df_indexed.index.duplicated(keep='first')]
    
    # Sortiranje indeksa
    df_indexed = df_indexed.sort_index()
    
    # Određivanje vremenskog raspona
    if start_time is None:
        start_time = df_indexed.index.min()
    else:
        start_time = pd.to_datetime(start_time)
        
    if end_time is None:
        end_time = df_indexed.index.max()
    else:
        end_time = pd.to_datetime(end_time)
    
    # Primena offseta ako je obezbeđen
    if offset:
        start_time = start_time + pd.Timedelta(minutes=offset)
    
    # Kreiranje novog indeksa sa zadatim time step-om
    new_index = pd.date_range(start=start_time, end=end_time, freq=f'{time_step}min')
    
    # Primena odgovarajuće metode
    if method == 'mean':
        # Resample sa offsetom
        resampled = df_indexed.resample(f'{time_step}min', offset=f'{offset}min')[col].mean()
        result_df = pd.DataFrame({
            'UTC': resampled.index,
            col: resampled.values
        })
        
    elif method == 'max':
        # Resample sa offsetom
        resampled = df_indexed.resample(f'{time_step}min', offset=f'{offset}min')[col].max()
        result_df = pd.DataFrame({
            'UTC': resampled.index,
            col: resampled.values
        })
        
    elif method == 'min':
        # Resample sa offsetom
        resampled = df_indexed.resample(f'{time_step}min', offset=f'{offset}min')[col].min()
        result_df = pd.DataFrame({
            'UTC': resampled.index,
            col: resampled.values
        })
        
    elif method == 'intrpl':
        # Parametri za interpolaciju
        reindex_params = {'method': None}  # Koristimo interpolate umesto method parametra
        
        # Reindex sa novim indeksom
        resampled = df_indexed.reindex(new_index, **reindex_params)
        
        # Interpolacija
        if intrpl_max is None:
            logger.warning(f"'intrpl' method requires intrpl_max parameter. Using unlimited interpolation")
            interpolated = resampled.interpolate(method='linear', limit_direction='both')
        else:
            limit_periods = int(intrpl_max / time_step)  # Konvertujemo minute u broj perioda
            interpolated = resampled.interpolate(
                method='linear',
                limit=limit_periods,
                limit_direction='both'
            )
            
        result_df = pd.DataFrame({
            'UTC': interpolated.index,
            col: interpolated[col].values
        })
        
    elif method == 'nearest':
        # Prvo agregiramo duplikate
        df_aggregated = df_indexed.groupby(level=0)[col].mean()
        # Zatim reindex sa nearest metodom
        resampled = df_aggregated.reindex(new_index, method='nearest')
        result_df = pd.DataFrame({
            'UTC': new_index,
            col: resampled.values
        })
        
    elif method == 'nearest (mean)':
        # Prvo agregiramo duplikate
        df_aggregated = df_indexed.groupby(level=0)[col].mean()
        # Zatim reindex sa nearest metodom
        nearest_vals = df_aggregated.reindex(new_index, method='nearest')
        rolling_mean = nearest_vals.rolling(window=2, min_periods=1).mean()
        result_df = pd.DataFrame({
            'UTC': new_index,
            col: rolling_mean.values
        })
        
    elif method == 'nearest (max. delta)':
        # Prvo agregiramo duplikate
        df_aggregated = df_indexed.groupby(level=0)[col].mean()
        
        # Parametri za reindex
        reindex_params = {'method': 'nearest'}
        
        # Dodajemo tolerance ako je intrpl_max definisan
        if intrpl_max is not None:
            reindex_params['tolerance'] = pd.Timedelta(minutes=intrpl_max)
        
        # Reindex sa nearest metodom i tolerancijom
        nearest_vals = df_aggregated.reindex(new_index, **reindex_params)
        
        # Jednostavnija implementacija koja koristi tolerance parametar
        # Vrednosti koje su dalje od tolerance će biti NaN
        result_df = pd.DataFrame({
            'UTC': new_index,
            col: nearest_vals.values
        })
        
    elif not method:
        # Ako je metoda prazna, generiramo sve vremenske korake s originalnim podacima
        
        # Kreiramo novi DataFrame s pravilnim vremenskim koracima
        result_df = pd.DataFrame(index=new_index)
        result_df['UTC'] = result_df.index
        
        # Reindeksiramo originalne podatke na novi indeks
        # Koristimo nearest metodu za reindeksiranje kako bismo sačuvali originalne vrijednosti
        # ali ne popunjavamo praznine između - one ostaju NaN
        if col in df_indexed.columns:
            # Koristimo reindex bez metode kako bismo dobili NaN za nedostajuće vrijednosti
            resampled = df_indexed[col].reindex(new_index)
            result_df[col] = resampled
        
        # Ako imamo originalnu kolonu, također je reindeksiramo
        original_col = f"{col}_original"
        if original_col in df_indexed.columns:
            original_resampled = df_indexed[original_col].reindex(new_index)
            result_df[original_col] = original_resampled
    else:
        # Ako metoda nije prepoznata, vraćamo originalne podatke
        result_df = df_single_col
    
    # Osiguravamo da imamo UTC kolonu
    if 'index' in result_df.columns and 'UTC' not in result_df.columns:
        result_df.rename(columns={'index': 'UTC'}, inplace=True)
    
    return result_df

# Kreiranje info zapisa za rezultate
def create_info_record(df, col, filename, time_step, offset):
    """Kreiranje info zapisa za rezultate"""
    total_points = len(df)
    numeric_points = df[col].count()
    numeric_ratio = (numeric_points / total_points * 100) if total_points > 0 else 0
    
    return {
        'Name der Datei': filename,
        'Name der Messreihe': col,
        'Startzeit (UTC)': df['UTC'].iloc[0].strftime(UTC_fmt) if len(df) > 0 else None,
        'Endzeit (UTC)': df['UTC'].iloc[-1].strftime(UTC_fmt) if len(df) > 0 else None,
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

# Glavna funkcija za obradu podataka sa detaljnim logovanjem
def process_data_detailed(data, filename, start_time=None, end_time=None, time_step=None, offset=None, methods={}, intrpl_max=None):
    try:
        # 1. Priprema podataka
        df, measurement_cols = prepare_data(data, filename)
        
        # 2. Filtriranje po vremenskom rasponu
        df = filter_by_time_range(df, start_time, end_time)
        
        # 3. Dobijanje metode obrade za ovaj fajl
        method = get_method_for_file(methods, filename)
        
        # Ako nemamo metodu, koristimo praznu metodu za generiranje svih vremenskih koraka
        if not method:
            method = ''  # Prazna metoda će generirati sve vremenske korake
        
        # 4. Obrada podataka za svaku kolonu merenja
        all_records = []
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
    finally:
        # Pokušaj očistiti privremeni fajl
        cleanup_old_files()
