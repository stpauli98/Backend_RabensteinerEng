import pandas as pd
import numpy as np
from datetime import datetime
import tempfile
import os
import csv
import logging
import traceback
import time
from io import StringIO, BytesIO
from flask import request, jsonify, send_file, Blueprint
from werkzeug.datastructures import FileStorage, ImmutableMultiDict
import json
from flask_cors import CORS
from flask import Flask

# Create Blueprint
bp = Blueprint('adjustmentsOfData_bp', __name__)

# API prefix
API_PREFIX_ADJUSTMENTS_OF_DATA = '/api/adjustmentsOfData'

# Configure temp upload folder
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'temp_uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global dictionaries to store data
adjustment_chunks = {}  # Store chunks during adjustment
temp_files = {}  # Store temporary files for download

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# UTC format
UTC_fmt = "%Y-%m-%d %H:%M:%S"

# Constants
UPLOAD_EXPIRY_TIME = 30 * 60  # 30 minuta u sekundama

# Global variables to store data
stored_data = {}
# Dictionary to store DataFrames
info_df = pd.DataFrame(columns=['Name der Datei', 'Name der Messreihe', 'Startzeit (UTC)', 'Endzeit (UTC)',
                                'Zeitschrittweite [min]', 'Offset [min]', 'Anzahl der Datenpunkte',
                                'Anzahl der numerischen Datenpunkte', 'Anteil an numerischen Datenpunkten'])  # DataFrame for file info

# Function to check if file is a CSV
def cleanup_old_files():
    """Clean up files older than UPLOAD_EXPIRY_TIME"""
    current_time = time.time()
    for file_id, file_info in list(temp_files.items()):
        if current_time - file_info.get('timestamp', 0) > UPLOAD_EXPIRY_TIME:
            try:
                os.remove(file_info['path'])
                del temp_files[file_id]
            except (OSError, KeyError) as e:
                logger.error(f"Error cleaning up file {file_id}: {str(e)}")

def allowed_file(filename):
    """Check if file has .csv extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() == 'csv'

# Function to detect delimiter
def detect_delimiter(file_content):
    """
    Detect the delimiter used in a CSV file content
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
    try:
        
        # Get upload metadata from form
        upload_id = request.form.get('uploadId')
        chunk_index = int(request.form.get('chunkIndex'))
        total_chunks = int(request.form.get('totalChunks'))
        
        # Validacija parametara
        if not all([upload_id, isinstance(chunk_index, int), isinstance(total_chunks, int)]):
            return jsonify({"error": "Missing or invalid required parameters"}), 400
            
        
        # Get the file chunk
        if 'files[]' not in request.files:
            logger.error(f"No file part. Available files: {list(request.files.keys())}")
            return jsonify({'error': 'No file part'}), 400
            
        file = request.files['files[]']
        if not file:
            return jsonify({'error': 'No selected file'}), 400
            
        # Read file content
        file_content = file.read()
        if not file_content:
            return jsonify({'error': 'Empty file content'}), 400
            
        # Create upload directory if it doesn't exist
        upload_dir = os.path.join(UPLOAD_FOLDER, upload_id)
        os.makedirs(upload_dir, exist_ok=True)
        
        # Save the chunk
        chunk_path = os.path.join(upload_dir, f'chunk_{chunk_index}')
        with open(chunk_path, 'wb') as f:
            f.write(file_content)
        
        # Check if all chunks have been received
        received_chunks = [f for f in os.listdir(upload_dir) if f.startswith('chunk_')]
        
        # If this was the last chunk and we have all chunks, combine them
        if len(received_chunks) == total_chunks:
            # Combine all chunks into final file
            final_path = os.path.join(upload_dir, 'complete_file.csv')
            with open(final_path, 'wb') as outfile:
                for i in range(total_chunks):
                    chunk_path = os.path.join(upload_dir, f'chunk_{i}')
                    try:
                        with open(chunk_path, 'rb') as infile:
                            outfile.write(infile.read())
                        # Delete chunk after combining
                        os.remove(chunk_path)
                    except Exception as e:
                        logger.error(f"Error processing chunk {i}: {str(e)}")
                        return jsonify({"error": f"Error processing chunk {i}"}), 500
            
            return jsonify({
                'status': 'complete',
                'message': 'File upload complete',
                'path': final_path
            })
        
        return jsonify({
            'status': 'chunk_received',
            'message': f'Received chunk {chunk_index + 1} of {total_chunks}',
            'chunksReceived': len(received_chunks)
        })
        
    except Exception as e:
        logger.error(f"Error in upload_chunk: {str(e)}\nTraceback: {traceback.format_exc()}")
        return jsonify({"error": str(e)}), 400

# First Step
@bp.route(f'{API_PREFIX_ADJUSTMENTS_OF_DATA}/upload-chunk', methods=['POST'])
def upload_chunk():
    # Clean up old files before new upload
    cleanup_old_files()
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
        tss = int(request.form.get('tss', 0))
        offset = int(request.form.get('offset', 0))
        mode_input = request.form.get('mode', '')
        intrpl_max = float(request.form.get('intrplMax', 60))
        
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
                # Create a temporary request context
                with app.test_request_context():
                    # Create a FileStorage object with the complete file
                    with open(final_path, 'rb') as f:
                        file_data = f.read()
                    file_obj = FileStorage(
                        stream=BytesIO(file_data),
                        filename=filename,
                        content_type='text/csv'
                    )
                    
                    # Set up the request context with our file and parameters
                    request.files = ImmutableMultiDict([('files[]', file_obj)])
                    request.form = ImmutableMultiDict([
                        ('tss', str(tss)),
                        ('offset', str(offset)),
                        ('mode', mode_input),
                        ('intrplMax', str(intrpl_max))
                    ])
                    
                    # Process the file using analyse_data
                    return analyse_data()
            except Exception as e:
                return jsonify({"error": str(e)}), 500
            
        return jsonify({
            'status': 'chunk_received',
            'message': f'Received chunk {chunk_index + 1} of {total_chunks}',
            'chunksReceived': len(received_chunks)
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# First Step Analyse
@bp.route(f'{API_PREFIX_ADJUSTMENTS_OF_DATA}/analyse-data', methods=['POST'])
def analyse_data():
    try:
        global stored_data, info_df
        
        # Get the file from request
        if not request.files:
            return jsonify({"error": "No files provided"}), 400
            
        files = request.files.getlist('files[]') if 'files[]' in request.files else [request.files['file']]
        
        all_file_info = []
        processed_data = []
        
        for file in files:
            if file and allowed_file(file.filename):
                try:
                    # Read file content
                    try:
                        file_content = file.read().decode('utf-8')
                        file.seek(0)  # Reset file pointer
                    except UnicodeDecodeError as e:
                        return jsonify({"error": f"Could not decode file {file.filename}. Make sure it's a valid UTF-8 encoded CSV file."}), 400
                    
                    # Detect delimiter from content
                    delimiter = detect_delimiter(file_content)
                    
                    # Read CSV with detected delimiter
                    df = pd.read_csv(StringIO(file_content), delimiter=delimiter)
                    
                    # Find time column
                    time_col = get_time_column(df)
                    if time_col is None:
                        raise ValueError(f"No time column found in file {file.filename}. Expected one of: UTC, Timestamp, Time, DateTime, Date, Zeit")
                    
                    # If time column is not 'UTC', rename it
                    if time_col != 'UTC':
                        df = df.rename(columns={time_col: 'UTC'})
                    
                    # Convert UTC column to datetime
                    df['UTC'] = pd.to_datetime(df['UTC'])
                    
                    # Store the DataFrame for later use
                    stored_data[file.filename] = df
                    
                    # Calculate time step
                    time_step = None
                    try:
                        # Calculate time differences and convert to minutes
                        time_diffs = pd.to_datetime(df['UTC']).diff().dropna()
                        time_diffs_minutes = time_diffs.dt.total_seconds() / 60
                        
                        # Get the most common time difference (mode)
                        time_step = int(time_diffs_minutes.mode()[0])
                    except Exception as e:
                        print(f"Error calculating time step: {str(e)}")
                    
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
                            'Name der Datei': str(file.filename),
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
                    for record in df.to_dict('records'):
                        converted_record = {}
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
                    
                except Exception as e:
                    return jsonify({"error": f"Error processing file {file.filename}: {str(e)}"}), 400
            else:
                return jsonify({"error": f"Invalid file format for {file.filename}"}), 400
        
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
        
        response_data = {
            'success': True,
            'data': {
                'info_df': all_file_info,
                'dataframe': processed_data[0] if processed_data else []
            }
        }
        return jsonify(response_data)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Second Step
@bp.route(f'{API_PREFIX_ADJUSTMENTS_OF_DATA}/adjust-data-chunk', methods=['POST'])
def adjust_data():
    try:
        global stored_data, info_df, adjustment_chunks
        
        data = request.json
        if not data:
            return jsonify({"error": "No JSON data received"}), 400

        # Get chunk parameters
        upload_id = data.get('uploadId')
        chunk_info = data.get('chunkInfo')
        chunk_data = data.get('dataChunk')
        
        # Get additional parameters that we'll need in complete_adjustment
        start_time = data.get('startTime')
        end_time = data.get('endTime')
        time_step_size = data.get('timeStepSize')
        offset = data.get('offset', 0)
        files = data.get('files', [])
        
        # Get methods from request or existing params
        methods = data.get('methods')
        if upload_id in adjustment_chunks and not methods:
            # If methods not in request, use existing methods
            methods = adjustment_chunks[upload_id]['params'].get('methods', {})
        else:
            # If no existing methods, use empty dict
            methods = methods or {}

        # Validate parameters
        if (upload_id is None or chunk_info is None or chunk_data is None):
            return jsonify({"error": "Missing required parameters"}), 400

        # Get chunk info
        total_chunks = chunk_info.get('totalChunks')
        current_chunk = chunk_info.get('currentChunk')

        # Initialize chunk storage if not exists
        if (upload_id not in adjustment_chunks):
            adjustment_chunks[upload_id] = {
                'chunks': {},
                'params': {
                    'startTime': start_time,
                    'endTime': end_time,
                    'files': files,  # Store files list
                    'timeStepSize': time_step_size,
                    'offset': offset,
                    'methods': methods,
                    'files': files
                }
            }

        # Store chunk
        # Add filename to each record in chunk_data
        for record in chunk_data:
            # Find which file this chunk belongs to based on the columns
            for file in files:
                if any(col in record for col in stored_data[file].columns):
                    record['filename'] = file
                    break
        
        # Store chunk data
        adjustment_chunks[upload_id]['chunks'][current_chunk] = chunk_data


        return jsonify({
            "message": f"Chunk {current_chunk} received",
            "remainingChunks": total_chunks - len(adjustment_chunks[upload_id]['chunks'])
        }), 200

    except Exception as e:
        logger.error(f"Error in receive_adjustment_chunk: {str(e)}\n{traceback.format_exc()}")
        return jsonify({"error": str(e)}), 400

# Route to complete adjustment
@bp.route(f'{API_PREFIX_ADJUSTMENTS_OF_DATA}/adjustdata/complete', methods=['POST'])
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
        global stored_data, info_df, adjustment_chunks
        
        data = request.json
        if not data:
            return jsonify({"error": "No JSON data received"}), 400

        # Obavezni parametri
        upload_id = data.get('uploadId')
        total_chunks = data.get('totalChunks')
        
        if not upload_id or total_chunks is None:
            return jsonify({"error": "Missing uploadId or totalChunks"}), 400
            
        # Get stored parameters
        if upload_id not in adjustment_chunks:
            return jsonify({"error": "Upload ID not found"}), 
        
        # Update methods if provided in the request
        if 'methods' in data and data['methods']:
            adjustment_chunks[upload_id]['params']['methods'] = data['methods']
        
        params = adjustment_chunks[upload_id]['params']
        
        files = params['files']
        requested_time_step = params['timeStepSize']
        requested_offset = params['offset']
        methods = params['methods']
        start_time = params['startTime']
        end_time = params['endTime']
        
        print("\n=== Parameters after combining chunks ===")
        print(f"Files: {files}")
        print(f"Start Time: {start_time}")
        print(f"End Time: {end_time}")
        print(f"Requested Time Step: {requested_time_step}")
        print(f"Requested Offset: {requested_offset}")
        print(f"Selected Methods: {methods}")
        
        # Get chunks data
        chunks = adjustment_chunks.get(upload_id, {}).get('chunks', {})
        if not chunks:
            # If no chunks in memory, try to read from filesystem
            upload_dir = os.path.join(UPLOAD_FOLDER, upload_id)
            if not os.path.exists(upload_dir):
                return jsonify({"error": f"No data found for upload ID: {upload_id}"}), 404
                
            # Load chunks from filesystem into memory
            chunks = {}
            for chunk_index in range(total_chunks):
                chunk_path = os.path.join(upload_dir, f'chunk_{chunk_index}')
                if os.path.exists(chunk_path):
                    with open(chunk_path, 'r', encoding='utf-8') as f:
                        chunks[chunk_index + 1] = f.read()
                        
            if not chunks:
                return jsonify({"error": "No chunks found"}), 404
                
            # Store chunks in memory for processing
            adjustment_chunks[upload_id] = {
                'chunks': chunks,
                'params': {
                    'startTime': data.get('startTime'),
                    'endTime': data.get('endTime'),
                    'timeStepSize': data.get('timeStepSize'),
                    'offset': data.get('offset', 0),
                    'methods': data.get('methods', {}),
                    'files': data.get('files', [])
                }
            }
            
        # Clean up methods by stripping whitespace
        if methods:
            methods = {k: v.strip() if isinstance(v, str) else v for k, v in methods.items()}
            
        # Define valid methods
        VALID_METHODS = {'mean', 'intrpl', 'nearest', 'nearest (mean)', 'nearest (max. delta)'}
        
        # Process each file separately
        for filename in files:
            # Get data for this file from all chunks
            file_data = []
            
            # Get all chunks from memory
            chunks_data = adjustment_chunks[upload_id]['chunks']
            
            # Convert chunks to DataFrame
            for chunk_index, chunk_content in sorted(chunks_data.items()):
                try:
                    # If chunk_content is a list (from adjust-data-chunk), convert to DataFrame directly
                    if isinstance(chunk_content, list):
                        chunk_data = pd.DataFrame(chunk_content)
                    else:
                        # If it's a string (from file upload), parse CSV
                        delimiter = detect_delimiter(chunk_content)
                        # Prvo pročitamo samo header da vidimo koje kolone stvarno postoje
                        header = pd.read_csv(StringIO(chunk_content), delimiter=delimiter, nrows=0)
                        # Učitamo samo kolone koje postoje u fajlu
                        chunk_data = pd.read_csv(StringIO(chunk_content), delimiter=delimiter, usecols=header.columns)
                        
                    if chunk_data.empty:
                        continue
                    file_data.append(chunk_data)
                except Exception as e:
                    logger.error(f"Error processing chunk {chunk_index}: {str(e)}")
                    return jsonify({"error": f"Error processing chunk {chunk_index}: {str(e)}"}), 500
                
            # Combine all chunks into a single DataFrame
            if not file_data:
                continue

            df = pd.concat(file_data, ignore_index=True)
            
            if 'UTC' not in df.columns:
                logger.error(f"No UTC column found in file {filename}")
                continue
                
            df['UTC'] = pd.to_datetime(df['UTC'])
            
            # Get time step and offset from info_df---------------------------
            file_info = info_df[info_df['Name der Datei'] == filename].iloc[0]
            file_time_step = file_info['Zeitschrittweite [min]']
            file_offset = file_info['Offset [min]']
            # Check if this file needs processing
            # Convert requested_offset to minutes from midnight if needed
            if requested_offset >= file_time_step:
                requested_offset = requested_offset % file_time_step
                
            # Check if file needs processing and has a valid method
            needs_processing = file_time_step != requested_time_step or file_offset != requested_offset
            method = methods.get(filename, '').strip()
            has_valid_method = method and method in VALID_METHODS
            
            if needs_processing and not has_valid_method:
                return jsonify({
                    "success": True,
                    "methodsRequired": True,
                    "message": f"Die Datei {filename} benötigt eine Verarbeitungsmethode (Zeitschrittweite: {file_time_step}->{requested_time_step}, Offset: {file_offset}->{requested_offset}).",
                    "data": {
                        "info_df": [{
                            "filename": filename,
                            "current_timestep": file_time_step,
                            "requested_timestep": requested_time_step,
                            "current_offset": file_offset,
                            "requested_offset": requested_offset,
                            "valid_methods": list(VALID_METHODS)
                        }],
                        "dataframe": []
                    }
                    }), 200
            else:
                print(f"\nFile {filename} does not need processing - parameters match")

        
        # Check if all files have methods assigned
        missing_methods = [f for f in files if f not in methods]
        if missing_methods:
            return jsonify({
                    "error": f"Missing methods for files: {', '.join(missing_methods)}"
                }), 400
            
        # Check if all chunks received
        if upload_id not in adjustment_chunks:
            return jsonify({"error": "Upload ID not found"}), 400
            
        chunks = adjustment_chunks[upload_id]['chunks']
        received_chunks = len(chunks)
        if received_chunks != total_chunks:
            return jsonify({"error": f"Missing chunks. Received {received_chunks} out of {total_chunks}"}), 400

        # Initialize data structures for each file
        data_by_file = {file: [] for file in files}
        
        # Combine chunks and separate by filename
        for i in range(1, total_chunks + 1):
            chunk = chunks.get(i)
            if chunk is None:
                return jsonify({"error": f"Missing chunk {i}"}), 400
            
            # Debug: Print first few records of each chunk
            print(f"\nProcessing chunk {i}:")
            for idx, record in enumerate(chunk[:5]):
                print(f"Record {idx}: {record}")
                
            # Distribute records to appropriate files
            for record in chunk:
                # Find which file this record belongs to by checking columns
                best_match = None
                max_matches = 0
                print("\nChecking record:", record)
                
                for filename, df in stored_data.items():
                    # Get non-UTC columns from the original DataFrame
                    data_columns = [col for col in df.columns if col != 'UTC']
                    print(f"Checking against {filename} with columns: {data_columns}")
                    
                    # Check how many columns match
                    matching_cols = [col for col in data_columns if col in record]
                    print(f"Found {len(matching_cols)} matching columns: {matching_cols}")
                    
                    # Update best match if this file has more matching columns
                    if len(matching_cols) > max_matches:
                        max_matches = len(matching_cols)
                        best_match = filename
                
                if best_match:
                    # Create a copy of the record to avoid modifying the original
                    record_copy = record.copy()
                    record_copy['filename'] = best_match
                    data_by_file[best_match].append(record_copy)
                else:
                    print(f"Warning: Could not determine file for record: {record}")

        # Use time step and offset from stored params
        time_step = params['timeStepSize']
        offset = params['offset']
        
        # Get methods from stored params
        methods = params['methods']

        # Debug: Print data distribution by file
        print("\nData distribution by file:")
        for f in files:
            print(f"File {f}: {len(data_by_file.get(f, []))} records")
            if f in data_by_file and data_by_file[f]:
                print("First record:", data_by_file[f][0])

        # Process data for each file
        all_results = []
        all_info_records = []
        
        for file in files:
            # Skip if no data for this file
            if not data_by_file[file]:
                logger.warning(f"No data found for file {file}")
                continue

            result_data, info_record = process_data_detailed(
                data_by_file[file],  # Send only data for this specific file
                file,
                start_time,
                end_time,
                time_step,
                offset,
                methods
            )
            print("result_data koji se salju na glavnu obradu: ", result_data[:10])
            all_results.extend(result_data)
            if info_record:
                all_info_records.append(info_record)

        # Nakon obrade, očistite spremljene chunkove
        del adjustment_chunks[upload_id]
        
        return jsonify({
            "success": True,
            "methodsRequired": False,
            "data": {
                "info_df": all_info_records,
                "dataframe": all_results
            }
        }), 200

    except Exception as e:
        logger.error(f"Error in complete_adjustment: {str(e)}\n{traceback.format_exc()}")
        return jsonify({"error": str(e)}), 400

# Function to process data with detailed logging
def process_data_detailed(data, filename, start_time=None, end_time=None, time_step=None, offset=None, methods={}):
    try:
        print(f"\n===Krece obrada | Processing data with parameters ===")
        print(f"DataFrame name: {filename}")
        
        # Debug: Print raw data structure
        print("Raw data first 5 records:")
        for i, record in enumerate(data[:5]):
            print(record)
            if i >= 4: break
        
        # Convert list of dictionaries to DataFrame
        df = pd.DataFrame(data)
        
        # Filter data to keep only records for this specific file
        df = df[df['filename'] == filename].copy()
        
        if len(df) == 0:
            raise ValueError(f"No data found for file {filename}")
            
        # Convert UTC column to datetime if it's not already
        df['UTC'] = pd.to_datetime(df['UTC'])
        
        # Identify the measurement column for this file from stored_data
        file_columns = stored_data[filename].columns
        measurement_cols = [col for col in file_columns if col != 'UTC']
        if not measurement_cols:
            raise ValueError(f"No measurement columns found for file {filename}")
        measurement_col = measurement_cols[0]
        
        # Keep only UTC and the relevant measurement column
        df = df[['UTC', measurement_col]]
        
        print(f"\nSelected measurement column for {filename}: {measurement_col}")
        print("Initial data shape:", df.shape)
        
        # Get the method for this file if available and strip whitespace
        method = methods.get(filename, '').strip() if methods else None
        
        # Apply the selected method if we have a time_step and a valid method
        if time_step and method:
            # Get measurement columns (non-UTC columns) pre obrade
            measurement_cols = [col for col in df.columns if col != 'UTC']
            if not measurement_cols:
                raise ValueError("No measurement columns found in DataFrame")
            measurement_col = measurement_cols[0]  # Take the first measurement column
            
            # Prvo postavimo UTC kao index za sve metode
            df.set_index('UTC', inplace=True)
            
            # Ako imamo duplikate u indexu, rešimo ih pre reindex-a
            if df.index.duplicated().any():
                if method in ['mean', 'nearest (mean)']:
                    # Za metode koje koriste mean, odmah grupišemo po UTC
                    # Isključujemo filename kolonu iz mean operacije
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    df = df[numeric_cols].groupby(level=0).mean()
                else:
                    # Za ostale metode, uzimamo prvi zapis za svaki timestamp
                    df = df.loc[~df.index.duplicated(keep='first')]
            
            # Sortiramo index da bude monoton rastuci
            df = df.sort_index()
            
            # Kreiramo novi vremenski index sa zadatim time step-om
            new_index = pd.date_range(start=df.index[0], end=df.index[-1], freq=f'{time_step}min')
            
            if method == 'mean':
                # Klasična metoda proseka
                df = df.resample(f'{time_step}min').mean()
                df = df.interpolate(method='linear')
                
            elif method == 'intrpl':
                # Linearna interpolacija
                df = df.reindex(new_index).interpolate(method='linear')
                
            elif method == 'nearest':
                # Najbliža vrednost
                df = df.reindex(new_index, method='nearest')
                
            elif method == 'nearest (mean)':
                # Prvo nađemo najbližu vrednost, pa onda prosek ako ima više vrednosti
                df = df.reindex(new_index, method='nearest')
                df = df.resample(f'{time_step}min').mean()
                
            elif method == 'nearest (max. delta)':
                # Najbliža vrednost sa maksimalnom dozvoljenom razlikom
                df = df.reindex(new_index, method='nearest', tolerance=pd.Timedelta(minutes=time_step))
                # Vrednosti koje su dalje od time_step će biti NaN
            
            # Reset index za sve metode i osiguraj da se kolona zove 'UTC'
            df.reset_index(inplace=True)
            if 'index' in df.columns:
                df.rename(columns={'index': 'UTC'}, inplace=True)
        
        # Get measurement columns (non-UTC columns)
        measurement_cols = [col for col in df.columns if col != 'UTC']
        if not measurement_cols:
            raise ValueError("No measurement columns found in DataFrame")
            
        measurement_col = measurement_cols[0]  # Take the first measurement column
        print(f"\nProcessing measurement column: {measurement_col}")
        print("Sample data:")
        print(df.head())
        
        # Convert measurement values to float, replacing non-numeric values with NaN
        df[measurement_col] = pd.to_numeric(df[measurement_col], errors='coerce')
        
        # Filter by time range if provided
        if start_time and end_time:
            start_time = pd.to_datetime(start_time)
            end_time = pd.to_datetime(end_time)
            df = df[(df['UTC'] >= start_time) & (df['UTC'] <= end_time)]
            print(f"\nFiltered by time range: {start_time} to {end_time}")
            print(f"Remaining points: {len(df)}")
        
        # Create regular time index
        if time_step:
            # Osiguraj da imamo UTC kolonu
            if 'index' in df.columns and 'UTC' not in df.columns:
                df.rename(columns={'index': 'UTC'}, inplace=True)
                
            # Create new time index
            if start_time is None:
                start_time = df['UTC'].min()
            if end_time is None:
                end_time = df['UTC'].max()
                
            # Apply offset if provided
            if offset:
                start_time = start_time + pd.Timedelta(minutes=offset)
                
            # Create new index with specified time step
            new_index = pd.date_range(start=start_time, end=end_time, freq=f'{time_step}min')
            
            # Resample data based on method
            if method == 'mean':
                # Set index and resample with mean
                df_indexed = df.set_index('UTC').sort_index()  # Sortiramo pre resample
                resampled = df_indexed.resample(f'{time_step}min', offset=f'{offset}min')[measurement_col].mean()
                result_df = pd.DataFrame({
                    'UTC': resampled.index,
                    measurement_col: resampled.values
                })
            elif method == 'max':
                resampled = df.set_index('UTC').sort_index().resample(f'{time_step}min', offset=f'{offset}min')[measurement_col].max()
                result_df = pd.DataFrame({
                    'UTC': resampled.index,
                    measurement_col: resampled.values
                })
            elif method == 'min':
                resampled = df.set_index('UTC').resample(f'{time_step}min', offset=f'{offset}min')[measurement_col].min()
                result_df = pd.DataFrame({
                    'UTC': resampled.index,
                    measurement_col: resampled.values
                })
            elif method == 'nearest':
                # First aggregate any duplicate timestamps by taking their mean
                df_indexed = df.groupby('UTC')[measurement_col].mean()
                # Then reindex with nearest method
                resampled = df_indexed.reindex(new_index, method='nearest')
                result_df = pd.DataFrame({
                    'UTC': new_index,
                    measurement_col: resampled.values
                })
            elif method == 'nearest (mean)':
                # First aggregate any duplicate timestamps by taking their mean
                df_indexed = df.groupby('UTC')[measurement_col].mean()
                # Then reindex with nearest method
                nearest_vals = df_indexed.reindex(new_index, method='nearest')
                rolling_mean = nearest_vals.rolling(window=2, min_periods=1).mean()
                result_df = pd.DataFrame({
                    'UTC': new_index,
                    measurement_col: rolling_mean.values
                })
            elif method == 'nearest (max. delta)':
                # First aggregate any duplicate timestamps by taking their mean
                df_indexed = df.groupby('UTC')[measurement_col].mean()
                # Then reindex with nearest method
                nearest_vals = df_indexed.reindex(new_index, method='nearest')
                # Calculate forward and backward deltas
                forward_deltas = nearest_vals.diff()
                backward_deltas = nearest_vals.diff(-1)
                # Get absolute deltas and take minimum of forward/backward
                deltas = pd.concat([forward_deltas.abs(), backward_deltas.abs()], axis=1).min(axis=1)
                max_delta = deltas.quantile(0.95)  # Use 95th percentile as threshold
                # Only mask values where both forward and backward deltas exceed threshold
                masked_values = nearest_vals.where(deltas <= max_delta)
                result_df = pd.DataFrame({
                    'UTC': new_index,
                    measurement_col: masked_values.values
                })
            else:
                result_df = df
        else:
            result_df = df
            
        # Calculate statistics
        total_points = len(result_df)
        numeric_points = result_df[measurement_col].count()
        numeric_ratio = (numeric_points / total_points * 100) if total_points > 0 else 0
       
        
        # Create info record
        original_name = filename
        info_record = {
            'Name der Datei': original_name,
            'Name der Messreihe': measurement_col,
            'Startzeit (UTC)': result_df['UTC'].iloc[0].strftime(UTC_fmt) if len(result_df) > 0 else None,
            'Endzeit (UTC)': result_df['UTC'].iloc[-1].strftime(UTC_fmt) if len(result_df) > 0 else None,
            'Zeitschrittweite [min]': time_step,  # Already a number from frontend
            'Offset [min]': offset,  # Already a number from frontend
            'Anzahl der Datenpunkte': int(total_points),
            'Anzahl der numerischen Datenpunkte': int(numeric_points),
            'Anteil an numerischen Datenpunkten': float(numeric_ratio)
        }
        
        # Convert to records
        records = []
        for _, row in result_df.iterrows():
            utc_timestamp = int(pd.to_datetime(row['UTC']).timestamp() * 1000)  # Convert to milliseconds
            value = float(row[measurement_col]) if pd.notnull(row[measurement_col]) else None
            record = {
                'UTC': utc_timestamp, 
                measurement_col: value,
                'filename': original_name  # Use original name for the record
            }
            records.append(record)
            
        return records, info_record
        
    except Exception as e:
        logger.error(f"Error in process_data_detailed: {str(e)}")
        traceback.print_exc()
        raise

# Route to prepare data for saving
@bp.route(f'{API_PREFIX_ADJUSTMENTS_OF_DATA}/prepare-save', methods=['POST'])
def prepare_save():
    # Clean up old files before saving new ones
    cleanup_old_files()
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

# Route to download file
@bp.route(f'{API_PREFIX_ADJUSTMENTS_OF_DATA}/download/<file_id>', methods=['GET'])
def download_file(file_id):
    # Clean up old files before download
    cleanup_old_files()
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
        if file_id in temp_files:
            try:
                os.unlink(temp_files[file_id])
                del temp_files[file_id]
            except Exception as ex:
                logger.error(f"Error cleaning up temp file: {ex}")
