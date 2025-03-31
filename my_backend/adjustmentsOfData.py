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
from flask import request, jsonify, send_file, Blueprint, current_app
from werkzeug.datastructures import FileStorage, ImmutableMultiDict
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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# UTC format
UTC_fmt = "%Y-%m-%d %H:%M:%S"

# Constants
UPLOAD_EXPIRY_TIME = 5 * 60  # 10 minuta u sekundama

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

# First Step
@bp.route('/upload-chunk', methods=['POST'])
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
        #logger.info(f"File content: {file_content[:100]}...")
        if not file_content:
            return jsonify({'error': 'Empty file content'}), 400
            
        # Create upload directory if it doesn't exist
        upload_dir = os.path.join(UPLOAD_FOLDER, upload_id)
        os.makedirs(upload_dir, exist_ok=True)
        #logger.info(f"Upload directory created in Folder {UPLOAD_FOLDER} with ID {upload_id} on path {upload_dir}")
        
        # Save the chunk
        chunk_path = os.path.join(upload_dir, f'chunk_{chunk_index}')
        logger.info(f"Chunk saved to {chunk_path}")
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
                logger.info(f"Starting analysis of file {final_path}")
                # Analyze the complete file
                result = analyse_data(final_path, upload_id)
                logger.info(f"Analysis complete for {final_path}")
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
        logger.info(f"Starting analyse_data for {file_path}")
        global stored_data, info_df
        
        # Clear stored data for new analysis
        stored_data.clear()
        
        all_file_info = []
        processed_data = []
        
        try:
            # Read file content
            logger.info(f"Reading file content from {file_path}")
            with open(file_path, 'r', encoding='utf-8') as file:
                file_content = file.read()
            logger.info(f"Successfully read {len(file_content)} bytes from {file_path}")
        except UnicodeDecodeError as e:
            logger.error(f"UnicodeDecodeError reading {file_path}: {str(e)}")
            raise ValueError(f"Could not decode file {file_path}. Make sure it's a valid UTF-8 encoded CSV file.")
        
        # Detect delimiter from content
        logger.info(f"Detecting delimiter for {file_path}")
        delimiter = detect_delimiter(file_content)
        logger.info(f"Detected delimiter '{delimiter}' for {file_path}")
        
        # Read CSV with detected delimiter
        logger.info(f"Reading CSV with delimiter '{delimiter}' for {file_path}")
        df = pd.read_csv(StringIO(file_content), delimiter=delimiter)
        logger.info(f"Successfully read CSV with {len(df)} rows and {len(df.columns)} columns")
        
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
        logger.info(f"Storing DataFrame for {filename}")
        stored_data[filename] = df
        
        # Store DataFrame in adjustment_chunks if upload_id provided
        if upload_id:
            logger.info(f"Storing DataFrame in adjustment_chunks for upload_id {upload_id}")
            if upload_id not in adjustment_chunks:
                adjustment_chunks[upload_id] = {'chunks': {}, 'params': {}, 'dataframes': {}}
            adjustment_chunks[upload_id]['dataframes'][filename] = df
            logger.info(f"Successfully stored DataFrame in adjustment_chunks for {filename}")
                    
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
        
        # Prepare response data
        dataframe_data = processed_data[0] if processed_data else []
        
        logger.info(f"Analysis complete for {file_path}:")
        logger.info(f"- Info records: {len(all_file_info)}")
        logger.info(f"- Data records: {len(dataframe_data)}")
        logger.info(f"- First record sample: {dataframe_data[0] if dataframe_data else 'No data'}")
        
        # Return just the data, not a response object
        return {
            'info_df': all_file_info,
            'dataframe': dataframe_data
        }
        
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Second Step
@bp.route('/adjust-data-chunk', methods=['POST'])
def adjust_data():
    try:
        global stored_data, info_df, adjustment_chunks
        
        data = request.json
        if not data:
            return jsonify({"error": "No JSON data received"}), 400

        # Get required parameters
        upload_id = data.get('uploadId')
        chunk_info = data.get('chunkInfo')
        chunk_data = data.get('dataChunk')
        
        # Get processing parameters
        start_time = data.get('startTime')
        end_time = data.get('endTime')
        time_step_size = data.get('timeStepSize')
        offset = data.get('offset', 0)
        intrpl_max = data.get('intrplMax', None)
        
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

        # Update or initialize chunk storage
        if upload_id not in adjustment_chunks:
            adjustment_chunks[upload_id] = {
                'chunks': {},
                'params': {
                    'startTime': start_time,
                    'endTime': end_time,
                    'timeStepSize': time_step_size,
                    'offset': offset,
                    'intrplMax': intrpl_max,
                    'methods': methods
                }
            }
        else:
            # Update parameters if they changed
            params = adjustment_chunks[upload_id]['params']
            if start_time is not None: params['startTime'] = start_time
            if end_time is not None: params['endTime'] = end_time
            if time_step_size is not None: params['timeStepSize'] = time_step_size
            if offset is not None: params['offset'] = offset
            if intrpl_max is not None: params['intrplMax'] = intrpl_max
            if methods: params['methods'].update(methods)

        # Add filename to each record based on 'Name der Datei'
        for record in chunk_data:
            if 'Name der Datei' in record:
                record['filename'] = record['Name der Datei']
            else:
                logger.warning(f"Record missing 'Name der Datei': {record}")
        
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
        
        requested_time_step = params['timeStepSize']
        requested_offset = params['offset']
        methods = params['methods']
        start_time = params['startTime']
        end_time = params['endTime']

        # Log the entire state for debugging
        logger.info(f"Complete adjustment called for upload_id: {upload_id}")
        logger.info(f"Current adjustment_chunks state: {adjustment_chunks}")
        
        # Get chunks data
        chunks = adjustment_chunks.get(upload_id, {}).get('chunks', {})
        
        if not chunks:
            return jsonify({"error": "No chunks found in memory for this upload ID"}), 404
            
        logger.info(f"Found {len(chunks)} chunks")
        
        # Extract and inspect chunks
        all_records = []
        for chunk_idx, chunk_data in chunks.items():
            logger.info(f"Chunk {chunk_idx} type: {type(chunk_data)}")
            logger.info(f"Chunk {chunk_idx} sample: {chunk_data[:2] if isinstance(chunk_data, list) else 'Not a list'}")
            if isinstance(chunk_data, list):
                all_records.extend(chunk_data)
        
        logger.info(f"Total records collected: {len(all_records)}")
        if all_records:
            logger.info(f"Sample record: {all_records[0]}")
            logger.info(f"Sample record keys: {all_records[0].keys() if isinstance(all_records[0], dict) else 'Not a dict'}")
        
        # Get filenames from records
        filenames = set()
        for record in all_records:
            if isinstance(record, dict) and 'Name der Datei' in record:
                filenames.add(record['Name der Datei'])
        
        logger.info(f"Found files in chunks: {filenames}")
        
        if not filenames:
            logger.error("No filenames found in records")
            return jsonify({"error": "No valid data found in chunks"}), 404
        
        # Log what we have
        logger.info(f"Found {len(stored_data)} files and {len(chunks)} chunks for upload_id {upload_id}")
        logger.info(f"Files: {list(stored_data.keys())}")
        logger.info(f"Processing parameters: {params}")
            
        # Clean up methods by stripping whitespace
        if methods:
            methods = {k: v.strip() if isinstance(v, str) else v for k, v in methods.items()}
            
        # Define valid methods
        VALID_METHODS = {'mean', 'intrpl', 'nearest', 'nearest (mean)', 'nearest (max. delta)'}
        
        # Process each file separately
        for filename in filenames:
            logger.info(f"Processing file: {filename}")
            # Get data for this file from all chunks
            file_data = [record for record in all_records if record.get('Name der Datei') == filename]
            logger.info(f"Found {len(file_data)} records for {filename}")
            
            # Convert to DataFrame
            if not file_data:  # Skip if no data for this file
                logger.warning(f"No data found for file {filename}")
                continue
                
            try:
                df = pd.DataFrame(file_data)
                logger.info(f"Created DataFrame for {filename} with columns: {df.columns.tolist()}")
            except Exception as e:
                logger.error(f"Error creating DataFrame for {filename}: {str(e)}")
                return jsonify({"error": f"Error processing file {filename}: {str(e)}"}), 500
                
            # Ensure we have the required columns
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
            method = methods.get(filename, {})
            method = method.get('method', '').strip() if isinstance(method, dict) else ''
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
        missing_methods = [f for f in filenames if f not in methods]
        if missing_methods:
            logger.warning(f"Missing methods for files: {missing_methods}")
            return jsonify({
                    "error": f"Missing methods for files: {', '.join(missing_methods)}"
                }), 400
            
        # Check if all chunks received
        chunks = adjustment_chunks[upload_id]['chunks']
        received_chunks = len(chunks)
        if received_chunks != total_chunks:
            logger.error(f"Missing chunks. Received {received_chunks} out of {total_chunks}")
            return jsonify({"error": f"Missing chunks. Received {received_chunks} out of {total_chunks}"}), 400

        # Initialize data structures for each file
        data_by_file = {filename: [] for filename in filenames}
        
        # Combine chunks and separate by filename
        for i in range(1, total_chunks + 1):
            chunk = chunks.get(i)
            if chunk is None:
                return jsonify({"error": f"Missing chunk {i}"}), 400
                
            # Distribute records to appropriate files
            for record in chunk:
                if 'Name der Datei' in record:
                    filename = record['Name der Datei']
                    if filename in data_by_file:
                        data_by_file[filename].append(record)
                    else:
                        logger.warning(f"Found record for unknown file {filename}: {record}")
                else:
                    logger.warning(f"Record missing 'Name der Datei': {record}")

        # Use time step and offset from stored params
        time_step = params['timeStepSize']
        offset = params['offset']
        
        # Get methods from stored params
        methods = params['methods']

        # Process data for each file
        all_results = []
        all_info_records = []
        
        for filename in filenames:
            # Skip if no data for this file
            if not data_by_file[filename]:
                logger.warning(f"No data found for file {filename}")
                continue

            result_data, info_record = process_data_detailed(
                data_by_file[filename],  # Send only data for this specific file
                filename,
                start_time,
                end_time,
                time_step,
                offset,
                methods,
                params.get('intrplMax')
            )
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
def process_data_detailed(data, filename, start_time=None, end_time=None, time_step=None, offset=None, methods={}, intrpl_max=None):
    try:
        print(f"\n===Krece obrada | Processing data with parameters ===")
        print(f"DataFrame name: {filename}")
        
        # Convert list of dictionaries to DataFrame
        df = pd.DataFrame(data)
        
        # Filter data to keep only records for this specific file
        df = df[df['filename'] == filename].copy()
        
        if len(df) == 0:
            raise ValueError(f"No data found for file {filename}")
            
        # Convert UTC column to datetime if it's not already
        df['UTC'] = pd.to_datetime(df['UTC'])
        
        # Identify the measurement column from data
        columns = df.columns
        measurement_cols = [col for col in columns if col not in ['UTC', 'filename', 'Name der Datei']]
        if not measurement_cols:
            raise ValueError(f"No measurement columns found for file {filename}")
        measurement_col = measurement_cols[0]
        logger.info(f"Using measurement column: {measurement_col} for file {filename}")
        
        # Keep only UTC and the relevant measurement column
        df = df[['UTC', measurement_col]]
        
        # Get the method for this file if available
        method_info = methods.get(filename, {})
        method = method_info.get('method', '').strip() if isinstance(method_info, dict) else None
        logger.info(f"Using method '{method}' for file {filename}")
        
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
                if intrpl_max is None:
                    logger.warning(f"'intrpl' method requires intrpl_max parameter. Using unlimited interpolation for {filename}")
                    df = df.reindex(new_index).interpolate(method='linear', limit_direction='both')
                else:
                    # Linearna interpolacija sa limitom baziranim na intrpl_max
                    limit_periods = int(intrpl_max / time_step)  # Konvertujemo minute u broj perioda
                    df = df.reindex(new_index).interpolate(
                        method='linear',
                        limit=limit_periods,  # Maksimalan broj uzastopnih NaN-ova za popunjavanje
                        limit_direction='both'  # Popunjavamo NaN-ove u oba smera
                    )
                
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
        
        # Convert measurement values to float, replacing non-numeric values with NaN
        df[measurement_col] = pd.to_numeric(df[measurement_col], errors='coerce')
        
        # Filter by time range if provided
        if start_time and end_time:
            start_time = pd.to_datetime(start_time)
            end_time = pd.to_datetime(end_time)
            df = df[(df['UTC'] >= start_time) & (df['UTC'] <= end_time)]
        
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
                
                # Then reindex with nearest method, using tolerance if available
                reindex_params = {
                    'method': 'nearest'
                }
                if intrpl_max is not None:
                    reindex_params['tolerance'] = pd.Timedelta(minutes=intrpl_max)
                    
                nearest_vals = df_indexed.reindex(new_index, **reindex_params)
                
                # Calculate forward and backward deltas
                forward_deltas = nearest_vals.diff()
                backward_deltas = nearest_vals.diff(-1)
                
                # Get absolute deltas and take minimum of forward/backward
                deltas = pd.concat([forward_deltas.abs(), backward_deltas.abs()], axis=1).min(axis=1)
                max_delta = deltas.quantile(0.95)  # Use 95th percentile as threshold
                
                # Only mask values where both forward and backward deltas exceed threshold
                # AND where the time difference is greater than intrpl_max
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
@bp.route('/prepare-save', methods=['POST'])
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
@bp.route('/download/<file_id>', methods=['GET'])
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
