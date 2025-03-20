import pandas as pd
import numpy as np
import math
import sys
from datetime import datetime
import statistics
import tempfile
import os
import csv
import logging
from flask import Flask, request, jsonify, send_file, Blueprint

# Create blueprint
bp = Blueprint('adjustments_of_data', __name__)
from io import StringIO
import traceback
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Dictionary to store temporary files
temp_files = {}

UTC_fmt = "%Y-%m-%d %H:%M:%S"

# Global dictionary to store DataFrames
stored_data = {}  # Dictionary to store DataFrames
info_df = pd.DataFrame()  # Initialize empty DataFrame for info

def allowed_file(filename):
    """Check if file has .csv extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() == 'csv'

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

@bp.route('/analyse-data', methods=['POST'])
def analyse_data():
    try:
        global stored_data, info_df
        logger.info("=== Starting file analysis ===")
        logger.info(f"Content-Type: {request.content_type}")
        logger.info(f"Files: {request.files}")
        logger.info(f"Form: {request.form}")
        
        # Get the file from request
        if not request.files:
            logger.error("No files in request")
            return jsonify({"error": "No files provided"}), 400
            
        files = request.files.getlist('files[]') if 'files[]' in request.files else [request.files['file']]
        logger.info(f"Processing {len(files)} files")
        
        all_file_info = []
        processed_data = []
        
        for file in files:
            if file and allowed_file(file.filename):
                try:
                    # Read file content
                    try:
                        file_content = file.read().decode('utf-8')
                        file.seek(0)  # Reset file pointer
                        logger.info(f"Successfully read file: {file.filename}")
                    except UnicodeDecodeError as e:
                        logger.error(f"Error decoding file {file.filename}: {str(e)}")
                        return jsonify({"error": f"Could not decode file {file.filename}. Make sure it's a valid UTF-8 encoded CSV file."}), 400
                    
                    # Detect delimiter from content
                    delimiter = detect_delimiter(file_content)
                    logger.info(f"Detected delimiter: {repr(delimiter)} for file {file.filename}")
                    logger.debug(f"First few lines:\n{file_content[:200]}")
                    
                    # Read CSV with detected delimiter
                    df = pd.read_csv(StringIO(file_content), delimiter=delimiter)
                    logger.info(f"Loaded file {file.filename} with columns: {df.columns.tolist()}")
                    
                    # Find time column
                    time_col = get_time_column(df)
                    if time_col is None:
                        raise ValueError(f"No time column found in file {file.filename}. Expected one of: UTC, Timestamp, Time, DateTime, Date, Zeit")
                    
                    # If time column is not 'UTC', rename it
                    if time_col != 'UTC':
                        df = df.rename(columns={time_col: 'UTC'})
                    
                    # Convert UTC column to datetime
                    df['UTC'] = pd.to_datetime(df['UTC'])
                    logger.info(f"Processed {len(df)} rows from {file.filename}")
                    
                    print(f"Sample data:\n{df.head()}")
                    
                    # Store the DataFrame for later use
                    stored_data[file.filename] = df
                    
                    # Calculate basic statistics
                    numeric_columns = df.select_dtypes(include=[np.number]).columns
                    numeric_count = int(df[numeric_columns].count().sum())
                    total_count = int(len(df) * len(numeric_columns))
                    numeric_percentage = float(numeric_count / total_count * 100) if total_count > 0 else 0.0
                    
                    # Calculate time step
                    time_step = None
                    try:
                        time_diffs = pd.to_datetime(df['UTC']).diff().dropna()
                        avg_time_diff = time_diffs.mean().total_seconds() / 60  # Convert to minutes
                        time_step = float(avg_time_diff)
                    except Exception as e:
                        print(f"Error calculating time step: {str(e)}")
                    
                    # Get the measurement column (second column or first non-time column)
                    measurement_col = None
                    for col in df.columns:
                        if col != 'UTC':
                            measurement_col = col
                            break

                    if measurement_col:
                        print(f"\nMeasurement column: {measurement_col}")
                        print(f"Measurement stats:\n{df[measurement_col].describe()}")

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
                        
                        print(f"\nFile info:")
                        print(json.dumps(file_info, indent=2))
                        
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
                    logger.error(f"Error processing file {file.filename}: {str(e)}")
                    logger.error(traceback.format_exc())
                    return jsonify({"error": f"Error processing file {file.filename}: {str(e)}"}), 400
            else:
                logger.error(f"Invalid file format for {file.filename}")
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
        logger.info(f"Sending response with {len(all_file_info)} files processed")
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Error in analyse_data: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 400

def process_data_detailed(df, filename, start_time=None, end_time=None, time_step=None, offset=None, method='mean'):
    try:
        print(f"\n=== Processing data with parameters ===")
        print(f"DataFrame name: {filename}")
        
        # Convert UTC column to datetime if it's not already
        df['UTC'] = pd.to_datetime(df['UTC'])
        
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
                resampled = df.set_index('UTC').resample(f'{time_step}min', offset=f'{offset}min')[measurement_col].mean()
                result_df = pd.DataFrame({
                    'UTC': resampled.index,
                    measurement_col: resampled.values
                })
            elif method == 'max':
                resampled = df.set_index('UTC').resample(f'{time_step}min', offset=f'{offset}min')[measurement_col].max()
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
                df_indexed = df.set_index('UTC')
                resampled = df_indexed[measurement_col].reindex(new_index, method='nearest')
                result_df = pd.DataFrame({
                    'UTC': new_index,
                    measurement_col: resampled.values
                })
            elif method == 'nearest (mean)':
                df_indexed = df.set_index('UTC')
                nearest_vals = df_indexed[measurement_col].reindex(new_index, method='nearest')
                rolling_mean = nearest_vals.rolling(window=2, min_periods=1).mean()
                result_df = pd.DataFrame({
                    'UTC': new_index,
                    measurement_col: rolling_mean.values
                })
            elif method == 'nearest (max. delta)':
                df_indexed = df.set_index('UTC')
                nearest_vals = df_indexed[measurement_col].reindex(new_index, method='nearest')
                deltas = nearest_vals.diff().abs()
                max_delta = deltas.quantile(0.95)  # Use 95th percentile as threshold
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
        
        print("\nStatistics:")
        print(f"Total points: {total_points}")
        print(f"Numeric points: {numeric_points}")
        print(f"Numeric ratio: {numeric_ratio:.1f}%")
        
        # Create info record
        original_name = filename
        info_record = {
            'Name der Datei': original_name,
            'Name der Messreihe': measurement_col,
            'Startzeit (UTC)': result_df['UTC'].iloc[0].strftime(UTC_fmt) if len(result_df) > 0 else None,
            'Endzeit (UTC)': result_df['UTC'].iloc[-1].strftime(UTC_fmt) if len(result_df) > 0 else None,
            'Zeitschrittweite [min]': float(time_step) if time_step else None,
            'Offset [min]': float(offset) if offset else 0.0,
            'Anzahl der Datenpunkte': int(total_points),
            'Anzahl der numerischen Datenpunkte': int(numeric_points),
            'Anteil an numerischen Datenpunkten': float(numeric_ratio)
        }
        
        print("\nInfo record:")
        print(json.dumps(info_record, indent=2))
        
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
            
        logger.info(f"Converted {len(records)} records")
        if records:
            logger.info(f"Sample record: {records[0]}")
            
        return records, info_record
        
    except Exception as e:
        logger.error(f"Error in process_data_detailed: {str(e)}")
        traceback.print_exc()
        raise

@bp.route('/adjust-data', methods=['POST'])
def adjust_data():
    try:
        global stored_data, info_df
        logger.info("=== Starting data adjustment ===")
        
        data = request.json
        if not data:
            return jsonify({"error": "No JSON data received"}), 400
            
        # Get parameters only once
        start_time = data.get('startTime')
        end_time = data.get('endTime')
        time_step = data.get('timeStepSize')
        offset = data.get('offset', 0)  # Default to 0
        methods = data.get('methods', {})  # Dictionary of filename -> method
        files = data.get('files', [])  # List of files to process
        
        print(f"\nReceived parameters:")
        print(f"Start time: {start_time}")
        print(f"End time: {end_time}")
        print(f"Time step: {time_step}")
        print(f"Offset: {offset}")
        print(f"Methods: {methods}")
        print(f"Files: {files}")
        
        if not stored_data or info_df.empty:
            return jsonify({"error": "No data available. Please upload a file first."}), 400

        # Create dictionary of files to process
        files_to_process = {}
        for filename in files:
            if filename in stored_data:
                df = stored_data[filename].copy()  # Make a copy to avoid modifying original
                df.name = filename  # Set name for the DataFrame
                files_to_process[filename] = df
            else:
                return jsonify({"error": f"File {filename} not found"}), 400

        all_processed_data = []
        all_processed_info = []
        files_needing_methods = []
        processed_files = set()  # Keep track of processed files

        # First pass: identify files needing method selection
        for filename, df in files_to_process.items():
            if filename in processed_files:
                continue

            try:
                file_info = info_df[info_df['Name der Datei'] == filename].iloc[0].to_dict()
                current_timestep = float(file_info.get('Zeitschrittweite [min]', 0))
                current_offset = float(file_info.get('Offset [min]', 0))
                
                needs_method = False
                if time_step is not None:
                    time_step_float = float(time_step)
                    offset_float = float(offset) if offset is not None else 0.0
                    needs_method = not (math.isclose(current_timestep, time_step_float, rel_tol=1e-9) and 
                                     math.isclose(current_offset, offset_float, rel_tol=1e-9))
                    
                    if needs_method:
                        # Check if we have a method for this file
                        file_method = methods.get(filename)
                        files_needing_methods.append({
                            'filename': filename,
                            'current_timestep': current_timestep,
                            'requested_timestep': time_step_float,
                            'current_offset': current_offset,
                            'requested_offset': offset_float,
                            'method': file_method
                        })
                        if not file_method:
                            continue

                # Process data if method is provided or not needed
                if not needs_method or (needs_method and methods.get(filename)):
                    actual_start = pd.to_datetime(start_time) if start_time else pd.to_datetime(file_info['Startzeit (UTC)'])
                    actual_end = pd.to_datetime(end_time) if end_time else pd.to_datetime(file_info['Endzeit (UTC)'])
                    
                    result, info_record = process_data_detailed(
                        df,
                        filename,
                        start_time=actual_start.strftime(UTC_fmt),
                        end_time=actual_end.strftime(UTC_fmt),
                        time_step=time_step_float if time_step is not None else None,
                        offset=offset_float if offset is not None else None,
                        method=methods.get(filename) if needs_method else 'mean'
                    )
                    
                    if result:
                        all_processed_data.extend(result)
                        all_processed_info.append(info_record)
                        processed_files.add(filename)
                
            except Exception as e:
                print(f"Error processing file {filename}: {str(e)}")
                traceback.print_exc()
                continue

        # Return appropriate response based on processing results
        if files_needing_methods and not all(f.get('method') for f in files_needing_methods):
            return jsonify({
                "success": True,
                "methodsRequired": True,
                "filesNeedingMethods": files_needing_methods,
                "data": None
            })
        elif all_processed_info:
            return jsonify({
                "success": True,
                "methodsRequired": False,
                "data": {
                    "info_df": all_processed_info,
                    "dataframe": all_processed_data
                }
            })
        else:
            return jsonify({"error": "No data was processed"}), 400

    except Exception as e:
        print(f"Error in adjust_data: {str(e)}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 400

@bp.route('/upload-chunk', methods=['POST'])
def upload_chunk():
    """
    Endpoint za prihvat pojedinačnih chunkova.
    Očekivani parametri (form data):
      - uploadId: jedinstveni ID za upload (string)
      - chunkIndex: redni broj chanka (int, počinje od 0)
      - totalChunks: ukupan broj chunkova (int)
      - tss, offset, mode, intrplMax: dodatni parametri za obradu
      - fileChunk: binarni sadržaj chanka
    Ako su svi chunkovi primljeni, oni se spajaju i obrađuju.
    """
    try:
        stream = request.stream
        boundary = None
        for key, value in request.headers.items():
            if key.lower() == 'content-type':
                logger.info(f"Parsing Content-Type: {value}")
                for part in value.split(';'):
                    logger.info(f"Checking part: {part}")
                    if 'boundary=' in part:
                        boundary = part.split('=')[1].strip()
                        logger.info(f"Found boundary: {boundary}")
                        break
        if not boundary:
            logger.error("No boundary found in Content-Type")
            raise ValueError("No boundary found in Content-Type")

        # Privremeno čuvamo podatke
        temp_data = {}
        current_key = None
        current_value = []
        file_data = None
        file_name = None
        reading_file = False
        reading_file_content = False
        
        logger.info("Starting to read stream...")
        line_count = 0
        for line in stream:
            line_count += 1
            
            # Ako čitamo sadržaj fajla
            if reading_file_content:
                if boundary.encode('utf-8') in line:
                    reading_file_content = False
                    reading_file = False
                    current_key = None
                    current_value = []
                    logger.info(f"Finished reading file content at line {line_count}")
                else:
                    if file_data is None:
                        file_data = line
                    else:
                        file_data += line
                continue
            
            # Pokušaj dekodirati liniju kao tekst
            try:
                line_str = line.decode('utf-8', errors='ignore').strip()
                logger.info(f"Line {line_count}: {line_str[:100]}..." if len(line_str) > 100 else f"Line {line_count}: {line_str}")
            except:
                continue

            # Nova sekcija počinje
            if boundary in line_str:
                logger.info(f"Found boundary at line {line_count}")
                if current_key and not reading_file:
                    temp_data[current_key] = ''.join(current_value)
                    logger.info(f"Saved value for key {current_key}: {temp_data[current_key][:100]}..." if len(temp_data[current_key]) > 100 else f"Saved value for key {current_key}: {temp_data[current_key]}")
                current_key = None
                current_value = []
                reading_file = False
                reading_file_content = False
                continue

            if 'Content-Disposition' in line_str:
                logger.info(f"Found Content-Disposition at line {line_count}")
                if 'name="' in line_str:
                    current_key = line_str.split('name="')[1].split('"')[0]
                    logger.info(f"Found field name: {current_key}")
                    # Proveri da li je ovo fajl polje (fileChunk ili file)
                    reading_file = ('filename="' in line_str and 
                                  (current_key == 'fileChunk' or current_key == 'file'))
                    if reading_file:
                        file_name = line_str.split('filename="')[1].split('"')[0]
                        logger.info(f"Found file field: {current_key} with filename: {file_name}")
                continue

            if line_str.startswith('Content-Type:'):
                logger.info(f"Found Content-Type at line {line_count}: {line_str}")
                if reading_file:
                    # Sledeca linija je prazna, pa onda pocinje sadrzaj fajla
                    reading_file_content = True
                continue

            # Ako je prazna linija posle Content-Type za fajl,
            # preskocimo je i cekamo sadrzaj
            if reading_file and not reading_file_content:
                continue

            if line_str and current_key is not None:
                if not reading_file:
                    if current_key == 'fileContent':
                        # Za CSV fajl, dodaj novi red
                        current_value.append(line_str + '\n')
                    else:
                        current_value.append(line_str)
                    pass  # Removed logging

        try:
            # Prvo proveri da li je ovo direktan upload ili chunk
            if 'uploadId' in temp_data:
                # Chunk upload
                upload_id = temp_data.get('uploadId')
                chunk_index = int(temp_data.get('chunkIndex', 0))
                total_chunks = int(temp_data.get('totalChunks', 0))
                tss = float(temp_data.get('tss', 0))
                offset = float(temp_data.get('offset', 0))
                mode_input = temp_data.get('mode', '')
                intrpl_max = float(temp_data.get('intrplMax', 60))
                
                if not upload_id or (not file_data and not temp_data.get('fileChunk')):
                    logger.error(f"Missing required chunk data: uploadId={bool(upload_id)}, fileData={bool(file_data)}")
                    return jsonify({"error": "uploadId i fileChunk su obavezni"}), 400

                # Ako imamo fileChunk kao string, tretiraj ga kao file_data
                if not file_data and temp_data.get('fileChunk'):
                    file_data = temp_data['fileChunk'].encode('utf-8')
            else:
                # Direktan upload
                tss = float(temp_data.get('tss', 0))
                offset = float(temp_data.get('offset', 0))
                mode_input = temp_data.get('mode', '')
                intrpl_max = float(temp_data.get('intrplMax', 60))
                file_content = temp_data.get('fileContent')

                if file_content:
                    # Ako imamo direktan file content, prosledi ga na obradu
                    logger.info("Processing direct file content")
                    # Ukloni poslednji newline ako postoji
                    if file_content.endswith('\n'):
                        file_content = file_content[:-1]
                    return process_csv(file_content, tss, offset, mode_input, intrpl_max)
                else:
                    logger.error("No file content found")
                    return jsonify({"error": "Keine Datei gefunden"}), 400
            
        except (ValueError, TypeError) as e:
            logger.error(f"Error parsing form values: {str(e)}")
            return jsonify({"error": "Invalid form values"}), 400

        logger.info(f"Prijem chunka {chunk_index+1}/{total_chunks} za uploadId {upload_id}")

        # Sačuvaj chunk
        chunk_filename = os.path.join(UPLOAD_FOLDER, f"{upload_id}_{chunk_index}.chunk")
        with open(chunk_filename, 'wb') as f:
            f.write(file_data)

        # Provjeri jesu li svi chunkovi primljeni
        received_chunks = [f for f in os.listdir(UPLOAD_FOLDER) if f.startswith(upload_id + "_")]
        if len(received_chunks) == total_chunks:
            logger.info(f"Svi chunkovi primljeni za uploadId {upload_id}. Spajanje...")
            chunks_sorted = sorted(received_chunks, key=lambda x: int(x.split("_")[1].split(".")[0]))
            full_content = b""
            try:
                for chunk_file in chunks_sorted:
                    chunk_path = os.path.join(UPLOAD_FOLDER, chunk_file)
                    with open(chunk_path, "rb") as cf:
                        chunk_content = cf.read()
                        full_content += chunk_content
                    os.remove(chunk_path)
                file_content = full_content.decode('utf-8')
                return process_csv(file_content, tss, offset, mode_input, intrpl_max)
            except Exception as e:
                # U slučaju greške, obriši sve chunkove
                for chunk_file in chunks_sorted:
                    try:
                        os.remove(os.path.join(UPLOAD_FOLDER, chunk_file))
                    except:
                        pass
                raise
        else:
            return jsonify({
                "message": f"Chunk {chunk_index+1}/{total_chunks} primljen",
                "uploadId": upload_id,
                "chunkIndex": chunk_index,
                "totalChunks": total_chunks,
                "remainingChunks": total_chunks - len(received_chunks)
            }), 200

    except Exception as e:
        error_msg = f"Unexpected error in upload_chunk: {str(e)}\nTraceback: {traceback.format_exc()}"
        logger.error(error_msg)
        return jsonify({
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 400

@bp.route('/prepare-save', methods=['POST'])
def prepare_save():
    try:
        data = request.json
        if not data or 'data' not in data:
            return jsonify({"error": "No data received"}), 400
        save_data = data['data']
        if not save_data:
            return jsonify({"error": "Empty data"}), 400

        # Kreiraj privremeni fajl i zapiši CSV podatke
        temp_file = tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.csv')
        writer = csv.writer(temp_file, delimiter=';')
        for row in save_data:
            writer.writerow(row)
        temp_file.close()

        # Generiši jedinstveni ID na osnovu trenutnog vremena
        file_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        temp_files[file_id] = temp_file.name

        return jsonify({"message": "File prepared for download", "fileId": file_id}), 200
    except Exception as e:
        logger.error(f"Error in prepare_save: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@bp.route('/download/<file_id>', methods=['GET'])
def download_file(file_id):
    try:
        if file_id not in temp_files:
            return jsonify({"error": "File not found"}), 404
        file_path = temp_files[file_id]
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