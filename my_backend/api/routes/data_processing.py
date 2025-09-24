import pandas as pd
import numpy as np
from datetime import datetime
from flask import Blueprint, request, Response, jsonify, send_file
from flask_socketio import emit
from flask import current_app
import io
import json
import csv
import tempfile
import os
import math
import logging
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

bp = Blueprint('data_processing', __name__)
UPLOAD_FOLDER = os.path.join(tempfile.gettempdir(), "upload_chunks")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Security configuration
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
ALLOWED_EXTENSIONS = {'.csv', '.txt'}
MAX_CHUNK_SIZE = 10 * 1024 * 1024  # 10MB per chunk (matching frontend)

def secure_path_join(base_dir, user_input):
    """Safely join paths preventing directory traversal attacks"""
    if not user_input:
        raise ValueError("Empty path component")

    # Check for path traversal attempts BEFORE cleaning
    if '..' in user_input or '/' in user_input or '\\' in user_input:
        raise ValueError("Path traversal attempt detected")

    # Additional check for encoded traversal attempts
    if '%2e%2e' in user_input.lower() or '%2f' in user_input.lower() or '%5c' in user_input.lower():
        raise ValueError("Encoded path traversal attempt detected")

    # Clean the input - only basename, remove dangerous chars
    clean_input = os.path.basename(user_input)

    # Remove any remaining dangerous characters
    dangerous_chars = ['<', '>', ':', '"', '|', '?', '*', '\x00']
    for char in dangerous_chars:
        clean_input = clean_input.replace(char, '')

    # Check if input is valid after cleaning
    if not clean_input or clean_input in ['.', '..', ''] or len(clean_input.strip()) == 0:
        raise ValueError("Invalid path component")

    full_path = os.path.join(base_dir, clean_input)

    # Ensure resolved path is within base directory
    base_real = os.path.realpath(base_dir)
    full_real = os.path.realpath(full_path)

    if not full_real.startswith(base_real + os.sep) and full_real != base_real:
        raise ValueError("Path traversal attempt detected")

    return full_path

def validate_processing_params(params):
    """Validate all numeric processing parameters"""
    validated = {}

    numeric_params = {
        'eqMax': (0, 10000, "Elimination max duration"),
        'elMax': (0, 1000000, "Upper limit value"),
        'elMin': (-1000000, 1000000, "Lower limit value"),
        'chgMax': (0, 10000, "Change rate max"),
        'lgMax': (0, 10000, "Length max"),
        'gapMax': (0, 10000, "Gap max duration")
    }

    for param, (min_val, max_val, description) in numeric_params.items():
        if param in params and params[param]:
            try:
                value = float(params[param])
                if not (min_val <= value <= max_val):
                    raise ValueError(f"{description} must be between {min_val} and {max_val}")
                validated[param] = value
            except (ValueError, TypeError) as e:
                raise ValueError(f"Invalid parameter {param}: must be a valid number")

    # Validate radio button parameters
    radio_params = ['radioValueNull', 'radioValueNotNull']
    for param in radio_params:
        if param in params:
            validated[param] = params[param]

    return validated

def validate_file_upload(file_chunk, filename):
    """Validate uploaded file security and format"""
    if not filename:
        raise ValueError("Empty filename")

    # Check file extension
    _, ext = os.path.splitext(filename.lower())
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {ext}. Only CSV and TXT files allowed.")

    # Check chunk size
    chunk_data = file_chunk.read()
    if len(chunk_data) > MAX_CHUNK_SIZE:
        raise ValueError("Chunk size too large")

    file_chunk.seek(0)  # Reset for further reading
    return chunk_data

def safe_error_response(error_msg, status_code=500):
    """Sanitize error messages to prevent information disclosure"""
    # Remove file paths from error messages
    sanitized = re.sub(r'/[^\s]+', '[PATH]', str(error_msg))
    # Remove detailed stack trace info - keep only first line
    sanitized = sanitized.split('\n')[0]

    logger.error(f"Error occurred: {error_msg}")  # Log full error for debugging

    return jsonify({"error": "A processing error occurred. Please check your input and try again."}), status_code

def clean_data(df, value_column, params, emit_progress_func=None, upload_id=None):
    logger.info("Starting data cleaning with parameters: %s", params)
    total_steps = 7  # Total number of cleaning steps
    current_step = 0
    
    def emit_cleaning_progress(step_name):
        nonlocal current_step
        current_step += 1
        if emit_progress_func and upload_id:
            progress = 75 + (current_step / total_steps) * 10  # 75-85%
            logger.info(f"Emitting progress: {progress}% - {step_name}")
            emit_progress_func(upload_id, 'cleaning', progress, f'{step_name}...')
    
    df["UTC"] = pd.to_datetime(df["UTC"], format="%Y-%m-%d %H:%M:%S")

    # ELIMINIERUNG VON MESSAUSFÄLLEN (GLEICHBLEIBENDE MESSWERTE)
    if params.get("eqMax"):
        emit_cleaning_progress("Eliminierung von Messausfällen")
        logger.info("Eliminierung von Messausfällen (gleichbleibende Messwerte)")
        eq_max = float(params["eqMax"])
        frm = 0
        for i in range(1, len(df)):
            if df.iloc[i-1][value_column] == df.iloc[i][value_column] and frm == 0:
                idx_strt = i-1
                frm = 1
            elif df.iloc[i-1][value_column] != df.iloc[i][value_column] and frm == 1:
                idx_end = i-1
                frm_width = (df.iloc[idx_end]["UTC"] - df.iloc[idx_strt]["UTC"]).total_seconds() / 60
                if frm_width >= eq_max:
                    for i_frm in range(idx_strt, idx_end+1):
                        df.iloc[i_frm, df.columns.get_loc(value_column)] = np.nan
                frm = 0
            elif i == len(df)-1 and frm == 1:
                idx_end = i
                frm_width = (df.iloc[idx_end]["UTC"] - df.iloc[idx_strt]["UTC"]).total_seconds() / 60
                if frm_width >= eq_max:
                    for i_frm in range(idx_strt, idx_end+1):
                        df.iloc[i_frm, df.columns.get_loc(value_column)] = np.nan

    # WERTE ÜBER DEM OBEREN GRENZWERT ENTFERNEN
    if params.get("elMax"):
        emit_cleaning_progress("Werte über dem oberen Grenzwert entfernen")
        logger.info("Werte über dem oberen Grenzwert entfernen")
        el_max = float(params["elMax"])
        for i in range(len(df)):
            try:
                if float(df.iloc[i][value_column]) > el_max:
                    df.iloc[i, df.columns.get_loc(value_column)] = np.nan
            except (ValueError, TypeError):
                df.iloc[i, df.columns.get_loc(value_column)] = np.nan

    # WERTE UNTER DEM UNTEREN GRENZWERT ENTFERNEN
    if params.get("elMin"):
        emit_cleaning_progress("Werte unter dem unteren Grenzwert entfernen")
        logger.info("Werte unter dem unteren Grenzwert entfernen")
        el_min = float(params["elMin"])
        for i in range(len(df)):
            try:
                if float(df.iloc[i][value_column]) < el_min:
                    df.iloc[i, df.columns.get_loc(value_column)] = np.nan
            except (ValueError, TypeError):
                df.iloc[i, df.columns.get_loc(value_column)] = np.nan

    # ELIMINIERUNG VON NULLWERTEN
    if params.get("radioValueNull") == "ja":
        emit_cleaning_progress("Eliminierung von Nullwerten")
        logger.info("Eliminierung von Nullwerten")
        for i in range(len(df)):
            if df.iloc[i][value_column] == 0:
                df.iloc[i, df.columns.get_loc(value_column)] = np.nan

    # ELIMINIERUNG VON NICHT NUMERISCHEN WERTEN
    if params.get("radioValueNotNull") == "ja":
        emit_cleaning_progress("Eliminierung von nicht numerischen Werten")
        logger.info("Eliminierung von nicht numerischen Werten")
        for i in range(len(df)):
            try:
                float(df.iloc[i][value_column])
                if math.isnan(float(df.iloc[i][value_column])):
                    df.iloc[i, df.columns.get_loc(value_column)] = np.nan
            except (ValueError, TypeError):
                df.iloc[i, df.columns.get_loc(value_column)] = np.nan

    # ELIMINIERUNG VON AUSREISSERN
    if params.get("chgMax") and params.get("lgMax"):
        emit_cleaning_progress("Eliminierung von Ausreissern")
        logger.info("Eliminierung von Ausreissern")
        chg_max = float(params["chgMax"])
        lg_max = float(params["lgMax"])
        frm = 0
        for i in range(1, len(df)):
            if pd.isna(df.iloc[i][value_column]) and frm == 0:
                pass
            elif pd.isna(df.iloc[i][value_column]) and frm == 1:
                idx_end = i-1
                for i_frm in range(idx_strt, idx_end+1):
                    df.iloc[i_frm, df.columns.get_loc(value_column)] = np.nan
                frm = 0
            elif pd.isna(df.iloc[i-1][value_column]):
                pass
            else:
                chg = abs(float(df.iloc[i][value_column]) - float(df.iloc[i-1][value_column]))
                t = (df.iloc[i]["UTC"] - df.iloc[i-1]["UTC"]).total_seconds() / 60
                if t > 0 and chg/t > chg_max and frm == 0:
                    idx_strt = i
                    frm = 1
                elif t > 0 and chg/t > chg_max and frm == 1:
                    idx_end = i-1
                    for i_frm in range(idx_strt, idx_end+1):
                        df.iloc[i_frm, df.columns.get_loc(value_column)] = np.nan
                    frm = 0
                elif frm == 1 and (df.iloc[i]["UTC"] - df.iloc[idx_strt]["UTC"]).total_seconds() / 60 > lg_max:
                    frm = 0

    # SCHLIESSEN VON MESSLÜCKEN
    if params.get("gapMax"):
        emit_cleaning_progress("Schließen von Messlücken")
        logger.info("Schließen von Messlücken")
        gap_max = float(params["gapMax"])
        frm = 0
        for i in range(1, len(df)):
            if pd.isna(df.iloc[i][value_column]) and frm == 0:
                idx_strt = i
                frm = 1
            elif not pd.isna(df.iloc[i][value_column]) and frm == 1:
                idx_end = i-1
                frm_width = (df.iloc[idx_end+1]["UTC"] - df.iloc[idx_strt-1]["UTC"]).total_seconds() / 60
                if frm_width <= gap_max and frm_width > 0:
                    dif = float(df.iloc[idx_end+1][value_column]) - float(df.iloc[idx_strt-1][value_column])
                    dif_min = dif/frm_width
                    for i_frm in range(idx_strt, idx_end+1):
                        gap_min = (df.iloc[i_frm]["UTC"] - df.iloc[idx_strt-1]["UTC"]).total_seconds() / 60
                        df.iloc[i_frm, df.columns.get_loc(value_column)] = float(df.iloc[idx_strt-1][value_column]) + gap_min*dif_min
                frm = 0

    logger.info("Data cleaning completed")
    return df

@bp.route("/api/dataProcessingMain/upload-chunk", methods=["POST"])
def upload_chunk():
    def emit_progress(upload_id, step, progress, message):
        """Emit progress update via Socket.IO"""
        if upload_id:
            try:
                # Import socketio directly from app module to avoid context issues
                from app import socketio
                logger.info(f"Emitting Socket.IO progress: {progress}% - {step} - {message} to room: {upload_id}")
                socketio.emit('processing_progress', {
                    'uploadId': upload_id,
                    'step': step,
                    'progress': progress,
                    'message': message
                }, room=upload_id)
            except Exception as e:
                logger.error(f"Error emitting progress: {e}")
                # Fallback: try current_app method
                try:
                    socketio = current_app.extensions.get('socketio')
                    if socketio:
                        logger.info(f"Fallback: Emitting Socket.IO progress via current_app: {progress}% - {step} - {message}")
                        socketio.emit('processing_progress', {
                            'uploadId': upload_id,
                            'step': step,
                            'progress': progress,
                            'message': message
                        }, room=upload_id)
                except Exception as fallback_error:
                    logger.error(f"Fallback emit also failed: {fallback_error}")
    try:
        # Minimal logging for performance
        logger.info(f"Processing chunk {request.form.get('chunkIndex')}/{request.form.get('totalChunks')}")

        if not all(key in request.form for key in ["uploadId", "chunkIndex", "totalChunks"]):
            missing_fields = [key for key in ["uploadId", "chunkIndex", "totalChunks"] if key not in request.form]
            logger.error(f"Missing required fields: {missing_fields}")
            return jsonify({"error": f"Missing required fields: {missing_fields}"}), 400

        upload_id = request.form["uploadId"]
        chunk_index = int(request.form["chunkIndex"])
        total_chunks = int(request.form["totalChunks"])
        
        if 'fileChunk' not in request.files:
            logger.error("No fileChunk in request.files")
            return safe_error_response("No file chunk provided", 400)

        file_chunk = request.files['fileChunk']
        if file_chunk.filename == '':
            logger.error("Empty filename in fileChunk")
            return safe_error_response("Empty filename", 400)

        # Validate and read the chunk
        try:
            chunk = validate_file_upload(file_chunk, file_chunk.filename)
            chunk_size = len(chunk)
        except ValueError as e:
            logger.error(f"File validation failed: {e}")
            return safe_error_response(str(e), 400)
        
        if chunk_size == 0:
            logger.error("Received empty chunk")
            return jsonify({"error": "Empty chunk received"}), 400

        # Create upload directory (secure path join)
        try:
            upload_dir = secure_path_join(UPLOAD_FOLDER, upload_id)
            os.makedirs(upload_dir, exist_ok=True)
        except ValueError as e:
            logger.error(f"Invalid upload_id: {upload_id}")
            return safe_error_response("Invalid upload identifier", 400)
        chunk_path = os.path.join(upload_dir, f"chunk_{chunk_index:04d}.part")
        
        # Save the chunk
        with open(chunk_path, "wb") as f:
            f.write(chunk)

        if chunk_index < total_chunks - 1:
            return jsonify({"status": "chunk received", "chunkIndex": chunk_index})

        # Combine all chunks
        logger.info("Starting to combine all chunks")
        emit_progress(upload_id, 'combining', 65, 'Combining uploaded chunks...')
        all_bytes = bytearray()
        total_size = 0
        
        for i in range(total_chunks):
            part_path = os.path.join(upload_dir, f"chunk_{i:04d}.part")
            if not os.path.exists(part_path):
                logger.error(f"Missing chunk file: {part_path}")
                return jsonify({"error": f"Missing chunk file: {i}"}), 400
                
            with open(part_path, "rb") as f:
                chunk_data = f.read()
                all_bytes.extend(chunk_data)
                total_size += len(chunk_data)

        if total_size == 0:
            logger.error("No data in combined chunks")
            return jsonify({"error": "No data in combined chunks"}), 400

        # Validate total file size after combination
        if total_size > MAX_FILE_SIZE:
            logger.error(f"Combined file size ({total_size} bytes) exceeds maximum allowed size ({MAX_FILE_SIZE} bytes)")
            # Clean up chunks before returning error
            for i in range(total_chunks):
                part_path = os.path.join(upload_dir, f"chunk_{i:04d}.part")
                try:
                    if os.path.exists(part_path):
                        os.remove(part_path)
                except OSError:
                    pass  # Ignore cleanup errors
            return safe_error_response("File too large", 413)

        # Process the combined data
        try:
            emit_progress(upload_id, 'processing', 70, 'Processing combined data...')
            content = all_bytes.decode("utf-8")
            lines = content.splitlines()
            
            if len(lines) < 2:
                logger.error("File has less than 2 lines")
                return jsonify({"error": "Invalid file format"}), 400

            separator = ";" if ";" in lines[0] else ","
            header = lines[0].split(separator)
            
            if len(header) < 2:
                logger.error("Invalid header format")
                return jsonify({"error": "Invalid header format"}), 400
                
            value_column = header[1].strip()
            data = [line.split(separator) for line in lines[1:] if line.strip()]
            
            if not data:
                logger.error("No data rows found")
                return jsonify({"error": "No data rows found"}), 400

            df = pd.DataFrame(data, columns=["UTC", value_column])
            df[value_column] = pd.to_numeric(df[value_column].str.replace(",", "."), errors='coerce')

            # Validate processing parameters
            raw_params = {
                "eqMax": request.form.get("eqMax"),
                "elMax": request.form.get("elMax"),
                "elMin": request.form.get("elMin"),
                "chgMax": request.form.get("chgMax"),
                "lgMax": request.form.get("lgMax"),
                "gapMax": request.form.get("gapMax"),
                "radioValueNull": request.form.get("radioValueNull"),
                "radioValueNotNull": request.form.get("radioValueNotNull")
            }

            try:
                params = validate_processing_params(raw_params)
            except ValueError as e:
                logger.error(f"Parameter validation failed: {e}")
                return safe_error_response(str(e), 400)

            emit_progress(upload_id, 'cleaning', 75, f'Cleaning data with {len(df)} rows...')
            df_clean = clean_data(df, value_column, params, emit_progress, upload_id)
            emit_progress(upload_id, 'cleaned', 85, f'Data cleaning completed. Processing {len(df_clean)} rows...')

            def generate():
                # Create a custom JSON encoder to handle Pandas Timestamp objects
                class CustomJSONEncoder(json.JSONEncoder):
                    def default(self, obj):
                        if isinstance(obj, pd.Timestamp):
                            return obj.strftime('%Y-%m-%d %H:%M:%S')
                        return super().default(obj)
                
                # Emit progress for streaming start
                emit_progress(upload_id, 'streaming', 90, f'Starting to stream {len(df_clean)} processed rows...')
                
                # First send total rows
                yield json.dumps({"total_rows": len(df_clean)}, cls=CustomJSONEncoder) + "\n"
                
                # Process data in larger chunks of 50000 rows
                chunk_size = 50000
                for i in range(0, len(df_clean), chunk_size):
                    # Emit progress for chunk processing
                    chunk_progress = 90 + ((i / len(df_clean)) * 8)  # 90-98%
                    emit_progress(upload_id, 'streaming', chunk_progress, f'Streaming chunk {i//chunk_size + 1}/{(len(df_clean)//chunk_size) + 1}...')
                    # Create a copy of the chunk and convert UTC in one step
                    chunk = df_clean.iloc[i:i + chunk_size].copy()
                    chunk.loc[:, 'UTC'] = chunk['UTC'].dt.strftime('%Y-%m-%d %H:%M:%S')
                    
                    # Log first 10 rows of the first chunk
                    if i == 0:
                        logger.info("First 10 rows of processed data:")
                        logger.info(chunk.head(10).to_string())
                    
                    # Convert to dict and ensure all values are JSON serializable
                    chunk_data = []
                    for _, row in chunk.iterrows():
                        # Ensure UTC is a string, not a Timestamp object
                        utc_value = row['UTC']
                        if isinstance(utc_value, pd.Timestamp):
                            utc_value = utc_value.strftime('%Y-%m-%d %H:%M:%S')
                            
                        record = {
                            'UTC': utc_value,
                            value_column: row[value_column] if pd.notnull(row[value_column]) else None
                        }
                        chunk_data.append(record)
                    
                    # Yield all records in the chunk at once
                    for record in chunk_data:
                        yield json.dumps(record, cls=CustomJSONEncoder) + "\n"
                
                # Send completion status
                emit_progress(upload_id, 'complete', 100, f'Processing completed! Generated {len(df_clean)} data points.')
                yield json.dumps({"status": "complete"}, cls=CustomJSONEncoder) + "\n"
                        
            return Response(generate(), mimetype="application/x-ndjson")
        

        except Exception as e:
            logger.error(f"Error processing data: {str(e)}")
            return safe_error_response("Error processing data", 500)

    except Exception as e:
        logger.error(f"Error in upload_chunk: {str(e)}")
        return safe_error_response("Error in upload", 500)

@bp.route("/api/dataProcessingMain/prepare-save", methods=["POST"])
def prepare_save():
    try:
        data = request.json.get("data")
        if not data:
            return jsonify({"error": "No data provided"}), 400

        file_name = data.get("fileName", "output.csv")
        rows = data.get("data", [])
        
        if not rows:
            return jsonify({"error": "No rows provided"}), 400

        logger.info(f"Preparing to save file: {file_name} with {len(rows)} rows")

        # Sanitize filename securely
        try:
            safe_file_name = os.path.basename(file_name).replace(" ", "_")
            file_path = secure_path_join(tempfile.gettempdir(), safe_file_name)
        except ValueError as e:
            logger.error(f"Invalid filename: {file_name}")
            return safe_error_response("Invalid filename", 400)

        # Write CSV
        with open(file_path, "w", newline='', encoding='utf-8') as f:
            writer = csv.writer(f, delimiter=";")
            for row in rows:
                writer.writerow(row)

        logger.info(f"File prepared successfully with ID: {safe_file_name}")
        return jsonify({"fileId": safe_file_name})

    except Exception as e:
        logger.error(f"Error preparing save: {str(e)}")
        return safe_error_response("Error preparing file", 500)


@bp.route("/api/dataProcessingMain/download/<file_id>", methods=["GET"])
def download_file(file_id):
    try:
        # Secure path validation
        try:
            path = secure_path_join(tempfile.gettempdir(), file_id)
        except ValueError as e:
            logger.error(f"Invalid file_id: {file_id}")
            return safe_error_response("Invalid file identifier", 400)

        if not os.path.exists(path):
            return safe_error_response("File not found", 404)

        logger.info(f"Sending file: {path}")
        return send_file(
            path,
            as_attachment=True,
            download_name=file_id,
            mimetype="text/csv"
        )

    except Exception as e:
        logger.error(f"Error downloading file: {str(e)}")
        return safe_error_response("Error downloading file", 500)
