import csv
import json
import logging
import math
import os
import re
import tempfile

import numpy as np
import pandas as pd
from flask import Blueprint, request, Response, jsonify, send_file, current_app

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

bp = Blueprint('data_processing', __name__)
UPLOAD_FOLDER = os.path.join(tempfile.gettempdir(), "upload_chunks")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

MAX_FILE_SIZE = 100 * 1024 * 1024
ALLOWED_EXTENSIONS = {'.csv', '.txt'}
MAX_CHUNK_SIZE = 10 * 1024 * 1024

PROGRESS_COMBINING = 65
PROGRESS_PROCESSING = 70
PROGRESS_CLEANING_START = 75
PROGRESS_CLEANED = 85
PROGRESS_STREAMING_START = 90
PROGRESS_COMPLETE = 100

STREAMING_CHUNK_SIZE = 50000
TOTAL_CLEANING_STEPS = 7
CLEANING_PROGRESS_RANGE = 10
STREAMING_PROGRESS_RANGE = 8

def secure_path_join(base_dir, user_input):
    """Safely join paths preventing directory traversal attacks"""
    if not user_input:
        raise ValueError("Empty path component")

    if '..' in user_input or '/' in user_input or '\\' in user_input:
        raise ValueError("Path traversal attempt detected")

    if '%2e%2e' in user_input.lower() or '%2f' in user_input.lower() or '%5c' in user_input.lower():
        raise ValueError("Encoded path traversal attempt detected")

    clean_input = os.path.basename(user_input)

    dangerous_chars = ['<', '>', ':', '"', '|', '?', '*', '\x00']
    for char in dangerous_chars:
        clean_input = clean_input.replace(char, '')

    if not clean_input or clean_input in ['.', '..', ''] or len(clean_input.strip()) == 0:
        raise ValueError("Invalid path component")

    full_path = os.path.join(base_dir, clean_input)

    base_real = os.path.realpath(base_dir)
    full_real = os.path.realpath(full_path)

    if not full_real.startswith(base_real + os.sep) and full_real != base_real:
        raise ValueError("Path traversal attempt detected")

    return full_path

def validate_processing_params(params):
    """Validate all numeric processing parameters"""
    validated = {}

    numeric_params = {
        'eqMax': (0, 1000000, "Elimination max duration"),
        'elMax': (-1000000, 1000000, "Upper limit value"),
        'elMin': (-1000000, 1000000, "Lower limit value"),
        'chgMax': (0, 1000000, "Change rate max"),
        'lgMax': (0, 1000000, "Length max"),
        'gapMax': (0, 1000000, "Gap max duration")
    }

    for param, (min_val, max_val, description) in numeric_params.items():
        if param in params and params[param] is not None and str(params[param]).strip() != '':
            try:
                value = float(params[param])
            except (ValueError, TypeError) as e:
                raise ValueError(f"Invalid parameter {param}: must be a valid number")

            if not (min_val <= value <= max_val):
                raise ValueError(f"{description} must be between {min_val} and {max_val}")
            validated[param] = value

    radio_params = ['radioValueNull', 'radioValueNotNull']
    for param in radio_params:
        if param in params:
            if params[param] in [None, '', 'undefined', 'null']:
                validated[param] = ''
            else:
                validated[param] = params[param]

    return validated

def validate_file_upload(file_chunk, filename):
    """Validate uploaded file security and format"""
    if not filename:
        raise ValueError("Empty filename")

    _, ext = os.path.splitext(filename.lower())
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {ext}. Only CSV and TXT files allowed.")

    chunk_data = file_chunk.read()
    if len(chunk_data) > MAX_CHUNK_SIZE:
        raise ValueError("Chunk size too large")

    file_chunk.seek(0)
    return chunk_data

def safe_error_response(error_msg, status_code=500, error_type=None):
    """Sanitize error messages to prevent information disclosure while preserving specific error types"""
    sanitized = re.sub(r'/[^\s]+', '[PATH]', str(error_msg))
    sanitized = sanitized.split('\n')[0]

    logger.error(f"Error occurred: {error_msg}")

    if error_type == 'validation' or 'validation' in str(error_msg).lower():
        if 'parameter' in str(error_msg).lower():
            return jsonify({"error": f"Parameter validation failed: {sanitized}"}), status_code
        elif 'csv' in str(error_msg).lower() or 'file format' in str(error_msg).lower():
            return jsonify({"error": f"CSV validation failed: {sanitized}"}), status_code
        elif 'file size' in str(error_msg).lower() or 'too large' in str(error_msg).lower():
            return jsonify({"error": f"File size validation failed: {sanitized}"}), status_code
        else:
            return jsonify({"error": f"Data validation failed: {sanitized}"}), status_code
    elif error_type == 'security' or 'security' in str(error_msg).lower():
        return jsonify({"error": f"Security validation failed: {sanitized}"}), status_code
    elif error_type == 'file' or status_code in [400, 413]:
        return jsonify({"error": sanitized}), status_code
    else:
        return jsonify({"error": "A processing error occurred. Please check your input and try again."}), status_code

def clean_data(df, value_column, params, emit_progress_func=None, upload_id=None):
    logger.info("Starting data cleaning with parameters: %s", params)
    current_step = 0

    def emit_cleaning_progress(step_name, removed_count=None):
        nonlocal current_step
        current_step += 1
        if emit_progress_func and upload_id:
            progress = PROGRESS_CLEANING_START + (current_step / TOTAL_CLEANING_STEPS) * CLEANING_PROGRESS_RANGE
            message = f"Step {current_step}/{TOTAL_CLEANING_STEPS}: {step_name}"
            if removed_count is not None:
                message += f" - Removed: {removed_count} values"
            logger.info(f"Emitting progress: {progress}% - {step_name}")
            emit_progress_func(upload_id, 'cleaning', progress, f'{message}...')
    
    df["UTC"] = pd.to_datetime(df["UTC"], format="%Y-%m-%d %H:%M:%S")

    if params.get("eqMax"):
        emit_cleaning_progress("Removing measurement failures")
        logger.info("Removing measurement failures (identical consecutive values)")
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

    if params.get("elMax") is not None:
        logger.info("Removing values above upper threshold")
        el_max = float(params["elMax"])
        values_to_remove = 0
        for i in range(len(df)):
            try:
                if float(df.iloc[i][value_column]) > el_max:
                    df.iloc[i, df.columns.get_loc(value_column)] = np.nan
                    values_to_remove += 1
            except:
                df.iloc[i, df.columns.get_loc(value_column)] = np.nan
                values_to_remove += 1
        emit_cleaning_progress("Removing values above upper threshold", values_to_remove)

    if params.get("elMin") is not None:
        logger.info("Removing values below lower threshold")
        el_min = float(params["elMin"])
        values_to_remove = 0
        for i in range(len(df)):
            try:
                if float(df.iloc[i][value_column]) < el_min:
                    df.iloc[i, df.columns.get_loc(value_column)] = np.nan
                    values_to_remove += 1
            except:
                df.iloc[i, df.columns.get_loc(value_column)] = np.nan
                values_to_remove += 1
        emit_cleaning_progress("Removing values below lower threshold", values_to_remove)

    if params.get("radioValueNull") == "ja":
        logger.info("Removing null values")
        values_to_remove = 0
        for i in range(len(df)):
            if df.iloc[i][value_column] == 0:
                df.iloc[i, df.columns.get_loc(value_column)] = np.nan
                values_to_remove += 1
        emit_cleaning_progress("Removing null values", values_to_remove)

    if params.get("radioValueNotNull") == "ja":
        logger.info("Removing non-numeric values")
        values_to_remove = 0
        for i in range(len(df)):
            try:
                float(df.iloc[i][value_column])
                if math.isnan(float(df.iloc[i][value_column])) == True:
                    df.iloc[i, df.columns.get_loc(value_column)] = np.nan
                    values_to_remove += 1
            except:
                df.iloc[i, df.columns.get_loc(value_column)] = np.nan
                values_to_remove += 1
        emit_cleaning_progress("Removing non-numeric values", values_to_remove)

    if params.get("chgMax") and params.get("lgMax"):
        emit_cleaning_progress("Removing outliers")
        logger.info("Removing outliers")
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

    if params.get("gapMax"):
        emit_cleaning_progress("Filling measurement gaps")
        logger.info("Filling measurement gaps")
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

    if params.get("elMin") is not None:
        el_min = float(params["elMin"])
        final_violations_min = (df[value_column] < el_min).sum()
        zero_values = (df[value_column] == 0).sum()
        logger.info(f"Final validation: Found {final_violations_min} values < {el_min} and {zero_values} zero values")
        if final_violations_min > 0:
            logger.info(f"Removing {final_violations_min} interpolated values below elMin threshold")
            df.loc[df[value_column] < el_min, value_column] = np.nan
    
    if params.get("elMax") is not None:
        el_max = float(params["elMax"])
        final_violations_max = (df[value_column] > el_max).sum()
        if final_violations_max > 0:
            logger.info(f"Removing {final_violations_max} interpolated values above elMax threshold")
            df.loc[df[value_column] > el_max, value_column] = np.nan

    logger.info("Data cleaning completed")
    return df

def _combine_chunks_efficiently(upload_dir, total_chunks):
    """Memory-efficient chunk combination using temporary file streaming"""
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    total_size = 0

    try:
        for i in range(total_chunks):
            chunk_path = os.path.join(upload_dir, f"chunk_{i:04d}.part")
            if not os.path.exists(chunk_path):
                temp_file.close()
                os.unlink(temp_file.name)
                raise FileNotFoundError(f"Missing chunk file: {i}")

            with open(chunk_path, "rb") as chunk_file:
                while True:
                    block = chunk_file.read(8192)
                    if not block:
                        break
                    temp_file.write(block)
                    total_size += len(block)

                    if total_size > MAX_FILE_SIZE:
                        temp_file.close()
                        os.unlink(temp_file.name)
                        raise ValueError(f"Combined file size exceeds {MAX_FILE_SIZE} bytes")

        temp_file.close()
        return temp_file.name, total_size

    except Exception:
        if not temp_file.closed:
            temp_file.close()
        if os.path.exists(temp_file.name):
            os.unlink(temp_file.name)
        raise

def _emit_progress_update(upload_id, step, progress, message):
    """Emit progress update via Socket.IO"""
    if upload_id:
        try:
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

@bp.route("/api/dataProcessingMain/upload-chunk", methods=["POST"])
def upload_chunk():
    try:
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

        try:
            chunk = validate_file_upload(file_chunk, file_chunk.filename)
            chunk_size = len(chunk)
        except ValueError as e:
            logger.error(f"File validation failed: {e}")
            return safe_error_response(str(e), 400, 'validation')
        
        if chunk_size == 0:
            logger.error("Received empty chunk")
            return jsonify({"error": "Empty chunk received"}), 400

        try:
            upload_dir = secure_path_join(UPLOAD_FOLDER, upload_id)
            os.makedirs(upload_dir, exist_ok=True)
        except ValueError as e:
            logger.error(f"Invalid upload_id: {upload_id}")
            return safe_error_response("Invalid upload identifier", 400, 'security')
        chunk_path = os.path.join(upload_dir, f"chunk_{chunk_index:04d}.part")
        
        with open(chunk_path, "wb") as f:
            f.write(chunk)

        if chunk_index < total_chunks - 1:
            return jsonify({"status": "chunk received", "chunkIndex": chunk_index})

        logger.info("Starting to combine all chunks")
        _emit_progress_update(upload_id, 'combining', PROGRESS_COMBINING, 'Combining uploaded chunks...')

        try:
            combined_file_path, total_size = _combine_chunks_efficiently(upload_dir, total_chunks)
        except (FileNotFoundError, ValueError) as e:
            logger.error(f"Chunk combination failed: {e}")
            for i in range(total_chunks):
                part_path = os.path.join(upload_dir, f"chunk_{i:04d}.part")
                try:
                    if os.path.exists(part_path):
                        os.remove(part_path)
                except OSError:
                    pass

            if "exceeds" in str(e):
                return safe_error_response("File too large", 413, 'validation')
            else:
                return jsonify({"error": f"Missing chunk file"}), 400

        if total_size == 0:
            logger.error("No data in combined chunks")
            os.unlink(combined_file_path)
            return jsonify({"error": "No data in combined chunks"}), 400

        try:
            _emit_progress_update(upload_id, 'processing', PROGRESS_PROCESSING, 'Processing combined data...')
            with open(combined_file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            os.unlink(combined_file_path)
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
            df[value_column] = df[value_column].replace('', np.nan)
            df[value_column] = pd.to_numeric(df[value_column].str.replace(",", "."), errors='coerce')

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
                return safe_error_response(str(e), 400, 'validation')

            _emit_progress_update(upload_id, 'cleaning', PROGRESS_CLEANING_START, f'Cleaning data with {len(df)} rows...')
            df_clean = clean_data(df, value_column, params, _emit_progress_update, upload_id)
            _emit_progress_update(upload_id, 'cleaned', PROGRESS_CLEANED, f'Data cleaning completed. Processing {len(df_clean)} rows...')

            def generate():
                class CustomJSONEncoder(json.JSONEncoder):
                    def default(self, obj):
                        if isinstance(obj, pd.Timestamp):
                            return obj.strftime('%Y-%m-%d %H:%M:%S')
                        return super().default(obj)
                
                _emit_progress_update(upload_id, 'streaming', PROGRESS_STREAMING_START, f'Starting to stream {len(df_clean)} processed rows...')
                
                yield json.dumps({"total_rows": len(df_clean)}, cls=CustomJSONEncoder) + "\n"
                
                chunk_size = STREAMING_CHUNK_SIZE
                for i in range(0, len(df_clean), chunk_size):
                    chunk_progress = PROGRESS_STREAMING_START + ((i / len(df_clean)) * STREAMING_PROGRESS_RANGE)
                    _emit_progress_update(upload_id, 'streaming', chunk_progress, f'Streaming chunk {i//chunk_size + 1}/{(len(df_clean)//chunk_size) + 1}...')
                    chunk = df_clean.iloc[i:i + chunk_size].copy()
                    chunk.loc[:, 'UTC'] = chunk['UTC'].dt.strftime('%Y-%m-%d %H:%M:%S')
                    
                    if i == 0:
                        logger.info("First 10 rows of processed data:")
                        logger.info(chunk.head(10).to_string())
                    
                    chunk_data = []
                    for _, row in chunk.iterrows():
                        utc_value = row['UTC']
                        if isinstance(utc_value, pd.Timestamp):
                            utc_value = utc_value.strftime('%Y-%m-%d %H:%M:%S')
                            
                        record = {
                            'UTC': utc_value,
                            value_column: row[value_column] if pd.notnull(row[value_column]) else None
                        }
                        chunk_data.append(record)
                    
                    for record in chunk_data:
                        yield json.dumps(record, cls=CustomJSONEncoder) + "\n"
                
                _emit_progress_update(upload_id, 'complete', PROGRESS_COMPLETE, f'Processing completed! Generated {len(df_clean)} data points.')
                yield json.dumps({"status": "complete"}, cls=CustomJSONEncoder) + "\n"
                        
            return Response(generate(), mimetype="application/x-ndjson")
        

        except Exception as e:
            logger.error(f"Error processing data: {str(e)}")
            return safe_error_response("Error processing data", 500, 'processing')

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

        try:
            safe_file_name = os.path.basename(file_name).replace(" ", "_")
            file_path = secure_path_join(tempfile.gettempdir(), safe_file_name)
        except ValueError as e:
            logger.error(f"Invalid filename: {file_name}")
            return safe_error_response("Invalid filename", 400, 'validation')

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
        try:
            path = secure_path_join(tempfile.gettempdir(), file_id)
        except ValueError as e:
            logger.error(f"Invalid file_id: {file_id}")
            return safe_error_response("Invalid file identifier", 400, 'security')

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
