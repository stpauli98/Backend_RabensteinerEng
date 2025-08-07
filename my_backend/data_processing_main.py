"""Data Processing Main Module

Handles chunked file upload and CSV data cleaning operations for the 
Rabensteiner Engineering backend. Supports real-time progress tracking
via Socket.IO and streaming responses for large datasets.

Key Features:
- Chunked file upload with progress tracking
- 7-step data cleaning pipeline
- Real-time Socket.IO progress updates
- Streaming JSON response for large datasets
- Robust error handling and logging
"""

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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Progress tracking constants
CHUNK_UPLOAD_MAX_PROGRESS = 65
CHUNK_COMBINE_PROGRESS = 68
DATA_DECODING_PROGRESS = 68
DATA_PARSING_PROGRESS = 70
DATAFRAME_CREATION_PROGRESS = 72
PARAMETER_LOADING_PROGRESS = 73
DATA_CLEANING_START_PROGRESS = 75
DATA_CLEANING_COMPLETE_PROGRESS = 85
STREAMING_START_PROGRESS = 90
STREAMING_COMPLETE_PROGRESS = 98
COMPLETE_PROGRESS = 100

# Processing constants
STREAMING_CHUNK_SIZE = 50000
TOTAL_CLEANING_STEPS = 7

bp = Blueprint('data_processing', __name__)
UPLOAD_FOLDER = os.path.join(tempfile.gettempdir(), "upload_chunks")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Custom JSON encoder for Pandas Timestamp objects
class CustomJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle Pandas Timestamp objects"""
    def default(self, obj):
        if isinstance(obj, pd.Timestamp):
            return obj.strftime('%Y-%m-%d %H:%M:%S')
        return super().default(obj)

def emit_progress_with_fallback(upload_id: str, step: str, progress: float, message: str) -> None:
    """Emit progress update via Socket.IO with fallback mechanism"""
    if not upload_id:
        return
        
    try:
        # Use current_app extensions first to avoid import cycles
        socketio = current_app.extensions.get('socketio')
        if socketio:
            logger.info(f"Primary: Emitting Socket.IO progress: {progress}% - {step} - {message} to room: {upload_id}")
            socketio.emit('processing_progress', {
                'uploadId': upload_id,
                'step': step,
                'progress': progress,
                'message': message
            }, room=upload_id)
        else:
            raise ValueError("SocketIO instance not found in current_app.extensions")
    except Exception as e:
        logger.error(f"Primary emit failed: {e}")
        # Fallback: try importing socketio directly
        try:
            from app import socketio
            logger.info(f"Fallback: Emitting Socket.IO progress via app import: {progress}% - {step} - {message} to room: {upload_id}")
            socketio.emit('processing_progress', {
                'uploadId': upload_id,
                'step': step,
                'progress': progress,
                'message': message
            }, room=upload_id)
        except Exception as fallback_error:
            logger.error(f"Fallback emit also failed: {fallback_error}")

def handle_processing_error(upload_id: str, error: Exception, context: str) -> dict:
    """Handle processing errors with consistent logging and response"""
    error_message = f"Error {context}: {str(error)}"
    logger.error(error_message)
    
    # Emit error via Socket.IO
    if upload_id:
        emit_progress_with_fallback(upload_id, 'error', 0, f'Fehler {context}: {str(error)}')
    
    return {"error": error_message}

def validate_required_fields(form_data: dict, required_fields: list) -> tuple[bool, str]:
    """Validate that all required fields are present in form data"""
    missing_fields = [field for field in required_fields if field not in form_data]
    if missing_fields:
        error_msg = f"Missing required fields: {missing_fields}"
        logger.error(error_msg)
        return False, error_msg
    return True, ""

def clean_data(df: pd.DataFrame, value_column: str, params: dict, emit_progress_func=None, upload_id: str = None) -> pd.DataFrame:
    """Clean DataFrame based on provided parameters with progress tracking
    
    Args:
        df: Input DataFrame to clean
        value_column: Name of the column containing values to clean
        params: Dictionary of cleaning parameters
        emit_progress_func: Function to emit progress updates
        upload_id: Unique identifier for progress tracking
        
    Returns:
        Cleaned DataFrame
    """
    logger.info("Starting data cleaning with parameters: %s", params)
    current_step = 0
    initial_row_count = len(df)
    
    def emit_detailed_progress(step_name, details=None, current_row=None, total_rows=None, step_progress=None):
        """Emit detailed progress with step information"""
        nonlocal current_step
        if emit_progress_func and upload_id:
            # Calculate overall progress
            if step_progress is not None:
                progress = step_progress
            else:
                base_progress = DATA_CLEANING_START_PROGRESS + (current_step / TOTAL_CLEANING_STEPS) * (DATA_CLEANING_COMPLETE_PROGRESS - DATA_CLEANING_START_PROGRESS)
                
                # If processing rows, add sub-progress
                if current_row and total_rows:
                    sub_progress = (current_row / total_rows) * ((DATA_CLEANING_COMPLETE_PROGRESS - DATA_CLEANING_START_PROGRESS) / TOTAL_CLEANING_STEPS)
                    progress = base_progress + sub_progress
                else:
                    progress = base_progress
                
            message = f'{step_name}'
            if details:
                message += f' - {details}'
            if current_row and total_rows:
                message += f' [{current_row}/{total_rows}]'
                
            logger.info(f"Emitting progress: {progress:.1f}% - {message}")
            emit_progress_func(upload_id, 'cleaning', progress, message)
    
    def emit_step_start(step_name, params_info=None):
        """Emit when starting a new cleaning step"""
        nonlocal current_step
        current_step += 1
        details = f"Parameter: {params_info}" if params_info else "Gestartet"
        emit_detailed_progress(f"Schritt {current_step}/{TOTAL_CLEANING_STEPS}: {step_name}", details)
    
    def emit_step_complete(step_name, result_info):
        """Emit when completing a cleaning step"""
        emit_detailed_progress(f"✓ {step_name}", result_info)
    
    # Convert UTC column to datetime - MUST match original format
    # This ensures consistent datetime handling throughout the cleaning process
    UTC_fmt = "%Y-%m-%d %H:%M:%S"
    df["UTC"] = pd.to_datetime(df["UTC"], format=UTC_fmt)

    # STEP 1: ELIMINIERUNG VON MESSAUSFÄLLEN (GLEICHBLEIBENDE MESSWERTE)
    # Remove measurement failures where values remain constant for extended periods
    if params.get("eqMax"):
        emit_step_start("Eliminierung von Messausfällen", f"Schwellwert: {params['eqMax']} min")
        logger.info("Eliminierung von Messausfällen (gleichbleibende Messwerte)")
        eq_max = float(params["eqMax"])
        frm = 0  # Keep original variable name for exact algorithm match
        removed_count = 0
        for i in range(1, len(df)):
            # Use proper column names to avoid FutureWarning
            if df.iloc[i-1][value_column] == df.iloc[i][value_column] and frm == 0:
                idx_strt = i-1
                frm = 1
            elif df.iloc[i-1][value_column] != df.iloc[i][value_column] and frm == 1:
                idx_end = i-1
                # Use proper column name for UTC
                frm_width = (df.iloc[idx_end]["UTC"] - df.iloc[idx_strt]["UTC"]).total_seconds() / 60
                if frm_width >= eq_max:
                    for i_frm in range(idx_strt, idx_end+1):
                        df.iloc[i_frm, df.columns.get_loc(value_column)] = "nan"
                        removed_count += 1
                frm = 0
            elif i == len(df)-1 and frm == 1:
                idx_end = i
                frm_width = (df.iloc[idx_end]["UTC"] - df.iloc[idx_strt]["UTC"]).total_seconds() / 60
                if frm_width >= eq_max:
                    for i_frm in range(idx_strt, idx_end+1):
                        df.iloc[i_frm, df.columns.get_loc(value_column)] = "nan"
                        removed_count += 1
        emit_step_complete("✓ Messausfälle eliminiert", f"Entfernt: {removed_count} Werte")

    # STEP 2: WERTE ÜBER DEM OBEREN GRENZWERT ENTFERNEN
    # Remove values that exceed the specified upper limit
    if params.get("elMax"):
        emit_step_start("Werte über dem oberen Grenzwert entfernen", f"Maximum: {params['elMax']}")
        logger.info("Werte über dem oberen Grenzwert entfernen")
        el_max = float(params["elMax"])
        removed_count = 0
        for i in range(len(df)):
            try:
                if float(df.iloc[i][value_column]) > el_max:
                    df.iloc[i, df.columns.get_loc(value_column)] = "nan"
                    removed_count += 1
            except:
                df.iloc[i, df.columns.get_loc(value_column)] = "nan"
                removed_count += 1
        emit_step_complete("Werte über dem oberen Grenzwert entfernt", f"Entfernt: {removed_count} Werte")
    else:
        emit_step_start("Werte über dem oberen Grenzwert", "Deaktiviert - übersprungen")
        emit_step_complete("✓ Werte über dem oberen Grenzwert", "Keine Entfernung - Parameter nicht gesetzt")

    # STEP 3: WERTE UNTER DEM UNTEREN GRENZWERT ENTFERNEN
    # Remove values that fall below the specified lower limit
    if params.get("elMin"):
        emit_step_start("Werte unter dem unteren Grenzwert entfernen", f"Minimum: {params['elMin']}")
        logger.info("Werte unter dem unteren Grenzwert entfernen")
        el_min = float(params["elMin"])
        removed_count = 0
        for i in range(len(df)):
            try:
                if float(df.iloc[i][value_column]) < el_min:
                    df.iloc[i, df.columns.get_loc(value_column)] = "nan"
                    removed_count += 1
            except:
                df.iloc[i, df.columns.get_loc(value_column)] = "nan"
                removed_count += 1
        emit_step_complete("Werte unter dem unteren Grenzwert entfernt", f"Entfernt: {removed_count} Werte")
    else:
        emit_step_start("Werte unter dem unteren Grenzwert", "Deaktiviert - übersprungen")
        emit_step_complete("✓ Werte unter dem unteren Grenzwert", "Keine Entfernung - Parameter nicht gesetzt")

    # STEP 4: ELIMINIERUNG VON NULLWERTEN
    # Remove zero values if enabled by user configuration
    if params.get("radioValueNull") == "ja":
        emit_step_start("Eliminierung von Nullwerten", "Aktiviert")
        logger.info("Eliminierung von Nullwerten")
        removed_count = 0
        # Use same loop range as original: range(0, len(df))
        for i in range(0, len(df)):
            if df.iloc[i][value_column] == 0:
                df.iloc[i, df.columns.get_loc(value_column)] = "nan"
                removed_count += 1
        emit_step_complete("Nullwerte eliminiert", f"Entfernt: {removed_count} Werte")
    else:
        emit_step_start("Eliminierung von Nullwerten", "Deaktiviert - übersprungen")
        emit_step_complete("✓ Nullwerte", "Keine Entfernung - deaktiviert")

    # STEP 5: ELIMINIERUNG VON NICHT NUMERISCHEN WERTEN
    # Remove non-numeric values and NaN entries if enabled
    if params.get("radioValueNotNull") == "ja":
        emit_step_start("Eliminierung von nicht numerischen Werten", "Aktiviert")
        logger.info("Eliminierung von nicht numerischen Werten")
        removed_count = 0
        # Use same loop range as original: range(0, len(df))
        for i in range(0, len(df)):
            try:
                float(df.iloc[i][value_column])  # Use proper column name to avoid FutureWarning
                if math.isnan(float(df.iloc[i][value_column])) == True:  # Exact original condition with == True
                    df.iloc[i, df.columns.get_loc(value_column)] = "nan"  # Use proper column location
                    removed_count += 1
            except:
                df.iloc[i, df.columns.get_loc(value_column)] = "nan"  # Use proper column location
                removed_count += 1
        emit_step_complete("Nicht numerische Werte eliminiert", f"Entfernt: {removed_count} Werte")
    else:
        emit_step_start("Eliminierung von nicht numerischen Werten", "Deaktiviert - übersprungen")
        emit_step_complete("✓ Nicht numerische Werte", "Keine Entfernung - deaktiviert")

    # STEP 6: ELIMINIERUNG VON AUSREISSERN
    # Remove outliers based on rate of change and duration thresholds  
    if params.get("chgMax") and params.get("lgMax"):
        emit_step_start("Eliminierung von Ausreissern", f"Änderung: {params['chgMax']}/min, Dauer: {params['lgMax']} min")
        logger.info("Eliminierung von Ausreissern")
        chg_max = float(params["chgMax"])  # Keep original variable names
        lg_max = float(params["lgMax"])
        frm = 0  # Keep original variable name
        removed_count = 0
        for i in range(1, len(df)):
            # nan im aktuellen Zeitschritt und Identifikationsrahmen ist nicht offen
            if df.iloc[i][value_column] == "nan" and frm == 0:
                pass
            # nan im aktuellen Zeitschritt und Identifikationsrahmen ist offen
            elif df.iloc[i][value_column] == "nan" and frm == 1:
                idx_end = i-1
                for i_frm in range(idx_strt, idx_end+1):
                    df.iloc[i_frm, df.columns.get_loc(value_column)] = "nan"  # Use proper column location
                    removed_count += 1
                frm = 0  # Identifikationsrahmen wird geschlossen
            # nan im letzten Zeitschritt
            elif df.iloc[i-1][value_column] == "nan":
                pass
            # Kein nan im letzten und aktuellen Zeitschritt
            else:
                # Änderung des Messwertes im aktuellen Zeitschritt
                chg = abs(float(df.iloc[i][value_column]) - float(df.iloc[i-1][value_column]))  # Keep original variable names
                # Zeitschrittweite vom letzten zum aktuellen Zeitschritt [min]
                t = (df.iloc[i]["UTC"] - df.iloc[i-1]["UTC"]).total_seconds() / 60  # Use proper column name
                # Minimal division by zero protection (t = 0 means identical timestamps)
                if t > 0 and chg/t > chg_max and frm == 0:
                    idx_strt = i
                    frm = 1
                elif t > 0 and chg/t > chg_max and frm == 1:
                    idx_end = i-1
                    for i_frm in range(idx_strt, idx_end+1):
                        df.iloc[i_frm, df.columns.get_loc(value_column)] = "nan"  # Use proper column location
                    frm = 0  # Identifikationsrahmen wird geschlossen
                elif frm == 1 and (df.iloc[i]["UTC"] - df.iloc[idx_strt]["UTC"]).total_seconds() / 60 > lg_max:
                    frm = 0
        emit_step_complete("Ausreisser eliminiert", f"Entfernt: {removed_count} Werte")
    else:
        emit_step_start("Eliminierung von Ausreissern", "Deaktiviert - übersprungen")
        emit_step_complete("✓ Ausreisser", "Keine Entfernung - Parameter nicht gesetzt")

    # STEP 7: SCHLIESSEN VON MESSLÜCKEN
    # Fill measurement gaps using linear interpolation for small gaps
    if params.get("gapMax"):
        emit_step_start("Schließen von Messlücken", f"Maximale Lücke: {params['gapMax']} min")
        logger.info("Schließen von Messlücken")
        gap_max = float(params["gapMax"])
        frm = 0  # Keep original variable name
        filled_count = 0
        for i in range(1, len(df)):
            # Kein Messwert für den aktuellen Zeitschritt vorhanden und Identifikationsrahmen ist geschlossen
            if df.iloc[i][value_column] == "nan" and frm == 0:
                idx_strt = i
                frm = 1
            # Messwert für den aktuellen Zeitschritt vorhanden und Identifikationsrahmen ist offen
            elif df.iloc[i][value_column] != "nan" and frm == 1:
                idx_end = i-1
                # Länge des Identifikationsrahmens [min] - use proper column name
                frm_width = (df.iloc[idx_end+1]["UTC"] - df.iloc[idx_strt-1]["UTC"]).total_seconds() / 60
                # Remove division by zero check - match original exactly
                if frm_width <= gap_max:
                    # Absolute Änderung des Messwertes
                    dif = float(df.iloc[idx_end+1][value_column]) - float(df.iloc[idx_strt-1][value_column])  # Use original variable names
                    # Änderung des Messwertes pro Minute
                    dif_min = dif/frm_width
                    # Lineare Interpolation
                    for i_frm in range(idx_strt, idx_end+1):
                        gap_min = (df.iloc[i_frm]["UTC"] - df.iloc[idx_strt-1]["UTC"]).total_seconds() / 60  # Use proper column name
                        df.iloc[i_frm, df.columns.get_loc(value_column)] = float(df.iloc[idx_strt-1][value_column]) + gap_min*dif_min  # Use proper column location
                        filled_count += 1
                frm = 0
        emit_step_complete("Messlücken geschlossen", f"Gefüllt: {filled_count} Werte")
    else:
        emit_step_start("Schließen von Messlücken", "Deaktiviert - übersprungen")
        emit_step_complete("✓ Messlücken", "Keine Füllung - Parameter nicht gesetzt")

    logger.info("Data cleaning completed")
    return df

@bp.route("/api/dataProcessingMain/upload-chunk", methods=["POST"])
def upload_chunk():
    """Handle chunked file upload with data processing"""
    try:
        # Get chunk information
        chunk_index = request.form.get('chunkIndex')
        total_chunks = request.form.get('totalChunks')
        upload_id = request.form.get("uploadId")
        
        # Emit chunk receive notification
        logger.info(f"Processing chunk {chunk_index}/{total_chunks}")
        if upload_id:
            chunk_progress = (int(chunk_index) / int(total_chunks)) * CHUNK_UPLOAD_MAX_PROGRESS
            emit_progress_with_fallback(upload_id, 'chunk_received', chunk_progress, 
                         f"Chunk {int(chunk_index)+1}/{total_chunks} empfangen")

        required_fields = ["uploadId", "chunkIndex", "totalChunks"]
        is_valid, error_msg = validate_required_fields(request.form, required_fields)
        if not is_valid:
            return jsonify({"error": error_msg}), 400

        upload_id = request.form["uploadId"]
        chunk_index = int(request.form["chunkIndex"])
        total_chunks = int(request.form["totalChunks"])
        
        if 'fileChunk' not in request.files:
            logger.error("No fileChunk in request.files")
            return jsonify({"error": "No file chunk provided"}), 400
            
        file_chunk = request.files['fileChunk']
        if file_chunk.filename == '':
            logger.error("Empty filename in fileChunk")
            return jsonify({"error": "Empty filename"}), 400

        # Read the chunk
        chunk = file_chunk.read()
        chunk_size = len(chunk)
        
        if chunk_size == 0:
            logger.error("Received empty chunk")
            return jsonify({"error": "Empty chunk received"}), 400

        # Create upload directory
        upload_dir = os.path.join(UPLOAD_FOLDER, upload_id)
        os.makedirs(upload_dir, exist_ok=True)
        chunk_path = os.path.join(upload_dir, f"chunk_{chunk_index:04d}.part")
        
        # Save the chunk
        with open(chunk_path, "wb") as f:
            f.write(chunk)

        if chunk_index < total_chunks - 1:
            return jsonify({"status": "chunk received", "chunkIndex": chunk_index})

        # Combine all chunks
        logger.info("Starting to combine all chunks")
        emit_progress_with_fallback(upload_id, 'combining', CHUNK_UPLOAD_MAX_PROGRESS, 'Combining uploaded chunks...')
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

        # Process the combined data
        try:
            emit_progress_with_fallback(upload_id, 'decoding', DATA_DECODING_PROGRESS, f'Dekodiere {total_size} Bytes...')
            content = all_bytes.decode("utf-8")
            emit_progress_with_fallback(upload_id, 'parsing', DATA_PARSING_PROGRESS, 'Parse CSV Daten...')
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

            emit_progress_with_fallback(upload_id, 'dataframe', DATAFRAME_CREATION_PROGRESS, f'Erstelle DataFrame mit {len(data)} Zeilen...')
            df = pd.DataFrame(data, columns=["UTC", value_column])
            # Convert comma to dot for decimal values but keep original format
            df[value_column] = df[value_column].str.replace(",", ".")
            
            emit_progress_with_fallback(upload_id, 'parameters', PARAMETER_LOADING_PROGRESS, 'Lade Bereinigungsparameter...')
            params = {
                "eqMax": request.form.get("eqMax"),
                "elMax": request.form.get("elMax"),
                "elMin": request.form.get("elMin"),
                "chgMax": request.form.get("chgMax"),
                "lgMax": request.form.get("lgMax"),
                "gapMax": request.form.get("gapMax"),
                "radioValueNull": request.form.get("radioValueNull"),
                "radioValueNotNull": request.form.get("radioValueNotNull")
            }

            emit_progress_with_fallback(upload_id, 'cleaning', DATA_CLEANING_START_PROGRESS, f'Cleaning data with {len(df)} rows...')
            df_clean = clean_data(df, value_column, params, emit_progress_with_fallback, upload_id)
            emit_progress_with_fallback(upload_id, 'cleaned', DATA_CLEANING_COMPLETE_PROGRESS, f'Data cleaning completed. Processing {len(df_clean)} rows...')

            def generate():
                """Generate streaming JSON response with processed data
                
                Yields processed data in JSON format line by line for efficient
                client-side consumption of large datasets.
                """
                # Emit progress for streaming start
                emit_progress_with_fallback(upload_id, 'streaming', STREAMING_START_PROGRESS, f'Starting to stream {len(df_clean)} processed rows...')
                
                # First send total rows
                yield json.dumps({"total_rows": len(df_clean)}, cls=CustomJSONEncoder) + "\n"
                
                # Process data in larger chunks
                chunk_size = STREAMING_CHUNK_SIZE
                for i in range(0, len(df_clean), chunk_size):
                    # Emit progress for chunk processing
                    chunk_progress = STREAMING_START_PROGRESS + ((i / len(df_clean)) * (STREAMING_COMPLETE_PROGRESS - STREAMING_START_PROGRESS))
                    emit_progress_with_fallback(upload_id, 'streaming', chunk_progress, f'Streaming chunk {i//chunk_size + 1}/{(len(df_clean)//chunk_size) + 1}...')
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
                        
                        # Handle "nan" string values properly
                        value = row[value_column]
                        if value == "nan" or (isinstance(value, float) and pd.isna(value)):
                            value = None
                        else:
                            try:
                                value = float(value)
                            except (ValueError, TypeError):
                                value = None
                            
                        record = {
                            'UTC': utc_value,
                            value_column: value
                        }
                        chunk_data.append(record)
                    
                    # Yield all records in the chunk at once
                    for record in chunk_data:
                        yield json.dumps(record, cls=CustomJSONEncoder) + "\n"
                
                # Send completion status
                emit_progress_with_fallback(upload_id, 'complete', COMPLETE_PROGRESS, f'Processing completed! Generated {len(df_clean)} data points.')
                yield json.dumps({"status": "complete"}, cls=CustomJSONEncoder) + "\n"
                        
            return Response(generate(), mimetype="application/x-ndjson")
        

        except Exception as e:
            error_response = handle_processing_error(upload_id, e, "bei Datenverarbeitung")
            return jsonify(error_response), 500
            
    except Exception as e:
        error_response = handle_processing_error(upload_id, e, "beim Upload")
        return jsonify(error_response), 500

@bp.route("/api/dataProcessingMain/prepare-save", methods=["POST"])
def prepare_save():
    """Prepare processed data for download by creating a temporary CSV file"""
    try:
        data = request.json.get("data")
        if not data:
            return jsonify({"error": "No data provided"}), 400

        file_name = data.get("fileName", "output.csv")
        rows = data.get("data", [])
        
        if not rows:
            return jsonify({"error": "No rows provided"}), 400

        logger.info(f"Preparing to save file: {file_name} with {len(rows)} rows")

        # Sanitize filename
        safe_file_name = os.path.basename(file_name)
        safe_file_name = safe_file_name.replace(" ", "_")
        file_path = os.path.join(tempfile.gettempdir(), safe_file_name)

        # Write CSV
        with open(file_path, "w", newline='', encoding='utf-8') as f:
            writer = csv.writer(f, delimiter=";")
            for row in rows:
                writer.writerow(row)

        logger.info(f"File prepared successfully with ID: {safe_file_name}")
        return jsonify({"fileId": safe_file_name})

    except Exception as e:
        logger.error(f"Error preparing save: {str(e)}")
        return jsonify({"error": f"Error preparing file: {str(e)}"}), 500


@bp.route("/api/dataProcessingMain/download/<file_id>", methods=["GET"])
def download_file(file_id: str):
    """Download a previously prepared CSV file"""
    try:
        path = os.path.join(tempfile.gettempdir(), file_id)
        if not os.path.exists(path):
            return jsonify({"error": "File not found"}), 404

        logger.info(f"Sending file: {path}")
        return send_file(
            path,
            as_attachment=True,
            download_name=file_id,
            mimetype="text/csv"
        )

    except Exception as e:
        logger.error(f"Error downloading file: {str(e)}")
        return jsonify({"error": f"Error downloading file: {str(e)}"}), 500
