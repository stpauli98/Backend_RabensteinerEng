import os
import tempfile
import traceback
import logging
import json
import csv
import time
from io import StringIO
from datetime import datetime
from flask import Blueprint, request, jsonify, send_file, current_app, g
import pandas as pd
from flask_socketio import join_room
from middleware.auth import require_auth
from middleware.subscription import require_subscription, check_processing_limit
from utils.usage_tracking import increment_processing_count, update_storage_usage

def get_socketio():
    return current_app.extensions['socketio']


bp = Blueprint('load_row_data', __name__)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

temp_files = {}
chunk_storage = {}

UPLOAD_EXPIRY_TIME = 30 * 60

def cleanup_old_uploads():
    """Briše stare uploade koji nisu završeni"""
    current_time = time.time()
    for upload_id in list(chunk_storage.keys()):
        upload_info = chunk_storage[upload_id]
        if current_time - upload_info.get('last_activity', 0) > UPLOAD_EXPIRY_TIME:
            del chunk_storage[upload_id]

SUPPORTED_DATE_FORMATS = [
    '%Y-%m-%dT%H:%M:%S%z',
    '%Y-%m-%dT%H:%M%z',
    '%Y-%m-%dT%H:%M:%S',
    '%Y-%m-%d %H:%M:%S',
    '%d.%m.%Y %H:%M',
    '%Y-%m-%d %H:%M',
    '%d.%m.%Y %H:%M:%S',
    '%d.%m.%Y %H:%M:%S.%f',
    '%Y/%m/%d %H:%M:%S',
    '%d/%m/%Y %H:%M:%S',
    '%Y/%m/%d',
    '%d/%m/%Y',
    '%d-%m-%Y %H:%M:%S',
    '%d-%m-%Y %H:%M',
    '%Y/%m/%d %H:%M',
    '%d/%m/%Y %H:%M',
    '%d-%m-%Y',
    '%H:%M:%S',
    '%H:%M'
]

def check_date_format(sample_date):
    """
    Proverava da li je format datuma podržan.
    Returns:
        tuple: (bool, str) - (da li je format podržan, poruka o grešci)
    """
    if not isinstance(sample_date, str):
        sample_date = str(sample_date)
    
    for fmt in SUPPORTED_DATE_FORMATS:
        try:
            pd.to_datetime(sample_date, format=fmt)
            return True, None
        except ValueError:
            continue
    
    return False, {
        "error": "UNSUPPORTED_DATE_FORMAT",
        "message": f"Format nicht unterstützt. Beispielwert: %d.%m.%Y %H:%M:%S"
    }

def detect_delimiter(file_content, sample_lines=3):
    """
    Detektira delimiter na osnovu prvih nekoliko linija sadržaja fajla.
    """
    delimiters = [',', ';', '\t']
    lines = file_content.splitlines()[:sample_lines]
    counts = {d: sum(line.count(d) for line in lines) for d in delimiters}
    if max(counts.values()) > 0:
        return max(counts, key=counts.get)
    return ','

def clean_time(time_str):
    """
    Cleans time string by keeping only valid characters (numbers and separators).
    Example: '00:00:00.000Kdd' -> '00:00:00.000'
    """
    if not isinstance(time_str, str):
        return time_str
    
    cleaned = ''
    for c in str(time_str):
        if c.isdigit() or c in ':-+.T ':
            cleaned += c
    return cleaned

def clean_file_content(file_content, delimiter):
    """
    Uklanja višak delimitera i whitespace iz svake linije.
    """
    cleaned_lines = [line.rstrip(f"{delimiter};,") for line in file_content.splitlines()]
    return "\n".join(cleaned_lines)

def parse_datetime_column(df, datetime_col, custom_format=None):
    """
    Pokušava parsirati datetime kolonu pomoću custom formata ili podržanih formata.
    Vraca tuple: (success: bool, parsed_dates: Series ili None, error_message: str ili None)
    """
    try:
        df = df.copy()
        df[datetime_col] = df[datetime_col].astype(str).str.strip()
        sample_datetime = df[datetime_col].iloc[0]

        if custom_format:
            try:
                parsed_dates = pd.to_datetime(df[datetime_col], format=custom_format, errors='coerce')
                if not parsed_dates.isna().all():
                    return True, parsed_dates, None
            except Exception as e:
                return False, None, f"Fehler mit custom Format: {str(e)}. Beispielwert: {sample_datetime}"

        if SUPPORTED_DATE_FORMATS:
            for fmt in SUPPORTED_DATE_FORMATS:
                try:
                    parsed_dates = pd.to_datetime(df[datetime_col], format=fmt, errors='coerce')
                    if not parsed_dates.isna().all():
                        return True, parsed_dates, None
                except Exception:
                    continue

        return False, None, f"Format nicht unterstützt. Beispielwert: %d.%m.%Y %H:%M:%S"

    except Exception as e:
        return False, None, f"Fehler beim Parsen: {str(e)}"

def is_format_supported(value, formats):
    """
    Proverava da li je vrednost u nekom od podržanih formata.
    """
    if not isinstance(value, str):
        value = str(value)
    
    for fmt in formats:
        try:
            pd.to_datetime(value, format=fmt)
            return True, fmt
        except ValueError:
            continue
    return False, None

def parse_datetime(df, date_column, time_column, custom_format=None):
    """
    Kombinuje odvojene kolone datuma i vremena u jednu datetime kolonu.
    Args:
        df: DataFrame sa podacima
        date_column: Ime kolone sa datumom
        time_column: Ime kolone sa vremenom
        custom_format: Opcioni custom format za parsiranje
    Returns:
        DataFrame sa dodatom 'datetime' kolonom
    """
    try:
        df = df.copy()
        
        df['datetime'] = df[date_column].astype(str).str.strip() + ' ' + df[time_column].astype(str).str.strip()
        sample_datetime = df['datetime'].iloc[0]
        
        if custom_format:
            try:
                df['datetime'] = pd.to_datetime(df['datetime'], format=custom_format, errors='coerce')
                if not df['datetime'].isna().all():
                    return df
            except Exception as e:
                return jsonify({
                    "error": "CUSTOM_FORMAT_ERROR",
                    "message": f"Fehler mit custom Format: {str(e)}. Beispielwert: {sample_datetime}"
                }), 400
        
        is_supported, detected_format = is_format_supported(sample_datetime, SUPPORTED_DATE_FORMATS)
        if not is_supported:
            return jsonify({
                "error": "UNSUPPORTED_DATE_FORMAT",
                "message": f"Format nicht unterstützt. Beispielwert: %d.%m.%Y %H:%M:%S"
            }), 400
            
        try:
            df['datetime'] = pd.to_datetime(df['datetime'], format=detected_format, errors='coerce')
            if df['datetime'].isna().all():
                return jsonify({
                    "error": "INVALID_DATE_FORMAT",
                    "message": f"Ungültiges Datumsformat. Beispielwert: {sample_datetime}"
                }), 400
        except Exception as e:
            return jsonify({
                "error": "DATE_PARSING_ERROR",
                "message": f"Fehler beim Parsen: {str(e)}"
            }), 400
        
        return df
        
    except Exception as e:
        raise ValueError(f'Fehler beim Parsen von Datum/Zeit: {str(e)}')

def validate_datetime_format(datetime_str):
    """
    Proverava da li je format datuma i vremena podržan.
    """
    if not isinstance(datetime_str, str):
        datetime_str = str(datetime_str)
    
    for fmt in SUPPORTED_DATE_FORMATS:
        try:
            pd.to_datetime(datetime_str, format=fmt)
            return True
        except ValueError:
            continue
    return False

def convert_to_utc(df, date_column, timezone='UTC'):
    """
    Konvertuje datetime kolonu u UTC.
    Ako datetime nema vremensku zonu, lokalizira ga prema zadanom timezone-u.
    """
    try:
        if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
            df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
        if df[date_column].dt.tz is None:
            try:
                df[date_column] = df[date_column].dt.tz_localize(timezone, ambiguous='NaT', nonexistent='NaT')
            except Exception as e:
                logger.error(f"Nicht unterstützte Zeitzone '{timezone}': {e}")
                raise ValueError(f"Nicht unterstützte Zeitzone '{timezone}'. Bitte prüfen Sie die Eingabe auf der Frontend-Seite.")
            if timezone.upper() != 'UTC':
                df[date_column] = df[date_column].dt.tz_convert('UTC')
        else:
            if str(df[date_column].dt.tz) != 'UTC':
                df[date_column] = df[date_column].dt.tz_convert('UTC')
        return df
    except Exception as e:
        logger.error(f"Error converting to UTC: {e}")
        raise

@bp.route('/upload-chunk', methods=['POST'])
@require_auth
def upload_chunk():
    try:
        if 'fileChunk' not in request.files:
            return jsonify({"error": "Chunk file not found"}), 400
        
        required_params = ['uploadId', 'chunkIndex', 'totalChunks', 'delimiter', 'selected_columns', 'timezone', 'dropdown_count', 'hasHeader']
        missing_params = [param for param in required_params if param not in request.form]
        if missing_params:
            return jsonify({"error": f"Missing required parameters: {', '.join(missing_params)}"}), 400
        
        upload_id = request.form['uploadId']
        chunk_index = int(request.form['chunkIndex'])
        total_chunks = int(request.form['totalChunks'])
        
        if upload_id not in chunk_storage:
            try:
                selected_columns_str = request.form.get('selected_columns')
                if not selected_columns_str:
                    return jsonify({"error": "selected_columns parameter is required"}), 400
                selected_columns = json.loads(selected_columns_str)
                if not isinstance(selected_columns, dict):
                    return jsonify({"error": "selected_columns must be a JSON object"}), 400
                
                chunk_storage[upload_id] = {
                    'chunks': {},
                    'total_chunks': total_chunks,
                    'received_chunks': 0,
                    'last_activity': time.time(),
                    'parameters': {
                        'delimiter': request.form.get('delimiter'),
                        'timezone': request.form.get('timezone', 'UTC'),
                        'has_header': request.form.get('hasHeader', 'nein'),
                        'selected_columns': selected_columns,
                        'custom_date_format': request.form.get('custom_date_format'),
                        'value_column_name': request.form.get('valueColumnName', '').strip(),
                        'dropdown_count': int(request.form.get('dropdown_count', '2'))
                    }
                }
            except json.JSONDecodeError as e:
                return jsonify({"error": "Invalid JSON format for selected_columns"}), 400
        else:
            chunk_storage[upload_id]['total_chunks'] = max(chunk_storage[upload_id]['total_chunks'], total_chunks)

        file_chunk = request.files['fileChunk']
        chunk_content = file_chunk.read()
        chunk_storage[upload_id]['chunks'][chunk_index] = chunk_content
        chunk_storage[upload_id]['received_chunks'] += 1
        chunk_storage[upload_id]['last_activity'] = time.time()
        
        progress_percentage = int((chunk_index + 1) / chunk_storage[upload_id]['total_chunks'] * 100)
        socketio = get_socketio()
        socketio.emit('upload_progress', {
            'uploadId': upload_id,
            'fileName': file_chunk.filename or 'unknown',
            'progress': progress_percentage,
            'status': 'uploading',
            'message': f'Processing chunk {chunk_index + 1}/{chunk_storage[upload_id]["total_chunks"]}'
        }, room=upload_id)
        
        if chunk_storage[upload_id]['received_chunks'] == chunk_storage[upload_id]['total_chunks']:
            socketio.emit('upload_progress', {
                'uploadId': upload_id,
                'progress': 100,
                'status': 'uploading',
                'message': 'All chunks received, waiting for finalization...'
            }, room=upload_id)
        
        return jsonify({
            "message": f"Chunk {chunk_index + 1}/{chunk_storage[upload_id]['total_chunks']} received",
            "uploadId": upload_id,
            "remainingChunks": chunk_storage[upload_id]['total_chunks'] - chunk_storage[upload_id]['received_chunks']
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@bp.route('/finalize-upload', methods=['POST'])
@require_auth
@require_subscription
@check_processing_limit
def finalize_upload():
    try:
        data = request.get_json(force=True, silent=True)
        
        if not data or 'uploadId' not in data:
            return jsonify({"error": "uploadId is required"}), 400
            
        upload_id = data['uploadId']
        
        if upload_id not in chunk_storage:
            return jsonify({"error": "Upload not found or already processed"}), 404
            
        if chunk_storage[upload_id]['received_chunks'] != chunk_storage[upload_id]['total_chunks']:
            remaining = chunk_storage[upload_id]['total_chunks'] - chunk_storage[upload_id]['received_chunks']
            return jsonify({
                "error": f"Not all chunks received. Missing {remaining} chunks.",
                "received": chunk_storage[upload_id]['received_chunks'],
                "total": chunk_storage[upload_id]['total_chunks']
            }), 400
            
        return process_chunks(upload_id)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@bp.route('/cancel-upload', methods=['POST'])
@require_auth
def cancel_upload():
    try:
        data = request.get_json(force=True, silent=True)
        
        if not data or 'uploadId' not in data:
            return jsonify({"error": "uploadId is required"}), 400
            
        upload_id = data['uploadId']
        
        if upload_id in chunk_storage:
            del chunk_storage[upload_id]
            
            socketio = get_socketio()
            socketio.emit('upload_progress', {
                'uploadId': upload_id,
                'progress': 0,
                'status': 'error',
                'message': 'Upload canceled by user'
            }, room=upload_id)
        
        return jsonify({
            "success": True,
            "message": "Upload canceled successfully"
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400

def process_chunks(upload_id):
    try:
        socketio = get_socketio()
        socketio.emit('upload_progress', {
            'uploadId': upload_id,
            'progress': 50,
            'status': 'processing',
            'message': 'Processing file data...'
        }, room=upload_id)
        
        chunks = [chunk_storage[upload_id]['chunks'][i] for i in range(chunk_storage[upload_id]['total_chunks'])]
        
        encodings = ['utf-8', 'utf-16', 'utf-16le', 'utf-16be', 'latin1', 'cp1252']
        full_content = None
        
        for encoding in encodings:
            try:
                decoded_chunks = [chunk.decode(encoding) for chunk in chunks]
                full_content = "".join(decoded_chunks)
                break
            except UnicodeDecodeError:
                continue
        
        if full_content is None:
            socketio.emit('upload_progress', {
                'uploadId': upload_id,
                'progress': 0,
                'status': 'error',
                'message': 'Could not decode file content with any supported encoding'
            }, room=upload_id)
            return jsonify({"error": "Could not decode file content with any supported encoding"}), 400

        num_lines = len(full_content.split('\n'))
        params = chunk_storage[upload_id]['parameters']
        params['uploadId'] = upload_id
        del chunk_storage[upload_id]
        
        socketio.emit('upload_progress', {
            'uploadId': upload_id,
            'progress': 100,
            'status': 'completed',
            'message': 'File processing completed'
        }, room=upload_id)
        
        return upload_files(full_content, params)
    except Exception as e:
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 400

def check_upload_status(upload_id):
    try:
        if upload_id not in chunk_storage:
            return jsonify({
                "error": "Upload not found or already completed"
            }), 404
            
        upload_info = chunk_storage[upload_id]
        return jsonify({
            "success": True,
            "totalChunks": upload_info['total_chunks'],
            "receivedChunks": upload_info['received_chunks'],
            "isComplete": upload_info['received_chunks'] == upload_info['total_chunks']
        })
        
    except Exception as e:
        logger.error(f"Error checking upload status: {str(e)}")
        return jsonify({"error": str(e)}), 400

def upload_files(file_content, params):
    try:
        upload_id = params.get('uploadId')
        
        socketio = get_socketio()
        socketio.emit('upload_progress', {
            'uploadId': upload_id,
            'progress': 60,
            'status': 'processing',
            'message': 'Parsing CSV data...'
        }, room=upload_id)
        
        delimiter = params.get('delimiter')
        if not delimiter:
            return jsonify({"error": "Delimiter not provided"}), 400
        timezone = params.get('timezone', 'UTC')
        selected_columns = params.get('selected_columns', {})
        custom_date_format = params.get('custom_date_format')
        value_column_name = params.get('value_column_name', '').strip()
        dropdown_count = int(params.get('dropdown_count', '2'))
        has_separate_date_time = dropdown_count == 3
        has_header = params.get('has_header', False)

        date_column = selected_columns.get('column1')
        time_column = selected_columns.get('column2') if has_separate_date_time else None
        value_column = selected_columns.get('column3') if has_separate_date_time else selected_columns.get('column2') 


        detected_delimiter = detect_delimiter(file_content)
        if delimiter != detected_delimiter:
            return jsonify({"error": f"Incorrect delimiter! Detected: '{detected_delimiter}', provided: '{delimiter}'"}), 400

        cleaned_content = clean_file_content(file_content, delimiter)
        try:
            df = pd.read_csv(StringIO(cleaned_content),
                delimiter=delimiter,
                header=0 if has_header == 'ja' else None)
            
            if has_header == 'nein':
                df.columns = [str(i) for i in range(len(df.columns))]
            else:
                df.columns = [col.strip() for col in df.columns]
            
            df = df.dropna(axis=1, how='all')
            
            df.columns = df.columns.astype(str)
            
            df.columns = [col.strip() for col in df.columns]
        except Exception as e:
            return jsonify({"error": f"Error processing CSV: {str(e)}"}), 400

        if df.empty:
            return jsonify({"error": "No data loaded from file"}), 400

        if value_column and value_column in df.columns:
            df[value_column] = pd.to_numeric(df[value_column], errors='coerce')

        try:
            datetime_col = date_column or df.columns[0]
            
            if has_separate_date_time and date_column and time_column:
                df[time_column] = df[time_column].apply(clean_time)
                
                df[date_column] = df[date_column].apply(clean_time)
                
                df['datetime'] = df[date_column].astype(str) + ' ' + df[time_column].astype(str)
                
                success, parsed_dates, err = parse_datetime_column(df, 'datetime')
                
                if not success and custom_date_format:
                    success, parsed_dates, err = parse_datetime_column(df, 'datetime', custom_format=custom_date_format)
                
                if not success:
                    return jsonify({
                        "error": "UNSUPPORTED_DATE_FORMAT",
                        "message": f"Format nicht unterstützt. Beispielwert: %d.%m.%Y %H:%M:%S"
                    }), 400
            else:
                success, parsed_dates, err = parse_datetime_column(df, datetime_col, custom_format=custom_date_format)
                if not success:
                    return jsonify({
                        "error": "UNSUPPORTED_DATE_FORMAT",
                        "message": err
                    }), 400
                
            df['datetime'] = parsed_dates
        except Exception as e:
            return jsonify({"error": f"Error parsing date/time: {str(e)}"}), 400

        try:
            socketio.emit('upload_progress', {
                'uploadId': upload_id,
                'progress': 80,
                'status': 'processing',
                'message': 'Converting to UTC...'
            }, room=upload_id)
                
            df = convert_to_utc(df, 'datetime', timezone)
        except Exception as e:
            socketio.emit('upload_progress', {
                'uploadId': upload_id,
                'progress': 0,
                'status': 'error',
                'message': f'Error: {str(e)}'
            }, room=upload_id)
                
            return jsonify({
                "error": "Überprüfe dein Datumsformat eingabe",
                "message": f"Fehler bei der Konvertierung in UTC: {str(e)}"
            }), 400

        if not value_column or value_column not in df.columns:
            return jsonify({"error": f"Datum, Wert 1 oder Wert 2 nicht ausgewählt"}), 400

        result_df = pd.DataFrame()
        result_df['UTC'] = df['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
        final_value_column = value_column_name if value_column_name else value_column
        result_df[final_value_column] = df[value_column].apply(lambda x: str(x) if pd.notnull(x) else "")
        result_df.dropna(subset=['UTC'], inplace=True)
        result_df.sort_values('UTC', inplace=True)

        headers = result_df.columns.tolist()
        data_list = [headers] + result_df.values.tolist()
        
        socketio.emit('upload_progress', {
            'uploadId': upload_id,
            'progress': 100,
            'status': 'completed',
            'message': 'Data processing completed'
        }, room=upload_id)

        # Track operations (processing) and storage usage
        try:
            # Increment operations count (replaces upload count)
            increment_processing_count(g.user_id)
            logger.info(f"✅ Tracked operation (processing) for user {g.user_id}")

            # Calculate file size in MB from file_content
            file_size_bytes = len(file_content.encode('utf-8'))
            file_size_mb = file_size_bytes / (1024 * 1024)

            update_storage_usage(g.user_id, file_size_mb)
            logger.info(f"✅ Tracked storage for user {g.user_id}: {file_size_mb:.2f} MB")
        except Exception as e:
            logger.error(f"⚠️ Failed to track usage: {str(e)}")
            # Don't fail the upload if tracking fails

        return jsonify({"data": data_list, "fullData": data_list})
    except Exception as e:
        socketio.emit('upload_progress', {
            'uploadId': params['uploadId'],
            'progress': 0,
            'status': 'error',
            'message': f'Error: {str(e)}'
        }, room=params['uploadId'])
            
        return jsonify({"error": str(e)}), 400

@bp.route('/prepare-save', methods=['POST'])
@require_auth
def prepare_save():
    try:
        data = request.json

        if not data or 'data' not in data:
            return jsonify({"error": "No data received"}), 400
        
        data_wrapper = data['data']
        save_data = data_wrapper.get('data', [])
        file_name = data_wrapper.get('fileName', '')

        logger.info(f"IME : Received data for file: {file_name}")
        
        if not save_data:
            return jsonify({"error": "Empty data"}), 400

        temp_file = tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.csv')
        writer = csv.writer(temp_file, delimiter=';')
        for row in save_data:
            writer.writerow(row)
        temp_file.close()

        file_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        temp_files[file_id] = {
            'path': temp_file.name,
            'fileName': file_name or f"data_{file_id}.csv",  
            'timestamp': time.time()
        }

        return jsonify({"message": "File prepared for download", "fileId": file_id}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@bp.route('/download/<file_id>', methods=['GET'])
@require_auth
def download_file(file_id):
    try:
        if file_id not in temp_files:
            return jsonify({"error": "File not found"}), 404
        file_info = temp_files[file_id]
        file_path = file_info['path']
        if not os.path.exists(file_path):
            return jsonify({"error": "File not found"}), 404

        download_name = file_info['fileName']
        return send_file(
            file_path,
            as_attachment=True,
            download_name=download_name,
            mimetype='text/csv'
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if file_id in temp_files:
            try:
                os.unlink(file_info['path'])
                del temp_files[file_id]
            except Exception as ex:
                return jsonify({"error": str(ex)}), 500
