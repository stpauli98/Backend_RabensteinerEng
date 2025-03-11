import os
import tempfile
import traceback
import logging
import json
import csv
import time
from io import StringIO
from datetime import datetime
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd

# Konfiguracija aplikacije i logginga
app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API configuration
API_PREFIX_LOAD_ROW_DATA = '/api/loadRowData'

# Globalni rječnici za privremene fajlove i chunk-ove
temp_files = {}
chunk_storage = {}

# Vrijeme nakon kojeg se brišu stari uploadi (30 minuta)
UPLOAD_EXPIRY_TIME = 30 * 60  # sekundi

def cleanup_old_uploads():
    """Briše stare uploade koji nisu završeni"""
    current_time = time.time()
    for upload_id in list(chunk_storage.keys()):
        upload_info = chunk_storage[upload_id]
        if current_time - upload_info.get('last_activity', 0) > UPLOAD_EXPIRY_TIME:
            del chunk_storage[upload_id]

# Lista podržanih formata datuma
SUPPORTED_DATE_FORMATS = [
    '%Y-%m-%dT%H:%M:%S%z',  # ISO format with timezone and seconds
    '%Y-%m-%dT%H:%M%z',     # ISO format with timezone
    '%Y-%m-%dT%H:%M:%S',    # ISO format without timezone
    '%Y-%m-%d %H:%M:%S',
    '%d.%m.%Y %H:%M',
    '%Y-%m-%d %H:%M',
    '%d.%m.%Y %H:%M:%S',
    '%Y/%m/%d %H:%M:%S',
    '%d/%m/%Y %H:%M:%S'
]
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

def clean_file_content(file_content, delimiter):
    """
    Uklanja višak delimitera i whitespace iz svake linije.
    """
    cleaned_lines = [line.rstrip(f"{delimiter};,") for line in file_content.splitlines()]
    return "\n".join(cleaned_lines)

def parse_datetime_column(df, datetime_col, custom_format=None):
    """
    Pokušava parsirati datetime kolonu pomoću custom formata (ako je zadan),
    automatski ili prema listi podržanih formata.
    Vraca tuple: (success: bool, parsed_dates: Series ili None, error_message: str ili None)
    """
    def clean_time(time_str):
        if not isinstance(time_str, str):
            return time_str
        # Očisti vrijeme od nevažećih znakova, zadrži samo brojeve i separatore
        cleaned = ''
        for c in time_str:
            if c.isdigit() or c in ':-+.T ':
                cleaned += c
        return cleaned

    # Očisti podatke prije parsiranja
    df = df.copy()
    df[datetime_col] = df[datetime_col].apply(clean_time)

    if custom_format:
        try:
            parsed_dates = pd.to_datetime(df[datetime_col], format=custom_format, errors='coerce')
            if parsed_dates.notna().any():
                logger.info(f"Successfully parsed using custom format: {custom_format}")
                return True, parsed_dates, None
        except Exception as e:
            logger.error(f"Error parsing datetime with custom format {custom_format}: {e}")

    # Prvo pokušaj sa podržanim formatima
    for fmt in SUPPORTED_DATE_FORMATS:
        try:
            parsed_dates = pd.to_datetime(df[datetime_col], format=fmt, errors='coerce')
            if parsed_dates.notna().any():
                logger.info(f"Successfully parsed dates using format: {fmt}")
                return True, parsed_dates, None
        except Exception as e:
            logger.debug(f"Failed to parse with format {fmt}: {str(e)}")
            continue

    # Ako podržani formati ne rade, probaj automatsko parsiranje
    try:
        parsed_dates = pd.to_datetime(df[datetime_col], errors='coerce')
        if parsed_dates.notna().any():
            logger.info("Successfully parsed dates using automatic detection")
            return True, parsed_dates, None
    except Exception as e:
        logger.error(f"Auto parsing of datetime failed: {e}")

    # Ako nijedan format ne odgovara, vrati NaT za sve vrijednosti
    parsed_dates = pd.Series([pd.NaT] * len(df), index=df.index)
    warning_msg = (
        f"Could not parse some datetime values in column '{datetime_col}'. "
        f"Invalid dates will be marked as NaT (Not a Time). Example value: '{df[datetime_col].iloc[0]}'"
    )
    return True, parsed_dates, warning_msg

def convert_to_utc(df, date_column, timezone='UTC'):
    """
    Konvertuje datetime kolonu u UTC.
    Ako datetime nema vremensku zonu, lokalizira ga prema zadanom timezone-u.
    """
    try:
        if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
            df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
        if df[date_column].dt.tz is None:
            if timezone.upper() == 'UTC':
                df[date_column] = df[date_column].dt.tz_localize('UTC', ambiguous='NaT', nonexistent='NaT')
            else:
                df[date_column] = df[date_column].dt.tz_localize(timezone, ambiguous='NaT', nonexistent='NaT')
                df[date_column] = df[date_column].dt.tz_convert('UTC')
        else:
            if str(df[date_column].dt.tz) != 'UTC':
                df[date_column] = df[date_column].dt.tz_convert('UTC')
        return df
    except Exception as e:
        logger.error(f"Error converting to UTC: {e}")
        raise

@app.route(f'{API_PREFIX_LOAD_ROW_DATA}/upload-chunk', methods=['POST'])
def upload_chunk(request):
    try:
        if 'fileChunk' not in request.files:
            return jsonify({"error": "Chunk file not found"}), 400
        
        required_params = ['uploadId', 'chunkIndex', 'totalChunks', 'delimiter', 'selected_columns', 'timezone', 'dropdown_count']
        missing_params = [param for param in required_params if param not in request.form]
        if missing_params:
            return jsonify({"error": f"Missing required parameters: {', '.join(missing_params)}"}), 400
        
        upload_id = request.form['uploadId']
        chunk_index = int(request.form['chunkIndex'])
        total_chunks = int(request.form['totalChunks'])
        file_chunk = request.files['fileChunk']
        
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
                        'selected_columns': selected_columns,
                        'custom_date_format': request.form.get('custom_date_format'),
                        'value_column_name': request.form.get('valueColumnName', '').strip(),
                        'dropdown_count': int(request.form.get('dropdown_count', '2'))
                    }
                }
            except json.JSONDecodeError as e:
                return jsonify({"error": "Invalid JSON format for selected_columns"}), 400
        
        chunk_content = file_chunk.read().decode('utf-8')
        chunk_storage[upload_id]['chunks'][chunk_index] = chunk_content
        chunk_storage[upload_id]['received_chunks'] += 1
        chunk_storage[upload_id]['last_activity'] = time.time()
        
        if chunk_storage[upload_id]['received_chunks'] == total_chunks:
            return process_chunks(upload_id)
        
        return jsonify({
            "message": f"Chunk {chunk_index + 1}/{total_chunks} received",
            "uploadId": upload_id,
            "remainingChunks": total_chunks - chunk_storage[upload_id]['received_chunks']
        }), 200
    except Exception as e:
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 400

def process_chunks(upload_id):
    try:
        chunks = [chunk_storage[upload_id]['chunks'][i] for i in range(chunk_storage[upload_id]['total_chunks'])]
        full_content = "".join(chunks)
        params = chunk_storage[upload_id]['parameters']
        del chunk_storage[upload_id]
        
        return upload_files(full_content, params)
    except Exception as e:
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 400

    try:
        delimiter = params['delimiter']
        timezone = params['timezone']
        selected_columns = params['selected_columns']
        custom_date_format = params['custom_date_format']
        value_column_name = params['value_column_name']
        dropdown_count = params['dropdown_count']
        has_separate_date_time = dropdown_count == 3

        # Očekivani nazivi kolona prema odabiru
        date_column = selected_columns.get('column1')
        time_column = selected_columns.get('column2') if has_separate_date_time else None
        value_column = selected_columns.get('column3') if has_separate_date_time else selected_columns.get('column2')

        # Detektiraj i provjeri delimiter
        detected_delimiter = detect_delimiter(file_content)
        if delimiter != detected_delimiter:
            return jsonify({
                "error": f"Incorrect delimiter! Detected: '{detected_delimiter}', provided: '{delimiter}'."
            }), 400

        # Očisti sadržaj i učitaj u DataFrame
        cleaned_content = clean_file_content(file_content, delimiter)
        df = pd.read_csv(StringIO(cleaned_content), delimiter=delimiter)
        df = df.dropna(axis=1, how='all')
        df.columns = df.columns.str.strip()

        # Pretvaranje vrijednosti u numerički tip ako je moguće
        if value_column and value_column in df.columns:
            df[value_column] = pd.to_numeric(df[value_column], errors='coerce')

        # Parsiranje datuma i/ili vremena
        if has_separate_date_time and date_column and time_column:
            try:
                # Očisti vrijeme od nevažećih znakova
                def clean_time(time_str):
                    if not isinstance(time_str, str):
                        return time_str
                    cleaned = ''.join(c for c in str(time_str) if c.isdigit() or c in ':.,')                    
                    parts = cleaned.split(':')
                    if len(parts) > 1:
                        seconds_parts = parts[-1].replace('.', ',').split(',', 1)
                        if len(seconds_parts) > 1:
                            parts[-1] = seconds_parts[0] + '.' + seconds_parts[1]
                        cleaned = ':'.join(parts)
                    return cleaned

                # Očisti i kombinuj datum i vrijeme
                cleaned_time = df[time_column].apply(clean_time)
                combined = df[date_column].astype(str) + " " + cleaned_time

                if custom_date_format:
                    try:
                        if '%f' in custom_date_format:
                            def extend_microseconds(x):
                                parts = x.split('.')
                                if len(parts) > 1:
                                    return parts[0] + '.' + parts[1].ljust(6, '0')
                                return x
                            combined = combined.apply(extend_microseconds)
                        
                        df['datetime'] = pd.to_datetime(combined, format=custom_date_format, errors='coerce')
                        if df['datetime'].isna().all():
                            return jsonify({
                                "error": f"Could not parse any dates with format '{custom_date_format}'. Example value: {combined.iloc[0]}",
                                "needs_custom_format": True,
                                "supported_formats": SUPPORTED_DATE_FORMATS
                            }), 400
                    except Exception as e:
                        return jsonify({
                            "error": f"Error parsing with format '{custom_date_format}': {str(e)}. Example value: {combined.iloc[0]}",
                            "needs_custom_format": True,
                            "supported_formats": SUPPORTED_DATE_FORMATS
                        }), 400
                else:
                    try:
                        df['datetime'] = pd.to_datetime(combined, errors='coerce')
                        if df['datetime'].isna().all():
                            return jsonify({
                                "error": "Could not automatically parse dates. Please provide a custom format.",
                                "needs_custom_format": True,
                                "supported_formats": SUPPORTED_DATE_FORMATS,
                                "example_value": combined.iloc[0] if not combined.empty else ""
                            }), 400
                    except Exception:
                        return jsonify({
                            "error": "Could not parse dates. Please provide a custom format.",
                            "needs_custom_format": True,
                            "supported_formats": SUPPORTED_DATE_FORMATS,
                            "example_value": combined.iloc[0] if not combined.empty else ""
                        }), 400
            except Exception as e:
                return jsonify({
                    "error": f"Error combining date and time: {str(e)}",
                    "needs_custom_format": True
                }), 400
        else:
            # Ako je samo jedna datetime kolona
            datetime_col = date_column or df.columns[0]
            success, parsed_dates, err = parse_datetime_column(df, datetime_col, custom_format=custom_date_format)
            if not success:
                return jsonify({
                    "error": err,
                    "needs_custom_format": True,
                    "supported_formats": SUPPORTED_DATE_FORMATS
                }), 400
            df['datetime'] = parsed_dates

        # Konverzija u UTC
        try:
            df = convert_to_utc(df, 'datetime', timezone)
        except Exception as e:
            return jsonify({"error": f"Error converting to UTC: {str(e)}"}), 400

        # Provjera postojanja value kolone i priprema rezultata
        if not value_column or value_column not in df.columns:
            return jsonify({"error": f"Value column '{value_column}' not found in data"}), 400

        result_df = pd.DataFrame()
        result_df['UTC'] = df['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
        final_value_column = value_column_name if value_column_name else value_column
        result_df[final_value_column] = df[value_column].apply(lambda x: str(x) if pd.notnull(x) else "")
        result_df.dropna(subset=['UTC'], inplace=True)
        result_df.sort_values('UTC', inplace=True)

        from flask import Response, stream_with_context
        
        def generate_chunks():
            CHUNK_SIZE = 1000  # Broj redova po chunk-u
            total_rows = len(result_df)
            total_chunks = (total_rows + CHUNK_SIZE - 1) // CHUNK_SIZE
            
            # Pošalji metapodatke
            yield json.dumps({
                'total_rows': total_rows,
                'total_chunks': total_chunks,
                'chunk_size': CHUNK_SIZE,
                'message': 'Daten werden gestreamt'
            }) + '\n'
            
            # Pošalji podatke u delovima
            for chunk_idx in range(total_chunks):
                start_idx = chunk_idx * CHUNK_SIZE
                end_idx = min(start_idx + CHUNK_SIZE, total_rows)
                
                chunk_data = {
                    'chunk_index': chunk_idx,
                    'data': df.iloc[start_idx:end_idx].to_dict('records')
                }
                
                yield json.dumps(chunk_data) + '\n'
            
            # Pošalji završnu poruku
            yield json.dumps({
                'message': 'Daten wurden erfolgreich verarbeitet',
                'status': 'complete'
            }) + '\n'
        
        return Response(
            stream_with_context(generate_chunks()),
            mimetype='application/x-ndjson'
        )

    except Exception as e:
        response = jsonify({'error': str(e)})
        return response, 400

@app.route(f'{API_PREFIX_LOAD_ROW_DATA}/upload-status/<upload_id>', methods=['GET'])
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

@app.route(f'{API_PREFIX_LOAD_ROW_DATA}/upload', methods=['POST'])
def upload_files(file_content, params):
    try:
        # Preuzimanje parametara iz memorije umesto request.form
        delimiter = params.get('delimiter')
        if not delimiter:
            return jsonify({"error": "Delimiter not provided"}), 400
        timezone = params.get('timezone', 'UTC')
        selected_columns = params.get('selected_columns', {})
        custom_date_format = params.get('custom_date_format')
        value_column_name = params.get('value_column_name', '').strip()
        dropdown_count = int(params.get('dropdown_count', '2'))
        has_separate_date_time = dropdown_count == 3

        # Očekivani nazivi kolona prema odabiru
        date_column = selected_columns.get('column1')
        time_column = selected_columns.get('column2') if has_separate_date_time else None
        value_column = selected_columns.get('column3') if has_separate_date_time else selected_columns.get('column2')

        # Pročitaj CSV sadržaj iz primljenog stringa
        detected_delimiter = detect_delimiter(file_content)
        if delimiter != detected_delimiter:
            return jsonify({"error": f"Incorrect delimiter! Detected: '{detected_delimiter}', provided: '{delimiter}'"}), 400

        # Očisti sadržaj i učitaj u DataFrame
        cleaned_content = clean_file_content(file_content, delimiter)
        try:
            df = pd.read_csv(StringIO(cleaned_content), delimiter=delimiter)
            df = df.dropna(axis=1, how='all')
            df.columns = df.columns.str.strip()
        except Exception as e:
            return jsonify({"error": f"Error processing CSV: {str(e)}"}), 400

        if df.empty:
            return jsonify({"error": "No data loaded from file"}), 400

        # Pretvaranje vrijednosti u numerički tip ako je moguće
        if value_column and value_column in df.columns:
            df[value_column] = pd.to_numeric(df[value_column], errors='coerce')

        # Parsiranje datuma i vremena
        if has_separate_date_time and date_column and time_column:
            try:
                df = parse_datetime(df, date_column, time_column, custom_date_format)
            except Exception as e:
                return jsonify({"error": f"Error parsing date/time: {str(e)}"}), 400
        else:
            datetime_col = date_column or df.columns[0]
            success, parsed_dates, err = parse_datetime_column(df, datetime_col, custom_format=custom_date_format)
            if not success:
                return jsonify({"error": err}), 400
            df['datetime'] = parsed_dates

        # Konverzija u UTC
        try:
            df = convert_to_utc(df, 'datetime', timezone)
        except Exception as e:
            return jsonify({"error": f"Error converting to UTC: {str(e)}"}), 400

        # Provera postojanja value kolone
        if not value_column or value_column not in df.columns:
            return jsonify({"error": f"Value column '{value_column}' not found in data"}), 400

        result_df = pd.DataFrame()
        result_df['UTC'] = df['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
        final_value_column = value_column_name if value_column_name else value_column
        result_df[final_value_column] = df[value_column].apply(lambda x: str(x) if pd.notnull(x) else "")
        result_df.dropna(subset=['UTC'], inplace=True)
        result_df.sort_values('UTC', inplace=True)

        # Pretvori DataFrame u listu listi (prvi red su headeri)
        headers = result_df.columns.tolist()
        data_list = [headers] + result_df.values.tolist()
        return jsonify({"data": data_list, "fullData": data_list})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

def prepare_save(request):
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

@app.route(f'{API_PREFIX_LOAD_ROW_DATA}/download/<file_id>', methods=['GET'])
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
