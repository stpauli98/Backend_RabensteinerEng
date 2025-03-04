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
def upload_chunk():
    try:
        # Provjeri da li su svi potrebni parametri prisutni
        if 'fileChunk' not in request.files:
            return jsonify({"error": "No file chunk provided"}), 400
            
        upload_id = request.form.get('uploadId')
        chunk_index = request.form.get('chunkIndex')
        total_chunks = request.form.get('totalChunks')
        
        if not all([upload_id, chunk_index, total_chunks]):
            return jsonify({"error": "Missing required parameters"}), 400
            
        chunk_index = int(chunk_index)
        total_chunks = int(total_chunks)
        
        # Očisti stare uploade
        cleanup_old_uploads()
        
        # Inicijaliziraj storage za ovaj upload ako ne postoji
        if upload_id not in chunk_storage:
            chunk_storage[upload_id] = {
                'chunks': {},
                'total_chunks': total_chunks,
                'received_chunks': 0,
                'last_activity': time.time(),
                'parameters': {
                    'delimiter': request.form.get('delimiter'),
                    'timezone': request.form.get('timezone', 'UTC'),
                    'selected_columns': json.loads(request.form.get('selected_columns', '{}')),
                    'custom_date_format': request.form.get('custom_date_format'),
                    'value_column_name': request.form.get('valueColumnName', '').strip(),
                    'dropdown_count': int(request.form.get('dropdown_count', '2'))
                }
            }
        
        # Spremi chunk i ažuriraj vrijeme zadnje aktivnosti
        file_chunk = request.files['fileChunk']
        chunk_content = file_chunk.read().decode('utf-8')
        chunk_storage[upload_id]['chunks'][chunk_index] = chunk_content
        chunk_storage[upload_id]['received_chunks'] += 1
        chunk_storage[upload_id]['last_activity'] = time.time()
        
        # Provjeri da li smo primili sve chunk-ove
        if chunk_storage[upload_id]['received_chunks'] == total_chunks:
            # Spoji sve chunk-ove u jedan fajl
            complete_content = ''
            for i in range(total_chunks):
                complete_content += chunk_storage[upload_id]['chunks'][i]
            
            # Uzmi parametre iz storage-a
            params = chunk_storage[upload_id]['parameters']
            
            # Procesiraj spojeni fajl
            result = process_complete_file(complete_content, params)
            
            # Očisti storage
            del chunk_storage[upload_id]
            
            return result
        
        return jsonify({
            "success": True,
            "message": f"Chunk {chunk_index + 1}/{total_chunks} received"
        })
        
    except Exception as e:
        logger.error(f"Error in upload_chunk: {str(e)}")
        return jsonify({"error": str(e)}), 400

def process_complete_file(file_content, params):
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

        return jsonify({
            "success": True,
            "data": result_df.values.tolist(),
            "headers": result_df.columns.tolist()
        })

    except Exception as e:
        logger.error(f"Error processing complete file: {str(e)}")
        return jsonify({"error": str(e)}), 400

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
def upload_files():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        files = request.files.getlist('file')
        if not files:
            return jsonify({"error": "No files selected"}), 400

        # Preuzimanje parametara iz forme
        delimiter = request.form.get('delimiter')
        if not delimiter:
            return jsonify({"error": "Delimiter not provided"}), 400
        timezone = request.form.get('timezone', 'UTC')
        selected_columns = json.loads(request.form.get('selected_columns', '{}'))
        custom_date_format = request.form.get('custom_date_format')
        value_column_name = request.form.get('valueColumnName', '').strip()
        dropdown_count = int(request.form.get('dropdown_count', '2'))
        has_separate_date_time = dropdown_count == 3

        # Očekivani nazivi kolona prema odabiru
        date_column = selected_columns.get('column1')
        time_column = selected_columns.get('column2') if has_separate_date_time else None
        value_column = selected_columns.get('column3') if has_separate_date_time else selected_columns.get('column2')

        all_data_df = pd.DataFrame()

        for file in files:
            # Pročitaj sadržaj fajla jednom
            file_content = file.read().decode('utf-8')
            detected_delimiter = detect_delimiter(file_content)
            if delimiter != detected_delimiter:
                error_message = (
                    f"Incorrect delimiter in file {file.filename}! Detected: '{detected_delimiter}', "
                    f"provided: '{delimiter}'."
                )
                return jsonify({"error": error_message}), 400

            # Očisti sadržaj i učitaj u DataFrame
            cleaned_content = clean_file_content(file_content, delimiter)
            try:
                df = pd.read_csv(StringIO(cleaned_content), delimiter=delimiter)
                df = df.dropna(axis=1, how='all')
                df.columns = df.columns.str.strip()
                if all_data_df.empty:
                    all_data_df = df
                else:
                    if set(all_data_df.columns) != set(df.columns):
                        return jsonify({"error": f"File {file.filename} has different columns than previous files"}), 400
                    all_data_df = pd.concat([all_data_df, df], ignore_index=True)
            except Exception as e:
                logger.error(f"Error processing file {file.filename}: {e}")
                return jsonify({"error": f"Error processing file {file.filename}: {str(e)}"}), 400

        if all_data_df.empty:
            return jsonify({"error": "No data loaded from files"}), 400

        # Pretvaranje vrijednosti u numerički tip ako je moguće
        if value_column and value_column in all_data_df.columns:
            all_data_df[value_column] = pd.to_numeric(all_data_df[value_column], errors='coerce')

        # Parsiranje datuma i/ili vremena
        if has_separate_date_time and date_column and time_column:
            try:
                # Očisti vrijeme od nevažećih znakova
                def clean_time(time_str):
                    if not isinstance(time_str, str):
                        return time_str
                    # Zadrži samo brojeve i dozvoljene separatore
                    cleaned = ''.join(c for c in str(time_str) if c.isdigit() or c in ':.,')
                    # Osiguraj da imamo samo jednu decimalnu točku
                    parts = cleaned.split(':')
                    if len(parts) > 1:
                        seconds_parts = parts[-1].replace('.', ',').split(',', 1)
                        if len(seconds_parts) > 1:
                            parts[-1] = seconds_parts[0] + '.' + seconds_parts[1]
                        cleaned = ':'.join(parts)
                    return cleaned

                # Očisti i kombinuj datum i vrijeme
                cleaned_time = all_data_df[time_column].apply(clean_time)
                combined = all_data_df[date_column].astype(str) + " " + cleaned_time

                if custom_date_format:
                    try:
                        # Ako format sadrži mikrosekundu (%f), proširi decimalne brojeve na 6 znamenki
                        if '%f' in custom_date_format:
                            def extend_microseconds(x):
                                parts = x.split('.')
                                if len(parts) > 1:
                                    return parts[0] + '.' + parts[1].ljust(6, '0')
                                return x
                            combined = combined.apply(extend_microseconds)
                        
                        # Koristi custom format za parsiranje
                        all_data_df['datetime'] = pd.to_datetime(combined, format=custom_date_format, errors='coerce')
                        if all_data_df['datetime'].isna().all():
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
                        all_data_df['datetime'] = pd.to_datetime(combined, errors='coerce')
                        if all_data_df['datetime'].isna().all():
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
            datetime_col = date_column or all_data_df.columns[0]
            success, parsed_dates, err = parse_datetime_column(all_data_df, datetime_col, custom_format=custom_date_format)
            if not success:
                return jsonify({
                    "error": err,
                    "needs_custom_format": True,
                    "supported_formats": SUPPORTED_DATE_FORMATS
                }), 400
            all_data_df['datetime'] = parsed_dates

        # Konverzija u UTC
        try:
            all_data_df = convert_to_utc(all_data_df, 'datetime', timezone)
        except Exception as e:
            return jsonify({"error": f"Error converting to UTC: {str(e)}"}), 400

        # Provjera postojanja value kolone i priprema rezultata
        if not value_column or value_column not in all_data_df.columns:
            return jsonify({"error": f"Value column '{value_column}' not found in data"}), 400

        result_df = pd.DataFrame()
        result_df['UTC'] = all_data_df['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
        final_value_column = value_column_name if value_column_name else value_column
        result_df[final_value_column] = all_data_df[value_column].apply(lambda x: str(x) if pd.notnull(x) else "")
        result_df.dropna(subset=['UTC'], inplace=True)
        result_df.sort_values('UTC', inplace=True)

        # Pretvori DataFrame u listu listi (prvi redak su headeri)
        headers = result_df.columns.tolist()
        data_list = [headers] + result_df.values.tolist()
        return jsonify({"data": data_list, "fullData": data_list})
    except Exception as e:
        logger.error(f"Unexpected error in upload_files: {str(e)}")
        logger.error(traceback.format_exc())
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
