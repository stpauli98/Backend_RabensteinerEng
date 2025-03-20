import os
import tempfile
import traceback
import logging
import json
import csv
import time
from io import StringIO
from datetime import datetime
from flask import Flask, request, jsonify, send_file, Blueprint
import pandas as pd

# Konfiguracija logginga
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Globalni rječnici za privremene fajlove i chunk-ove
temp_files = {}
chunk_storage = {}

bp = Blueprint('load_row_data', __name__)

# Vrijeme nakon kojeg se brišu stari uploadi (30 minuta)
UPLOAD_EXPIRY_TIME = 30 * 60  # sekundi

def cleanup_old_uploads():
    """Briše stare uploade koji nisu završeni"""
    current_time = time.time()
    for upload_id in list(chunk_storage.keys()):
        upload_info = chunk_storage[upload_id]
        if current_time - upload_info.get('last_activity', 0) > UPLOAD_EXPIRY_TIME:
            del chunk_storage[upload_id]

# Lista podržanih formata datuma i vremena
SUPPORTED_DATE_FORMATS = [
    '%Y-%m-%dT%H:%M:%S%z',  # ISO format with timezone and seconds
    '%Y-%m-%dT%H:%M%z',     # ISO format with timezone
    '%Y-%m-%dT%H:%M:%S',    # ISO format without timezone
    '%Y-%m-%d %H:%M:%S',
    '%d.%m.%Y %H:%M',
    '%Y-%m-%d %H:%M',
    '%d.%m.%Y %H:%M:%S',
    '%d.%m.%Y %H:%M:%S.%f',  # Added for milliseconds
    '%Y/%m/%d %H:%M:%S',
    '%d/%m/%Y %H:%M:%S',
    '%Y/%m/%d',
    '%d/%m/%Y',
    '%d-%m-%Y %H:%M:%S',  # Added missing formats
    '%d-%m-%Y %H:%M',
    '%Y/%m/%d %H:%M',
    '%d/%m/%Y %H:%M',
    '%d-%m-%Y',
    '%H:%M:%S',  # Pure time format
    '%H:%M'       # Pure time format
]

def check_date_format(sample_date):
    """
    Proverava da li je format datuma podržan.
    Returns:
        tuple: (bool, str) - (da li je format podržan, poruka o grešci)
    """
    if not isinstance(sample_date, str):
        sample_date = str(sample_date)
    
    # Pokušaj parsiranje sa svim podržanim formatima
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
    
    # Očisti vrijeme od nevažećih znakova, zadrži samo brojeve i separatore
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

# Datum i vrijeme su spojeni
def parse_datetime_column(df, datetime_col, custom_format=None):
    """
    Pokušava parsirati datetime kolonu pomoću custom formata ili podržanih formata.
    Vraca tuple: (success: bool, parsed_dates: Series ili None, error_message: str ili None)
    """
    try:
        # Očisti podatke prije parsiranja
        df = df.copy()
        df[datetime_col] = df[datetime_col].astype(str).str.strip()
        sample_datetime = df[datetime_col].iloc[0]

        # Prvo probamo sa custom formatom ako je prosleđen
        if custom_format:
            try:
                parsed_dates = pd.to_datetime(df[datetime_col], format=custom_format, errors='coerce')
                if not parsed_dates.isna().all():
                    return True, parsed_dates, None
            except Exception as e:
                return False, None, f"Fehler mit custom Format: {str(e)}. Beispielwert: {sample_datetime}"

        # Ako nemamo custom format ili nije uspeo, probamo sa podržanim formatima
        if SUPPORTED_DATE_FORMATS:  # Samo ako imamo podržane formate
            for fmt in SUPPORTED_DATE_FORMATS:
                try:
                    parsed_dates = pd.to_datetime(df[datetime_col], format=fmt, errors='coerce')
                    if not parsed_dates.isna().all():
                        return True, parsed_dates, None
                except Exception:
                    continue

        # Ako nijedan format nije uspeo, vrati grešku
        return False, None, f"Format nicht unterstützt. Beispielwert: %d.%m.%Y %H:%M:%S"

    except Exception as e:
        return False, None, f"Fehler beim Parsen: {str(e)}"

# Datum i vrijeme su odvojeni
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

# Datum i vrijeme razdvojeni
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
        
        # Kombinujemo datum i vreme u jednu kolonu
        df['datetime'] = df[date_column].astype(str).str.strip() + ' ' + df[time_column].astype(str).str.strip()
        sample_datetime = df['datetime'].iloc[0]
        
        # Prvo probamo sa custom formatom ako je prosleđen
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
        
        # Ako nemamo custom format ili nije uspeo, probamo sa podržanim formatima
        is_supported, detected_format = is_format_supported(sample_datetime, SUPPORTED_DATE_FORMATS)
        if not is_supported:
            return jsonify({
                "error": "UNSUPPORTED_DATE_FORMAT",
                "message": f"Format nicht unterstützt. Beispielwert: %d.%m.%Y %H:%M:%S"
            }), 400
            
        # Parsiranje sa detektovanim formatom
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

@bp.route('/upload-chunk', methods=['POST'])
def upload_chunk():
    try:
        print("\n=== Request Form Data ===")
        print(f"All form data: {dict(request.form)}")
        if 'fileChunk' not in request.files:
            return jsonify({"error": "Chunk file not found"}), 400
        
        required_params = ['uploadId', 'chunkIndex', 'totalChunks', 'delimiter', 'selected_columns', 'timezone', 'dropdown_count', 'hasHeader']
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
                print(f"\nRaw selected_columns from form: {selected_columns_str}")
                if not selected_columns_str:
                    return jsonify({"error": "selected_columns parameter is required"}), 400
                selected_columns = json.loads(selected_columns_str)
                print(f"Parsed selected_columns: {selected_columns}")
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
                        'has_header': request.form.get('hasHeader', 'nein'),  # Direktno koristimo string vrijednost
                        'selected_columns': selected_columns,
                        'custom_date_format': request.form.get('custom_date_format'),
                        'value_column_name': request.form.get('valueColumnName', '').strip(),
                        'dropdown_count': int(request.form.get('dropdown_count', '2'))
                    }
                }
            except json.JSONDecodeError as e:
                return jsonify({"error": "Invalid JSON format for selected_columns"}), 400

        chunk_content = file_chunk.read()
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
        
        # Try different encodings
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
            return jsonify({"error": "Could not decode file content with any supported encoding"}), 400

        print("\n=== Decoded File Content ===")
        num_lines = len(full_content.split('\n'))
        print(f"Decoded content: {num_lines} lines long")
        print("=== End Decoded File Content ===")
            
        params = chunk_storage[upload_id]['parameters']
        del chunk_storage[upload_id]
        
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
        print("\n=== Processing Upload ===")
        print(f"Received params: {json.dumps(params, indent=2)}")
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
        has_header = params.get('has_header', False)

        print(f"\nSelected columns from frontend:")
        print(f"selected_columns: {selected_columns}")
        print(f"has_header: {has_header}")

        # Postavi kolone prema odabiru s frontenda
        if has_header == 'ja':
            # Kad ima header, koristi točno ona imena kolona koja je korisnik odabrao
            date_column = selected_columns.get('column1')  # npr. 'Datum'
            time_column = selected_columns.get('column2') if has_separate_date_time else None
            value_column = selected_columns.get('column3') if has_separate_date_time else selected_columns.get('column2') 
        else:
            # Kad nema header, koristi indekse kolona koje je korisnik odabrao
            # Frontend šalje indeks kolone (npr. '4' za petu kolonu)
            date_column = selected_columns.get('column1', '0')
            time_column = selected_columns.get('column2', '1')  if has_separate_date_time else None
            value_column = selected_columns.get('column3', '2') if has_separate_date_time else selected_columns.get('column2') 

            
        print(f"\nSelected columns from frontend:")
        print(f"date_column: {date_column}")
        print(f"time_column: {time_column}")
        print(f"value_column: {value_column}")
        print(f"has_separate_date_time: {has_separate_date_time}")
        
        print(f"Using columns - Date: {date_column}, Time: {time_column}, Value: {value_column}")

        # Pročitaj CSV sadržaj iz primljenog stringa
        detected_delimiter = detect_delimiter(file_content)
        if delimiter != detected_delimiter:
            return jsonify({"error": f"Incorrect delimiter! Detected: '{detected_delimiter}', provided: '{delimiter}'"}), 400

        # Očisti sadržaj i učitaj u DataFrame
        cleaned_content = clean_file_content(file_content, delimiter)
        try:
            print(f"\n=== Reading CSV with parameters ===\ndelimiter: {delimiter}, has_header: {has_header}")
            print(f"\nReading CSV file...")
            
            # Pročitaj prvu liniju da vidimo je li header
            with StringIO(cleaned_content) as f:
                first_line = f.readline().strip()
                first_line_values = first_line.split(delimiter)
                # Provjeri izgleda li prvi red kao header
                looks_like_header = all(not val.replace('.', '').replace(',', '').replace('-', '').isdigit() 
                                       for val in first_line_values if val.strip())
                
                if looks_like_header and has_header == 'nein':
                    
                    # Ponovno učitaj CSV bez prve linije
                    df = pd.read_csv(StringIO(cleaned_content),
                        delimiter=delimiter,
                        header=None,
                        skiprows=1)
                    print("First line looks like header but has_header is 'nein'. Skipping first line.")
                else:
                    df = pd.read_csv(StringIO(cleaned_content),
                        delimiter=delimiter,
                        header=0 if has_header == 'ja' else None)
            
            print(f"Original columns: {df.columns.tolist()}")
            print(f"Column types: {df.dtypes}")

            if has_header == 'nein':
                df.columns = [str(i) for i in range(len(df.columns))]
            else:
                df.columns = [col.strip() for col in df.columns]
            
            # Očisti prazne kolone
            df = df.dropna(axis=1, how='all')
            
            # Konvertiraj imena kolona u stringove ako nisu
            df.columns = df.columns.astype(str)
            
            # Očisti whitespace iz imena kolona
            df.columns = [col.strip() for col in df.columns]
            
            print(f"Cleaned columns: {df.columns.tolist()}")
            
            print(f"\nDetected columns: {df.columns.tolist()}")
            print(f"First row sample: {df.iloc[0].to_dict()}")
        except Exception as e:
            return jsonify({"error": f"Error processing CSV: {str(e)}"}), 400

        if df.empty:
            return jsonify({"error": "No data loaded from file"}), 400

        # Pretvaranje vrijednosti u numerički tip ako je moguće
        if value_column and value_column in df.columns:
            df[value_column] = pd.to_numeric(df[value_column], errors='coerce')

        # Parsiranje datuma i vremena
        try:
            datetime_col = date_column or df.columns[0]
            
            if has_separate_date_time and date_column and time_column:
                try:
                    # Clean time format first
                    df[time_column] = df[time_column].apply(clean_time)
                    
                    # Clean date format
                    df[date_column] = df[date_column].apply(clean_time)
                    
                    # Combine date and time columns
                    df['datetime'] = df[date_column].astype(str) + ' ' + df[time_column].astype(str)
                    
                    # Try parsing with default formats first
                    success, parsed_dates, err = parse_datetime_column(df, 'datetime')
                    
                    # If that fails and we have a custom format, try that
                    if not success and custom_date_format:
                        success, parsed_dates, err = parse_datetime_column(df, 'datetime', custom_format=custom_date_format)
                    
                    if not success:
                        return jsonify({
                            "error": "UNSUPPORTED_DATE_FORMAT",
                            "message": f"Format nicht unterstützt. Beispielwert: %d.%m.%Y %H:%M:%S"
                        }), 400
                except Exception as e:
                    return jsonify({"error": f"Error parsing date/time: {str(e)}"}), 400
            else:
                # Try to parse as a combined datetime
                success, parsed_dates, err = parse_datetime_column(df, datetime_col, custom_format=custom_date_format)
                if not success:
                    return jsonify({
                        "error": "UNSUPPORTED_DATE_FORMAT",
                        "message": err
                    }), 400
                
            df['datetime'] = parsed_dates
        except Exception as e:
            return jsonify({"error": f"Error parsing date/time: {str(e)}"}), 400

        # Konverzija u UTC
        try:
            df = convert_to_utc(df, 'datetime', timezone)
        except Exception as e:
            return jsonify({
                "error": "Überprüfe dein Datumsformat eingabe",
                "message": f"Fehler bei der Konvertierung in UTC: {str(e)}"
            }), 400

        # Provera postojanja value kolone
        if not value_column or value_column not in df.columns:
            return jsonify({"error": f"Datum, Wert 1 oder Wert 2 nicht ausgewählt"}), 400

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
        print(f"Error in prepare_save: {str(e)}")
        print(traceback.format_exc())
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
        print(f"Error in download_file: {str(e)}")
        return jsonify({"error": str(e)}), 500
    finally:
        # Pokušaj očistiti privremeni fajl
        if file_id in temp_files:
            try:
                os.unlink(temp_files[file_id])
                del temp_files[file_id]
            except Exception as ex:
                print(f"Error cleaning up temp file: {ex}")
