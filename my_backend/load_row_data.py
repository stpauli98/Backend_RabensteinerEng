import os
import tempfile
import traceback
import logging
import json
import csv
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

# Globalni rječnik za privremene fajlove (nije thread-safe – za produkciju koristiti sigurniji pristup)
temp_files = {}

# Lista podržanih formata datuma
SUPPORTED_DATE_FORMATS = [
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
    if custom_format:
        try:
            parsed_dates = pd.to_datetime(df[datetime_col], format=custom_format)
            if parsed_dates.notna().all():
                return True, parsed_dates, None
        except Exception as e:
            logger.error(f"Error parsing datetime with custom format: {e}")
            return False, None, str(e)
    try:
        parsed_dates = pd.to_datetime(df[datetime_col])
        if parsed_dates.notna().all():
            return True, parsed_dates, None
    except Exception as e:
        logger.error(f"Auto parsing of datetime failed: {e}")

    # Pokušaj svih podržanih formata
    for fmt in SUPPORTED_DATE_FORMATS:
        try:
            parsed_dates = pd.to_datetime(df[datetime_col], format=fmt)
            if parsed_dates.notna().all():
                return True, parsed_dates, None
        except Exception:
            continue
    error_msg = (
        f"Unrecognized datetime format. Please use one of the supported formats: "
        f"{', '.join(SUPPORTED_DATE_FORMATS)} or provide a custom format."
    )
    return False, None, error_msg

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
                combined = all_data_df[date_column].astype(str) + " " + all_data_df[time_column].astype(str)
                if custom_date_format:
                    # Custom format očekuje oblik "date_format time_format"
                    parts = custom_date_format.split(" ", 1)
                    fmt = parts[0] + " " + (parts[1] if len(parts) > 1 else "%H:%M:%S")
                    success, parsed_dates, err = parse_datetime_column(
                        pd.DataFrame({'combined': combined}), 'combined', custom_format=fmt
                    )
                    if not success:
                        return jsonify({
                            "error": f"Error parsing datetime with custom format: {err}",
                            "needs_custom_format": True,
                            "supported_formats": SUPPORTED_DATE_FORMATS
                        }), 400
                    all_data_df['datetime'] = parsed_dates
                else:
                    all_data_df['datetime'] = pd.to_datetime(combined)
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
