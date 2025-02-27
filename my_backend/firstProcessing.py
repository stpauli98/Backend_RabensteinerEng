import os
import pandas as pd
import numpy as np
import json
import gzip
import traceback
import logging
import tempfile
import csv

# Konfigurisi logging da bude manje opširan
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
from io import StringIO
from datetime import datetime as dat, timedelta
from flask import Flask, request, jsonify, Response, send_file
from flask_cors import CORS
from werkzeug.formparser import FormDataParser

# Globalni rečnik za čuvanje privremenih fajlova
temp_files = {}

def clean_for_json(obj):
    """Konvertuje numpy i pandas tipove u Python native tipove."""
    if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32,
                         np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    elif isinstance(obj, (pd.Timestamp)):
        return obj.isoformat()
    elif pd.isna(obj):
        return None
    return obj


# Konfiguracija logginga
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define API prefix
API_PREFIX_FIRST_PROCESSING = '/api/firstProcessing'

# Folder za privremeno spremanje chunkova
UPLOAD_FOLDER = "chunk_uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def process_csv(file_content, tss, offset, mode_input, intrpl_max):
    """
    Obradjuje CSV sadržaj te vraća rezultat kao gzip-komprimiran JSON odgovor.
    Koristi vektorizirane Pandas operacije za bolju performansu.
    """
    try:
        try:
            # Učitaj CSV podatke u DataFrame i konvertuj vremena
            df = pd.read_csv(StringIO(file_content), delimiter=';', skipinitialspace=True)
            df.columns = df.columns.str.strip()
            value_col_name = df.columns[1]
            
            # Konvertuj UTC kolonu u datetime i vrednosti u numerički format
            df['UTC'] = pd.to_datetime(df['UTC'], format='%Y-%m-%d %H:%M:%S')
            df[value_col_name] = pd.to_numeric(df[value_col_name], errors='coerce')
        except Exception as e:
            logger.error(f"Error parsing CSV data: {str(e)}")
            return jsonify({"error": "Invalid CSV format"}), 400
        
        # Očisti i sortiraj podatke
        df = df.drop_duplicates(subset=['UTC']).sort_values('UTC').reset_index(drop=True)
        
        if df.empty:
            return jsonify({"error": "Nema valjanih vremenskih oznaka"}), 400
            
        # Pripremi vremenski raspon sa offsetom
        time_min_raw = df['UTC'].iloc[0]
        time_max_raw = df['UTC'].iloc[-1]
        
        # Prvo resetujemo sekunde i mikrosekunde
        time_min = time_min_raw.replace(second=0, microsecond=0)
        
        # Izračunaj početno vreme sa offsetom
        # Prvo dodajemo offset
        time_with_offset = time_min + pd.Timedelta(minutes=offset)
        
        # Zaokružujemo na najbliži interval koji je višestruk od tss
        minutes_since_hour = time_with_offset.minute
        adjusted_minutes = ((minutes_since_hour + tss - 1) // tss) * tss
        time_min = time_with_offset.replace(minute=adjusted_minutes)
        
        # Kreiraj novi DataFrame sa željenim vremenskim intervalima
        time_range = pd.date_range(
            start=time_min,
            end=time_max_raw,
            freq=f'{int(tss)}min'
        )
        df_resampled = pd.DataFrame({'UTC': time_range})
        
        # Postavi UTC kao index za efikasnije operacije
        df.set_index('UTC', inplace=True)
        
        if mode_input == "mean":
            # Koristi resample za računanje proseka
            resampled = df[value_col_name].resample(
                rule=f'{int(tss)}min',
                origin=time_min,
                closed='right',
                label='right'
            ).mean()
            df_resampled = pd.DataFrame({'UTC': resampled.index, value_col_name: resampled.values})
            
        elif mode_input == "intrpl":
            # Koristi merge_asof i interpolate za interpolaciju
            df_resampled = pd.merge_asof(
                df_resampled, df.reset_index(),
                left_on='UTC',
                right_on='UTC',
                direction='nearest',
                tolerance=pd.Timedelta(minutes=intrpl_max)
            )
            # Postavi UTC kao index za vremensku interpolaciju
            df_resampled.set_index('UTC', inplace=True)
            # Primeni linearnu interpolaciju gde je moguće
            df_resampled[value_col_name] = df_resampled[value_col_name].interpolate(
                method='time',
                limit=int(intrpl_max * 60 / tss)  # Konvertuj minute u broj intervala
            )
            # Resetuj index da bi UTC bio kolona
            df_resampled.reset_index(inplace=True)
            
        elif mode_input in ["nearest", "nearest (mean)"]:
            if mode_input == "nearest":
                # Koristi merge_asof za najbliže vrednosti
                df_resampled = pd.merge_asof(
                    df_resampled, df.reset_index(),
                    left_on='UTC',
                    right_on='UTC',
                    direction='nearest',
                    tolerance=pd.Timedelta(minutes=tss/2)
                )
            else:  # nearest (mean)
                # Za svaki interval, nađi prosek najbližih vrednosti
                resampled = df[value_col_name].resample(
                    rule=f'{int(tss)}min',
                    origin=time_min,
                    closed='right',
                    label='right'
                ).mean()
                df_resampled = pd.DataFrame({'UTC': resampled.index, value_col_name: resampled.values})
        
        # Konvertuj rezultate u JSON format sa specificnim formatom vremena
        result = df_resampled.apply(
            lambda row: {
                "UTC": row["UTC"].strftime("%Y-%m-%d %H:%M:%S"),
                value_col_name: clean_for_json(row[value_col_name])
            },
            axis=1
        ).tolist()
        
        # Kompresuj i vrati rezultat
        result_json = json.dumps(result)
        compressed_data = gzip.compress(result_json.encode('utf-8'))
        
        response = Response(compressed_data)
        response.headers['Content-Encoding'] = 'gzip'
        response.headers['Content-Type'] = 'application/json'
        return response
    except Exception as e:
        error_msg = f"Error processing CSV: {str(e)}"
        logger.error(error_msg)
        return jsonify({"error": str(e)}), 400

def upload_chunk(request):
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
        file_id = dat.now().strftime('%Y%m%d_%H%M%S')
        temp_files[file_id] = temp_file.name

        return jsonify({"message": "File prepared for download", "fileId": file_id}), 200
    except Exception as e:
        logger.error(f"Error in prepare_save: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

def download_file(file_id, request):
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



