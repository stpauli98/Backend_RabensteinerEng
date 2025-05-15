import os
import pandas as pd
import numpy as np
import json
import gzip
import traceback
import logging
import tempfile
import csv 
from io import StringIO
from datetime import datetime
import time
from flask import request, jsonify, Response, send_file, Blueprint

# Create blueprint
bp = Blueprint('first_processing', __name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Globalni rečnik za čuvanje privremenih fajlova
temp_files = {}

def clean_for_json(obj):
    """Konvertuje numpy i pandas tipove u Python native tipove."""
    if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32,
                         np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, (np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    elif isinstance(obj, (pd.Timestamp)):
        return obj.isoformat()
    elif pd.isna(obj):
        return None
    return obj

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
        
        # Resetujemo sekunde i mikrosekunde
        time_min_base = time_min_raw.replace(second=0, microsecond=0)
        
        # Dodajemo offset za početno vreme
        logger.info(f"Applying offset of {offset} minutes to {time_min_base}")
        time_min = time_min_base + pd.Timedelta(minutes=int(offset))  # Osiguraj da je offset int
        logger.info(f"Resulting start time: {time_min}")
        
        # Kreiraj novi DataFrame sa željenim vremenskim intervalima
        time_range = pd.date_range(
            start=time_min,
            end=time_max_raw,
            freq=f'{int(tss)}min'
        )
        
        # Kreiraj DataFrame sa željenim vremenima
        df_utc = pd.DataFrame({'UTC': time_range})
        
        if mode_input == "mean":
            # Optimizovana mean kalkulacija koristeći resample
            df.set_index('UTC', inplace=True)
            
            # Direktno resample sa offset-om kao početkom
            df_resampled = df[value_col_name].resample(
                rule=f'{int(tss)}min',
                origin=time_min,
                closed='right',
                label='right'
            ).mean().to_frame()
            
            # Filtriraj samo vremena nakon offset-a
            df_resampled = df_resampled[df_resampled.index >= time_min]
            
            # Resetuj index da dobijemo UTC kolonu
            df_resampled.reset_index(inplace=True)
            
        elif mode_input == "intrpl":
            # UTC als Index setzen für Originaldaten
            df.set_index("UTC", 
                        inplace = True)
            
            # First, merge with nearest values within tolerance
            df_resampled = pd.merge_asof(
                            df_utc,
                            df.reset_index(),
                            left_on     ='UTC',
                            right_on    ='UTC',
                            direction   ='nearest',
                            tolerance   = pd.Timedelta(minutes = tss/2)
                            )
            
            df_resampled.set_index("UTC", 
                        inplace = True)
            
            # Create a copy of the dataframe before interpolation to identify gaps
            df_before_interp = df_resampled.copy()
            
            # Find the actual measurement points (non-NaN values)
            actual_measurements = df_before_interp[~df_before_interp[value_col_name].isna()]
            
            # Calculate time differences between consecutive measurements
            if len(actual_measurements) > 1:
                # Get the timestamps of actual measurements
                measurement_times = actual_measurements.index.to_list()
                
                # Calculate time differences between consecutive measurements in minutes
                time_diffs = np.diff(measurement_times) / pd.Timedelta(minutes=1)
                
                # Find gaps larger than intrpl_max
                large_gaps = np.where(time_diffs > float(intrpl_max))[0]
                
                logger.info(f"Found {len(large_gaps)} large measurement gaps exceeding {intrpl_max} minutes")
                
                # For each large gap, create a mask to prevent interpolation
                for gap_idx in large_gaps:
                    gap_start = measurement_times[gap_idx]
                    gap_end = measurement_times[gap_idx + 1]
                    
                    # Create a mask for timestamps that fall within the large gap
                    # We'll allow interpolation for up to intrpl_max minutes after the last measurement
                    max_interp_time = gap_start + pd.Timedelta(minutes=float(intrpl_max))
                    
                    # Points that are in the gap but beyond the max interpolation time should not be interpolated
                    no_interp_mask = ((df_resampled.index > max_interp_time) & 
                                      (df_resampled.index < gap_end))
                    
                    # Set these points to NaN to prevent interpolation
                    if no_interp_mask.any():
                        logger.info(f"Preventing interpolation for {no_interp_mask.sum()} points in gap between "
                                   f"{gap_start} and {gap_end} (beyond {intrpl_max} minutes)")
            
            # Now perform the interpolation with the limit
            df_resampled = df_resampled.interpolate(method = "time",
                                                   limit = int(intrpl_max/tss))
            
            # Reset index to get UTC column back for JSON conversion
            df_resampled.reset_index(inplace=True)
            
        elif mode_input in ["nearest", "nearest (mean)"]:
            # Postavi UTC kao index za originalne podatke
            df.set_index('UTC', inplace=True)
            
            if mode_input == "nearest":
                    # Koristi merge_asof za najbliže vrednosti sa tačnim vremenima
                    df_resampled = pd.merge_asof(
                        df_utc,
                        df.reset_index(),
                        left_on='UTC',
                        right_on='UTC',
                        direction='nearest',
                        tolerance=pd.Timedelta(minutes=tss/2)
                    )
            else:  # nearest (mean)
                    # Proveri da li je UTC već index
                    if df.index.name != 'UTC':
                        if 'UTC' in df.columns:
                            df.set_index('UTC', inplace=True)
                        else:
                            # Ako je UTC već index ali nije imenovan
                            df.index.name = 'UTC'
                    
                    # Kreiraj resampler sa offset vremenom kao početkom
                    resampled = df[value_col_name].resample(
                        rule=f'{int(tss)}min',
                        origin=time_min,
                        closed='right',
                        label='right'
                    ).mean()
                    
                    # Filtriraj samo vremena nakon offset-a
                    mask = resampled.index >= time_min
                    resampled = resampled[mask]
                    
                    # Kreiraj finalni DataFrame
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

@bp.route('/upload_chunk', methods=['POST'])
def upload_chunk():
    """
    Endpoint za prihvat i obradu CSV podataka u delovima (chunks).
    Očekivani parametri (form data):
      - uploadId: jedinstveni ID za upload (string)
      - chunkIndex: redni broj chunka (int, počinje od 0)
      - totalChunks: ukupan broj chunkova (int)
      - fileChunk: Blob/File sa delom CSV podataka
      - tss: Time step size u minutama (float)
      - offset: Offset u minutama (float)
      - mode: Način obrade ('mean', 'intrpl', 'nearest', 'nearest (mean)')
      - intrplMax: Maksimalno vreme za interpolaciju u minutama (float, default 60)
    """
    try:
        # Proveri da li imamo sve potrebne parametre
        if 'fileChunk' not in request.files:
            return jsonify({"error": "No file chunk found"}), 400

        # Učitaj parametre iz form data
        try:
            upload_id = request.form.get('uploadId')
            chunk_index = int(request.form.get('chunkIndex', 0))
            total_chunks = int(request.form.get('totalChunks', 0))
            tss = float(request.form.get('tss', 0))
            offset = float(request.form.get('offset', 0))
            mode = request.form.get('mode', '')
            intrpl_max = float(request.form.get('intrplMax', 60))
        except (ValueError, TypeError) as e:
            logger.error(f"Error parsing parameters: {e}")
            return jsonify({"error": f"Invalid parameter values: {str(e)}"}), 400

        # Validacija parametara
        if not all([upload_id, mode, tss > 0]):
            return jsonify({"error": "Missing required parameters"}), 400

        # Uzmi chunk fajl
        chunk = request.files['fileChunk']
        if not chunk:
            return jsonify({"error": "Empty chunk received"}), 400

        # Kreiraj folder ako ne postoji
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)

        # Sačuvaj chunk
        chunk_filename = os.path.join(UPLOAD_FOLDER, f"{upload_id}_{chunk_index}.chunk")
        chunk.save(chunk_filename)

        logger.info(f"Saved chunk {chunk_index + 1}/{total_chunks} for upload {upload_id}")

        # Proveri da li su svi chunkovi primljeni
        received_chunks = [f for f in os.listdir(UPLOAD_FOLDER) 
                         if f.startswith(upload_id + "_")]

        if len(received_chunks) == total_chunks:
            logger.info(f"All chunks received for upload {upload_id}, processing...")
            
            try:
                # Sortiraj chunkove po indeksu
                chunks_sorted = sorted(received_chunks, 
                                     key=lambda x: int(x.split("_")[1].split(".")[0]))
                
                # Spoji sve chunkove
                full_content = ""
                for chunk_file in chunks_sorted:
                    chunk_path = os.path.join(UPLOAD_FOLDER, chunk_file)
                    with open(chunk_path, 'rb') as f:
                        chunk_content = f.read().decode('utf-8')
                        full_content += chunk_content
                    # Obriši chunk fajl nakon čitanja
                    os.remove(chunk_path)

                # Obradi spojeni sadržaj
                return process_csv(full_content, tss, offset, mode, intrpl_max)
                
            except Exception as e:
                # U slučaju greške, obriši sve chunkove
                for chunk_file in chunks_sorted:
                    try:
                        os.remove(os.path.join(UPLOAD_FOLDER, chunk_file))
                    except:
                        pass
                raise

        # Vrati status o primljenom chunk-u
        return jsonify({
            "message": f"Chunk {chunk_index + 1}/{total_chunks} received",
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
        data_wrapper = data['data']
        save_data = data_wrapper.get('data', [])
        file_name = data_wrapper.get('fileName', '')
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
        temp_files[file_id] = {
            'path': temp_file.name,
            'fileName': file_name or f"data_{file_id}.csv",  # Koristi poslato ime ili default
            'timestamp': time.time()
        }

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
            
        file_info = temp_files[file_id]
        file_path = file_info['path']
        
        if not os.path.exists(file_path):
            return jsonify({"error": "File not found"}), 404

        # Koristi sačuvano ime fajla
        download_name = file_info['fileName']
        
        response = send_file(
            file_path,
            as_attachment=True,
            download_name=download_name,
            mimetype='text/csv'
        )
        
        # Cleanup nakon slanja
        try:
            os.unlink(file_info['path'])
        except Exception as e:
            logger.error(f"Error cleaning up temp file: {e}")
        return response

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


