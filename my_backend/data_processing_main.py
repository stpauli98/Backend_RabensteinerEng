import pandas as pd
from datetime import datetime
import time
from io import StringIO
import numpy as np
from flask import Blueprint, jsonify, send_file, request, Response
import tempfile
import csv
import os
import traceback
import logging
import json

# Create Blueprint
bp = Blueprint('data_processing_main_bp', __name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Directory for temporary chunk storage
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'temp_uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Dictionary to store temporary files
temp_files = {}
# Format for UTC dates
UTC_fmt = "%Y-%m-%d %H:%M:%S"

@bp.route('/upload-chunk', methods=['POST'])
def handle_upload_chunk():
    try:
        return upload_chunk(request)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

def upload_chunk(req):
    try:
        # Validate parameters
        upload_id = req.form.get('uploadId')
        chunk_index = int(req.form.get('chunkIndex', '0') or '0')
        total_chunks = int(req.form.get('totalChunks', '0') or '0')
        
        # Helper function to safely convert to float
        def safe_float(value, default='0'):
            try:
                return float(value) if value.strip() else float(default)
            except (ValueError, AttributeError):
                return float(default)
        
        # Convert form values to float, using 0 as default for empty or invalid values
        EQ_MAX = safe_float(req.form.get('eqMax'))
        CHG_MAX = safe_float(req.form.get('chgMax'))
        LG_MAX = safe_float(req.form.get('lgMax'))
        GAP_MAX = safe_float(req.form.get('gapMax'))
        ELMAX = safe_float(req.form.get('elMax'))
        ELMIN = safe_float(req.form.get('elMin'))
        
        # Get radio button values
        EL0 = req.form.get('radioValueNull')
        ELNN = req.form.get('radioValueNotNull')

        # Convert radio button values
        EL0 = 1 if EL0 == "ja" else 0
        ELNN = 1 if ELNN == "ja" else 0

        # If upload_id is provided, it's a chunk upload
        if upload_id:
            if 'fileChunk' not in req.files:
                return jsonify({"error": "Chunk file not found"}), 400

            chunk = req.files['fileChunk']
            if not chunk:
                return jsonify({"error": "Empty chunk received"}), 400

            os.makedirs(UPLOAD_FOLDER, exist_ok=True)
            chunk_filename = os.path.join(UPLOAD_FOLDER, f"{upload_id}_{chunk_index}.chunk")
            chunk.save(chunk_filename)
            received_chunks = [f for f in os.listdir(UPLOAD_FOLDER) if f.startswith(upload_id + "_")]
            if len(received_chunks) == total_chunks:
                chunks_sorted = sorted(received_chunks, key=lambda x: int(x.split("_")[1].split(".")[0]))
                full_content = ""
                try:
                    for chunk_file in chunks_sorted:
                        chunk_path = os.path.join(UPLOAD_FOLDER, chunk_file)
                        with open(chunk_path, 'rb') as f:
                            full_content += f.read().decode('utf-8')
                        os.remove(chunk_path)
                except Exception as e:
                    for cf in chunks_sorted:
                        try:
                            os.remove(os.path.join(UPLOAD_FOLDER, cf))
                        except Exception:
                            pass
                    return jsonify({"error": f"Error processing chunks: {str(e)}"}), 400

                form_params = {
                    'eqMax': str(EQ_MAX),
                    'chgMax': str(CHG_MAX),
                    'lgMax': str(LG_MAX),
                    'gapMax': str(GAP_MAX),
                    'elMax': str(ELMAX),
                    'elMin': str(ELMIN),
                    'radioValueNull': 'ja' if EL0 == 1 else 'nein',
                    'radioValueNotNull': 'ja' if ELNN == 1 else 'nein'
                }
                mock_request = create_mock_request(form_params, full_content, 'combined_chunks.csv')

                try:
                    result = zweite_bearbeitung(mock_request)
                    return result
                except Exception as e:
                    for cf in chunks_sorted:
                        try:
                            os.remove(os.path.join(UPLOAD_FOLDER, cf))
                        except Exception:
                            pass
                    return jsonify({"error": f"Error processing chunks: {str(e)}"}), 400

            # Not all chunks received yet; return status
            return jsonify({
                "message": f"Chunk {chunk_index + 1}/{total_chunks} received",
                "uploadId": upload_id,
                "chunkIndex": chunk_index,
                "totalChunks": total_chunks,
                "remainingChunks": total_chunks - len(received_chunks)
            }), 200

        else:
            # Direct upload
            if 'file' not in req.files:
                return jsonify({"error": "Keine Datei gefunden"}), 400
            file = req.files['file']
            if not file:
                return jsonify({"error": "Keine Datei gefunden"}), 400
            file_content = file.stream.read().decode('utf-8')
            const_form_params = {
                'eqMax': str(EQ_MAX),
                'chgMax': str(CHG_MAX),
                'lgMax': str(LG_MAX),
                'gapMax': str(GAP_MAX),
                'elMax': str(ELMAX),
                'elMin': str(ELMIN),
                'radioValueNull': 'ja' if EL0 == 1 else 'nein',
                'radioValueNotNull': 'ja' if ELNN == 1 else 'nein'
            }
            mock_request = create_mock_request(const_form_params, file_content, file.filename)
            try:
                result = zweite_bearbeitung(mock_request)
                return result
            except Exception as e:
                return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 400

    except Exception as e:
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 400

def zweite_bearbeitung(req):
    try:
        file_key = 'file' if 'file' in req.files else 'fileChunk' if 'fileChunk' in req.files else None
        if not file_key:
            return jsonify({"error": "No file uploaded (neither 'file' nor 'fileChunk' found)"}), 400

        file = req.files[file_key]
        file_content = file.stream.read().decode('utf-8')
        if not file_content.strip():
            return jsonify({"error": "Empty file content"}), 400

        def safe_float(value):
            if value and value.strip():
                try:
                    return float(value)
                except ValueError:
                    return None
            return None

        EQ_MAX = safe_float(req.form.get('eqMax'))
        CHG_MAX = safe_float(req.form.get('chgMax'))
        LG_MAX = safe_float(req.form.get('lgMax'))
        GAP_MAX = safe_float(req.form.get('gapMax'))
        ELMAX = safe_float(req.form.get('elMax'))
        ELMIN = safe_float(req.form.get('elMin'))
        EL0 = req.form.get('radioValueNull')
        ELNN = req.form.get('radioValueNotNull')
        EL0 = 1 if EL0 == "ja" else 0
        ELNN = 1 if ELNN == "ja" else 0


        content_lines = file_content.splitlines()
        if not content_lines:
            return jsonify({"error": "No data received"}), 400

        first_line = content_lines[0].strip()
        if not first_line:
            return jsonify({"error": "Empty first line"}), 400

        if ';' in first_line:
            delimiter = ';'
        elif ',' in first_line:
            delimiter = ','
        else:
            return jsonify({"error": "No valid delimiter (comma or semicolon) found in data"}), 400

        try:
            cleaned_content = file_content.replace('\r', '').strip()
            df = pd.read_csv(StringIO(cleaned_content),
                             delimiter=delimiter,
                             header=0,
                             skipinitialspace=True,
                             skip_blank_lines=True)
            if df.empty:
                return jsonify({"error": "No data found in file"}), 400
            if len(df.columns) < 2:
                return jsonify({"error": f"Data must have at least 2 columns, but found only {len(df.columns)}"}), 400

            time_column = df.columns[0]
            data_column = df.columns[1]

            
            # Convert to string and clean
            df[data_column] = df[data_column].astype(str)
            
            
            # Clean the data
            df[data_column] = (df[data_column]
                               .str.strip()
                               .str.replace('\r', '')
                               .str.replace('\n', '')
                               .str.replace(',', '.'))
            
            # Convert to numeric
            df[data_column] = pd.to_numeric(df[data_column], errors='coerce'    )
            if df[data_column].isna().all():
                return jsonify({"error": "Could not convert any values to numeric format"}), 400
        except Exception as e:
            err_msg = f"Error processing data: {str(e)}\n{traceback.format_exc()}"
            return jsonify({"error": err_msg}), 400

        # Konvertiraj stupac u numerički format (potrebno za daljnju obradu)
        df.iloc[:, 1] = pd.to_numeric(df.iloc[:, 1], errors='coerce')

        # Primjeni gornju i donju granicu samo ako su definirane (veće od 0)
        try:
            if ELMAX > 0:
                df.iloc[:, 1] = df.iloc[:, 1].mask(df.iloc[:, 1] > ELMAX, np.nan)
                
            if ELMIN > 0:
                df.iloc[:, 1] = df.iloc[:, 1].mask(df.iloc[:, 1] < ELMIN, np.nan)
        except Exception as e:
            df.iloc[:, 1] = np.nan


        if EQ_MAX is not None and EQ_MAX > 0:
            
            # Status des Identifikationsrahmens (frm...frame)
            frm = 0
            """
                0...Im aktuellen Zeitschritt ist kein Identifikationsrahmen für
                    gleichbleibende Messwerte offen
                1...Im aktuellen Zeitschritt ist ein Identifikationsrahmen für 
                    gleichbleibende Messwerte offen
            """
            
            # Konvertuj vremenske kolone u datetime format za brže procesiranje
            df[time_column] = pd.to_datetime(df[time_column])
            
            # Kreiraj masku za konstantne vrednosti
            constant_mask = df[data_column].eq(df[data_column].shift())
            
            # Nađi početke konstantnih segmenata
            segment_starts = constant_mask & ~constant_mask.shift(1, fill_value=False)
            start_indices = segment_starts[segment_starts].index.tolist()
            
            # Ako postoje konstantni segmenti
            if start_indices:
                for start_idx in start_indices:
                    # Nađi kraj trenutnog konstantnog segmenta
                    end_mask = ~constant_mask[start_idx:]
                    if end_mask.any():
                        end_idx = end_mask.idxmax()
                    else:
                        end_idx = len(df) - 1
                    
                    # Izračunaj širinu segmenta u minutama
                    segment_width = (df.loc[end_idx, time_column] - 
                                   df.loc[start_idx, time_column]).total_seconds() / 60
                    
                    # Ako je segment prevelik, postavi vrednosti na NaN
                    if segment_width >= EQ_MAX:
                        df.loc[start_idx:end_idx, data_column] = np.nan
                        
            # Interpolacija se ne primenjuje na konstantne segmente
            # Samo interpoliramo ostale praznine
            non_constant_mask = ~constant_mask
            if non_constant_mask.any():
                df.loc[non_constant_mask, data_column] = df.loc[non_constant_mask, data_column].interpolate(
                    method='linear',
                    limit=10,
                    limit_direction='both'
                )



       
        ##############################################################################
        # ELIMINIERUNG VON NULLWERTEN #################################################
        ##############################################################################

        if EL0 == 1:
            
            # Durchlauf des gesamten Datenrahmens
            df.loc[df[data_column] == 0, data_column] = np.nan


        ##############################################################################
        # ELIMINIERUNG VON NICHT NUMERISCHEN WERTEN ###################################
        ##############################################################################

        if ELNN == 1:
            
            # Konvertuj kolonu u numerički format, nevalidne vrijednosti postaju NaN
            df[data_column] = pd.to_numeric(df[data_column], errors='coerce')

        ##############################################################################
        # ELIMINIERUNG VON EXTREMEN ###################################################
        ##############################################################################

        if all(x is not None and x > 0 for x in [CHG_MAX, LG_MAX]):
            
            # Konvertuj vremenske kolone u datetime format
            df[time_column] = pd.to_datetime(df[time_column])
            
            # Izračunaj promjene vrijednosti po jedinici vremena (rate of change)
            value_changes = df[data_column].diff().abs()
            time_diffs = df[time_column].diff().dt.total_seconds() / 60
            change_rates = value_changes / time_diffs
            
            # Inicijaliziraj masku za označavanje NaN vrijednosti
            nan_mask = pd.Series(False, index=df.index)
            
            # Nađi tačke gdje je rate of change veći od CHG_MAX
            extreme_points = change_rates > CHG_MAX
            
            if extreme_points.any():
                # Nađi početke segmenata sa ekstremnim promjenama
                segment_starts = extreme_points & ~extreme_points.shift(1, fill_value=False)
                start_indices = segment_starts[segment_starts].index.tolist()
                
                for start_idx in start_indices:
                    # Nađi kraj trenutnog segmenta
                    segment_end = (~extreme_points[start_idx:]).idxmax() if (~extreme_points[start_idx:]).any() else len(df) - 1
                    
                    # Izračunaj širinu segmenta u minutama
                    segment_width = (df.loc[segment_end, time_column] - 
                                   df.loc[start_idx, time_column]).total_seconds() / 60
                    
                    # Ako je segment širi od LG_MAX, označi ga za NaN
                    if segment_width > LG_MAX:
                        nan_mask.loc[start_idx:segment_end] = True
                    
                    # Ako nije širi od LG_MAX, ali ima ekstremne promjene,
                    # označi samo tačke s ekstremnim promjenama
                    else:
                        nan_mask.loc[start_idx:segment_end] = extreme_points.loc[start_idx:segment_end]
            
            # Primijeni NaN masku na podatke
            df.loc[nan_mask, data_column] = np.nan

        ##############################################################################
        # ELIMINIERUNG VON GAPS ######################################################
        ##############################################################################

        if GAP_MAX is not None and GAP_MAX > 0:

            # Status des Identifikationsrahmens (frm...frame)
            frm = 0
            """
                0...Im aktuellen Zeitschritt ist kein Identifikationsrahmen für
                    Messlücken offen
                1...Im aktuellen Zeitschritt ist ein Identifikationsrahmen für 
                    Messlücken offen
            """
            
            # Konvertuj vremenske kolone u datetime format za brže procesiranje
            df[time_column] = pd.to_datetime(df[time_column], format=UTC_fmt)
            
            # Identifikuj početke praznina (gdje vrijednost postaje NaN)
            gap_starts = df[data_column].isna() & df[data_column].shift(1).notna()
            gap_start_indices = gap_starts[gap_starts].index
            
            # Identifikuj krajeve praznina (gdje vrijednost prestaje biti NaN)
            gap_ends = df[data_column].notna() & df[data_column].shift(1).isna()
            gap_end_indices = gap_ends[gap_ends].index
            
            # Procesiraj svaku prazninu
            for start, end in zip(gap_start_indices, gap_end_indices):
                if start >= end:
                    continue
                    
                # Izračunaj širinu praznine u minutama
                frm_width = (df.loc[end, time_column] - df.loc[start-1, time_column]).total_seconds() / 60
                
                # Primijeni linearnu interpolaciju ako je praznina dovoljno mala
                if frm_width <= GAP_MAX:
                    # Uzmi vrijednosti prije i poslije praznine
                    start_val = df.loc[start-1, data_column]
                    end_val = df.loc[end, data_column]
                    
                    # Izračunaj vremensku razliku za svaku tačku u praznini
                    time_deltas = (df.loc[start:end-1, time_column] - df.loc[start-1, time_column]).dt.total_seconds() / 60
                    
                    # Izračunaj i primijeni linearnu interpolaciju
                    slope = (end_val - start_val) / frm_width
                    df.loc[start:end-1, data_column] = start_val + time_deltas * slope

                    # Ende des Datensatzes ist erreicht und Identifikationsrahmen ist offen
        
        # === TIME-BASED INTERPOLATION ON A REGULAR GRID ===
        # Parameters for the time grid
        tss = float(req.form.get('tss', 1))  # Time step in minutes, provided by the user
        intrpl_max = float(req.form.get('intrplMax', 10))  # Max gap (in minutes) for interpolation, provided by the user

        # Prepare a regular time grid from the minimum to the maximum timestamp
        df[time_column] = pd.to_datetime(df[time_column])
        start_time = df[time_column].min()
        end_time = df[time_column].max()
        utc_grid = pd.date_range(start=start_time, end=end_time, freq=f'{int(tss)}min')
        df_utc = pd.DataFrame({time_column: utc_grid})

        # Merge the original data onto the regular time grid using nearest match within half the step size
        df_for_merge = df[[time_column, data_column]].copy()
        df_for_merge = df_for_merge.sort_values(time_column)
        df_resampled = pd.merge_asof(
            df_utc,
            df_for_merge,
            left_on=time_column,
            right_on=time_column,
            direction='nearest',
            tolerance=pd.Timedelta(minutes=tss/2)
        )
        df_resampled.set_index(time_column, inplace=True)

        # Perform time-based interpolation on the resampled data
        df_resampled[data_column] = df_resampled[data_column].interpolate(
            method='time',
            limit=int(intrpl_max/tss)
        )
        df_resampled.reset_index(inplace=True)
        df_resampled[time_column] = df_resampled[time_column].dt.strftime(UTC_fmt)
        df_resampled = df_resampled.replace({np.nan: None})

        def generate_chunks():
            CHUNK_SIZE = 1000  # Number of rows per chunk in the stream
            total_rows = len(df_resampled)
            total_chunks = (total_rows + CHUNK_SIZE - 1) // CHUNK_SIZE
            # Send metadata as the first JSON object
            yield json.dumps({
                'total_rows': total_rows,
                'total_chunks': total_chunks,
                'chunk_size': CHUNK_SIZE,
                'message': 'Daten werden gestreamt',
                'type': 'metadata'
            }, separators=(',', ':')) + '\n'
            # Stream each chunk of data as a separate JSON line
            for chunk_idx in range(total_chunks):
                start_idx = chunk_idx * CHUNK_SIZE
                end_idx = min(start_idx + CHUNK_SIZE, total_rows)
                chunk_data = {
                    'chunk_index': chunk_idx,
                    'data': df_resampled.iloc[start_idx:end_idx].to_dict('records'),
                    'type': 'data'
                }
                json_line = json.dumps(chunk_data, separators=(',', ':')) + '\n'
                yield json_line
            # Send a final message to indicate completion
            yield json.dumps({
                'message': 'Daten wurden erfolgreich verarbeitet',
                'status': 'complete',
                'type': 'complete'
            }, separators=(',', ':')) + '\n'

        return Response(
            generate_chunks(),
            mimetype='application/x-ndjson'
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 400

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
        print(f"Error in prepare_save: {str(e)}")
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@bp.route('/download/<file_id>', methods=['GET'])
def download_file(file_id):
    if file_id not in temp_files:
        return jsonify({"error": "File not found"}), 404
    file_info = temp_files[file_id]
    file_path = file_info['path']
    if not os.path.exists(file_path):
        del temp_files[file_id]
        return jsonify({"error": "File not found"}), 404
    try:
        download_name = file_info['fileName']
        response = send_file(
            file_path,
            as_attachment=True,
            download_name=download_name,
            mimetype='text/csv'
        )
        response.headers['Access-Control-Expose-Headers'] = 'Content-Disposition'
        return response
    except Exception as e:
        return jsonify({"error": f"Error downloading file: {str(e)}"}), 500
    finally:
        try:
            os.unlink(file_path)
            del temp_files[file_id]
        except Exception as e:
            return jsonify({"error": f"Error cleaning up temp file: {str(e)}"}), 500

def create_mock_request(form_params, file_content, filename):
    from werkzeug.datastructures import FileStorage
    from io import BytesIO

    class MockRequest:
        def __init__(self):
            self.files = {}
            self.form = {}

    mock_request = MockRequest()
    mock_request.form = form_params
    mock_request.files['file'] = FileStorage(
        stream=BytesIO(file_content.encode('utf-8')),
        filename=filename,
        content_type='text/csv'
    )
    return mock_request
