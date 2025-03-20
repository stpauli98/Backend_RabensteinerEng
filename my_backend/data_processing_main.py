import pandas as pd
from datetime import datetime as dat
from io import StringIO
import numpy as np
from flask import Blueprint, jsonify, send_file, request, Response, stream_with_context
import tempfile
import csv
import os
import traceback
import logging
import json

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

# Create Blueprint
bp = Blueprint('data_processing_main', __name__)

@bp.route('/upload-chunk', methods=['POST'])
def handle_upload_chunk():
    try:
        logger.info(f"Received upload request with form data: {request.form}")
        logger.info(f"Files in request: {request.files}")
        return upload_chunk(request)
    except Exception as e:
        logger.error(f"Error in handle_upload_chunk: {str(e)}\nTraceback: {traceback.format_exc()}")
        return jsonify({"error": str(e)}), 400

def upload_chunk(req):
    try:
        logger.info("Starting upload_chunk processing")
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
            logger.info(f"Processing chunk upload. Upload ID: {upload_id}, Index: {chunk_index}, Total: {total_chunks}")
            if 'fileChunk' not in req.files:
                logger.error(f"No fileChunk in request.files. Available: {list(req.files.keys())}")
                return jsonify({"error": "Chunk file not found"}), 400

            chunk = req.files['fileChunk']
            if not chunk:
                logger.error("Received empty chunk")
                return jsonify({"error": "Empty chunk received"}), 400

            logger.info(f"Received chunk with filename: {chunk.filename}, content type: {chunk.content_type}")

            # Save the chunk
            os.makedirs(UPLOAD_FOLDER, exist_ok=True)
            chunk_filename = os.path.join(UPLOAD_FOLDER, f"{upload_id}_{chunk_index}.chunk")
            chunk.save(chunk_filename)

            # Check if all chunks have been received
            received_chunks = [f for f in os.listdir(UPLOAD_FOLDER) if f.startswith(upload_id + "_")]
            if len(received_chunks) == total_chunks:
                # Combine all chunks in sorted order
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
                    logger.error(f"Error processing chunks: {str(e)}")
                    return jsonify({"error": f"Error processing chunks: {str(e)}"}), 400

                # Prepare form parameters dictionary
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
                logger.info(f"Creating mock request with content length: {len(full_content)}")
                mock_request = create_mock_request(form_params, full_content, 'combined_chunks.csv')

                try:
                    logger.info("Starting zweite_bearbeitung processing")
                    result = zweite_bearbeitung(mock_request)
                    logger.info(f"zweite_bearbeitung returned result of type: {type(result)}")
                    # Return result directly (assume streaming response)
                    return result
                except Exception as e:
                    for cf in chunks_sorted:
                        try:
                            os.remove(os.path.join(UPLOAD_FOLDER, cf))
                        except Exception:
                            pass
                    logger.error(f"Error processing chunks: {str(e)}")
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
                logger.error(f"Error in direct upload: {str(e)}\n{traceback.format_exc()}")
                return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 400

    except Exception as e:
        logger.error(f"Error in upload_chunk: {str(e)}\n{traceback.format_exc()}")
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 400

def zweite_bearbeitung(req):
    try:
        logger.info("Starting zweite_bearbeitung function")
        logger.info(f"Available files in request: {list(req.files.keys())}")
        file_key = 'file' if 'file' in req.files else 'fileChunk' if 'fileChunk' in req.files else None
        if not file_key:
            return jsonify({"error": "No file uploaded (neither 'file' nor 'fileChunk' found)"}), 400

        file = req.files[file_key]
        logger.info("Reading file content")
        file_content = file.stream.read().decode('utf-8')
        logger.info(f"File content length: {len(file_content)}")
        if not file_content.strip():
            return jsonify({"error": "Empty file content"}), 400

        try:
            EQ_MAX = float(req.form.get('eqMax', '0'))
            CHG_MAX = float(req.form.get('chgMax', '0'))
            LG_MAX = float(req.form.get('lgMax', '0'))
            GAP_MAX = float(req.form.get('gapMax', '0'))
            ELMAX = float(req.form.get('elMax', '0'))
            ELMIN = float(req.form.get('elMin', '0'))
            EL0 = req.form.get('radioValueNull')
            ELNN = req.form.get('radioValueNotNull')
            EL0 = 1 if EL0 == "ja" else 0
            ELNN = 1 if ELNN == "ja" else 0
        except (TypeError, ValueError) as e:
            return jsonify({"error": f"Invalid parameter value: {str(e)}"}), 400

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
            logger.info("Cleaning data...")
            cleaned_content = file_content.replace('\r', '').strip()
            logger.info("Sample of cleaned data:")
            for line in cleaned_content.split('\n')[:5]:
                logger.info(f"  {repr(line)}")
            logger.info(f"Attempting to read CSV with delimiter: {delimiter}")
            df = pd.read_csv(StringIO(cleaned_content),
                             delimiter=delimiter,
                             header=0,
                             skipinitialspace=True,
                             skip_blank_lines=True)
            logger.info(f"Successfully read CSV with shape: {df.shape}")
            logger.info(f"Columns: {df.columns.tolist()}")
            if df.empty:
                return jsonify({"error": "No data found in file"}), 400
            if len(df.columns) < 2:
                return jsonify({"error": f"Data must have at least 2 columns, but found only {len(df.columns)}"}), 400

            time_column = df.columns[0]
            data_column = df.columns[1]

            logger.info(f"Converting {data_column} to numeric")
            df[data_column] = (df[data_column].astype(str)
                               .str.strip()
                               .str.replace('\r', '')
                               .str.replace('\n', '')
                               .str.replace(',', '.'))
            df[data_column] = pd.to_numeric(df[data_column], errors='coerce')
            logger.info(f"Sample values after conversion: {df[data_column].head().tolist()}")
            logger.info(f"Data column info:\n{df[data_column].describe()}")
            if df[data_column].isna().all():
                return jsonify({"error": "Could not convert any values to numeric format"}), 400
        except Exception as e:
            err_msg = f"Error processing data: {str(e)}\n{traceback.format_exc()}"
            logger.error(err_msg)
            return jsonify({"error": err_msg}), 400

        # Konvertiraj stupac u numerički format (potrebno za daljnju obradu)
        df.iloc[:, 1] = pd.to_numeric(df.iloc[:, 1], errors='coerce')
        logger.info(f"Column converted to numeric. NaN count: {df.iloc[:, 1].isna().sum()}")

        # Primjeni gornju i donju granicu samo ako su definirane (veće od 0)
        try:
            if ELMAX > 0:
                logger.info(f"Applying upper limit: ELMAX={ELMAX}")
                df.iloc[:, 1] = df.iloc[:, 1].mask(df.iloc[:, 1] > ELMAX, np.nan)
                logger.info(f"Applied upper limit {ELMAX}. NaN count: {df.iloc[:, 1].isna().sum()}")
                
            if ELMIN > 0:
                logger.info(f"Applying lower limit: ELMIN={ELMIN}")
                df.iloc[:, 1] = df.iloc[:, 1].mask(df.iloc[:, 1] < ELMIN, np.nan)
                logger.info(f"Applied lower limit {ELMIN}. NaN count: {df.iloc[:, 1].isna().sum()}")
        except Exception as e:
            logger.error(f"Error applying limits: {str(e)}\nTraceback: {traceback.format_exc()}")
            # U slučaju greške, postavi sve na NaN
            df.iloc[:, 1] = np.nan


        if "EQ_MAX" in locals():
            logger.info(f"Processing equal values with EQ_MAX={EQ_MAX}")
            
            # Status des Identifikationsrahmens (frm...frame)
            frm = 0
            """
                0...Im aktuellen Zeitschritt ist kein Identifikationsrahmen für
                    gleichbleibende Messwerte offen
                1...Im aktuellen Zeitschritt ist ein Identifikationsrahmen für 
                    gleichbleibende Messwerte offen
            """
            
            # Konvertuj vremenske kolone u datetime format za brže procesiranje
            try:
                logger.info(f"Converting time column '{time_column}' to datetime")
                df[time_column] = pd.to_datetime(df[time_column])
                logger.info("Time column conversion successful")
            except Exception as e:
                logger.error(f"Error converting time column: {str(e)}\nFirst few values: {df[time_column].head()}")
                raise
            
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

        if "CHG_MAX" in locals() and "LG_MAX" in locals():
            # Status des Identifikationsrahmens (frm...frame)
            frm = 0
            """
                0...Im aktuellen Zeitschritt ist kein Identifikationsrahmen für
                    Ausreisser offen
                1...Im aktuellen Zeitschritt ist ein Identifikationsrahmen für 
                    Ausreisser offen
            """
            
                       # Konvertuj vremenske kolone u datetime format za brže procesiranje
            df[time_column] = pd.to_datetime(df[time_column])
            
            # Izračunaj promjene vrijednosti i vremenske razlike
            value_changes = df[data_column].diff().abs()
            time_diffs = df[time_column].diff().dt.total_seconds() / 60
            
            # Izračunaj promjene vrijednosti
            value_changes = df[data_column].diff().abs()
            
            # Nađi tačke gde razlika premašuje CHG_MAX
            extreme_points = value_changes > CHG_MAX
            
            if extreme_points.any():
                # Nađi indekse ekstrema
                extreme_indices = extreme_points[extreme_points].index.tolist()
                
                if extreme_indices:
                    # Grupiši susedne ekstremne tačke
                    segments = []
                    current_segment = [extreme_indices[0]-1]  # Počni sa tačkom pre prvog ekstrema
                    
                    for i in range(len(extreme_indices)):
                        current_idx = extreme_indices[i]
                        current_segment.append(current_idx)
                        
                        # Ako je ovo poslednji indeks ili sledeći indeks nije susedan
                        if (i == len(extreme_indices)-1 or 
                            extreme_indices[i+1] > current_idx + 1):
                            segments.append(current_segment)
                            if i < len(extreme_indices)-1:
                                current_segment = [extreme_indices[i+1]-1]
                    
                    # Postavi NaN za sve tačke u segmentima sa ekstremima
                    for segment in segments:
                        start_idx = segment[0]
                        end_idx = segment[-1]
                        df.loc[start_idx:end_idx, data_column] = np.nan

        ##############################################################################
        # ELIMINIERUNG VON GAPS ######################################################
        ##############################################################################

        if "GAP_MAX" in locals():

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
        
        # Konvertuj DataFrame u format pogodan za JSON
        df = df.replace({np.nan: None})
        if pd.api.types.is_datetime64_any_dtype(df[time_column]):
            df[time_column] = df[time_column].dt.strftime(UTC_fmt)
        else:
            try:
                df[time_column] = pd.to_datetime(df[time_column]).dt.strftime(UTC_fmt)
            except Exception:
                pass

        logger.info("Preparing to stream data in chunks")
        def generate_chunks():
            logger.info("Starting to generate chunks")
            CHUNK_SIZE = 1000  # Number of rows per chunk
            total_rows = len(df)
            total_chunks = (total_rows + CHUNK_SIZE - 1) // CHUNK_SIZE
            logger.info(f"Data will be sent in {total_chunks} chunks of {CHUNK_SIZE} rows each")
            # Send metadata as the first JSON object
            yield json.dumps({
                'total_rows': total_rows,
                'total_chunks': total_chunks,
                'chunk_size': CHUNK_SIZE,
                'message': 'Daten werden gestreamt',
                'type': 'metadata'
            }, separators=(',', ':')) + '\n'
            for chunk_idx in range(total_chunks):
                start_idx = chunk_idx * CHUNK_SIZE
                end_idx = min(start_idx + CHUNK_SIZE, total_rows)
                chunk_data = {
                    'chunk_index': chunk_idx,
                    'data': df.iloc[start_idx:end_idx].to_dict('records'),
                    'type': 'data'
                }
                json_line = json.dumps(chunk_data, separators=(',', ':')) + '\n'
                logger.info(f"Sending chunk {chunk_idx + 1}/{total_chunks}, sample: {repr(json_line[:200])}")
                yield json_line
            yield json.dumps({
                'message': 'Daten wurden erfolgreich verarbeitet',
                'status': 'complete',
                'type': 'complete'
            }, separators=(',', ':')) + '\n'
        
        logger.info("Starting to stream response using generator")
        return Response(
            generate_chunks(),
            mimetype='application/x-ndjson'
        )
    except Exception as e:
        logger.error(f"Error in zweite_bearbeitung: {str(e)}\n{traceback.format_exc()}")
        return jsonify({'error': str(e)}), 400

@bp.route('/prepare-save', methods=['POST'])
def prepare_save():
    """Prepare CSV file for download from JSON data."""
    try:
        logger.info(f"Received prepare-save request")
        
        # Check if request has JSON data
        if not request.is_json:
            logger.error("Request does not contain JSON data")
            return jsonify({"error": "Request must be JSON"}), 400
            
        # Get the JSON data
        try:
            data = request.get_json()
            logger.info(f"Received data structure: {type(data)}")
        except Exception as e:
            logger.error(f"Error parsing JSON data: {str(e)}")
            return jsonify({"error": "Invalid JSON format"}), 400
            
        # Validate data structure
        if not isinstance(data, dict):
            logger.error(f"Data is not a dictionary: {type(data)}")
            return jsonify({"error": "Invalid data format: expected JSON object"}), 400
            
        if 'data' not in data:
            logger.error("Missing 'data' key in request")
            return jsonify({"error": "Missing 'data' field in request"}), 400
            
        save_data = data['data']
        
        # Validate save_data
        if not isinstance(save_data, list):
            logger.error(f"save_data is not a list: {type(save_data)}")
            return jsonify({"error": "Invalid data format: expected array"}), 400
            
        if not save_data:
            logger.error("Empty save_data array")
            return jsonify({"error": "Empty data array"}), 400
            
        logger.info(f"Processing {len(save_data)} rows of data")
        
        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.csv')
        writer = csv.writer(temp_file, delimiter=';')
        
        try:
            # Write data to CSV
            for i, row in enumerate(save_data):
                if not isinstance(row, list):
                    raise ValueError(f"Invalid row format at index {i}: expected array")
                writer.writerow(row)
                
            temp_file.close()
            
            # Generate file ID and store reference
            file_id = dat.now().strftime('%Y%m%d_%H%M%S')
            temp_files[file_id] = temp_file.name
            
            logger.info(f"Successfully prepared file with ID: {file_id}")
            return jsonify({
                "message": "File prepared for download",
                "fileId": file_id,
                "rowCount": len(save_data)
            }), 200
            
        except Exception as e:
            # Clean up the temporary file if there's an error
            try:
                temp_file.close()
                os.unlink(temp_file.name)
            except:
                pass
            logger.error(f"Error writing data to CSV: {str(e)}")
            return jsonify({"error": f"Error writing to CSV: {str(e)}"}), 500
            
    except Exception as e:
        logger.error(f"Unexpected error in prepare_save: {str(e)}\n{traceback.format_exc()}")
        return jsonify({"error": "Internal server error"}), 500

@bp.route('/download/<file_id>', methods=['GET'])
def download_file(file_id):
    if file_id not in temp_files:
        return jsonify({"error": "File not found"}), 404
    file_path = temp_files[file_id]
    if not os.path.exists(file_path):
        del temp_files[file_id]
        return jsonify({"error": "File not found"}), 404
    try:
        download_name = f"data_{file_id}.csv"
        return send_file(
            file_path,
            as_attachment=True,
            download_name=download_name,
            mimetype='text/csv'
        )
    except Exception as e:
        return jsonify({"error": f"Error downloading file: {str(e)}"}), 500
    finally:
        try:
            os.unlink(file_path)
            del temp_files[file_id]
        except Exception:
            pass

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
