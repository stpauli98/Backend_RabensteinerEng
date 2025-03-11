import pandas as pd
from datetime import datetime as dat
from io import StringIO
import numpy as np
from flask import jsonify, send_file, request
import tempfile
import csv
import os
import traceback
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Directory for temporary chunk storage
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'temp_uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Dictionary to store temporary files
temp_files = {}
# Format for UTC dates
UTC_fmt = "%Y-%m-%d %H:%M:%S"

def zweite_bearbeitung(request):
    try:
        # Check if file is in request.files
        if 'file' not in request.files:
            response = jsonify({"error": "No file uploaded"})
            return response, 400

        file = request.files['file']

        # Read file content
        file_content = file.stream.read().decode('utf-8')   

        if not file_content.strip():
            response = jsonify({"error": "Empty file content"})
            return response, 400

        # Get parameters from form data
        try:
            EQ_MAX = float(request.form.get('eqMax', '0'))
            CHG_MAX = float(request.form.get('chgMax', '0'))
            LG_MAX = float(request.form.get('lgMax', '0'))
            GAP_MAX = float(request.form.get('gapMax', '0'))
            EL0 = request.form.get('radioValueNull')
            ELNN = request.form.get('radioValueNotNull')
        except (TypeError, ValueError) as e:
            response = jsonify({"error": f"Invalid parameter value: {str(e)}"})
            return response, 400

        EL0 = 1 if EL0 == "ja" else 0
        ELNN = 1 if ELNN == "ja" else 0
        content_lines = file_content.splitlines()
        
        EL0 = request.form.get('radioValueNull')
        ELNN = request.form.get('radioValueNotNull')
        try:
            EQ_MAX = float(request.form.get('eqMax'))
            CHG_MAX = float(request.form.get('chgMax'))
            LG_MAX = float(request.form.get('lgMax'))
            GAP_MAX = float(request.form.get('gapMax'))
        except (TypeError, ValueError) as e:
            response = jsonify({"error": f"Invalid numeric value in parameters: {str(e)}"})
            return response, 400

        EL0 = 1 if EL0 == "ja" else 0
        ELNN = 1 if ELNN == "ja" else 0


        ##############################################################################
        # DATEN LADEN #################################################################
        ##############################################################################

        if not content_lines:
            return jsonify({"error": "No data received"}), 400

        # Detect the delimiter by checking the first line
        first_line = content_lines[0].strip()
        
        if not first_line:
            response = jsonify({"error": "Empty first line"})
            return response, 400
            
        # Try to detect delimiter
        if ';' in first_line:
            delimiter = ';'
        elif ',' in first_line:
            delimiter = ','
        else:
            response = jsonify({"error": "No valid delimiter (comma or semicolon) found in data"})
            return response, 400
        
        try:
            # Try to read the CSV data
            df = pd.read_csv(StringIO(file_content), delimiter=delimiter, header=0)
            
            # Basic validation
            if df.empty:
                response = jsonify({"error": "No data found in file"})
                return response, 400
            
            if len(df.columns) < 2:
                response = jsonify({"error": f"Data must have at least 2 columns, but found only {len(df.columns)}"})
                return response, 400
                
            # Get column names
            time_column = df.columns[0]
            data_column = df.columns[1]
            
            # Convert data column to numeric, handling both . and , as decimal separators
            df[data_column] = pd.to_numeric(df[data_column].astype(str).str.replace(',', '.'), errors='coerce')
            
            # Check if conversion was successful
            if df[data_column].isna().all():
                response = jsonify({"error": "Could not convert any values to numeric format"})
                return response, 400
          
        except Exception as e:
            response = jsonify({"error": f"Error processing data: {str(e)}"})
            return response, 400

       

        ##############################################################################
        # ELIMINIERUNG VON MESSAUSFÄLLEN (GLEICHBLEIBENDE MESSWERTE) ##################
        ##############################################################################

        if "EQ_MAX" in locals():
            
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
        # ELIMINIERUNG VON AUSREISSERN ################################################
        ##############################################################################

        if "CHG_MAX" and "LG_MAX" in locals():
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
        # SCHLIESSEN VON MESSLÜCKEN ###################################################
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

        # Na kraju funkcije, prije except bloka:
        # Konvertujemo DataFrame u format pogodan za JSON
        # Prvo zamijenimo np.nan sa None da bi se moglo serijalizovati u JSON
        df = df.replace({np.nan: None})
        
        # Konvertuj datetime kolonu u željeni format prije konverzije u dictionary
        if pd.api.types.is_datetime64_any_dtype(df[time_column]):
            df[time_column] = df[time_column].dt.strftime(UTC_fmt)
        else:
            try:
                df[time_column] = pd.to_datetime(df[time_column]).dt.strftime(UTC_fmt)
            except Exception:
                pass
        
        # Implementacija paginacije
        page = request.args.get('page', default=1, type=int)
        per_page = request.args.get('per_page', default=1000, type=int)  # Default 1000 redova po strani
        
        # Ograniči maksimalan broj redova po strani
        per_page = min(per_page, 5000)  # Maksimalno 5000 redova po strani
        
        # Izračunaj ukupan broj stranica
        total_rows = len(df)
        total_pages = (total_rows + per_page - 1) // per_page
        
        # Osiguraj da je zatražena stranica validna
        page = max(1, min(page, total_pages))
        
        # Izračunaj indekse za trenutnu stranicu
        start_idx = (page - 1) * per_page
        end_idx = min(start_idx + per_page, total_rows)
        
        # Uzmi samo podatke za trenutnu stranicu
        page_data = df.iloc[start_idx:end_idx]
        
        processed_data = {
            'data': page_data.to_dict('records'),
            'pagination': {
                'page': page,
                'per_page': per_page,
                'total_pages': total_pages,
                'total_rows': total_rows
            },
            'message': 'Daten wurden erfolgreich verarbeitet'
        }
        
        response = jsonify(processed_data)
        return response

    except Exception as e:
        response = jsonify({'error': str(e)})
        return response, 400  
                  
def prepare_save(request):
    """Prepare CSV file for download from JSON data."""
    try:
        # Validate request
        if not request.is_json:
            response = jsonify({"error": "Request must be JSON"})
            return response, 400
            
        data = request.json
        if not data or 'data' not in data:
            response = jsonify({"error": "No data received"})
            return response, 400
            
        save_data = data['data']
        if not save_data:
            response = jsonify({"error": "Empty data"})
            return response, 400
            
        # Create and write to temporary file
        temp_file = tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.csv')
        writer = csv.writer(temp_file, delimiter=';')
        
        try:
            for row in save_data:
                writer.writerow(row)
        except Exception as e:
            os.unlink(temp_file.name)  # Clean up file if write fails
            response = jsonify({"error": f"Error writing to CSV: {str(e)}"})
            return response, 500
            
        temp_file.close()
        
        # Generate unique file ID
        file_id = dat.now().strftime('%Y%m%d_%H%M%S')
        temp_files[file_id] = temp_file.name

        response = jsonify({"message": "File prepared for download", "fileId": file_id})
        return response, 200
    except Exception as e:
        response = jsonify({"error": f"Error preparing file: {str(e)}"})
        return response, 500

def download_file(file_id, request):
    """Download a previously prepared CSV file."""
    if file_id not in temp_files:
        response = jsonify({"error": "File not found"})
        return response, 404
        
    file_path = temp_files[file_id]
    if not os.path.exists(file_path):
        del temp_files[file_id]  # Clean up reference if file doesn't exist
        response = jsonify({"error": "File not found"})
        return response, 404

    try:
        download_name = f"data_{file_id}.csv"
        return send_file(
            file_path,
            as_attachment=True,
            download_name=download_name,
            mimetype='text/csv'
        )
    except Exception as e:
        response = jsonify({"error": f"Error downloading file: {str(e)}"})
        return response, 500
    finally:
        # Clean up temporary file
        try:
            os.unlink(file_path)
            del temp_files[file_id]
        except Exception:
            pass  # Ignore cleanup errors

def upload_chunk(request):
    """
    Endpoint za prihvat i obradu fajla. Podržava i direktan upload i upload u delovima (chunks).
    Očekivani parametri za direktan upload (form data):
      - file: CSV fajl za obradu
      - eqMax: maksimalna vrijednost za jednake vrijednosti (float)
      - chgMax: maksimalna vrijednost za promjene (float)
      - lgMax: maksimalna vrijednost za logaritamske promjene (float)
      - gapMax: maksimalna vrijednost za praznine (float)
      - radioValueNull: eliminacija nul vrijednosti (ja/nein)
      - radioValueNotNull: eliminacija ne-numeričkih vrijednosti (ja/nein)
    
    Očekivani parametri za chunk upload:
      - fileChunk: Deo fajla
      - uploadId: Jedinstveni ID za ovaj upload
      - chunkIndex: Redni broj chunka (počinje od 0)
      - totalChunks: Ukupan broj chunkova
      + svi parametri kao za direktan upload
    """

    try:
        # Validacija parametara
        upload_id = request.form.get('uploadId')
        chunk_index = int(request.form.get('chunkIndex', 0))
        total_chunks = int(request.form.get('totalChunks', 0))
        EQ_MAX = float(request.form.get('eqMax', '0'))
        CHG_MAX = float(request.form.get('chgMax', '0'))
        LG_MAX = float(request.form.get('lgMax', '0'))
        GAP_MAX = float(request.form.get('gapMax', '0'))
        EL0 = request.form.get('radioValueNull')
        ELNN = request.form.get('radioValueNotNull')

        # Konvertuj radio button vrednosti
        EL0 = 1 if EL0 == "ja" else 0
        ELNN = 1 if ELNN == "ja" else 0

        # Proveri da li je chunk upload
        if upload_id:
            # Chunk upload
            chunk_index = int(request.form.get('chunkIndex', 0))
            total_chunks = int(request.form.get('totalChunks', 0))
            
            if 'fileChunk' not in request.files:
                response = jsonify({"error": "Chunk file not found"})
                return response, 400
                
            chunk = request.files['fileChunk']
            if not chunk:
                response = jsonify({"error": "Empty chunk received"})
                return response, 400

            # Sačuvaj chunk
            os.makedirs(UPLOAD_FOLDER, exist_ok=True)
            chunk_filename = os.path.join(UPLOAD_FOLDER, f"{upload_id}_{chunk_index}.chunk")
            chunk.save(chunk_filename)

            # Proveri da li su svi chunkovi primljeni
            received_chunks = [f for f in os.listdir(UPLOAD_FOLDER) if f.startswith(upload_id + "_")]
            
            if len(received_chunks) == total_chunks:
                # Spoji sve chunkove
                chunks_sorted = sorted(received_chunks, key=lambda x: int(x.split("_")[1].split(".")[0]))
                full_content = ""
                
                try:
                    for chunk_file in chunks_sorted:
                        chunk_path = os.path.join(UPLOAD_FOLDER, chunk_file)
                        with open(chunk_path, 'rb') as f:
                            full_content += f.read().decode('utf-8')
                        os.remove(chunk_path)
                        
                    # Kreiraj mock request za obradu
                    class MockRequest:
                        def __init__(self):
                            self.files = {}
                            self.form = {}

                    mock_request = MockRequest()
                    mock_request.form = {
                        'eqMax': str(EQ_MAX),
                        'chgMax': str(CHG_MAX),
                        'lgMax': str(LG_MAX),
                        'gapMax': str(GAP_MAX),
                        'radioValueNull': 'ja' if EL0 == 1 else 'nein',
                        'radioValueNotNull': 'ja' if ELNN == 1 else 'nein'
                    }

                    from werkzeug.datastructures import FileStorage
                    from io import BytesIO
                    mock_request.files['file'] = FileStorage(
                        stream=BytesIO(full_content.encode('utf-8')),
                        filename='combined_chunks.csv',
                        content_type='text/csv',
                    )

                    try:
                        response = zweite_bearbeitung(mock_request)
                        if isinstance(response, tuple):
                            response_data, status_code = response
                            return jsonify(response_data), status_code
                        return response
                    except Exception as e:
                        # U slučaju greške, obriši sve chunkove
                        for chunk_file in chunks_sorted:
                            try:
                                os.remove(os.path.join(UPLOAD_FOLDER, chunk_file))
                            except:
                                pass
                        logger.error(f"Error processing chunks: {str(e)}")
                        return jsonify({"error": f"Error processing chunks: {str(e)}"}), 400

                except Exception as e:
                    # U slučaju greške, obriši sve chunkove
                    for chunk_file in chunks_sorted:
                        try:
                            os.remove(os.path.join(UPLOAD_FOLDER, chunk_file))
                        except:
                            pass
                    logger.error(f"Error processing chunks: {str(e)}")
                    return jsonify({"error": f"Error processing chunks: {str(e)}"}), 400

            # Ako nisu svi chunkovi primljeni, vrati status
            response = jsonify({
                "message": f"Chunk {chunk_index + 1}/{total_chunks} received",
                "uploadId": upload_id,
                "chunkIndex": chunk_index,
                "totalChunks": total_chunks,
                "remainingChunks": total_chunks - len(received_chunks)
            })
            return response, 200

        else:
            # Direktan upload
            try:
                if 'file' not in request.files:
                    response = jsonify({"error": "Keine Datei gefunden"})
                    return response, 400

                file = request.files['file']
                if not file:
                    response = jsonify({"error": "Keine Datei gefunden"})
                    return response, 400

                # Pročitaj sadržaj fajla
                file_content = file.stream.read().decode('utf-8')

                # Kreiraj mock request za zweite_bearbeitung
                class MockRequest:
                    def __init__(self):
                        self.files = {}
                        self.form = {}

                mock_request = MockRequest()
                mock_request.form = {
                    'eqMax': str(EQ_MAX),
                    'chgMax': str(CHG_MAX),
                    'lgMax': str(LG_MAX),
                    'gapMax': str(GAP_MAX),
                    'radioValueNull': 'ja' if EL0 else 'nein',
                    'radioValueNotNull': 'ja' if ELNN else 'nein'
                }

                # Kreiraj FileStorage objekat sa sadržajem
                from werkzeug.datastructures import FileStorage
                from io import BytesIO
                mock_request.files['file'] = FileStorage(
                    stream=BytesIO(file_content.encode('utf-8')),
                    filename='uploaded_file.csv',
                    content_type='text/csv',
                )

                try:
                    response_data = zweite_bearbeitung(mock_request)
                    # Konvertuj response u format koji frontend očekuje
                    if isinstance(response_data, tuple):
                        response, status_code = response_data
                        if status_code == 200:
                            response = jsonify({"data": response})
                            return response, 200
                        return response, status_code
                    response = jsonify({"data": response_data})
                    return response, 200
                except Exception as e:
                    error_msg = f"Error in direct upload: {str(e)}\nTraceback: {traceback.format_exc()}"
                    logger.error(error_msg)
                    response = jsonify({
                        "error": str(e),
                        "traceback": traceback.format_exc()
                    })
                    return response, 400

            except Exception as e:
                error_msg = f"Error in direct upload: {str(e)}\nTraceback: {traceback.format_exc()}"
                logger.error(error_msg)
                response = jsonify({
                    "error": str(e),
                    "traceback": traceback.format_exc()
                })
                return response, 400

    except Exception as e:
        error_msg = f"Error in upload_chunk: {str(e)}\nTraceback: {traceback.format_exc()}"
        logger.error(error_msg)
        response = jsonify({
            "error": str(e),
            "traceback": traceback.format_exc()
        })
        return response, 400