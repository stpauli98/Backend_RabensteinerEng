import pandas as pd
from datetime import datetime as dat
from io import StringIO
import numpy as np
from flask import jsonify, send_file
import tempfile
import csv
import os
import traceback

# Dictionary to store temporary files
temp_files = {}
# Format for UTC dates
UTC_fmt = "%Y-%m-%d %H:%M:%S"

def zweite_bearbeitung(request):
    try:

        # Check if file is in request.files
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files['file']

        # Read file content
        file_content = file.stream.read().decode('utf-8')   

        if not file_content.strip():
            return jsonify({"error": "Empty file content"}), 400

        # Get parameters from form data
        try:
            EQ_MAX = float(request.form.get('eqMax', '0'))
            CHG_MAX = float(request.form.get('chgMax', '0'))
            LG_MAX = float(request.form.get('lgMax', '0'))
            GAP_MAX = float(request.form.get('gapMax', '0'))
            EL0 = request.form.get('radioValueNull')
            ELNN = request.form.get('radioValueNotNull')
        except (TypeError, ValueError) as e:
            return jsonify({"error": f"Invalid parameter value: {str(e)}"}), 400

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
            return jsonify({"error": f"Invalid numeric value in parameters: {str(e)}"}), 400

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
            return jsonify({"error": "Empty first line"}), 400
            
        # Try to detect delimiter
        if ';' in first_line:
            delimiter = ';'
        elif ',' in first_line:
            delimiter = ','
        else:
            return jsonify({"error": "No valid delimiter (comma or semicolon) found in data"}), 400
        
        try:
            # Try to read the CSV data
            df = pd.read_csv(StringIO(file_content), delimiter=delimiter, header=0)
            
            # Basic validation
            if df.empty:
                return jsonify({"error": "No data found in file"}), 400
            
            if len(df.columns) < 2:
                return jsonify({"error": f"Data must have at least 2 columns, but found only {len(df.columns)}"}), 400
                
            # Get column names
            time_column = df.columns[0]
            data_column = df.columns[1]
            
            # Convert data column to numeric, handling both . and , as decimal separators
            df[data_column] = pd.to_numeric(df[data_column].astype(str).str.replace(',', '.'), errors='coerce')
            
            # Check if conversion was successful
            if df[data_column].isna().all():
                return jsonify({"error": "Could not convert any values to numeric format"}), 400
          
        except Exception as e:
            return jsonify({"error": f"Error processing data: {str(e)}"}), 400

       

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
            # Ako kolona nije datetime tip, pokušaj konvertovati
            try:
                df[time_column] = pd.to_datetime(df[time_column]).dt.strftime(UTC_fmt)
            except Exception:
                pass  # Zadrži originalni format ako konverzija ne uspije
        
        processed_data = {
            'data': df.to_dict('records'),  # Konvertuje DataFrame u listu dictionary-ja
            'message': 'Daten wurden erfolgreich verarbeitet'
        }
        
        return jsonify(processed_data)

    except Exception as e:
        return jsonify({'error': str(e)}), 400  
                  
def prepare_save(request):
    """Prepare CSV file for download from JSON data."""
    try:
        # Validate request
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400
            
        data = request.json
        if not data or 'data' not in data:
            return jsonify({"error": "No data received"}), 400
            
        save_data = data['data']
        if not save_data:
            return jsonify({"error": "Empty data"}), 400
            
        # Create and write to temporary file
        temp_file = tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.csv')
        writer = csv.writer(temp_file, delimiter=';')
        
        try:
            for row in save_data:
                writer.writerow(row)
        except Exception as e:
            os.unlink(temp_file.name)  # Clean up file if write fails
            return jsonify({"error": f"Error writing to CSV: {str(e)}"}), 500
            
        temp_file.close()
        
        # Generate unique file ID
        file_id = dat.now().strftime('%Y%m%d_%H%M%S')
        temp_files[file_id] = temp_file.name

        return jsonify({"message": "File prepared for download", "fileId": file_id}), 200
    except Exception as e:
        return jsonify({"error": f"Error preparing file: {str(e)}"}), 500

def download_file(file_id, request):
    """Download a previously prepared CSV file."""
    if file_id not in temp_files:
        return jsonify({"error": "File not found"}), 404
        
    file_path = temp_files[file_id]
    if not os.path.exists(file_path):
        del temp_files[file_id]  # Clean up reference if file doesn't exist
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
        # Clean up temporary file
        try:
            os.unlink(file_path)
            del temp_files[file_id]
        except Exception:
            pass  # Ignore cleanup errors