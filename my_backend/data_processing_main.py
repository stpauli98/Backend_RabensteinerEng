import pandas as pd
from datetime import datetime as dat
from io import StringIO
import numpy as np
from flask import jsonify, send_file
import tempfile
import csv
import os
import traceback
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Dictionary to store temporary files
temp_files = {}

def zweite_bearbeitung(request):
    try:
        # Log request information
        logger.info(f"Request method: {request.method}")
        logger.info(f"Content type: {request.content_type}")
        logger.info(f"Files: {request.files}")
        logger.info(f"Form data: {request.form}")

        # Check if file is in request.files
        if 'file' not in request.files:
            logger.error("No file in request.files")
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files['file']
        logger.info(f"Received file: {file.filename}")

        # Read file content
        file_content = file.stream.read().decode('utf-8')
        logger.info(f"File content preview: {file_content[:200]}...")

        if not file_content.strip():
            logger.error("Empty file content")
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
            logger.error(f"Parameter conversion error: {e}")
            return jsonify({"error": f"Invalid parameter value: {str(e)}"}), 400

        logger.info(f"Parameters: EQ_MAX={EQ_MAX}, CHG_MAX={CHG_MAX}, LG_MAX={LG_MAX}, GAP_MAX={GAP_MAX}")
        logger.info(f"Radio buttons: EL0={EL0}, ELNN={ELNN}")

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

        print(f"EL0: {EL0}")
        print(f"ELNN: {ELNN}")


        ##############################################################################
        # DATEN LADEN #################################################################
        ##############################################################################

        if not content_lines:
            return jsonify({"error": "No data received"}), 400

        # Detect the delimiter by checking the first line
        first_line = content_lines[0].strip()
        print(f"First line: '{first_line}'")
        
        if not first_line:
            return jsonify({"error": "Empty first line"}), 400
            
        # Try to detect delimiter
        if ';' in first_line:
            delimiter = ';'
        elif ',' in first_line:
            delimiter = ','
        else:
            print("No valid delimiter found in first line")
            return jsonify({"error": "No valid delimiter (comma or semicolon) found in data"}), 400
        
        try:
            # Try to read the CSV data
            df = pd.read_csv(StringIO(file_content), delimiter=delimiter, header=0)
            
            # Basic validation
            if df.empty:
                return jsonify({"error": "No data found in file"}), 400
            
            if len(df.columns) < 2:
                print(f"Error: Not enough columns. Found only: {len(df.columns)}")
                return jsonify({"error": f"Data must have at least 2 columns, but found only {len(df.columns)}"}), 400
                
            # Get column names
            time_column = df.columns[0]
            data_column = df.columns[1]
            
            # Convert data column to numeric, handling both . and , as decimal separators
            df[data_column] = pd.to_numeric(df[data_column].astype(str).str.replace(',', '.'), errors='coerce')
            
            # Check if conversion was successful
            if df[data_column].isna().all():
                return jsonify({"error": "Could not convert any values to numeric format"}), 400
                
            print("\nAfter numeric conversion:")
            print(df.head())
            
        except Exception as e:
            print(f"Error processing data: {str(e)}")
            return jsonify({"error": f"Error processing data: {str(e)}"}), 400

        UTC_fmt = "%Y-%m-%d %H:%M:%S"

        ##############################################################################
        # ELIMINIERUNG VON MESSAUSFÄLLEN (GLEICHBLEIBENDE MESSWERTE) ##################
        ##############################################################################

        if "EQ_MAX" in locals():
            print("\nPočinjem EQ_MAX obradu...")
            print(f"EQ_MAX vrednost: {EQ_MAX}")
            
            # Status des Identifikationsrahmens (frm...frame)
            frm = 0
            """
                0...Im aktuellen Zeitschritt ist kein Identifikationsrahmen für
                    gleichbleibende Messwerte offen
                1...Im aktuellen Zeitschritt ist ein Identifikationsrahmen für 
                    gleichbleibende Messwerte offen
            """
            
            # Durchlauf des gesamten Datenrahmens             
            for i in range (1, len(df)):
                
                # Aktueller Messwert ist gleich dem letzen Messwert und 
                # Identifikationsrahmen ist geschlossen → Identifikationsrahmen wird
                # geöffnet
                if df.iloc[i-1][data_column] == df.iloc[i][data_column] and frm == 0:           
                    
                    idx_strt = i-1
                    frm = 1
                
                # Aktueller Messwert ist ungleich dem letzten Messwert und
                # Identifikationsrahmen ist offen → Identifikationsrahmen wird
                # geschlossen und ausgewertet
                elif df.iloc[i-1][data_column] != df.iloc[i][data_column] and frm == 1:
                    
                    idx_end = i-1
                    
                    # Länge des Identifikationsrahmens [min]
                    frm_width = (dat.strptime(df.iloc[idx_end][time_column], UTC_fmt)-\
                                dat.strptime(df.iloc[idx_strt][time_column], UTC_fmt)).total_seconds()/60
                    
                    # Der Identifikationsrahmen ist zu groß, so dass eine Messlücke
                    # eingefügt werden muss
                    if frm_width >= EQ_MAX:
                        for i_frm in range (idx_strt, idx_end+1):
                            df.at[i_frm, data_column] = np.nan
                    frm = 0
        
                # Ende des Datensatzes ist erreicht und Identifikationsrahmen ist offen
                elif i == len(df)-1 and frm == 1: 
                    idx_end = i
                    
                    # Länge des Identifikationsrahmens [min]
                    frm_width = (dat.strptime(df.iloc[idx_end][time_column], UTC_fmt)-\
                                dat.strptime(df.iloc[idx_strt][time_column], UTC_fmt)).total_seconds()/60
                    
                    # Der Identifikationsrahmen ist zu groß, so dass eine Messlücke
                    # eingefügt werden muss
                    if frm_width >= EQ_MAX:
                        for i_frm in range (idx_strt, idx_end+1):
                            df.at[i_frm, data_column] = np.nan
                            
        print("\nNakon EQ_MAX obrade:")
        print(df.head())

        ##############################################################################
        # ELIMINIERUNG VON NULLWERTEN #################################################
        ##############################################################################

        if EL0 == 1:
            print("\nPočinjem EL0 obradu...")
            
            # Durchlauf des gesamten Datenrahmens
            for i in range (0, len(df)):  
                if df.iloc[i][data_column] == 0:
                    df.at[i, data_column] = np.nan

        print("\nNakon EL0 obrade:")
        print(df.head())

        ##############################################################################
        # ELIMINIERUNG VON NICHT NUMERISCHEN WERTEN ###################################
        ##############################################################################

        if ELNN == 1:
            print("\nPočinjem ELNN obradu...")
            
            # Durchlauf des gesamten Datenrahmens
            for i in range (0, len(df)):  
                try:
                    # Pokušaj konverzije u float
                    float_value = float(df.iloc[i][data_column])
                    # Ako je NaN, ostavi ga kao NaN
                    if pd.isna(float_value):
                        df.at[i, data_column] = np.nan
                except (ValueError, TypeError):
                    # Ako konverzija ne uspije, postavi na NaN
                    df.at[i, data_column] = np.nan      

            print("\nNakon ELNN obrade:")
            print(df.head())

        ##############################################################################
        # ELIMINIERUNG VON AUSREISSERN ################################################
        ##############################################################################

        if "CHG_MAX" and "LG_MAX" in locals():
            print("\nPočinjem CHG_MAX i LG_MAX obradu...")
            print(f"CHG_MAX vrednost: {CHG_MAX}")
            print(f"LG_MAX vrednost: {LG_MAX}")
        
            # Status des Identifikationsrahmens (frm...frame)
            frm = 0
            """
                0...Im aktuellen Zeitschritt ist kein Identifikationsrahmen für
                    Ausreisser offen
                1...Im aktuellen Zeitschritt ist ein Identifikationsrahmen für 
                    Ausreisser offen
            """
            
            # Durchlauf des gesamten Datenrahmens                         
            for i in range (1, len(df)):  
                
                # nan im aktuellen Zeitschritt und Identifikationsrahmen ist im
                # aktuellen Zeitschritt nicht offen
                if pd.isna(df.iloc[i][data_column]) and frm == 0:
                    pass
                
                # nan im aktuellen Zeitschritt und Identifikationsrahmen ist im
                # aktuellen Zeitschritt offen → Identifikationsrahmen wird geschlossen
                # und ausgewertet
                elif pd.isna(df.iloc[i][data_column]) and frm == 1:
                    idx_end = i-1
        
                    for i_frm in range (idx_strt, idx_end+1):
                        df.at[i_frm, data_column] = np.nan

                    # Identifikationsrahmen wird geschlossen
                    frm = 0
                
                # nan im letzten Zeitschritt
                elif pd.isna(df.iloc[i-1][data_column]):
                    pass
                
                # Kein nan im letzten und aktuellen Zeitschritt
                else:
                    # Änderung des Messwertes im aktuellen Zeitschritt
                    chg = abs(df.iloc[i][data_column] - df.iloc[i-1][data_column])
                    
                    # Zeitschrittweite vom letzten zum aktuellen Zeitschritt [min]
                    t = (dat.strptime(df.iloc[i][time_column], UTC_fmt)-\
                        dat.strptime(df.iloc[i-1][time_column], UTC_fmt)).total_seconds()/60
                    
                    # Änderung im aktuellen Zeitschritt ist zu groß und
                    # Identifikationsrahmen ist geschlossen → Identifikationsrahmen
                    # wird geöffnet
                    if chg/t > CHG_MAX and frm == 0:
                        idx_strt = i
                        frm = 1
                        
                    # Änderung im aktuellen Zeitschritt ist zu groß und
                    # Identifikationsrahmen ist offen → nan einfügen
                    elif chg/t > CHG_MAX and frm == 1:
                        idx_end = i-1
                        
                        for i_frm in range (idx_strt, idx_end+1):
                            df.at[i_frm, data_column] = np.nan
                        
                        # Identifikationsrahmen wird geschlossen
                        frm = 0
                    
                    # Identifikationsrahmen ist offen und die maximale Breite des 
                    # Identifikationsrahmens wurde erreicht → Identifikationsrahmen
                    # wird geschlossen
                    elif frm == 1 and (dat.strptime(df.iloc[i][time_column], \
                                                    UTC_fmt)-\
                                    dat.strptime(df.iloc[idx_strt][time_column], \
                                                    UTC_fmt)).total_seconds()/60 > LG_MAX:
                        frm = 0
                            
        print("\nNakon CHG_MAX i LG_MAX obrade:")
        print(df.head())

        ##############################################################################
        # SCHLIESSEN VON MESSLÜCKEN ###################################################
        ##############################################################################

        if "GAP_MAX" in locals():
            print("\nPočinjem GAP_MAX obradu...")
            print(f"GAP_MAX vrednost: {GAP_MAX}")

            # Status des Identifikationsrahmens (frm...frame)
            frm = 0
            """
                0...Im aktuellen Zeitschritt ist kein Identifikationsrahmen für
                    Messlücken offen
                1...Im aktuellen Zeitschritt ist ein Identifikationsrahmen für 
                    Messlücken offen
            """
            
            # Durchlauf des gesamten Datenrahmens                   
            for i in range (1, len(df)):
                
                # Kein Messwert für den aktuellen Zeitschritt vorhanden und 
                # Identifikationsrahmen ist geschlossen → Identifikationsrahmen wird 
                # geöffnet
                if pd.isna(df.iloc[i][data_column]) and frm == 0:
                    idx_strt = i
                    frm = 1
                    
                # Messwert für den aktuellen Zeitschritt vorhanden und 
                # Identifikationsrahmen ist offen → Identifikationsrahmen wird
                # geschlossen und ausgewertet
                elif not pd.isna(df.iloc[i][data_column]) and frm == 1:
                    idx_end = i-1
                    
                    # Länge des Identifikationsrahmens [min]
                    frm_width = (dat.strptime(df.iloc[idx_end+1][time_column], \
                                            UTC_fmt)-\
                                dat.strptime(df.iloc[idx_strt-1][time_column], \
                                            UTC_fmt)).total_seconds()/60
                    
                    # Der Identifikationsrahmen ist klein genug, um die Anwendung einer
                    # linearen Interpolation zu erlauben, um die Messlücke zu füllen
                    if frm_width <= GAP_MAX:
                        
                        # Absolute Änderung des Messwertes
                        dif = float(df.iloc[idx_end+1][data_column])-\
                            float(df.iloc[idx_strt-1][data_column])

                        # Änderung des Messwertes pro Minute
                        dif_min = dif/frm_width
                        
                        # Lineare Interpolation
                        for i_frm in range (idx_strt, idx_end+1):
                            

                            gap_min = (dat.strptime(df.iloc[i_frm][time_column], UTC_fmt)-\
                                    dat.strptime(df.iloc[idx_strt-1][time_column], UTC_fmt)).total_seconds()/60
                            

                            df.at[i_frm, data_column] = float(df.iloc[idx_strt-1][data_column])+\
                                gap_min*dif_min                    

                            i_frm += 1
                    frm = 0

                    # Ende des Datensatzes ist erreicht und Identifikationsrahmen ist offen

        print("\nNakon GAP_MAX obrade:")
        print(df.head())

        # Na kraju funkcije, prije except bloka:
        # Konvertujemo DataFrame u format pogodan za JSON
        # Prvo zamijenimo np.nan sa None da bi se moglo serijalizovati u JSON
        df = df.replace({np.nan: None})
        
        print("\nRezultati obrade:")
        print("----------------")
        print(f"Broj redova nakon obrade: {len(df)}")
        print(f"Kolone: {df.columns.tolist()}")
        print("\nPrvih 5 redova:")
        print(df.head())
        
        processed_data = {
            'data': df.to_dict('records'),  # Konvertuje DataFrame u listu dictionary-ja
            'message': 'Daten wurden erfolgreich verarbeitet'
        }
        
        print("\nPrvi red processed_data:")
        print(processed_data['data'][0] if processed_data['data'] else "Nema podataka")
        
        return jsonify(processed_data)

    except Exception as e:
        print(f"FEHLER: {e}")
        return jsonify({'error': str(e)}), 400  
                  
def prepare_save(request):
    try:
        logger.info("Starting prepare_save function")
        logger.info(f"Request content type: {request.content_type}")
        
        if not request.is_json:
            logger.error("Request is not JSON")
            return jsonify({"error": "Request must be JSON"}), 400
            
        data = request.json
        logger.info(f"Received data: {data}")
        
        if not data or 'data' not in data:
            logger.error("No data field in request JSON")
            return jsonify({"error": "No data received"}), 400
            
        save_data = data['data']
        if not save_data:
            logger.error("Empty data array")
            return jsonify({"error": "Empty data"}), 400
            
        logger.info(f"Processing {len(save_data)} rows of data")

        # Kreiraj privremeni fajl i zapiši CSV podatke
        temp_file = tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.csv')
        writer = csv.writer(temp_file, delimiter=';')
        
        try:
            for row in save_data:
                writer.writerow(row)
        except Exception as e:
            logger.error(f"Error writing to CSV: {str(e)}")
            temp_file.close()
            return jsonify({"error": f"Error writing to CSV: {str(e)}"}), 500
            
        temp_file.close()
        logger.info(f"Successfully wrote data to temporary file: {temp_file.name}")

        # Generiši jedinstveni ID na osnovu trenutnog vremena
        file_id = dat.now().strftime('%Y%m%d_%H%M%S')
        temp_files[file_id] = temp_file.name
        logger.info(f"Generated file ID: {file_id}")

        return jsonify({"message": "File prepared for download", "fileId": file_id}), 200
    except Exception as e:
        logger.error(f"Error in prepare_save: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": "Fehler beim Vorbereiten der Datei"}), 500

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