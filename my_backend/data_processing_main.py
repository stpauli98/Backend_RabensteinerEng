import pandas as pd
import numpy as np
from datetime import datetime
from flask import Blueprint, request, Response, jsonify, send_file
import io
import json
import csv
import tempfile
import os
import math
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

bp = Blueprint('data_processing', __name__)
UPLOAD_FOLDER = os.path.join(tempfile.gettempdir(), "upload_chunks")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def clean_data(df, value_column, params):
    logger.info("Starting data cleaning with parameters: %s", params)
    df["UTC"] = pd.to_datetime(df["UTC"], format="%Y-%m-%d %H:%M:%S")

    # ELIMINIERUNG VON MESSAUSFÄLLEN (GLEICHBLEIBENDE MESSWERTE)
    if params.get("eqMax"):
        logger.info("Eliminierung von Messausfällen (gleichbleibende Messwerte)")
        eq_max = float(params["eqMax"])
        frm = 0
        for i in range(1, len(df)):
            if df.iloc[i-1][value_column] == df.iloc[i][value_column] and frm == 0:
                idx_strt = i-1
                frm = 1
            elif df.iloc[i-1][value_column] != df.iloc[i][value_column] and frm == 1:
                idx_end = i-1
                frm_width = (df.iloc[idx_end]["UTC"] - df.iloc[idx_strt]["UTC"]).total_seconds() / 60
                if frm_width >= eq_max:
                    for i_frm in range(idx_strt, idx_end+1):
                        df.iloc[i_frm, df.columns.get_loc(value_column)] = "nan"
                frm = 0
            elif i == len(df)-1 and frm == 1:
                idx_end = i
                frm_width = (df.iloc[idx_end]["UTC"] - df.iloc[idx_strt]["UTC"]).total_seconds() / 60
                if frm_width >= eq_max:
                    for i_frm in range(idx_strt, idx_end+1):
                        df.iloc[i_frm, df.columns.get_loc(value_column)] = "nan"

    # WERTE ÜBER DEM OBEREN GRENZWERT ENTFERNEN
    if params.get("elMax"):
        logger.info("Werte über dem oberen Grenzwert entfernen")
        el_max = float(params["elMax"])
        for i in range(len(df)):
            try:
                if float(df.iloc[i][value_column]) > el_max:
                    df.iloc[i, df.columns.get_loc(value_column)] = "nan"
            except:
                df.iloc[i, df.columns.get_loc(value_column)] = "nan"

    # WERTE UNTER DEM UNTEREN GRENZWERT ENTFERNEN
    if params.get("elMin"):
        logger.info("Werte unter dem unteren Grenzwert entfernen")
        el_min = float(params["elMin"])
        for i in range(len(df)):
            try:
                if float(df.iloc[i][value_column]) < el_min:
                    df.iloc[i, df.columns.get_loc(value_column)] = "nan"
            except:
                df.iloc[i, df.columns.get_loc(value_column)] = "nan"

    # ELIMINIERUNG VON NULLWERTEN
    if params.get("radioValueNull") == "ja":
        logger.info("Eliminierung von Nullwerten")
        for i in range(len(df)):
            if df.iloc[i][value_column] == 0:
                df.iloc[i, df.columns.get_loc(value_column)] = "nan"

    # ELIMINIERUNG VON NICHT NUMERISCHEN WERTEN
    if params.get("radioValueNotNull") == "ja":
        logger.info("Eliminierung von nicht numerischen Werten")
        for i in range(len(df)):
            try:
                float(df.iloc[i][value_column])
                if math.isnan(float(df.iloc[i][value_column])):
                    df.iloc[i, df.columns.get_loc(value_column)] = "nan"
            except:
                df.iloc[i, df.columns.get_loc(value_column)] = "nan"

    # ELIMINIERUNG VON AUSREISSERN
    if params.get("chgMax") and params.get("lgMax"):
        logger.info("Eliminierung von Ausreissern")
        chg_max = float(params["chgMax"])
        lg_max = float(params["lgMax"])
        frm = 0
        for i in range(1, len(df)):
            if df.iloc[i][value_column] == "nan" and frm == 0:
                pass
            elif df.iloc[i][value_column] == "nan" and frm == 1:
                idx_end = i-1
                for i_frm in range(idx_strt, idx_end+1):
                    df.iloc[i_frm, df.columns.get_loc(value_column)] = "nan"
                frm = 0
            elif df.iloc[i-1][value_column] == "nan":
                pass
            else:
                chg = abs(float(df.iloc[i][value_column]) - float(df.iloc[i-1][value_column]))
                t = (df.iloc[i]["UTC"] - df.iloc[i-1]["UTC"]).total_seconds() / 60
                if chg/t > chg_max and frm == 0:
                    idx_strt = i
                    frm = 1
                elif chg/t > chg_max and frm == 1:
                    idx_end = i-1
                    for i_frm in range(idx_strt, idx_end+1):
                        df.iloc[i_frm, df.columns.get_loc(value_column)] = "nan"
                    frm = 0
                elif frm == 1 and (df.iloc[i]["UTC"] - df.iloc[idx_strt]["UTC"]).total_seconds() / 60 > lg_max:
                    frm = 0

    # SCHLIESSEN VON MESSLÜCKEN
    if params.get("gapMax"):
        logger.info("Schließen von Messlücken")
        gap_max = float(params["gapMax"])
        frm = 0
        for i in range(1, len(df)):
            if df.iloc[i][value_column] == "nan" and frm == 0:
                idx_strt = i
                frm = 1
            elif df.iloc[i][value_column] != "nan" and frm == 1:
                idx_end = i-1
                frm_width = (df.iloc[idx_end+1]["UTC"] - df.iloc[idx_strt-1]["UTC"]).total_seconds() / 60
                if frm_width <= gap_max:
                    dif = float(df.iloc[idx_end+1][value_column]) - float(df.iloc[idx_strt-1][value_column])
                    dif_min = dif/frm_width
                    for i_frm in range(idx_strt, idx_end+1):
                        gap_min = (df.iloc[i_frm]["UTC"] - df.iloc[idx_strt-1]["UTC"]).total_seconds() / 60
                        df.iloc[i_frm, df.columns.get_loc(value_column)] = float(df.iloc[idx_strt-1][value_column]) + gap_min*dif_min
                frm = 0

    logger.info("Data cleaning completed")
    return df

@bp.route("/api/dataProcessingMain/upload-chunk", methods=["POST"])
def upload_chunk():
    try:
        # Minimal logging for performance
        logger.info(f"Processing chunk {request.form.get('chunkIndex')}/{request.form.get('totalChunks')}")

        if not all(key in request.form for key in ["uploadId", "chunkIndex", "totalChunks"]):
            missing_fields = [key for key in ["uploadId", "chunkIndex", "totalChunks"] if key not in request.form]
            logger.error(f"Missing required fields: {missing_fields}")
            return jsonify({"error": f"Missing required fields: {missing_fields}"}), 400

        upload_id = request.form["uploadId"]
        chunk_index = int(request.form["chunkIndex"])
        total_chunks = int(request.form["totalChunks"])
        
        if 'fileChunk' not in request.files:
            logger.error("No fileChunk in request.files")
            return jsonify({"error": "No file chunk provided"}), 400
            
        file_chunk = request.files['fileChunk']
        if file_chunk.filename == '':
            logger.error("Empty filename in fileChunk")
            return jsonify({"error": "Empty filename"}), 400

        # Read the chunk
        chunk = file_chunk.read()
        chunk_size = len(chunk)
        
        if chunk_size == 0:
            logger.error("Received empty chunk")
            return jsonify({"error": "Empty chunk received"}), 400

        # Create upload directory
        upload_dir = os.path.join(UPLOAD_FOLDER, upload_id)
        os.makedirs(upload_dir, exist_ok=True)
        chunk_path = os.path.join(upload_dir, f"chunk_{chunk_index:04d}.part")
        
        # Save the chunk
        with open(chunk_path, "wb") as f:
            f.write(chunk)

        if chunk_index < total_chunks - 1:
            return jsonify({"status": "chunk received", "chunkIndex": chunk_index})

        # Combine all chunks
        logger.info("Starting to combine all chunks")
        all_bytes = bytearray()
        total_size = 0
        
        for i in range(total_chunks):
            part_path = os.path.join(upload_dir, f"chunk_{i:04d}.part")
            if not os.path.exists(part_path):
                logger.error(f"Missing chunk file: {part_path}")
                return jsonify({"error": f"Missing chunk file: {i}"}), 400
                
            with open(part_path, "rb") as f:
                chunk_data = f.read()
                all_bytes.extend(chunk_data)
                total_size += len(chunk_data)

        if total_size == 0:
            logger.error("No data in combined chunks")
            return jsonify({"error": "No data in combined chunks"}), 400

        # Process the combined data
        try:
            content = all_bytes.decode("utf-8")
            lines = content.splitlines()
            
            if len(lines) < 2:
                logger.error("File has less than 2 lines")
                return jsonify({"error": "Invalid file format"}), 400

            separator = ";" if ";" in lines[0] else ","
            header = lines[0].split(separator)
            
            if len(header) < 2:
                logger.error("Invalid header format")
                return jsonify({"error": "Invalid header format"}), 400
                
            value_column = header[1].strip()
            data = [line.split(separator) for line in lines[1:] if line.strip()]
            
            if not data:
                logger.error("No data rows found")
                return jsonify({"error": "No data rows found"}), 400

            df = pd.DataFrame(data, columns=["UTC", value_column])
            df[value_column] = pd.to_numeric(df[value_column].str.replace(",", "."), errors='coerce')

            params = {
                "eqMax": request.form.get("eqMax"),
                "elMax": request.form.get("elMax"),
                "elMin": request.form.get("elMin"),
                "chgMax": request.form.get("chgMax"),
                "lgMax": request.form.get("lgMax"),
                "gapMax": request.form.get("gapMax"),
                "radioValueNull": request.form.get("radioValueNull"),
                "radioValueNotNull": request.form.get("radioValueNotNull")
            }

            df_clean = clean_data(df, value_column, params)

            def generate():
                # Create a custom JSON encoder to handle Pandas Timestamp objects
                class CustomJSONEncoder(json.JSONEncoder):
                    def default(self, obj):
                        if isinstance(obj, pd.Timestamp):
                            return obj.strftime('%Y-%m-%d %H:%M:%S')
                        return super().default(obj)
                
                # First send total rows
                yield json.dumps({"total_rows": len(df_clean)}, cls=CustomJSONEncoder) + "\n"
                
                # Process data in larger chunks of 50000 rows
                chunk_size = 50000
                for i in range(0, len(df_clean), chunk_size):
                    # Create a copy of the chunk and convert UTC in one step
                    chunk = df_clean.iloc[i:i + chunk_size].copy()
                    chunk.loc[:, 'UTC'] = chunk['UTC'].dt.strftime('%Y-%m-%d %H:%M:%S')
                    
                    # Log first 10 rows of the first chunk
                    if i == 0:
                        logger.info("First 10 rows of processed data:")
                        logger.info(chunk.head(10).to_string())
                    
                    # Convert to dict and ensure all values are JSON serializable
                    chunk_data = []
                    for _, row in chunk.iterrows():
                        # Ensure UTC is a string, not a Timestamp object
                        utc_value = row['UTC']
                        if isinstance(utc_value, pd.Timestamp):
                            utc_value = utc_value.strftime('%Y-%m-%d %H:%M:%S')
                            
                        record = {
                            'UTC': utc_value,
                            value_column: row[value_column] if pd.notnull(row[value_column]) else None
                        }
                        chunk_data.append(record)
                    
                    # Yield all records in the chunk at once
                    for record in chunk_data:
                        yield json.dumps(record, cls=CustomJSONEncoder) + "\n"
                
                # Send completion status
                yield json.dumps({"status": "complete"}, cls=CustomJSONEncoder) + "\n"
                        
            return Response(generate(), mimetype="application/x-ndjson")
        

        except Exception as e:
            logger.error(f"Error processing data: {str(e)}")
            return jsonify({"error": f"Error processing data: {str(e)}"}), 500
            
    except Exception as e:
        logger.error(f"Error in upload_chunk: {str(e)}")
        return jsonify({"error": f"Error in upload: {str(e)}"}), 500

@bp.route("/api/dataProcessingMain/prepare-save", methods=["POST"])
def prepare_save():
    try:
        data = request.json.get("data")
        if not data:
            return jsonify({"error": "No data provided"}), 400

        file_name = data.get("fileName", "output.csv")
        rows = data.get("data", [])
        
        if not rows:
            return jsonify({"error": "No rows provided"}), 400

        logger.info(f"Preparing to save file: {file_name} with {len(rows)} rows")

        # Sanitize filename
        safe_file_name = os.path.basename(file_name)
        safe_file_name = safe_file_name.replace(" ", "_")
        file_path = os.path.join(tempfile.gettempdir(), safe_file_name)

        # Write CSV
        with open(file_path, "w", newline='', encoding='utf-8') as f:
            writer = csv.writer(f, delimiter=";")
            for row in rows:
                writer.writerow(row)

        logger.info(f"File prepared successfully with ID: {safe_file_name}")
        return jsonify({"fileId": safe_file_name})

    except Exception as e:
        logger.error(f"Error preparing save: {str(e)}")
        return jsonify({"error": f"Error preparing file: {str(e)}"}), 500


@bp.route("/api/dataProcessingMain/download/<file_id>", methods=["GET"])
def download_file(file_id):
    try:
        path = os.path.join(tempfile.gettempdir(), file_id)
        if not os.path.exists(path):
            return jsonify({"error": "File not found"}), 404

        logger.info(f"Sending file: {path}")
        return send_file(
            path,
            as_attachment=True,
            download_name=file_id,
            mimetype="text/csv"
        )

    except Exception as e:
        logger.error(f"Error downloading file: {str(e)}")
        return jsonify({"error": f"Error downloading file: {str(e)}"}), 500
