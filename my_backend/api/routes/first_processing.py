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
from flask_socketio import emit

bp = Blueprint('first_processing', __name__)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


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

UPLOAD_FOLDER = "chunk_uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def process_csv(file_content, tss, offset, mode_input, intrpl_max, upload_id=None):
    """
    Obradjuje CSV sadržaj te vraća rezultat kao gzip-komprimiran JSON odgovor.
    Koristi vektorizirane Pandas operacije za bolju performansu.
    
    Args:
        upload_id: Optional upload ID for Socket.IO progress tracking
    """
    def emit_progress(step, progress, message):
        """Emit progress update via Socket.IO"""
        if upload_id:
            try:
                emit('processing_progress', {
                    'uploadId': upload_id,
                    'step': step,
                    'progress': progress,
                    'message': message
                }, room=upload_id)
            except Exception as e:
                logger.error(f"Error emitting progress: {e}")
    
    try:
        emit_progress('parsing', 10, 'Parsing CSV data...')
        
        try:
            lines = file_content.strip().split('\n')
            logger.info(f"Total lines in CSV: {len(lines)}")
            emit_progress('parsing', 20, f'Loaded {len(lines)} lines from CSV')
            
            if len(lines) > 0:
                header = lines[0]
                logger.info(f"Header: '{header}'")
                logger.info(f"Header fields: {header.split(';')}")
            
            problem_line_num = 264270
            if len(lines) > problem_line_num:
                problem_line = lines[problem_line_num - 1]
                logger.info(f"Line {problem_line_num}: '{problem_line}'")
                logger.info(f"Line {problem_line_num} fields: {problem_line.split(';')}")
                logger.info(f"Line {problem_line_num} field count: {len(problem_line.split(';'))}")
                
                for offset in [-2, -1, 1, 2]:
                    check_line_num = problem_line_num + offset
                    if 0 <= check_line_num - 1 < len(lines):
                        check_line = lines[check_line_num - 1]
                        logger.info(f"Line {check_line_num}: '{check_line}' (fields: {len(check_line.split(';'))})")
            
            emit_progress('parsing', 30, 'Parsing CSV with pandas...')
            try:
                df = pd.read_csv(StringIO(file_content), delimiter=';', skipinitialspace=True, on_bad_lines='skip')
                logger.info(f"Successfully parsed CSV with {len(df)} rows after skipping bad lines")
                emit_progress('parsing', 40, f'Successfully parsed {len(df)} rows')
            except Exception as pandas_error:
                logger.error(f"Even with on_bad_lines='skip', pandas failed: {str(pandas_error)}")
                import csv
                try:
                    df = pd.read_csv(StringIO(file_content), delimiter=';', skipinitialspace=True, 
                                   quoting=csv.QUOTE_NONE, on_bad_lines='skip')
                    logger.info(f"Successfully parsed CSV with QUOTE_NONE, {len(df)} rows")
                except Exception as final_error:
                    logger.error(f"All parsing attempts failed: {str(final_error)}")
                    raise pandas_error
            
            df.columns = df.columns.str.strip()
            
            if len(df.columns) < 2:
                raise ValueError(f"CSV must have at least 2 columns, found {len(df.columns)}: {list(df.columns)}")
            
            utc_col_name = df.columns[0]
            value_col_name = df.columns[1]
            logger.info(f"Using columns: UTC='{utc_col_name}', Value='{value_col_name}'")
            
            df = df.rename(columns={utc_col_name: 'UTC'})
            
            emit_progress('preprocessing', 50, 'Converting data types...')
            df['UTC'] = pd.to_datetime(df['UTC'], format='%Y-%m-%d %H:%M:%S')
            df[value_col_name] = pd.to_numeric(df[value_col_name], errors='coerce')
            
            initial_count = len(df)
            df = df.dropna(subset=['UTC', value_col_name])
            final_count = len(df)
            
            if initial_count != final_count:
                logger.info(f"Removed {initial_count - final_count} rows with invalid data")
                
        except Exception as e:
            logger.error(f"Error parsing CSV data: {str(e)}")
            return jsonify({"error": f"CSV parsing failed: {str(e)}"}), 400
        
        df = df.drop_duplicates(subset=['UTC']).sort_values('UTC').reset_index(drop=True)
        
        if df.empty:
            return jsonify({"error": "Nema valjanih vremenskih oznaka"}), 400
            
        time_min_raw = df['UTC'].iloc[0]
        time_max_raw = df['UTC'].iloc[-1]
        
        time_min_base = time_min_raw.replace(second=0, microsecond=0)
        
        logger.info(f"Applying offset of {offset} minutes to {time_min_base}")
        time_min = time_min_base + pd.Timedelta(minutes=int(offset))
        logger.info(f"Resulting start time: {time_min}")
        
        time_range = pd.date_range(
            start=time_min,
            end=time_max_raw,
            freq=f'{int(tss)}min'
        )
        
        df_utc = pd.DataFrame({'UTC': time_range})
        
        emit_progress('processing', 60, f'Starting {mode_input} processing...')
        
        if mode_input == "mean":
            emit_progress('processing', 65, 'Calculating mean values...')
            df.set_index('UTC', inplace=True)
            
            df_resampled = df[value_col_name].resample(
                rule=f'{int(tss)}min',
                origin=time_min,
                closed='right',
                label='right'
            ).mean().to_frame()
            
            df_resampled = df_resampled[df_resampled.index >= time_min]
            
            df_resampled.reset_index(inplace=True)
            
        elif mode_input == "intrpl":
            emit_progress('processing', 65, 'Starting interpolation...')
            df.set_index("UTC", 
                        inplace = True)
            
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
            
            df_before_interp = df_resampled.copy()
            
            actual_measurements = df_before_interp[~df_before_interp[value_col_name].isna()]
            
            if len(actual_measurements) > 1:
                measurement_times = actual_measurements.index.to_list()
                
                time_diffs = np.diff(measurement_times) / pd.Timedelta(minutes=1)
                
                large_gaps = np.where(time_diffs > float(intrpl_max))[0]
                
                logger.info(f"Found {len(large_gaps)} large measurement gaps exceeding {intrpl_max} minutes")
                
                for gap_idx in large_gaps:
                    gap_start = measurement_times[gap_idx]
                    gap_end = measurement_times[gap_idx + 1]
                    
                    max_interp_time = gap_start + pd.Timedelta(minutes=float(intrpl_max))
                    
                    no_interp_mask = ((df_resampled.index > max_interp_time) & 
                                      (df_resampled.index < gap_end))
                    
                    if no_interp_mask.any():
                        logger.info(f"Preventing interpolation for {no_interp_mask.sum()} points in gap between "
                                   f"{gap_start} and {gap_end} (beyond {intrpl_max} minutes)")
            
            df_resampled = df_resampled.interpolate(method = "time",
                                                   limit = int(intrpl_max/tss))
            
            df_resampled.reset_index(inplace=True)
            
        elif mode_input in ["nearest", "nearest (mean)"]:
            emit_progress('processing', 65, f'Processing {mode_input}...')
            df.set_index('UTC', inplace=True)
            
            if mode_input == "nearest":
                    df_resampled = pd.merge_asof(
                        df_utc,
                        df.reset_index(),
                        left_on='UTC',
                        right_on='UTC',
                        direction='nearest',
                        tolerance=pd.Timedelta(minutes=tss/2)
                    )
            else:
                    if df.index.name != 'UTC':
                        if 'UTC' in df.columns:
                            df.set_index('UTC', inplace=True)
                        else:
                            df.index.name = 'UTC'
                    
                    resampled = df[value_col_name].resample(
                        rule=f'{int(tss)}min',
                        origin=time_min,
                        closed='right',
                        label='right'
                    ).mean()
                    
                    mask = resampled.index >= time_min
                    resampled = resampled[mask]
                    
                    df_resampled = pd.DataFrame({'UTC': resampled.index, value_col_name: resampled.values})
        
        emit_progress('finalizing', 85, 'Converting results to JSON...')
        result = df_resampled.apply(
            lambda row: {
                "UTC": row["UTC"].strftime("%Y-%m-%d %H:%M:%S"),
                value_col_name: clean_for_json(row[value_col_name])
            },
            axis=1
        ).tolist()
        
        emit_progress('finalizing', 95, 'Compressing data...')
        result_json = json.dumps(result)
        compressed_data = gzip.compress(result_json.encode('utf-8'))
        
        emit_progress('complete', 100, f'Processing complete! Generated {len(result)} data points.')
        
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
        if 'fileChunk' not in request.files:
            return jsonify({"error": "No file chunk found"}), 400

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

        if not all([upload_id, mode, tss > 0]):
            return jsonify({"error": "Missing required parameters"}), 400

        chunk = request.files['fileChunk']
        if not chunk:
            return jsonify({"error": "Empty chunk received"}), 400

        os.makedirs(UPLOAD_FOLDER, exist_ok=True)

        chunk_filename = os.path.join(UPLOAD_FOLDER, f"{upload_id}_{chunk_index}.chunk")
        chunk.save(chunk_filename)

        logger.info(f"Saved chunk {chunk_index + 1}/{total_chunks} for upload {upload_id}")

        received_chunks = [f for f in os.listdir(UPLOAD_FOLDER) 
                         if f.startswith(upload_id + "_")]

        if len(received_chunks) == total_chunks:
            logger.info(f"All chunks received for upload {upload_id}, processing...")
            
            def extract_chunk_index(filename):
                try:
                    parts = filename.split("_")
                    chunk_part = parts[-1].split(".")[0]
                    return int(chunk_part)
                except (ValueError, IndexError) as e:
                    logger.error(f"Error parsing chunk filename {filename}: {e}")
                    return 0
            
            chunks_sorted = sorted(received_chunks, key=extract_chunk_index)
            
            try:
                
                full_content = ""
                logger.info(f"Assembling {len(chunks_sorted)} chunks: {chunks_sorted}")
                
                for i, chunk_file in enumerate(chunks_sorted):
                    chunk_path = os.path.join(UPLOAD_FOLDER, chunk_file)
                    logger.info(f"Processing chunk {i+1}/{len(chunks_sorted)}: {chunk_file}")
                    
                    with open(chunk_path, 'rb') as f:
                        chunk_bytes = f.read()
                        logger.info(f"Chunk {i+1} size: {len(chunk_bytes)} bytes")
                        
                        try:
                            chunk_content = chunk_bytes.decode('utf-8')
                            logger.info(f"Chunk {i+1} decoded successfully, content length: {len(chunk_content)}")
                            
                            if i == 0:
                                first_lines = chunk_content.split('\n')[:3]
                                logger.info(f"First chunk first 3 lines: {first_lines}")
                            
                            if i == len(chunks_sorted) - 1:
                                last_lines = chunk_content.split('\n')[-3:]
                                logger.info(f"Last chunk last 3 lines: {last_lines}")
                                
                        except UnicodeDecodeError as decode_error:
                            logger.error(f"Failed to decode chunk {i+1}: {decode_error}")
                            for encoding in ['latin-1', 'cp1252', 'iso-8859-1']:
                                try:
                                    chunk_content = chunk_bytes.decode(encoding)
                                    logger.info(f"Successfully decoded chunk {i+1} with {encoding}")
                                    break
                                except:
                                    continue
                            else:
                                raise decode_error
                        
                        if i < len(chunks_sorted) - 1 and not chunk_content.endswith('\n'):
                            chunk_content += '\n'
                        
                        full_content += chunk_content
                    
                    os.remove(chunk_path)
                
                logger.info(f"Final assembled content length: {len(full_content)}")
                
                final_lines = full_content.split('\n')
                logger.info(f"Final content total lines: {len(final_lines)}")
                if len(final_lines) > 0:
                    logger.info(f"Final content first line: '{final_lines[0]}'")
                if len(final_lines) > 1:
                    logger.info(f"Final content second line: '{final_lines[1]}'")
                if len(final_lines) > 2:
                    logger.info(f"Final content third line: '{final_lines[2]}'")

                return process_csv(full_content, tss, offset, mode, intrpl_max, upload_id)
                
            except Exception as e:
                for chunk_file in chunks_sorted:
                    try:
                        os.remove(os.path.join(UPLOAD_FOLDER, chunk_file))
                    except:
                        pass
                raise

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

        temp_file = tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.csv')
        writer = csv.writer(temp_file, delimiter=';')
        for row in save_data:
            writer.writerow(row)
        temp_file.close()

        file_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        temp_files[file_id] = {
            'path': temp_file.name,
            'fileName': file_name or f"data_{file_id}.csv",
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

        download_name = file_info['fileName']
        
        response = send_file(
            file_path,
            as_attachment=True,
            download_name=download_name,
            mimetype='text/csv'
        )
        
        try:
            os.unlink(file_info['path'])
        except Exception as e:
            logger.error(f"Error cleaning up temp file: {e}")
        return response

    except Exception as e:
        logger.error(f"Error in download_file: {str(e)}")
        return jsonify({"error": str(e)}), 500
    finally:
        if file_id in temp_files:
            try:
                os.unlink(temp_files[file_id])
                del temp_files[file_id]
            except Exception as ex:
                logger.error(f"Error cleaning up temp file: {ex}")

