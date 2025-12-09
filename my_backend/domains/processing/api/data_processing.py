"""
Data Processing API endpoints.
Handles chunked CSV upload and data cleaning operations.
"""
import json
import logging
import math
import os
import re
import time

import numpy as np
import pandas as pd
from flask import Blueprint, request, Response, jsonify, g

from shared.auth.jwt import require_auth
from shared.auth.subscription import require_subscription, check_processing_limit
from shared.tracking.usage import increment_processing_count, update_storage_usage

from domains.processing.config import (
    DATA_PROCESSING_UPLOAD_FOLDER,
    STREAMING_CHUNK_SIZE,
    BACKPRESSURE_DELAY
)
from domains.processing.services.progress import ProgressTracker
from domains.processing.services.data_cleaner import clean_data, validate_processing_params
from domains.processing.services.chunk_handler import (
    secure_path_join,
    validate_file_upload,
    combine_chunks_efficiently
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

bp = Blueprint('data_processing', __name__)

os.makedirs(DATA_PROCESSING_UPLOAD_FOLDER, exist_ok=True)


def safe_error_response(error_msg, status_code=500, error_type=None):
    """Sanitize error messages to prevent information disclosure"""
    sanitized = re.sub(r'/[^\s]+', '[PATH]', str(error_msg))
    sanitized = sanitized.split('\n')[0]

    logger.error(f"Error occurred: {error_msg}")

    if error_type == 'validation' or 'validation' in str(error_msg).lower():
        if 'parameter' in str(error_msg).lower():
            return jsonify({"error": f"Parameter validation failed: {sanitized}"}), status_code
        elif 'csv' in str(error_msg).lower() or 'file format' in str(error_msg).lower():
            return jsonify({"error": f"CSV validation failed: {sanitized}"}), status_code
        elif 'file size' in str(error_msg).lower() or 'too large' in str(error_msg).lower():
            return jsonify({"error": f"File size validation failed: {sanitized}"}), status_code
        else:
            return jsonify({"error": f"Data validation failed: {sanitized}"}), status_code
    elif error_type == 'security' or 'security' in str(error_msg).lower():
        return jsonify({"error": f"Security validation failed: {sanitized}"}), status_code
    elif error_type == 'file' or status_code in [400, 413]:
        return jsonify({"error": sanitized}), status_code
    else:
        return jsonify({"error": "A processing error occurred. Please check your input and try again."}), status_code


@bp.route("/api/dataProcessingMain/upload-chunk", methods=["POST"])
@require_auth
@require_subscription
@check_processing_limit
def upload_chunk():
    """
    Endpoint for receiving and processing CSV data in chunks.
    Uses ProgressTracker for real-time progress tracking with ETA calculation.
    """
    tracker = None

    try:
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
            return safe_error_response("No file chunk provided", 400)

        file_chunk = request.files['fileChunk']
        if file_chunk.filename == '':
            logger.error("Empty filename in fileChunk")
            return safe_error_response("Empty filename", 400)

        try:
            chunk = validate_file_upload(file_chunk, file_chunk.filename)
            chunk_size = len(chunk)
        except ValueError as e:
            logger.error(f"File validation failed: {e}")
            return safe_error_response(str(e), 400, 'validation')

        if chunk_size == 0:
            logger.error("Received empty chunk")
            return jsonify({"error": "Empty chunk received"}), 400

        try:
            upload_dir = secure_path_join(DATA_PROCESSING_UPLOAD_FOLDER, upload_id)
            os.makedirs(upload_dir, exist_ok=True)
        except ValueError as e:
            logger.error(f"Invalid upload_id: {upload_id}")
            return safe_error_response("Invalid upload identifier", 400, 'security')

        chunk_path = os.path.join(upload_dir, f"chunk_{chunk_index:04d}.part")

        with open(chunk_path, "wb") as f:
            f.write(chunk)

        if chunk_index < total_chunks - 1:
            return jsonify({"status": "chunk received", "chunkIndex": chunk_index})

        # === LAST CHUNK RECEIVED - CHECK IF ALL CHUNKS ARE ON DISK ===
        missing_chunks = []
        for i in range(total_chunks):
            part_path = os.path.join(upload_dir, f"chunk_{i:04d}.part")
            if not os.path.exists(part_path):
                missing_chunks.append(i)

        if missing_chunks:
            import time as time_module
            max_wait = 5
            wait_interval = 0.2
            waited = 0

            while waited < max_wait and missing_chunks:
                time_module.sleep(wait_interval)
                waited += wait_interval
                missing_chunks = [
                    i for i in range(total_chunks)
                    if not os.path.exists(os.path.join(upload_dir, f"chunk_{i:04d}.part"))
                ]

            if missing_chunks:
                logger.error(f"Timeout waiting for chunks: {missing_chunks}")
                return jsonify({
                    "error": "Missing chunks after timeout",
                    "missingChunks": missing_chunks
                }), 400

            logger.info(f"All chunks arrived after {waited:.1f}s wait")

        # === ALL CHUNKS RECEIVED - START PROCESSING ===
        logger.info("All chunks received, starting processing...")

        # Calculate total file size
        total_file_size = sum(
            os.path.getsize(os.path.join(upload_dir, f"chunk_{i:04d}.part"))
            for i in range(total_chunks)
            if os.path.exists(os.path.join(upload_dir, f"chunk_{i:04d}.part"))
        )

        # Initialize ProgressTracker
        tracker = ProgressTracker(
            upload_id=upload_id,
            file_size_bytes=total_file_size,
            total_chunks=total_chunks
        )

        # === PHASE 1: CHUNK ASSEMBLY (0-10%) ===
        tracker.start_phase('chunk_assembly')
        tracker.emit('chunk_assembly', 0, 'chunk_assembly_start', force=True, message_params={'totalChunks': total_chunks})

        try:
            combined_file_path, total_size = combine_chunks_efficiently(upload_dir, total_chunks)
            tracker.end_phase('chunk_assembly')
            tracker.emit('chunk_assembly', 10, 'chunk_assembly_complete', force=True)
        except (FileNotFoundError, ValueError) as e:
            logger.error(f"Chunk combination failed: {e}")
            for i in range(total_chunks):
                part_path = os.path.join(upload_dir, f"chunk_{i:04d}.part")
                try:
                    if os.path.exists(part_path):
                        os.remove(part_path)
                except OSError:
                    pass

            if "exceeds" in str(e):
                return safe_error_response("File too large", 413, 'validation')
            else:
                return jsonify({"error": "Missing chunk file"}), 400

        if total_size == 0:
            logger.error("No data in combined chunks")
            os.unlink(combined_file_path)
            return jsonify({"error": "No data in combined chunks"}), 400

        try:
            # === PHASE 2: PARSING (10-15%) ===
            tracker.start_phase('parsing')
            tracker.emit('parsing', 10, 'parsing_start', force=True)

            with open(combined_file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            os.unlink(combined_file_path)
            lines = content.splitlines()

            tracker.emit('parsing', 12, 'parsing_lines_loaded', message_params={'lineCount': len(lines)})

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

            tracker.end_phase('parsing')
            tracker.emit('parsing', 15, 'parsing_complete', force=True, message_params={'rowCount': len(data)})

            # === PHASE 3: PREPROCESSING (15-25%) ===
            tracker.start_phase('preprocessing')
            tracker.emit('preprocessing', 15, 'preprocessing_type_conversion')

            df = pd.DataFrame(data, columns=["UTC", value_column])
            df[value_column] = df[value_column].replace('', np.nan)
            df[value_column] = pd.to_numeric(df[value_column].str.replace(",", "."), errors='coerce')

            tracker.emit('preprocessing', 20, 'preprocessing_param_validation')

            raw_params = {
                "eqMax": request.form.get("eqMax"),
                "elMax": request.form.get("elMax"),
                "elMin": request.form.get("elMin"),
                "chgMax": request.form.get("chgMax"),
                "lgMax": request.form.get("lgMax"),
                "gapMax": request.form.get("gapMax"),
                "radioValueNull": request.form.get("radioValueNull"),
                "radioValueNotNull": request.form.get("radioValueNotNull")
            }

            decimal_precision = request.form.get('decimalPrecision', 'full')

            try:
                params = validate_processing_params(raw_params)
            except ValueError as e:
                logger.error(f"Parameter validation failed: {e}")
                return safe_error_response(str(e), 400, 'validation')

            tracker.end_phase('preprocessing')
            tracker.emit('preprocessing', 25, 'preprocessing_complete', force=True, message_params={'rowCount': len(df)})

            # === PHASE 4: CLEANING (25-85%) ===
            tracker.start_phase('cleaning')
            tracker.emit('cleaning', 25, 'cleaning_start', force=True, message_params={'rowCount': len(df)})

            df_clean = clean_data(df, value_column, params, tracker, decimal_precision)

            tracker.end_phase('cleaning')
            tracker.emit('cleaning', 90, 'cleaning_complete', force=True, message_params={'rowCount': len(df_clean)})

            # Reset step tracking for remaining phases
            tracker.current_step = 0
            tracker.total_steps = 0

            # Track processing usage
            try:
                increment_processing_count(g.user_id)
                logger.info(f"Tracked processing for user {g.user_id}")

                file_size_mb = total_size / (1024 * 1024)
                update_storage_usage(g.user_id, file_size_mb)
                logger.info(f"Tracked storage for user {g.user_id}: {file_size_mb:.2f} MB")
            except Exception as e:
                logger.error(f"Failed to track processing usage: {str(e)}")

            # Create closure for tracker so generator can use it
            tracker_ref = tracker

            def generate():
                """Generator for streaming NDJSON with progress tracking - OPTIMIZED"""

                def clean_value_for_json(val):
                    """Clean value for JSON - NaN/Inf -> None"""
                    if val is None:
                        return None
                    try:
                        if pd.isna(val):
                            return None
                    except (TypeError, ValueError):
                        pass
                    try:
                        if isinstance(val, (float, np.floating)):
                            if math.isnan(val) or math.isinf(val):
                                return None
                    except (TypeError, ValueError):
                        pass
                    return val

                class CustomJSONEncoder(json.JSONEncoder):
                    def default(self, obj):
                        if isinstance(obj, pd.Timestamp):
                            return obj.strftime('%Y-%m-%d %H:%M:%S')
                        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
                            return None
                        return super().default(obj)

                try:
                    # === PHASE 5: STREAMING (90-100%) ===
                    tracker_ref.start_phase('streaming')
                    tracker_ref.emit('streaming', 90, 'streaming_start', force=True, message_params={'rowCount': len(df_clean)})

                    yield json.dumps({"total_rows": len(df_clean)}, cls=CustomJSONEncoder) + "\n"

                    chunk_size = STREAMING_CHUNK_SIZE
                    total_chunks_to_stream = (len(df_clean) // chunk_size) + 1
                    streaming_start_time = time.time()

                    for i in range(0, len(df_clean), chunk_size):
                        try:
                            chunk_progress = 90 + ((i / len(df_clean)) * 9)
                            current_chunk = (i // chunk_size) + 1

                            streaming_eta = None
                            if current_chunk > 1:
                                elapsed = time.time() - streaming_start_time
                                chunks_done = current_chunk - 1
                                chunks_remaining = total_chunks_to_stream - current_chunk + 1
                                time_per_chunk = elapsed / chunks_done
                                streaming_eta = int(chunks_remaining * time_per_chunk)

                            tracker_ref.emit('streaming', chunk_progress, 'streaming_chunk',
                                            eta_seconds=streaming_eta,
                                            message_params={'currentChunk': current_chunk, 'totalChunks': total_chunks_to_stream})

                            chunk = df_clean.iloc[i:i + chunk_size]

                            if i == 0:
                                logger.info("First 10 rows of processed data:")
                                logger.info(chunk.head(10).to_string())

                            chunk_subset = chunk[['UTC', value_column]].copy()
                            chunk_subset['UTC'] = chunk_subset['UTC'].dt.strftime('%Y-%m-%d %H:%M:%S')

                            if decimal_precision != 'full':
                                chunk_subset[value_column] = chunk_subset[value_column].apply(
                                    lambda x: round(x, int(decimal_precision)) if pd.notna(x) else x
                                )

                            chunk_subset[value_column] = chunk_subset[value_column].astype(object)
                            chunk_subset[value_column] = chunk_subset[value_column].where(
                                pd.notna(chunk_subset[value_column]), None
                            )

                            for record in chunk_subset.to_dict('records'):
                                yield json.dumps(record) + "\n"

                            time.sleep(BACKPRESSURE_DELAY)

                        except Exception as chunk_error:
                            logger.error(f"Streaming error at chunk {i}: {chunk_error}")
                            yield json.dumps({"error": f"Chunk {i} failed: {str(chunk_error)}", "partial": True}, cls=CustomJSONEncoder) + "\n"
                            break

                    tracker_ref.end_phase('streaming')
                    tracker_ref.emit('complete', 100, 'processing_complete', force=True, message_params={'rowCount': len(df_clean)})
                    yield json.dumps({"status": "complete"}, cls=CustomJSONEncoder) + "\n"

                except GeneratorExit:
                    logger.info("Client disconnected during streaming")
                except BrokenPipeError:
                    logger.warning("Broken pipe - client forcefully disconnected")
                except Exception as e:
                    logger.error(f"Generator error: {e}")
                    try:
                        yield json.dumps({"error": str(e)}, cls=CustomJSONEncoder) + "\n"
                    except:
                        pass

            return Response(generate(), mimetype="application/x-ndjson")

        except Exception as e:
            logger.error(f"Error processing data: {str(e)}")
            return safe_error_response("Error processing data", 500, 'processing')

    except Exception as e:
        logger.error(f"Error in upload_chunk: {str(e)}")
        return safe_error_response("Error in upload", 500)
