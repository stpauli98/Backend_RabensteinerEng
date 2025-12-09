"""
First Processing API endpoints.
Handles chunked CSV upload and initial processing with various modes.
"""
import os
import csv
import json
import logging
import traceback
from io import StringIO

from flask import Blueprint, request, jsonify, g

from shared.auth.jwt import require_auth
from shared.auth.subscription import require_subscription, check_processing_limit
from shared.tracking.usage import increment_processing_count, update_storage_usage
from shared.storage.service import storage_service

from domains.processing.config import CHUNK_UPLOAD_FOLDER
from domains.processing.services.progress import ProgressTracker
from domains.processing.services.csv_processor import process_csv
from domains.processing.services.chunk_handler import (
    get_upload_lock,
    cleanup_upload_lock,
    extract_chunk_index
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

bp = Blueprint('first_processing', __name__)

os.makedirs(CHUNK_UPLOAD_FOLDER, exist_ok=True)


@bp.route('/upload_chunk', methods=['POST'])
@require_auth
@require_subscription
@check_processing_limit
def upload_chunk():
    """
    Endpoint for receiving and processing CSV data in chunks.
    Expected parameters (form data):
      - uploadId: unique upload ID (string)
      - chunkIndex: chunk index (int, starts from 0)
      - totalChunks: total number of chunks (int)
      - fileChunk: Blob/File with CSV data part
      - tss: Time step size in minutes (float)
      - offset: Offset in minutes (float)
      - mode: Processing mode ('mean', 'intrpl', 'nearest', 'nearest (mean)')
      - intrplMax: Maximum time for interpolation in minutes (float, default 60)
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
            decimal_precision = request.form.get('decimalPrecision', 'full')
        except (ValueError, TypeError) as e:
            logger.error(f"Error parsing parameters: {e}")
            return jsonify({"error": f"Invalid parameter values: {str(e)}"}), 400

        if not all([upload_id, mode, tss > 0]):
            return jsonify({"error": "Missing required parameters"}), 400

        chunk = request.files['fileChunk']
        if not chunk:
            return jsonify({"error": "Empty chunk received"}), 400

        os.makedirs(CHUNK_UPLOAD_FOLDER, exist_ok=True)

        chunk_filename = os.path.join(CHUNK_UPLOAD_FOLDER, f"{upload_id}_{chunk_index}.chunk")
        chunk.save(chunk_filename)

        logger.info(f"Saved chunk {chunk_index + 1}/{total_chunks} for upload {upload_id}")

        # Use lock to prevent race condition when multiple chunks finish simultaneously
        upload_lock = get_upload_lock(upload_id)

        # Try to acquire lock - if fails immediately, someone else is processing
        if not upload_lock.acquire(blocking=False):
            logger.info(f"Upload {upload_id} is already being processed by another thread")
            return jsonify({"success": True, "message": "Chunk saved, processing in progress"}), 200

        lock_acquired = True
        try:
            received_chunks = [f for f in os.listdir(CHUNK_UPLOAD_FOLDER)
                             if f.startswith(upload_id + "_")]

            if len(received_chunks) != total_chunks:
                return jsonify({"success": True, "message": "Chunk saved"}), 200

            logger.info(f"All chunks received for upload {upload_id}, processing...")

            # Calculate total file size
            total_file_size = sum(
                os.path.getsize(os.path.join(CHUNK_UPLOAD_FOLDER, f))
                for f in received_chunks
            )

            # Initialize ProgressTracker with file size
            tracker = ProgressTracker(
                upload_id=upload_id,
                file_size_bytes=total_file_size,
                total_chunks=total_chunks
            )
            tracker.start_phase('chunk_assembly')
            tracker.emit('chunk_assembly', 0, 'chunk_assembly_start', force=True, message_params={'totalChunks': total_chunks})

            chunks_sorted = sorted(received_chunks, key=extract_chunk_index)

            try:
                full_content = ""
                logger.info(f"Assembling {len(chunks_sorted)} chunks: {chunks_sorted}")

                for i, chunk_file in enumerate(chunks_sorted):
                    chunk_path = os.path.join(CHUNK_UPLOAD_FOLDER, chunk_file)
                    logger.info(f"Processing chunk {i+1}/{len(chunks_sorted)}: {chunk_file}")

                    # Progress update for chunk assembly (0-10%)
                    chunk_progress = (i / len(chunks_sorted)) * 10
                    if i % max(1, len(chunks_sorted) // 10) == 0:
                        tracker.emit('chunk_assembly', chunk_progress,
                                   'chunk_assembly_progress', message_params={'current': i+1, 'total': len(chunks_sorted)})

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

                    # Safe cleanup - handle race condition with multiple workers
                    try:
                        if os.path.exists(chunk_path):
                            os.remove(chunk_path)
                    except (FileNotFoundError, OSError) as e:
                        logger.debug(f"Chunk already cleaned up by another worker: {chunk_path}")

                logger.info(f"Final assembled content length: {len(full_content)}")

                # End chunk assembly phase
                tracker.end_phase('chunk_assembly')
                tracker.emit('chunk_assembly', 10, 'chunk_assembly_complete', force=True)

                final_lines = full_content.split('\n')
                logger.info(f"Final content total lines: {len(final_lines)}")
                if len(final_lines) > 0:
                    logger.info(f"Final content first line: '{final_lines[0]}'")
                if len(final_lines) > 1:
                    logger.info(f"Final content second line: '{final_lines[1]}'")
                if len(final_lines) > 2:
                    logger.info(f"Final content third line: '{final_lines[2]}'")

                # Pass tracker to process_csv
                result = process_csv(full_content, tss, offset, mode, intrpl_max, upload_id, tracker, decimal_precision)

                # Track processing and storage usage
                try:
                    increment_processing_count(g.user_id)
                    logger.info(f"Tracked processing for user {g.user_id}")

                    file_size_bytes = len(full_content.encode('utf-8'))
                    file_size_mb = file_size_bytes / (1024 * 1024)
                    update_storage_usage(g.user_id, file_size_mb)
                    logger.info(f"Tracked storage for user {g.user_id}: {file_size_mb:.2f} MB")
                except Exception as e:
                    logger.error(f"Failed to track processing usage: {str(e)}")

                return result

            except Exception as e:
                for chunk_file in chunks_sorted:
                    try:
                        os.remove(os.path.join(CHUNK_UPLOAD_FOLDER, chunk_file))
                    except:
                        pass
                raise
        finally:
            if lock_acquired:
                try:
                    upload_lock.release()
                    cleanup_upload_lock(upload_id)
                except RuntimeError:
                    pass

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
@require_auth
def prepare_save():
    """
    Prepare processed data for download.
    Saves CSV data to Supabase Storage for persistent access.
    """
    try:
        data = request.json
        if not data or 'data' not in data:
            return jsonify({"error": "No data received"}), 400

        data_wrapper = data['data']
        save_data = data_wrapper.get('data', [])
        file_name = data_wrapper.get('fileName', '')

        if not save_data:
            return jsonify({"error": "Empty data"}), 400

        # Convert data array to CSV string
        output = StringIO()
        writer = csv.writer(output, delimiter=';')
        for row in save_data:
            writer.writerow(row)
        csv_content = output.getvalue()

        # Upload to Supabase Storage
        user_id = g.user_id

        file_id = storage_service.upload_csv(
            user_id=user_id,
            csv_content=csv_content,
            original_filename=file_name or "processed_data.csv",
            metadata={
                'totalRows': len(save_data) - 1,
                'source': 'first-processing-prepare-save'
            }
        )

        if not file_id:
            return jsonify({"error": "Failed to save file to storage"}), 500

        logger.info(f"File prepared for download: {file_id}")

        return jsonify({
            "message": "File prepared for download",
            "fileId": file_id,
            "totalRows": len(save_data) - 1
        }), 200

    except Exception as e:
        logger.error(f"Error in prepare_save: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@bp.route('/download/<path:file_id>', methods=['GET'])
@require_auth
def download_file(file_id: str):
    """
    Get download URL for prepared CSV file from Supabase Storage.
    Returns JSON with signed URL for frontend to use directly.
    """
    try:
        logger.info(f"Download request for file: {file_id}")

        # Get signed URL from Supabase Storage (valid for 1 hour)
        signed_url = storage_service.get_download_url(file_id, expires_in=3600)

        if signed_url:
            logger.info(f"Generated signed URL for: {file_id}")

            return jsonify({
                "success": True,
                "downloadUrl": signed_url,
                "fileId": file_id
            }), 200

        logger.warning(f"Signed URL failed for: {file_id}")
        return jsonify({"error": "Failed to generate download URL"}), 500

    except Exception as e:
        logger.error(f"Error in download_file: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@bp.route('/cleanup-files', methods=['POST'])
@require_auth
def cleanup_files():
    """
    Delete files from Supabase Storage after successful download.
    """
    try:
        data = request.json

        if not data:
            return jsonify({"error": "No data received"}), 400

        file_ids = data.get('fileIds', [])

        if not file_ids:
            return jsonify({"message": "No files to delete", "deletedCount": 0}), 200

        deleted_count = 0
        failed_ids = []

        for file_id in file_ids:
            try:
                if storage_service.delete_file(file_id):
                    deleted_count += 1
                    logger.info(f"Cleaned up file: {file_id}")
                else:
                    failed_ids.append(file_id)
            except Exception as del_error:
                logger.error(f"Failed to delete file {file_id}: {del_error}")
                failed_ids.append(file_id)

        return jsonify({
            "message": "Cleanup complete",
            "deletedCount": deleted_count,
            "totalRequested": len(file_ids),
            "failedIds": failed_ids
        }), 200

    except Exception as e:
        logger.error(f"Error in cleanup_files: {str(e)}")
        return jsonify({"error": str(e)}), 500
