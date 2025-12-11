"""
First Processing API endpoints.
Handles chunked CSV upload and initial processing with various modes.

Chunks are stored in Supabase Storage (temp-chunks bucket) for multi-instance
Cloud Run support. Chunks are automatically cleaned up after processing.
"""
import csv
import logging
import traceback
from io import StringIO

from flask import Blueprint, request, jsonify, g

from shared.auth.jwt import require_auth
from shared.auth.subscription import require_subscription, check_processing_limit
from shared.tracking.usage import increment_processing_count, update_storage_usage
from shared.storage.service import storage_service
from shared.storage.chunk_service import chunk_storage_service

from domains.processing.services.progress import ProgressTracker
from domains.processing.services.csv_processor import process_csv

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

bp = Blueprint('first_processing', __name__)


@bp.route('/upload_chunk', methods=['POST'])
@require_auth
@require_subscription
@check_processing_limit
def upload_chunk():
    """
    Endpoint for receiving and processing CSV data in chunks.

    Chunks are stored in Supabase Storage (temp-chunks bucket) for multi-instance
    Cloud Run support. Once all chunks are received, they are assembled and processed.
    Chunks are automatically cleaned up after successful processing or on error.

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

        # Read chunk data
        chunk_data = chunk.read()

        # Upload chunk to Supabase Storage
        if not chunk_storage_service.upload_chunk(upload_id, chunk_index, chunk_data):
            return jsonify({"error": "Failed to save chunk to storage"}), 500

        logger.debug(f"Saved chunk {chunk_index + 1}/{total_chunks} for upload {upload_id[:8]}...")

        # Check if all chunks received
        received_chunks = chunk_storage_service.list_chunks(upload_id)

        if len(received_chunks) != total_chunks:
            return jsonify({"success": True, "message": "Chunk saved"}), 200

        logger.info(f"All chunks received for upload {upload_id[:8]}..., processing...")

        # Estimate total file size from chunk sizes
        total_file_size = len(chunk_data) * total_chunks  # Approximate

        # Initialize ProgressTracker with file size
        tracker = ProgressTracker(
            upload_id=upload_id,
            file_size_bytes=total_file_size,
            total_chunks=total_chunks
        )
        tracker.start_phase('chunk_assembly')
        tracker.emit('chunk_assembly', 0, 'chunk_assembly_start', force=True, message_params={'totalChunks': total_chunks})

        try:
            # Download and combine all chunks from Supabase Storage
            full_content = chunk_storage_service.download_all_chunks_as_string(
                upload_id, total_chunks, encoding='utf-8'
            )

            if full_content is None:
                raise ValueError("Failed to download and assemble chunks from storage")

            logger.debug(f"Assembled {total_chunks} chunks: {len(full_content)} bytes total")

            # End chunk assembly phase
            tracker.end_phase('chunk_assembly')
            tracker.emit('chunk_assembly', 10, 'chunk_assembly_complete', force=True)

            final_lines = full_content.split('\n')
            logger.debug(f"Final content total lines: {len(final_lines)}")
            if len(final_lines) > 0:
                logger.debug(f"Final content first line: '{final_lines[0]}'")
            if len(final_lines) > 1:
                logger.debug(f"Final content second line: '{final_lines[1]}'")
            if len(final_lines) > 2:
                logger.debug(f"Final content third line: '{final_lines[2]}'")

            # Pass tracker to process_csv
            result = process_csv(full_content, tss, offset, mode, intrpl_max, upload_id, tracker, decimal_precision)

            # Cleanup chunks from Supabase Storage after successful processing
            deleted = chunk_storage_service.delete_upload_chunks(upload_id)
            logger.debug(f"Cleaned up {deleted} chunks for upload {upload_id[:8]}...")

            # Track processing and storage usage
            try:
                increment_processing_count(g.user_id)
                logger.debug(f"Tracked processing for user {g.user_id}")

                file_size_bytes = len(full_content.encode('utf-8'))
                file_size_mb = file_size_bytes / (1024 * 1024)
                update_storage_usage(g.user_id, file_size_mb)
                logger.debug(f"Tracked storage for user {g.user_id}: {file_size_mb:.2f} MB")
            except Exception as e:
                logger.error(f"Failed to track processing usage: {str(e)}")

            return result

        except Exception as e:
            # Cleanup chunks on error - scheduled job will handle any missed ones
            try:
                chunk_storage_service.delete_upload_chunks(upload_id)
            except Exception:
                pass  # Scheduled cleanup will handle it
            raise

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

        logger.debug(f"File prepared for download: {file_id}")

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
        logger.debug(f"Download request for file: {file_id}")

        # Get signed URL from Supabase Storage (valid for 1 hour)
        signed_url = storage_service.get_download_url(file_id, expires_in=3600)

        if signed_url:
            logger.debug(f"Generated signed URL for: {file_id}")

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
                    logger.debug(f"Cleaned up file: {file_id}")
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
