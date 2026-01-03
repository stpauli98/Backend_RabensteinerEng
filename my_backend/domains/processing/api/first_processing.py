"""
First Processing API endpoints.
Handles chunked CSV upload and initial processing with various modes.

Chunks are stored on local filesystem for fast processing.
Session Affinity ensures all chunks from same user go to same instance.
Chunks are automatically cleaned up after processing.
"""
import csv
import logging
import traceback
from io import StringIO

from flask import Blueprint, request, jsonify, g, Response

from shared.auth.jwt import require_auth
from shared.auth.subscription import require_subscription, check_processing_limit
from shared.tracking.usage import increment_processing_count, update_storage_usage
from shared.storage.service import storage_service
from domains.processing.services.local_chunk_service import local_chunk_service

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

    Chunks are stored on local filesystem for fast processing.
    Session Affinity ensures all chunks from same user go to same instance.
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

        # Upload chunk to local filesystem
        if not local_chunk_service.upload_chunk(upload_id, chunk_index, chunk_data):
            return jsonify({"error": "Failed to save chunk to storage"}), 500

        logger.debug(f"Saved chunk {chunk_index + 1}/{total_chunks} for upload {upload_id[:8]}...")

        # Check if all chunks received
        received_chunks = local_chunk_service.list_chunks(upload_id)

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
            # Download and combine all chunks from local filesystem
            full_content = local_chunk_service.download_all_chunks_as_string(
                upload_id, total_chunks, encoding='utf-8'
            )

            if full_content is None:
                raise ValueError("Failed to assemble chunks from filesystem")

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

            # Cleanup chunks from local filesystem after successful processing
            deleted = local_chunk_service.delete_upload_chunks(upload_id)
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
            # Cleanup chunks on error - TTL cleanup will handle any missed ones
            try:
                local_chunk_service.delete_upload_chunks(upload_id)
            except Exception:
                pass  # TTL cleanup will handle it
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
    Saves CSV data to local filesystem for fast access and no size limits.
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

        # Save to local filesystem
        user_id = g.user_id

        import uuid
        file_id = f"{user_id}_{uuid.uuid4().hex[:8]}"

        if not local_chunk_service.save_processed_result(
            file_id=file_id,
            csv_content=csv_content,
            metadata={
                'totalRows': len(save_data) - 1,
                'source': 'first-processing-prepare-save',
                'originalFilename': file_name or "processed_data.csv"
            }
        ):
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
    Download prepared CSV file.
    First tries local storage, then falls back to Supabase Storage for legacy files.
    Returns the file directly for local storage, or signed URL for Supabase.
    """
    try:
        logger.debug(f"Download request for file: {file_id}")

        # First, try to get from local storage (new files)
        csv_content = local_chunk_service.get_processed_result(file_id)
        
        if csv_content:
            logger.debug(f"Serving file from local storage: {file_id}")
            
            # Extract filename from file_id for download
            filename = file_id.split('/')[-1] if '/' in file_id else file_id
            if not filename.endswith('.csv'):
                filename += '.csv'
            
            return Response(
                csv_content,
                mimetype='text/csv',
                headers={
                    'Content-Disposition': f'attachment; filename="{filename}"',
                    'Content-Type': 'text/csv; charset=utf-8'
                }
            )

        # Fallback: Try Supabase Storage for legacy files
        logger.debug(f"File not in local storage, trying Supabase: {file_id}")
        signed_url = storage_service.get_download_url(file_id, expires_in=3600)

        if signed_url:
            logger.debug(f"Generated signed URL for legacy file: {file_id}")

            return jsonify({
                "success": True,
                "downloadUrl": signed_url,
                "fileId": file_id
            }), 200

        logger.warning(f"File not found in local or Supabase storage: {file_id}")
        return jsonify({"error": "File not found"}), 404

    except Exception as e:
        logger.error(f"Error in download_file: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@bp.route('/cleanup-files', methods=['POST'])
@require_auth
def cleanup_files():
    """
    Delete files after successful download.
    Tries local storage first, then falls back to Supabase Storage for legacy files.
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
                # First try local storage (new files)
                if local_chunk_service.delete_processed_result(file_id):
                    deleted_count += 1
                    logger.debug(f"Cleaned up file from local storage: {file_id}")
                # Fallback to Supabase Storage (legacy files)
                elif storage_service.delete_file(file_id):
                    deleted_count += 1
                    logger.debug(f"Cleaned up file from Supabase: {file_id}")
                else:
                    # File not found in either location - consider it already deleted
                    logger.debug(f"File not found (already deleted?): {file_id}")
                    deleted_count += 1
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
