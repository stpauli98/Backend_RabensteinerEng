"""
Refaktorisan RowData Blueprint sa servisnim slojem
"""
import os
import tempfile
import logging
import time
import csv
from datetime import datetime
from flask import Blueprint, request, jsonify, send_file, current_app
from flask_socketio import emit, join_room

# Import servisa i utility modula
from .services import FileUploadService, DateParsingService, DataProcessingService
from .repositories.repository_factory import get_repository, get_storage_info
from .utils import (
    UploadValidator, SecurityValidator,
    handle_exception, require_auth, require_permission, apply_rate_limit,
    setup_cors_headers, limiter
)
from .utils.exceptions import (
    ValidationError, UploadError, ProcessingError,
    AuthenticationError, RateLimitError
)
from .config import RATE_LIMITS, SECURITY_CONFIG

# Inicijalizacija
logger = logging.getLogger(__name__)
bp = Blueprint('rowdata', __name__)

# Inicijalizacija servisa (lazy loading)
_services = {}

def get_services():
    """Lazy inicijalizacija servisa"""
    if not _services:
        # Koristi factory za automatski izbor repository-ja
        repository = get_repository()
        _services['repository'] = repository
        _services['file_service'] = FileUploadService(repository)
        _services['date_service'] = DateParsingService()
        _services['processing_service'] = DataProcessingService(
            _services['file_service'],
            _services['date_service']
        )
        
        # Log koji storage koristimo
        storage_info = get_storage_info()
        logger.info(f"RowData using {storage_info['backend']} storage: {storage_info['description']}")
    return _services

def get_socketio():
    """Dohvata Socket.IO instancu"""
    return current_app.extensions.get('socketio')

# Temporarno skladište za download fajlove (trebalo bi premestiti u Redis)
temp_files = {}

# Error handlers
@bp.errorhandler(ValidationError)
@bp.errorhandler(UploadError)
@bp.errorhandler(ProcessingError)
def handle_rowdata_error(error):
    """Handler za RowData greške"""
    response, status_code = handle_exception(error)
    return jsonify(response), status_code

@bp.errorhandler(AuthenticationError)
def handle_auth_error(error):
    """Handler za autentifikacione greške"""
    return jsonify({
        "error": error.code,
        "message": error.message
    }), error.status_code

@bp.errorhandler(RateLimitError)
def handle_rate_limit_error(error):
    """Handler za rate limit greške"""
    response = jsonify({
        "error": "RATE_LIMIT_EXCEEDED",
        "message": error.message,
        "retry_after": error.retry_after
    })
    response.headers['Retry-After'] = str(error.retry_after)
    return response, 429

# Middleware
@bp.after_request
def after_request(response):
    """Postavlja CORS header-e"""
    return setup_cors_headers(response)

# Socket.IO event za join room
def register_socketio_events(socketio):
    """Registruje Socket.IO event handler-e"""
    
    @socketio.on('join_upload_room')
    def handle_join_room(data):
        upload_id = data.get('uploadId')
        if upload_id:
            join_room(upload_id)
            emit('joined_room', {'uploadId': upload_id}, room=upload_id)

# API Endpoints

@bp.route('/upload-chunk', methods=['POST'])
@require_auth
@apply_rate_limit(RATE_LIMITS.get('upload_chunk', '100 per minute'))
def upload_chunk():
    """
    Endpoint za upload pojedinačnog chunk-a
    
    Expects:
        - fileChunk: File chunk
        - uploadId: Unique upload identifier
        - chunkIndex: Index of current chunk
        - totalChunks: Total number of chunks
        - delimiter: CSV delimiter
        - ... other parameters
    """
    try:
        services = get_services()
        
        # Validacija request-a
        validated_params = UploadValidator.validate_upload_request(
            request.form.to_dict(),
            request.files
        )
        
        # Dohvati chunk
        file_chunk = request.files['fileChunk']
        
        # Sačuvaj chunk
        chunk_info = services['file_service'].store_chunk(
            upload_id=validated_params['upload_id'],
            chunk_index=validated_params['chunk_index'],
            chunk_data=file_chunk.read(),
            metadata={
                'total_chunks': validated_params['total_chunks'],
                'delimiter': validated_params['delimiter'],
                'timezone': validated_params['timezone'],
                'has_header': validated_params['has_header'],
                'selected_columns': validated_params['selected_columns'],
                'dropdown_count': validated_params['dropdown_count'],
                'custom_date_format': validated_params['custom_date_format'],
                'value_column_name': validated_params['value_column_name']
            }
        )
        
        # Emituj progress preko Socket.IO
        socketio = get_socketio()
        if socketio:
            progress = int((validated_params['chunk_index'] + 1) / validated_params['total_chunks'] * 100)
            socketio.emit('upload_progress', {
                'uploadId': validated_params['upload_id'],
                'fileName': SecurityValidator.sanitize_filename(file_chunk.filename or 'unknown'),
                'progress': progress,
                'status': 'uploading',
                'message': f"Processing chunk {validated_params['chunk_index'] + 1}/{validated_params['total_chunks']}"
            }, room=validated_params['upload_id'])
        
        return jsonify({
            "success": True,
            "message": f"Chunk {validated_params['chunk_index'] + 1} received",
            "uploadId": validated_params['upload_id'],
            "chunkInfo": {
                "size": chunk_info['size'],
                "hash": chunk_info['hash'],
                "duplicate": chunk_info.get('duplicate', False)
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Error in upload_chunk: {str(e)}", exc_info=True)
        if isinstance(e, (ValidationError, UploadError)):
            raise
        return jsonify({"error": "Upload failed", "message": "An error occurred during upload"}), 500

@bp.route('/finalize-upload', methods=['POST'])
@require_auth
@apply_rate_limit(RATE_LIMITS.get('finalize_upload', '10 per minute'))
def finalize_upload():
    """
    Finalizuje upload i pokreće procesiranje
    
    Expects:
        - uploadId: Upload identifier
    """
    try:
        services = get_services()
        
        # Validacija
        data = request.get_json(force=True, silent=True) or {}
        upload_id = data.get('uploadId')
        
        if not upload_id:
            raise ValidationError("uploadId is required")
        
        # Verifikuj integritet
        is_valid, error_msg = services['file_service'].verify_upload_integrity(upload_id)
        if not is_valid:
            raise UploadError(f"Upload integrity check failed: {error_msg}", upload_id)
        
        # Dohvati metadata
        metadata = services['repository'].get_upload_metadata(upload_id)
        if not metadata:
            raise UploadError("Upload not found", upload_id)
        
        # Emituj status
        socketio = get_socketio()
        if socketio:
            socketio.emit('upload_progress', {
                'uploadId': upload_id,
                'progress': 50,
                'status': 'processing',
                'message': 'Processing file data...'
            }, room=upload_id)
        
        # Procesiraj podatke
        result_data = []
        for row in services['processing_service'].process_upload_data(upload_id, metadata):
            # Preskoči _row_number iz rezultata
            clean_row = {k: v for k, v in row.items() if not k.startswith('_')}
            result_data.append(clean_row)
        
        # Sortiraj po UTC
        result_data.sort(key=lambda x: x['UTC'])
        
        # Pripremi response
        headers = list(result_data[0].keys()) if result_data else ['UTC']
        data_list = [headers]
        
        for row in result_data:
            data_list.append([row.get(h, '') for h in headers])
        
        # Sačuvaj rezultat
        services['repository'].store_processing_result(upload_id, {
            'row_count': len(result_data),
            'headers': headers,
            'processing_time': datetime.utcnow().isoformat()
        })
        
        # Cleanup
        services['file_service'].cleanup_upload(upload_id)
        
        # Emituj završetak
        if socketio:
            socketio.emit('upload_progress', {
                'uploadId': upload_id,
                'progress': 100,
                'status': 'completed',
                'message': 'Data processing completed'
            }, room=upload_id)
        
        return jsonify({
            "success": True,
            "data": data_list,
            "fullData": data_list,
            "rowCount": len(result_data),
            "uploadId": upload_id
        }), 200
        
    except Exception as e:
        logger.error(f"Error in finalize_upload: {str(e)}", exc_info=True)
        
        # Emituj grešku
        socketio = get_socketio()
        if socketio and upload_id:
            socketio.emit('upload_progress', {
                'uploadId': upload_id,
                'progress': 0,
                'status': 'error',
                'message': str(e) if isinstance(e, (ValidationError, UploadError)) else 'Processing failed'
            }, room=upload_id)
        
        if isinstance(e, (ValidationError, UploadError, ProcessingError)):
            raise
        return jsonify({"error": "Processing failed", "message": "An error occurred during processing"}), 500

@bp.route('/cancel-upload', methods=['POST'])
@require_auth
def cancel_upload():
    """
    Otkazuje upload u toku
    
    Expects:
        - uploadId: Upload identifier
    """
    try:
        services = get_services()
        
        data = request.get_json(force=True, silent=True) or {}
        upload_id = data.get('uploadId')
        
        if not upload_id:
            raise ValidationError("uploadId is required")
        
        # Cleanup
        services['file_service'].cleanup_upload(upload_id)
        
        # Emituj otkazivanje
        socketio = get_socketio()
        if socketio:
            socketio.emit('upload_progress', {
                'uploadId': upload_id,
                'progress': 0,
                'status': 'error',
                'message': 'Upload canceled by user'
            }, room=upload_id)
        
        return jsonify({
            "success": True,
            "message": "Upload canceled successfully"
        }), 200
        
    except Exception as e:
        logger.error(f"Error in cancel_upload: {str(e)}")
        return jsonify({"error": "Failed to cancel upload"}), 500

@bp.route('/check-status/<upload_id>', methods=['GET'])
@require_auth
def check_upload_status(upload_id):
    """Proverava status upload-a"""
    try:
        services = get_services()
        
        metadata = services['repository'].get_upload_metadata(upload_id)
        if not metadata:
            return jsonify({
                "error": "Upload not found"
            }), 404
        
        received_chunks = len(services['repository'].get_received_chunks(upload_id))
        total_chunks = metadata.get('total_chunks', 0)
        
        return jsonify({
            "success": True,
            "uploadId": upload_id,
            "totalChunks": total_chunks,
            "receivedChunks": received_chunks,
            "isComplete": received_chunks == total_chunks,
            "metadata": {
                "created_at": metadata.get('created_at'),
                "last_activity": metadata.get('last_activity')
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Error checking status: {str(e)}")
        return jsonify({"error": "Failed to check status"}), 500

@bp.route('/prepare-save', methods=['POST'])
@require_auth
def prepare_save():
    """
    Priprema podatke za download
    
    Expects:
        - data: Array of data to save
        - fileName: Optional filename
    """
    try:
        data = request.json
        
        if not data or 'data' not in data:
            raise ValidationError("No data provided")
        
        data_wrapper = data['data']
        save_data = data_wrapper.get('data', [])
        file_name = SecurityValidator.sanitize_filename(
            data_wrapper.get('fileName', f'data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
        )
        
        if not save_data:
            raise ValidationError("Empty data")
        
        # Kreiraj temp fajl
        temp_file = tempfile.NamedTemporaryFile(
            mode='w+',
            delete=False,
            suffix='.csv',
            dir=tempfile.gettempdir()
        )
        
        # Piši CSV
        writer = csv.writer(temp_file, delimiter=';')
        for row in save_data:
            writer.writerow(row)
        temp_file.close()
        
        # Generiši file ID
        file_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Sačuvaj info (trebalo bi u Redis)
        temp_files[file_id] = {
            'path': temp_file.name,
            'fileName': file_name,
            'timestamp': time.time()
        }
        
        # Cleanup starih fajlova
        current_time = time.time()
        for fid, info in list(temp_files.items()):
            if current_time - info['timestamp'] > 3600:  # 1 sat
                try:
                    os.unlink(info['path'])
                    del temp_files[fid]
                except:
                    pass
        
        return jsonify({
            "success": True,
            "message": "File prepared for download",
            "fileId": file_id
        }), 200
        
    except Exception as e:
        logger.error(f"Error in prepare_save: {str(e)}")
        if isinstance(e, ValidationError):
            raise
        return jsonify({"error": "Failed to prepare file"}), 500

@bp.route('/download/<file_id>', methods=['GET'])
@require_auth
@apply_rate_limit(RATE_LIMITS.get('download_file', '50 per minute'))
def download_file(file_id):
    """Download pripremljen fajl"""
    try:
        # Validacija file_id
        if not file_id or not file_id.isalnum():
            raise ValidationError("Invalid file ID")
        
        if file_id not in temp_files:
            return jsonify({"error": "File not found"}), 404
        
        file_info = temp_files[file_id]
        file_path = file_info['path']
        
        # Validacija putanje
        SecurityValidator.validate_path_traversal(file_path)
        
        if not os.path.exists(file_path):
            return jsonify({"error": "File not found"}), 404
        
        try:
            return send_file(
                file_path,
                as_attachment=True,
                download_name=file_info['fileName'],
                mimetype='text/csv'
            )
        finally:
            # Cleanup nakon download-a
            try:
                os.unlink(file_path)
                del temp_files[file_id]
            except:
                pass
                
    except Exception as e:
        logger.error(f"Error in download_file: {str(e)}")
        if isinstance(e, ValidationError):
            raise
        return jsonify({"error": "Download failed"}), 500

@bp.route('/cleanup', methods=['POST'])
@require_auth
@require_permission('admin')
def cleanup_old_uploads():
    """Admin endpoint za cleanup starih upload-ova"""
    try:
        services = get_services()
        
        cleaned_count = services['file_service'].cleanup_old_uploads()
        
        return jsonify({
            "success": True,
            "message": f"Cleaned up {cleaned_count} old uploads"
        }), 200
        
    except Exception as e:
        logger.error(f"Error in cleanup: {str(e)}")
        return jsonify({"error": "Cleanup failed"}), 500

@bp.route('/stats', methods=['GET'])
@require_auth
@require_permission('admin')  
def get_statistics():
    """Admin endpoint za statistiku"""
    try:
        services = get_services()
        
        stats = services['repository'].get_upload_statistics()
        
        return jsonify({
            "success": True,
            "statistics": stats
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting stats: {str(e)}")
        return jsonify({"error": "Failed to get statistics"}), 500

@bp.route('/storage-info', methods=['GET'])
def get_storage_backend_info():
    """Informacije o storage backend-u"""
    try:
        storage_info = get_storage_info()
        
        return jsonify({
            "success": True,
            "storage": storage_info
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting storage info: {str(e)}")
        return jsonify({"error": "Failed to get storage info"}), 500