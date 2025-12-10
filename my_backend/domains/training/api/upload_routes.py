"""
Upload routes for training API.

Contains 5 endpoints for file upload and CSV management:
- upload-chunk
- csv-files/<session_id> GET
- csv-files POST
- csv-files/<file_id> PUT
- csv-files/<file_id> DELETE
"""

import tempfile
from flask import Blueprint

from .common import (
    os, json, request, jsonify, g, logging,
    require_auth, require_subscription, check_processing_limit,
    increment_processing_count,
    sanitize_filename,
    create_error_response, create_success_response,
    get_logger
)

from domains.training.services.session import get_csv_files_for_session
from domains.training.services.upload import (
    process_chunk_upload, create_csv_file_record,
    update_csv_file_record, delete_csv_file_record
)

bp = Blueprint('training_upload', __name__)
logger = get_logger(__name__)


@bp.route('/upload-chunk', methods=['POST'])
@require_auth
@require_subscription
@check_processing_limit
def upload_chunk():
    """Handle chunk upload from frontend - saving locally."""
    try:
        if 'chunk' not in request.files:
            return create_error_response('No chunk in request', 400)

        chunk_file = request.files['chunk']
        if not chunk_file.filename:
            return create_error_response('No chunk file selected', 400)

        if 'metadata' not in request.form:
            return create_error_response('No metadata provided', 400)

        metadata = json.loads(request.form['metadata'])
        chunk_data = chunk_file.read()
        additional_data = {}

        if 'additionalData' in request.form:
            additional_data = json.loads(request.form['additionalData'])

        for key, value in request.form.items():
            if key not in ['metadata', 'additionalData']:
                try:
                    additional_data[key] = json.loads(value)
                except (json.JSONDecodeError, TypeError):
                    additional_data[key] = value

        for key, value in request.args.items():
            additional_data[key] = value

        result = process_chunk_upload(chunk_data, metadata, additional_data)

        if result.get('assembled'):
            increment_processing_count(g.user_id)
            logger.info(f"Tracked CSV upload for user {g.user_id}")

        return jsonify({
            'success': True,
            'message': result['message']
        })

    except ValueError as e:
        logger.error(f"Validation error in chunk upload: {str(e)}")
        return create_error_response(str(e), 400)

    except Exception as e:
        logger.error(f"Error processing chunk upload: {str(e)}")
        return create_error_response(str(e), 500)


@bp.route('/csv-files/<session_id>', methods=['GET'])
@require_auth
def get_csv_files_endpoint(session_id):
    """Get all CSV files for a session."""
    try:
        file_type = request.args.get('type', None)
        files = get_csv_files_for_session(session_id, file_type)

        if files:
            return jsonify({
                'success': True,
                'data': files
            })
        else:
            return jsonify({
                'success': True,
                'data': [],
                'message': f'No CSV files found for session {session_id}'
            })

    except ValueError as e:
        return create_error_response(str(e), 400)
    except Exception as e:
        logger.error(f"Error getting CSV files for session {session_id}: {str(e)}")
        return create_error_response(str(e), 500)


@bp.route('/csv-files', methods=['POST'])
@require_auth
@require_subscription
@check_processing_limit
def create_csv_file():
    """Create a new CSV file entry."""
    try:
        from shared.database.operations import save_file_info, save_csv_file_content

        if 'file' in request.files:
            file = request.files['file']
            session_id = request.form.get('sessionId')
            file_data = request.form.to_dict()
        else:
            data = request.get_json()
            if not data:
                return create_error_response('No data provided', 400)

            session_id = data.get('sessionId')
            file_data = data.get('fileData', {})
            file = None

        if not session_id:
            return create_error_response('Session ID is required', 400)

        success, file_uuid = save_file_info(session_id, file_data)
        if not success:
            return create_error_response('Failed to save file metadata', 500)

        if file and file_uuid:
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                file.save(temp_file.name)
                temp_path = temp_file.name

            try:
                file_type = file_data.get('type', 'input')
                # Sanitiziraj ime datoteke prije spremanja
                safe_filename = sanitize_filename(file.filename)
                storage_success = save_csv_file_content(
                    file_uuid, session_id, safe_filename, temp_path, file_type
                )

                if not storage_success:
                    logger.warning(f"Failed to upload file to storage for file {file_uuid}")

            finally:
                try:
                    os.unlink(temp_path)
                except:
                    pass

        increment_processing_count(g.user_id)
        logger.info(f"Tracked CSV file creation for user {g.user_id}")

        return jsonify({
            'success': True,
            'data': {
                'id': file_uuid,
                'message': 'CSV file created successfully'
            }
        })

    except Exception as e:
        logger.error(f"Error creating CSV file: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@bp.route('/csv-files/<file_id>', methods=['PUT'])
@require_auth
@require_subscription
def update_csv_file(file_id):
    """Update CSV file metadata."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400

        updated_file = update_csv_file_record(file_id, data)

        return jsonify({
            'success': True,
            'data': updated_file
        })

    except ValueError as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

    except Exception as e:
        logger.error(f"Error updating CSV file {file_id}: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@bp.route('/csv-files/<file_id>', methods=['DELETE'])
@require_auth
def delete_csv_file(file_id):
    """Delete CSV file from database and storage."""
    try:
        result = delete_csv_file_record(file_id)

        return jsonify({
            'success': True,
            'message': result['message']
        })

    except ValueError as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 404

    except Exception as e:
        logger.error(f"Error deleting CSV file {file_id}: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
