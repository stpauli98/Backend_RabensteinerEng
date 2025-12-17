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

        # Check for duplicate file_name + bezeichnung on first chunk only
        chunk_index = metadata.get('chunkIndex', 0)
        if chunk_index == 0:
            file_metadata = additional_data.get('fileMetadata', {})
            file_name = metadata.get('fileName', '')
            bezeichnung = file_metadata.get('bezeichnung', '')
            session_id = metadata.get('sessionId', '')

            if file_name and bezeichnung and session_id:
                from shared.database.client import get_supabase_client
                from shared.database.operations import create_or_get_session_uuid

                try:
                    uuid_session_id = create_or_get_session_uuid(session_id, user_id=g.user_id if hasattr(g, 'user_id') else None)
                    if uuid_session_id:
                        supabase = get_supabase_client()
                        existing = supabase.table('files')\
                            .select('id')\
                            .eq('session_id', uuid_session_id)\
                            .eq('file_name', file_name)\
                            .eq('bezeichnung', bezeichnung)\
                            .execute()

                        if existing.data and len(existing.data) > 0:
                            return create_error_response(
                                f'File "{file_name}" with bezeichnung "{bezeichnung}" already exists in this session',
                                400
                            )
                except Exception as dup_check_error:
                    logger.warning(f"Duplicate check failed (non-blocking): {dup_check_error}")

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


@bp.route('/instant-upload', methods=['POST'])
@require_auth
@require_subscription
@check_processing_limit
def instant_upload():
    """Upload a single CSV file immediately to database and Storage.

    This endpoint is used for instant file uploads when a user adds a file.
    It performs deduplication via file hash check before uploading.

    Request: FormData with:
    - file: CSV file
    - sessionId: string (required)
    - fileMetadata: JSON string with file metadata
    - fileHash: SHA-256 hash of file content (optional)

    Response:
    - success: bool
    - data:
        - fileId: UUID of the file
        - isNew: bool (true if newly uploaded, false if duplicate)
        - storagePath: string path in storage
    """
    try:
        from shared.database.operations import save_file_info, save_csv_file_content
        from shared.database.storage import check_file_exists_by_hash
        from shared.database.session import resolve_session_id

        # Validate request
        if 'file' not in request.files:
            return create_error_response('No file provided', 400)

        file = request.files['file']
        if not file.filename:
            return create_error_response('No file selected', 400)

        session_id = request.form.get('sessionId')
        if not session_id:
            return create_error_response('Session ID is required', 400)

        # Parse file metadata
        file_metadata = {}
        if 'fileMetadata' in request.form:
            try:
                file_metadata = json.loads(request.form['fileMetadata'])
            except json.JSONDecodeError:
                return create_error_response('Invalid fileMetadata JSON', 400)

        file_hash = request.form.get('fileHash', '')

        # Resolve session ID to UUID
        uuid_session_id = resolve_session_id(session_id)
        if not uuid_session_id:
            return create_error_response(f'Invalid session ID: {session_id}', 400)

        # Check for duplicate by hash - if found, create NEW DB record with SHARED storage
        if file_hash:
            existing_file = check_file_exists_by_hash(uuid_session_id, file_hash)
            if existing_file:
                logger.info(f"File with hash {file_hash[:16]}... already exists, creating new record with shared storage")

                # Prepare file data for NEW database record (shared storage)
                safe_filename = sanitize_filename(file.filename)
                bezeichnung = file_metadata.get('bezeichnung', safe_filename)
                file_type = file_metadata.get('type', 'input')

                # Check for duplicate bezeichnung in this session
                from shared.database.client import get_supabase_client
                supabase = get_supabase_client()
                existing_bezeichnung = supabase.table('files')\
                    .select('id')\
                    .eq('session_id', uuid_session_id)\
                    .eq('bezeichnung', bezeichnung)\
                    .eq('type', file_type)\
                    .execute()

                if existing_bezeichnung.data and len(existing_bezeichnung.data) > 0:
                    return create_error_response(
                        f'A file with bezeichnung "{bezeichnung}" already exists in this session',
                        400
                    )

                # Create NEW file record with SHARED storage_path
                file_data = {
                    'fileName': safe_filename,
                    'bezeichnung': bezeichnung,
                    'type': file_type,
                    'file_hash': file_hash,
                    'storage_path': existing_file.get('storage_path', ''),  # Share storage path
                    **file_metadata
                }

                from shared.database.operations import save_file_info
                success, file_uuid = save_file_info(session_id, file_data)
                if not success:
                    return create_error_response('Failed to save file metadata', 500)

                # Update storage_path in the new record
                try:
                    supabase.table('files').update({
                        'storage_path': existing_file.get('storage_path', ''),
                        'file_hash': file_hash
                    }).eq('id', file_uuid).execute()
                except Exception as update_err:
                    logger.warning(f"Could not update storage_path for shared file: {update_err}")

                increment_processing_count(g.user_id)
                logger.info(f"Created new record {file_uuid} with shared storage from {existing_file['id']}")

                return jsonify({
                    'success': True,
                    'data': {
                        'fileId': file_uuid,  # NEW ID
                        'isNew': True,  # Logically new record
                        'storagePath': existing_file.get('storage_path', ''),
                        'sharedStorage': True,
                        'message': 'File record created with shared storage'
                    }
                })

        # Prepare file data for database
        safe_filename = sanitize_filename(file.filename)
        bezeichnung = file_metadata.get('bezeichnung', safe_filename)
        file_type = file_metadata.get('type', 'input')

        # Check for duplicate bezeichnung in this session
        from shared.database.client import get_supabase_client
        supabase = get_supabase_client()
        existing_bezeichnung = supabase.table('files')\
            .select('id')\
            .eq('session_id', uuid_session_id)\
            .eq('bezeichnung', bezeichnung)\
            .eq('type', file_type)\
            .execute()

        if existing_bezeichnung.data and len(existing_bezeichnung.data) > 0:
            return create_error_response(
                f'A file with bezeichnung "{bezeichnung}" already exists in this session',
                400
            )

        file_data = {
            'fileName': safe_filename,
            'bezeichnung': bezeichnung,
            'type': file_type,
            **file_metadata
        }

        # Add file hash to metadata if provided
        if file_hash:
            file_data['file_hash'] = file_hash

        # Save file metadata to database
        success, file_uuid = save_file_info(session_id, file_data)
        if not success:
            return create_error_response('Failed to save file metadata', 500)

        # Save file content to Storage
        storage_path = ''
        if file_uuid:
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                file.save(temp_file.name)
                temp_path = temp_file.name

            try:
                file_type = file_data.get('type', 'input')
                storage_success = save_csv_file_content(
                    file_uuid, uuid_session_id, safe_filename, temp_path, file_type
                )

                if storage_success:
                    storage_path = f"{uuid_session_id}/{safe_filename}"
                else:
                    logger.warning(f"Failed to upload file to storage for file {file_uuid}")

            finally:
                try:
                    os.unlink(temp_path)
                except:
                    pass

        # Update file record with hash if provided
        if file_hash and file_uuid:
            try:
                from shared.database.session import get_supabase_client
                supabase = get_supabase_client()
                supabase.table('files').update({
                    'file_hash': file_hash
                }).eq('id', file_uuid).execute()
            except Exception as hash_err:
                logger.warning(f"Could not save file hash: {hash_err}")

        increment_processing_count(g.user_id)
        logger.info(f"Instant upload successful for user {g.user_id}, file {file_uuid}")

        return jsonify({
            'success': True,
            'data': {
                'fileId': file_uuid,
                'isNew': True,
                'storagePath': storage_path,
                'message': 'File uploaded successfully'
            }
        })

    except ValueError as e:
        logger.error(f"Validation error in instant upload: {str(e)}")
        return create_error_response(str(e), 400)

    except Exception as e:
        logger.error(f"Error in instant upload: {str(e)}")
        return create_error_response(str(e), 500)
