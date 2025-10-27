import os
import json
import logging
import hashlib
import uuid
import pandas as pd
import numpy as np
from datetime import datetime
from io import BytesIO, StringIO
from flask import Blueprint, request, jsonify, current_app, send_file
from werkzeug.utils import secure_filename
import tempfile
import glob
import shutil
import re
import time
from pathlib import Path
from typing import Optional
from utils.database import save_session_to_supabase, get_string_id_from_uuid, create_or_get_session_uuid, get_supabase_client
from middleware.auth import require_auth
from middleware.subscription import require_subscription, check_processing_limit, check_training_limit
from utils.usage_tracking import increment_processing_count, increment_training_count
from utils.validation import validate_session_id, create_error_response, create_success_response
from utils.metadata_utils import extract_file_metadata_fields, extract_file_metadata

from services.training.visualization import Visualizer

from services.training.scaler_manager import get_session_scalers, create_scaler_download_package, scale_new_data

from services.training.model_manager import save_models_to_storage, get_models_list, download_model_file

from services.training.session_manager import (
    initialize_session, finalize_session, get_sessions_list,
    get_session_info, get_session_from_database, delete_session,
    delete_all_sessions, update_session_name, save_time_info_data,
    save_zeitschritte_data, get_time_info_data, get_zeitschritte_data,
    get_csv_files_for_session, get_session_status, create_database_session,
    get_session_uuid, get_upload_status
)

from services.training.upload_manager import (
    process_chunk_upload, verify_file_hash, assemble_file_locally,
    save_session_metadata_locally, get_session_metadata_locally,
    update_session_metadata, verify_session_files,
    create_csv_file_record, update_csv_file_record, delete_csv_file_record
)

from services.training.dataset_generator import generate_violin_plots_for_session
from services.training.training_orchestrator import run_model_training_async

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

bp = Blueprint('training', __name__)

#     """Validate session ID format - should be either UUID or session_XXXXXX_XXXXXX format"""
#     """Create standardized error response"""
#     """Create standardized success response"""

#     """
#     Helper function to extract standardized file metadata fields from a file metadata dictionary.
#
#     Args:
#         file_metadata: Dictionary containing file metadata
#
#     Returns:
#         dict: Dictionary containing standardized file metadata fields
#     """

def calculate_n_dat_from_session(session_id):
    """
    Calculate n_dat (total number of data samples) from uploaded CSV files in a session.
    This mimics the n_dat = i_array_3D.shape[0] calculation from training_original.py
    
    Args:
        session_id: ID of the session
        
    Returns:
        int: Total number of data samples (n_dat)
    """
    try:
        upload_dir = os.path.join(UPLOAD_BASE_DIR, session_id)
        if not os.path.exists(upload_dir):
            logger.warning(f"Upload directory not found for session {session_id}")
            return 0
        
        total_samples = 0
        csv_files = []
        
        for file_name in os.listdir(upload_dir):
            if file_name.lower().endswith('.csv') and os.path.isfile(os.path.join(upload_dir, file_name)):
                csv_files.append(file_name)
        
        if not csv_files:
            logger.warning(f"No CSV files found in session {session_id}")
            return 0
        
        for csv_file in csv_files:
            file_path = os.path.join(upload_dir, csv_file)
            try:
                df = pd.read_csv(file_path)
                file_samples = len(df)
                total_samples += file_samples
                logger.info(f"File {csv_file}: {file_samples} samples")
            except Exception as e:
                logger.error(f"Error reading CSV file {csv_file}: {str(e)}")
                continue
        
        logger.info(f"Session {session_id} total n_dat: {total_samples}")
        return total_samples
        
    except Exception as e:
        logger.error(f"Error calculating n_dat for session {session_id}: {str(e)}")
        return 0

#     """
#     Extracts file metadata from session metadata.
#
#     Args:
#         session_id: ID of the session to extract metadata from
#
#     Returns:
#         dict: Dictionary containing file metadata fields or None if not found
#     """

MAX_SESSIONS_TO_RETURN = 50

UPLOAD_BASE_DIR = 'uploads/file_uploads'

def verify_file_hash(file_data: bytes, expected_hash: str) -> bool:
    """Verify the hash of file data matches the expected hash."""
    try:
        file_hash = hashlib.sha256(file_data).hexdigest()
        return file_hash == expected_hash
    except Exception as e:
        logger.error(f"Error verifying file hash: {str(e)}")
        return False

def assemble_file_locally(upload_id: str, filename: str) -> str:
    """
    Securely assemble a complete file from its chunks.

    Args:
        upload_id: Unique identifier for the upload session
        filename: Name of the file to assemble

    Returns:
        str: Path to the assembled file
    """
    safe_filename = secure_filename(filename)
    if not safe_filename or safe_filename != filename:
        raise ValueError(f"Invalid filename: {filename}")

    if not upload_id.replace('_', '').replace('-', '').isalnum():
        raise ValueError(f"Invalid upload_id format: {upload_id}")

    upload_dir = os.path.join(UPLOAD_BASE_DIR, upload_id)

    upload_dir = os.path.abspath(upload_dir)
    base_dir = os.path.abspath(UPLOAD_BASE_DIR)
    if not upload_dir.startswith(base_dir):
        raise ValueError("Path traversal attempt detected")

    if not os.path.exists(upload_dir):
        raise FileNotFoundError(f"Upload directory not found: {upload_dir}")

    metadata_path = os.path.join(upload_dir, 'metadata.json')
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            chunks_metadata = json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise ValueError(f"Invalid metadata file: {e}")

    file_chunks = [c for c in chunks_metadata if c.get('fileName') == filename]
    if not file_chunks:
        raise FileNotFoundError(f"No chunks found for file {filename}")

    file_chunks.sort(key=lambda x: x.get('chunkIndex', 0))
    expected_chunks = file_chunks[0].get('totalChunks', 0)

    if len(file_chunks) != expected_chunks:
        raise ValueError(
            f"Missing chunks for {filename}. "
            f"Expected {expected_chunks}, found {len(file_chunks)}"
        )

    for i, chunk in enumerate(file_chunks):
        if chunk.get('chunkIndex') != i:
            raise ValueError(f"Non-sequential chunk index: expected {i}, got {chunk.get('chunkIndex')}")

    assembled_file_path = os.path.join(upload_dir, safe_filename)
    assembled_file_path = os.path.abspath(assembled_file_path)

    if not assembled_file_path.startswith(upload_dir):
        raise ValueError("Invalid file path detected")

    BUFFER_SIZE = 64 * 1024

    try:
        with open(assembled_file_path, 'wb') as output_file:
            for chunk_info in file_chunks:
                chunk_path = chunk_info.get('filePath')
                if not chunk_path or not os.path.exists(chunk_path):
                    raise FileNotFoundError(f"Chunk file not found: {chunk_path}")

                chunk_path = os.path.abspath(chunk_path)
                if not chunk_path.startswith(base_dir):
                    raise ValueError("Invalid chunk path detected")

                with open(chunk_path, 'rb') as chunk_file:
                    while True:
                        chunk_data = chunk_file.read(BUFFER_SIZE)
                        if not chunk_data:
                            break
                        output_file.write(chunk_data)

        file_size = os.path.getsize(assembled_file_path)
        logger.info(
            f"Successfully assembled {safe_filename} from {len(file_chunks)} chunks, "
            f"total size: {file_size:,} bytes"
        )

        return assembled_file_path

    except Exception as e:
        if os.path.exists(assembled_file_path):
            try:
                os.remove(assembled_file_path)
            except OSError:
                pass
        logger.error(f"Error assembling file locally: {str(e)}")
        raise

def save_session_metadata_locally(session_id: str, metadata: dict) -> bool:
    """
    Securely save session metadata to local storage.

    Args:
        session_id: Session identifier (will be validated)
        metadata: Metadata dictionary to save

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        if not session_id or not isinstance(session_id, str):
            raise ValueError("Invalid session_id")

        clean_session_id = session_id.replace('..', '').replace('/', '').replace('\\', '')
        if clean_session_id != session_id:
            raise ValueError("Invalid characters in session_id")

        upload_dir = os.path.join(UPLOAD_BASE_DIR, clean_session_id)

        upload_dir = os.path.abspath(upload_dir)
        base_dir = os.path.abspath(UPLOAD_BASE_DIR)
        if not upload_dir.startswith(base_dir):
            raise ValueError("Path traversal attempt detected")

        os.makedirs(upload_dir, mode=0o755, exist_ok=True)

        session_metadata_path = os.path.join(upload_dir, 'session_metadata.json')
        temp_path = session_metadata_path + '.tmp'

        try:
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)

            os.rename(temp_path, session_metadata_path)

            logger.info(f"Session metadata saved: {session_metadata_path}")
            return True

        except Exception as e:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass
            raise e

    except Exception as e:
        logger.error(f"Error saving session metadata: {e}")
        return False

def get_session_metadata_locally(session_id: str) -> Optional[dict]:
    """
    Securely retrieve session metadata from local storage.

    Args:
        session_id: Session identifier

    Returns:
        dict: Metadata if found, None if not found
    """
    try:
        if not session_id or not isinstance(session_id, str):
            logger.warning("Invalid session_id provided")
            return {}

        clean_session_id = session_id.replace('..', '').replace('/', '').replace('\\', '')
        if clean_session_id != session_id:
            logger.warning("Invalid characters in session_id")
            return {}

        upload_dir = os.path.join(UPLOAD_BASE_DIR, clean_session_id)

        upload_dir = os.path.abspath(upload_dir)
        base_dir = os.path.abspath(UPLOAD_BASE_DIR)
        if not upload_dir.startswith(base_dir):
            logger.warning("Path traversal attempt detected")
            return {}

        if not os.path.exists(upload_dir):
            logger.debug(f"Upload directory does not exist: {upload_dir}")
            return {}

        session_metadata_path = os.path.join(upload_dir, 'session_metadata.json')
        if not os.path.exists(session_metadata_path):
            logger.debug(f"Session metadata file not found: {session_metadata_path}")
            return {}

        with open(session_metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        logger.debug(f"Session metadata loaded for: {session_id}")
        return metadata

    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        logger.error(f"Error parsing session metadata for {session_id}: {e}")
        return {}
    except Exception as e:
        logger.error(f"Error retrieving session metadata for {session_id}: {e}")
        return {}

def print_session_files(session_id, files_data):
    """
    Print information about files in a session for debugging purposes.
    
    Args:
        session_id: ID of the session
        files_data: Dictionary of file names and their data
    """
    try:
        
        if not logger.isEnabledFor(logging.DEBUG):
            return
            
        metadata = get_session_metadata_locally(session_id)
                
        time_info = metadata.get('timeInfo', {})
        if time_info:
            pass
        
        for file_name, file_data in files_data.items():
            
            try:
                df = pd.read_csv(BytesIO(file_data), encoding='utf-8')
            except Exception as e:
                pass
                
            try:
                file_type = magic.from_buffer(file_data[:1024], mime=True)
            except ImportError:
                _, ext = os.path.splitext(file_name)
                    
            try:
                preview = file_data.decode('utf-8')[:1000]
            except UnicodeDecodeError:
                pass
                
    except Exception as e:
        logger.error(f"Error processing session files: {str(e)}")

@bp.route('/upload-chunk', methods=['POST'])
@require_auth
@require_subscription
@check_processing_limit
def upload_chunk():
    """
    Handle chunk upload from frontend - saving locally.
    
    Refactored: Business logic moved to upload_manager.process_chunk_upload()
    """
    try:
        if 'chunk' not in request.files:
            return jsonify({'success': False, 'error': 'No chunk in request'}), 400
        
        chunk_file = request.files['chunk']
        if not chunk_file.filename:
            return jsonify({'success': False, 'error': 'No chunk file selected'}), 400
        
        if 'metadata' not in request.form:
            return jsonify({'success': False, 'error': 'No metadata provided'}), 400
        
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
            from flask import g
            increment_processing_count(g.user_id)
            logger.info(f"Tracked CSV upload for user {g.user_id}")
        
        return jsonify({
            'success': True,
            'message': result['message']
        })
        
    except ValueError as e:
        logger.error(f"Validation error in chunk upload: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 400
        
    except Exception as e:
        logger.error(f"Error processing chunk upload: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

def update_session_metadata(session_id, data):
    """
    Update session metadata with finalization info from request data.
    
    Args:
        session_id: ID of the session
        data: Request data containing timeInfo and zeitschritte
        
    Returns:
        dict: Updated metadata
    """
    existing_metadata = get_session_metadata_locally(session_id)
    
    finalization_metadata = {
        **existing_metadata,
        'finalized': True,
        'finalizationTime': datetime.now().isoformat(),
        'timeInfo': data.get('timeInfo', existing_metadata.get('timeInfo', {})),
        'zeitschritte': data.get('zeitschritte', existing_metadata.get('zeitschritte', {}))
    }
    
    save_session_metadata_locally(session_id, finalization_metadata)
    return finalization_metadata

def verify_session_files(session_id, metadata):
    """
    Verify files in the session directory and update metadata accordingly.
    
    Args:
        session_id: ID of the session
        metadata: Session metadata
        
    Returns:
        tuple: (updated_metadata, file_count)
    """
    upload_dir = os.path.join(UPLOAD_BASE_DIR, session_id)
    file_count = 0
    files_metadata = metadata.get('files', [])
    
    if not files_metadata:
        for file_name in os.listdir(upload_dir):
            if os.path.isfile(os.path.join(upload_dir, file_name)) and not file_name.endswith(('_metadata.json', '.json')) and '_' not in file_name:
                file_count += 1
                files_metadata.append({
                    'fileName': file_name,
                    'createdAt': datetime.now().isoformat()
                })
    else:
        for file_info in files_metadata:
            filename = file_info.get('fileName')
            if filename:
                file_path = os.path.join(upload_dir, filename)
                if os.path.exists(file_path):
                    file_count += 1
    
    metadata['files'] = files_metadata
    
    save_session_metadata_locally(session_id, metadata)
    
    return metadata, file_count

def save_session_to_database(session_id, n_dat=None, file_count=None):
    """
    Save session data to Supabase database.

    Args:
        session_id: ID of the session
        n_dat: Total number of data samples (optional)
        file_count: Number of files in the session (optional)

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        supabase_result = save_session_to_supabase(session_id, n_dat, file_count)
        if supabase_result:
            return True
        else:
            logger.warning(f"Failed to save session {session_id} data to Supabase")
            return False
    except Exception as e:
        logger.error(f"Error saving session data to Supabase: {str(e)}")
        return False

@bp.route('/finalize-session', methods=['POST'])
def finalize_session_endpoint():
    """
    Refactored: Business logic moved to session_manager.finalize_session()
    
    Finalize a session after all files have been uploaded.
    """
    try:
        data = request.json
        if not data or 'sessionId' not in data:
            return jsonify({'success': False, 'error': 'No session ID provided'}), 400
            
        session_id = data['sessionId']
        
        result = finalize_session(session_id, data)
        
        return jsonify({
            'success': True,
            'message': result['message'],
            'sessionId': result['session_id'],
            'n_dat': result['n_dat'],
            'file_count': result['file_count']
        })
        
    except ValueError as e:
        return jsonify({'success': False, 'error': str(e)}), 400
    except Exception as e:
        logger.error(f"Error finalizing session: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500



@bp.route('/list-sessions', methods=['GET'])
def list_sessions():
    """
    List all available training sessions from Supabase database.
    
    Refactored: Business logic moved to session_manager.get_sessions_list()
    """
    try:
        limit = request.args.get('limit', 50, type=int)
        
        sessions = get_sessions_list(limit=limit)
        
        return jsonify({
            'success': True,
            'sessions': sessions,
            'total_count': len(sessions)
        })
        
    except Exception as e:
        logger.error(f"Error listing sessions: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'Failed to retrieve sessions from database'
        }), 500

@bp.route('/session/<session_id>', methods=['GET'])
def get_session_endpoint(session_id):
    """
    Refactored: Business logic moved to session_manager.get_session_info()
    
    Get detailed information about a specific session from local storage.
    """
    try:
        if not session_id:
            return jsonify({'success': False, 'error': 'No session ID provided'}), 400

        try:
            import uuid
            uuid.UUID(session_id)
            string_session_id = get_string_id_from_uuid(session_id)
            if not string_session_id:
                return jsonify({'success': False, 'error': 'Session mapping not found for UUID'}), 404
        except (ValueError, TypeError):
            string_session_id = session_id

        session_metadata = get_session_info(string_session_id)
        
        upload_dir = os.path.join(UPLOAD_BASE_DIR, string_session_id)
        files = []
        
        if os.path.exists(upload_dir):
            for file_name in os.listdir(upload_dir):
                if os.path.isfile(os.path.join(upload_dir, file_name)) and not file_name.endswith('.json'):
                    file_path = os.path.join(upload_dir, file_name)
                    file_size = os.path.getsize(file_path)
                    files.append({
                        'fileName': file_name,
                        'size': file_size,
                        'createdAt': datetime.fromtimestamp(os.path.getctime(file_path)).isoformat()
                    })
        
        session_info = {
            'sessionId': string_session_id,
            'files': files,
            'timeInfo': session_metadata.get('timeInfo', {}),
            'zeitschritte': session_metadata.get('zeitschritte', {}),
            'finalized': session_metadata.get('finalized', False),
            'n_dat': session_metadata.get('n_dat', 0),
            'file_count': len(files),
            'createdAt': datetime.fromtimestamp(os.path.getctime(upload_dir)).isoformat() if os.path.exists(upload_dir) else None,
            'lastUpdated': datetime.fromtimestamp(os.path.getmtime(upload_dir)).isoformat() if os.path.exists(upload_dir) else None
        }
        
        return jsonify({
            'success': True,
            'session': session_info
        })
        
    except FileNotFoundError as e:
        return jsonify({'success': False, 'error': str(e)}), 404
    except Exception as e:
        logger.error(f"Error getting session {session_id}: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500



@bp.route('/session/<session_id>/database', methods=['GET'])
def get_session_from_database_endpoint(session_id):
    """
    Get detailed information about a specific session from Supabase database.
    NOTE: This endpoint aggregates data from multiple tables (sessions, files, time_info, zeitschritte)
    """
    try:
        if not session_id:
            return jsonify({'success': False, 'error': 'No session ID provided'}), 400

        try:
            import uuid
            uuid.UUID(session_id)
            string_session_id = get_string_id_from_uuid(session_id)
            if not string_session_id:
                return jsonify({'success': False, 'error': 'Session mapping not found for UUID'}), 404
            database_session_id = session_id
        except (ValueError, TypeError):
            string_session_id = session_id
            database_session_id = create_or_get_session_uuid(session_id)
            if not database_session_id:
                return jsonify({'success': False, 'error': 'Could not find or create database session'}), 404

        supabase = get_supabase_client()
        if not supabase:
            return jsonify({'success': False, 'error': 'Database connection not available'}), 500

        session_response = supabase.table('sessions').select('*').eq('id', database_session_id).execute()
        if not session_response.data:
            return jsonify({'success': False, 'error': 'Session not found in database'}), 404

        session_data = session_response.data[0]
        files_response = supabase.table('files').select('*').eq('session_id', database_session_id).execute()
        time_info_response = supabase.table('time_info').select('*').eq('session_id', database_session_id).execute()
        zeitschritte_response = supabase.table('zeitschritte').select('*').eq('session_id', database_session_id).execute()

        session_info = {
            'sessionId': string_session_id,
            'databaseSessionId': database_session_id,
            'n_dat': session_data.get('n_dat', 0),
            'finalized': session_data.get('finalized', False),
            'file_count': session_data.get('file_count', 0),
            'files': [{'id': f['id'], 'fileName': f['file_name'], 'bezeichnung': f['bezeichnung'], 
                      'min': f['min'], 'max': f['max'], 'datenpunkte': f['datenpunkte'], 'type': f['type']} 
                     for f in (files_response.data or [])],
            'timeInfo': {k: (time_info_response.data[0] if time_info_response.data else {}).get(k, False if k != 'zeitzone' else 'UTC') 
                        for k in ['jahr', 'monat', 'woche', 'tag', 'feiertag', 'zeitzone', 'category_data']},
            'zeitschritte': {k: (zeitschritte_response.data[0] if zeitschritte_response.data else {}).get(k, '') 
                            for k in ['eingabe', 'ausgabe', 'zeitschrittweite', 'offset']},
            'createdAt': session_data.get('created_at'),
            'updatedAt': session_data.get('updated_at')
        }
        
        return jsonify({'success': True, 'session': session_info})
        
    except Exception as e:
        logger.error(f"Error getting session from database {session_id}: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500



@bp.route('/session-status/<session_id>', methods=['GET'])
def session_status(session_id):
    """
    Get the upload status of a session.
    
    Refactored: Business logic moved to session_manager.get_upload_status()
    """
    try:
        if not session_id:
            return jsonify({
                'status': 'error',
                'progress': 0,
                'message': 'Missing session ID'
            }), 400
        
        status_info = get_upload_status(session_id)
        
        return jsonify(status_info)
        
    except ValueError as e:
        return jsonify({
            'status': 'error',
            'progress': 0,
            'message': str(e)
        }), 404
        
    except Exception as e:
        logger.error(f"Error getting session status for {session_id}: {str(e)}")
        return jsonify({
            'status': 'error',
            'progress': 0,
            'message': f'Error: {str(e)}'
        }), 500

@bp.route('/init-session', methods=['POST'])
def init_session():
    """
    Initialize a new upload session.

    Refactored: Business logic moved to session_manager.initialize_session()
    """
    try:
        data = request.json
        session_id = data.get('sessionId')
        time_info = data.get('timeInfo', {})
        zeitschritte = data.get('zeitschritte', {})

        result = initialize_session(session_id, time_info, zeitschritte)

        return jsonify({
            'success': True,
            'sessionId': result['session_id'],
            'message': result['message']
        })

    except ValueError as e:
        return jsonify({'success': False, 'error': str(e)}), 400
    except Exception as e:
        logger.error(f"Error initializing session: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@bp.route('/save-time-info', methods=['POST'])
def save_time_info_endpoint():
    """
    Save time information via API endpoint.

    Refactored: Business logic moved to session_manager.save_time_info_data()
    """
    try:
        if not request.is_json:
            logger.error("Request is not JSON")
            return jsonify({'success': False, 'error': 'Request must be JSON'}), 400

        try:
            data = request.get_json(force=True)
        except Exception as e:
            logger.error(f"Failed to parse JSON: {str(e)}")
            return jsonify({'success': False, 'error': f'Invalid JSON: {str(e)}'}), 400

        if not data or 'sessionId' not in data or 'timeInfo' not in data:
            logger.error("Missing required fields in request")
            return jsonify({'success': False, 'error': 'Missing sessionId or timeInfo'}), 400

        session_id = data['sessionId']
        time_info = data['timeInfo']

        save_time_info_data(session_id, time_info)

        return jsonify({'success': True, 'message': 'Time info saved successfully'})

    except ValueError as e:
        return jsonify({'success': False, 'error': str(e)}), 400
    except Exception as e:
        logger.error(f"Error saving time info: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@bp.route('/create-database-session', methods=['POST'])
def create_database_session_endpoint():
    """
    Create a new session in Supabase database and return UUID.

    Refactored: Business logic moved to session_manager.create_database_session()
    """
    try:
        data = request.json
        session_id = data.get('sessionId') if data else None
        session_name = data.get('sessionName') if data else None

        session_uuid = create_database_session(session_id, session_name)

        return jsonify({
            'success': True,
            'sessionUuid': session_uuid,
            'message': f'Database session created with UUID: {session_uuid}'
        })

    except ValueError as e:
        return jsonify({'success': False, 'error': str(e)}), 400
    except Exception as e:
        logger.error(f"Error creating database session: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@bp.route('/get-session-uuid/<session_id>', methods=['GET'])
def get_session_uuid_endpoint(session_id):
    """
    Get the UUID session ID for a string session ID.

    Refactored: Business logic moved to session_manager.get_session_uuid()
    """
    try:
        try:
            import uuid
            uuid.UUID(session_id)
            return jsonify({
                'success': True,
                'sessionUuid': session_id,
                'message': 'Session ID is already in UUID format'
            })
        except (ValueError, TypeError):
            session_uuid = get_session_uuid(session_id)

            return jsonify({
                'success': True,
                'sessionUuid': session_uuid,
                'message': f'Found/created UUID for session: {session_uuid}'
            })

    except ValueError as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 404
    except Exception as e:
        logger.error(f"Error getting session UUID: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@bp.route('/save-zeitschritte', methods=['POST'])
def save_zeitschritte_endpoint():
    """
    Save zeitschritte information via API endpoint.

    Refactored: Business logic moved to session_manager.save_zeitschritte_data()
    """
    try:
        if not request.is_json:
            logger.error("Request is not JSON")
            return jsonify({'success': False, 'error': 'Request must be JSON'}), 400

        try:
            data = request.get_json(force=True)
        except Exception as e:
            logger.error(f"Failed to parse JSON: {str(e)}")
            return jsonify({'success': False, 'error': f'Invalid JSON: {str(e)}'}), 400

        if not data or 'sessionId' not in data or 'zeitschritte' not in data:
            logger.error("Missing required fields in request")
            return jsonify({'success': False, 'error': 'Missing sessionId or zeitschritte'}), 400

        session_id = data['sessionId']
        zeitschritte = data['zeitschritte']

        save_zeitschritte_data(session_id, zeitschritte)

        return jsonify({'success': True, 'message': 'Zeitschritte saved successfully'})

    except ValueError as e:
        return jsonify({'success': False, 'error': str(e)}), 400
    except Exception as e:
        logger.error(f"Error saving zeitschritte: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@bp.route('/session/<session_id>/delete', methods=['POST'])
def delete_session_endpoint(session_id):
    """
    Refactored: Business logic moved to session_manager.delete_session()
    
    Delete a specific session and all its files from local storage and Supabase database.
    """
    try:
        result = delete_session(session_id)
        
        if result.get('warnings'):
            return jsonify({
                'success': True,
                'message': result['message'],
                'warnings': result['warnings']
            })
        else:
            return jsonify({
                'success': True,
                'message': result['message']
            })
        
    except ValueError as e:
        return jsonify({'success': False, 'error': str(e)}), 404
    except Exception as e:
        logger.error(f"Error deleting session {session_id}: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500



@bp.route('/get-zeitschritte/<session_id>', methods=['GET'])
def get_zeitschritte_endpoint(session_id):
    """
    Refactored: Business logic moved to session_manager.get_zeitschritte_data()
    
    Get zeitschritte data for a session.
    """
    try:
        if not validate_session_id(session_id):
            return create_error_response('Invalid session ID format', 400)
        
        zeitschritte = get_zeitschritte_data(session_id)
        return create_success_response(zeitschritte)
        
    except ValueError as e:
        return create_error_response(str(e), 404)
    except Exception as e:
        logger.error(f"Error getting zeitschritte for {session_id}: {str(e)}")
        return create_error_response(f'Internal server error: {str(e)}', 500)



@bp.route('/get-time-info/<session_id>', methods=['GET'])
def get_time_info_endpoint(session_id):
    """
    Refactored: Business logic moved to session_manager.get_time_info_data()
    
    Get time info data for a session.
    """
    try:
        time_info = get_time_info_data(session_id)
        return jsonify({
            'success': True,
            'data': time_info
        })
        
    except ValueError as e:
        return jsonify({
            'success': False,
            'data': None,
            'message': str(e)
        }), 404
    except Exception as e:
        logger.error(f"Error getting time info for {session_id}: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500




@bp.route('/csv-files/<session_id>', methods=['GET'])
def get_csv_files_endpoint(session_id):
    """
    Refactored: Business logic moved to session_manager.get_csv_files_for_session()
    
    Get all CSV files for a session.
    """
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
        return jsonify({'success': False, 'error': str(e)}), 400
    except Exception as e:
        logger.error(f"Error getting CSV files for session {session_id}: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500



@bp.route('/csv-files', methods=['POST'])
@require_auth
@require_subscription
@check_processing_limit
def create_csv_file():
    """Create a new CSV file entry."""
    try:
        from utils.database import save_file_info, save_csv_file_content
        
        if 'file' in request.files:
            file = request.files['file']
            session_id = request.form.get('sessionId')
            file_data = request.form.to_dict()
        else:
            data = request.get_json()
            if not data:
                return jsonify({'success': False, 'error': 'No data provided'}), 400
            
            session_id = data.get('sessionId')
            file_data = data.get('fileData', {})
            file = None
        
        if not session_id:
            return jsonify({'success': False, 'error': 'Session ID is required'}), 400
            
        success, file_uuid = save_file_info(session_id, file_data)
        if not success:
            return jsonify({'success': False, 'error': 'Failed to save file metadata'}), 500
            
        if file and file_uuid:
            import tempfile
            import os
            
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                file.save(temp_file.name)
                temp_path = temp_file.name
            
            try:
                file_type = file_data.get('type', 'input')
                storage_success = save_csv_file_content(
                    file_uuid, session_id, file.filename, temp_path, file_type
                )
                
                if not storage_success:
                    logger.warning(f"Failed to upload file to storage for file {file_uuid}")
                    
            finally:
                try:
                    os.unlink(temp_path)
                except:
                    pass

        from flask import g
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
def update_csv_file(file_id):
    """
    Update CSV file metadata.
    
    Refactored: Business logic moved to upload_manager.update_csv_file_record()
    """
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
def delete_csv_file(file_id):
    """
    Delete CSV file from database and storage.
    
    Refactored: Business logic moved to upload_manager.delete_csv_file_record()
    """
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

@bp.route('/results/<session_id>', methods=['GET'])
def get_training_results(session_id):
    """
    Get training results for a session.
    Checks both UUID and string session IDs.
    NEW: Downloads full results from Storage bucket if available.
    """
    try:
        from utils.database import get_supabase_client, create_or_get_session_uuid
        from utils.training_storage import download_training_results
        supabase = get_supabase_client()

        uuid_session_id = create_or_get_session_uuid(session_id)

        logger.info(f"Getting training results for session {session_id} (UUID: {uuid_session_id})")

        response = supabase.table('training_results')\
            .select('id, session_id, status, created_at, updated_at, '
                   'results_file_path, file_size_bytes, compressed, results_metadata')\
            .eq('session_id', uuid_session_id)\
            .order('created_at.desc')\
            .limit(1)\
            .execute()

        if response.data and len(response.data) > 0:
            record = response.data[0]

            if record.get('results_file_path'):
                try:
                    logger.info(f"📥 Downloading full results from storage: {record['results_file_path']}")
                    full_results = download_training_results(
                        file_path=record['results_file_path'],
                        decompress=record.get('compressed', False)
                    )
                    record['results'] = full_results
                    logger.info(f"✅ Full results loaded from storage successfully")
                except Exception as download_error:
                    logger.error(f"Failed to download results from storage: {download_error}")
                    record['results'] = record.get('results_metadata', {})
                    logger.warning("Returning metadata only (full results unavailable)")
            else:
                logger.info("No storage file path - checking for legacy JSONB results")
                legacy_response = supabase.table('training_results')\
                    .select('results')\
                    .eq('id', record['id'])\
                    .single()\
                    .execute()
                if legacy_response.data and legacy_response.data.get('results'):
                    record['results'] = legacy_response.data['results']
                    logger.info("Loaded results from legacy JSONB column")
                else:
                    record['results'] = record.get('results_metadata', {})
                    logger.warning("No results available in storage or JSONB")

            return jsonify({
                'success': True,
                'results': [record],
                'count': 1
            })
        else:
            logger.info(f"No training results found for session {session_id}")
            return jsonify({
                'success': True,
                'message': 'No training results yet - training may not have been started',
                'results': [],
                'count': 0
            }), 200

    except Exception as e:
        logger.error(f"Error getting training results for {session_id}: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@bp.route('/get-training-results/<session_id>', methods=['GET'])
def get_training_results_details(session_id):
    """
    Get detailed training results for a session (used by frontend)
    This is an alias for /results/<session_id> to maintain compatibility
    """
    return get_training_results(session_id)


@bp.route('/plot-variables/<session_id>', methods=['GET'])
def get_plot_variables(session_id):
    """
    Get available input and output variables for plotting
    Refactored: business logic moved to Visualizer.get_available_variables()
    """
    try:
        visualizer = Visualizer()
        variables = visualizer.get_available_variables(session_id)

        return jsonify({
            'success': True,
            'session_id': session_id,
            'input_variables': variables['input_variables'],
            'output_variables': variables['output_variables']
        })

    except Exception as e:
        logger.error(f"Error getting plot variables for {session_id}: {str(e)}")
        return jsonify({
            'success': True,
            'session_id': session_id,
            'input_variables': ['Temperature', 'Load'],
            'output_variables': ['Predicted_Load']
        })




@bp.route('/visualizations/<session_id>', methods=['GET'])
def get_training_visualizations(session_id):
    """
    Get training visualizations (violin plots) for a session
    Refactored: business logic moved to Visualizer.get_session_visualizations()
    """
    try:
        visualizer = Visualizer()
        viz_data = visualizer.get_session_visualizations(session_id)

        if not viz_data.get('plots'):
            return jsonify({
                'session_id': session_id,
                'plots': {},
                'message': viz_data.get('message', 'No visualizations found for this session')
            }), 404

        return jsonify({
            'session_id': session_id,
            'plots': viz_data['plots'],
            'metadata': viz_data['metadata'],
            'created_at': viz_data['created_at'],
            'message': viz_data['message']
        })

    except Exception as e:
        logger.error(f"Error retrieving visualizations for {session_id}: {str(e)}")
        return create_error_response(f'Failed to retrieve training visualizations: {str(e)}', 500)




@bp.route('/generate-plot', methods=['POST'])
def generate_plot():
    """
    Generate plot based on user selections
    Refactored: business logic moved to Visualizer.generate_custom_plot()

    Expected request body:
    {
        'session_id': str,
        'plot_settings': {
            'num_sbpl': int,
            'x_sbpl': str ('UTC' or 'ts'),
            'y_sbpl_fmt': str ('original' or 'skaliert'),
            'y_sbpl_set': str ('gemeinsame Achse' or 'separate Achsen')
        },
        'df_plot_in': dict,  # {feature_name: bool, ...}
        'df_plot_out': dict,  # {feature_name: bool, ...}
        'df_plot_fcst': dict  # {feature_name: bool, ...}
    }
    """
    try:
        data = request.json
        session_id = data.get('session_id')

        if not session_id:
            return create_error_response('Session ID is required', 400)

        plot_settings = data.get('plot_settings', {})
        df_plot_in = data.get('df_plot_in', {})
        df_plot_out = data.get('df_plot_out', {})
        df_plot_fcst = data.get('df_plot_fcst', {})
        model_id = data.get('model_id')

        visualizer = Visualizer()
        result = visualizer.generate_custom_plot(
            session_id=session_id,
            plot_settings=plot_settings,
            df_plot_in=df_plot_in,
            df_plot_out=df_plot_out,
            df_plot_fcst=df_plot_fcst,
            model_id=model_id
        )

        return jsonify({
            'success': True,
            'session_id': session_id,
            'plot_data': result['plot_data'],
            'message': result['message']
        })

    except ValueError as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400
    except Exception as e:
        logger.error(f"Error in generate_plot endpoint: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return create_error_response(f'Failed to generate plot: {str(e)}', 500)




@bp.route('/status/<session_id>', methods=['GET'])
@bp.route('/session-status/<session_id>', methods=['GET'])
def get_training_status(session_id: str):
    """
    Get training status for a session

    Args:
        session_id: Session identifier

    Returns:
        JSON response with training status
    """
    try:
        from utils.database import get_supabase_client, create_or_get_session_uuid
        supabase = get_supabase_client()
        uuid_session_id = create_or_get_session_uuid(session_id)

        results_response = supabase.table('training_results').select('*').eq('session_id', uuid_session_id).order('created_at.desc').limit(1).execute()

        logs_response = supabase.table('training_logs').select('*').eq('session_id', uuid_session_id).order('created_at.desc').limit(1).execute()

        if results_response.data and len(results_response.data) > 0:
            result_data = results_response.data[0]
            status = {
                'session_id': session_id,
                'status': result_data.get('status', 'completed'),
                'progress': 100,
                'current_step': 'Training completed',
                'total_steps': 7,
                'completed_steps': 7,
                'started_at': result_data.get('created_at'),
                'completed_at': result_data.get('completed_at'),
                'message': 'Training completed successfully'
            }
        elif logs_response.data and len(logs_response.data) > 0:
            log_data = logs_response.data[0]
            progress_data = log_data.get('progress', {})
            status = {
                'session_id': session_id,
                'status': 'in_progress',
                'progress': progress_data.get('overall', 0) if isinstance(progress_data, dict) else 0,
                'current_step': progress_data.get('current_step', 'Processing') if isinstance(progress_data, dict) else 'Processing',
                'total_steps': progress_data.get('total_steps', 7) if isinstance(progress_data, dict) else 7,
                'completed_steps': progress_data.get('completed_steps', 0) if isinstance(progress_data, dict) else 0,
                'started_at': log_data.get('created_at'),
                'completed_at': None,
                'message': 'Training in progress'
            }
        else:
            status = {
                'session_id': session_id,
                'status': 'not_found',
                'progress': 0,
                'current_step': 'Not started',
                'total_steps': 7,
                'completed_steps': 0,
                'started_at': None,
                'completed_at': None,
                'message': 'No training found for this session'
            }

        return jsonify(status)

    except Exception as e:
        logger.error(f"Error getting training status for {session_id}: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Failed to get training status',
            'message': str(e),
            'session_id': session_id,
            'status': 'error'
        }), 500


@bp.route('/generate-datasets/<session_id>', methods=['POST'])
@require_auth
@require_subscription
@check_processing_limit
def generate_datasets(session_id):
    """
    Generate datasets and violin plots WITHOUT training models.
    This is phase 1 of the training workflow - data visualization only.
    
    Refactored: Business logic moved to dataset_generator.generate_violin_plots_for_session()
    """
    try:
        data = request.json
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400
        
        model_parameters = data.get('model_parameters', {})
        training_split = data.get('training_split', {})
        
        from services.training.dataset_generator import generate_violin_plots_for_session
        
        result = generate_violin_plots_for_session(
            session_id=session_id,
            model_parameters=model_parameters,
            training_split=training_split
        )
        
        violin_plots = result.get('violin_plots', {})
        if violin_plots:
            from services.training.training_api import save_visualization_to_database
            
            for plot_name, plot_data in violin_plots.items():
                try:
                    if plot_data:
                        save_visualization_to_database(session_id, plot_name, plot_data)
                except Exception as viz_error:
                    logger.error(f"Failed to save visualization {plot_name}: {str(viz_error)}")
        
        from flask import g
        increment_processing_count(g.user_id)
        logger.info(f"Tracked dataset generation for user {g.user_id}")
        
        return jsonify({
            'success': True,
            'message': 'Datasets generated successfully',
            'dataset_count': 10,
            'violin_plots': violin_plots
        })
        
    except ValueError as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400
        
    except Exception as e:
        logger.error(f"Error in generate_datasets: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@bp.route('/train-models/<session_id>', methods=['POST'])
@require_auth
@require_subscription
@check_training_limit
def train_models(session_id):
    """
    Train models with user-specified parameters.
    This is phase 2 of the training workflow.
    
    Refactored: Business logic moved to training_orchestrator.run_model_training_async()
    """
    try:
        data = request.json
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400
        
        model_parameters = data.get('model_parameters', {})
        training_split = data.get('training_split', {})
        
        from flask import current_app
        socketio_instance = current_app.extensions.get('socketio')
        
        actf = model_parameters.get('ACTF')
        if actf:
            actf = actf.lower()
        
        model_config = {
            'MODE': model_parameters.get('MODE', 'Linear'),
            'LAY': model_parameters.get('LAY'),
            'N': model_parameters.get('N'),
            'EP': model_parameters.get('EP'),
            'ACTF': actf,
            'K': model_parameters.get('K'),
            'KERNEL': model_parameters.get('KERNEL'),
            'C': model_parameters.get('C'),
            'EPSILON': model_parameters.get('EPSILON'),
            'random_dat': not training_split.get('shuffle', True)
        }
        
        from flask import g
        increment_training_count(g.user_id)
        logger.info(f"Tracked training run for user {g.user_id}")
        
        import threading
        from services.training.training_orchestrator import run_model_training_async
        
        training_thread = threading.Thread(
            target=run_model_training_async,
            args=(session_id, model_config, training_split, socketio_instance)
        )
        training_thread.daemon = True
        training_thread.start()
        
        return jsonify({
            'success': True,
            'message': f'Model training started for session {session_id}',
            'note': 'Training is running in background, listen for SocketIO events for progress'
        })
        
    except Exception as e:
        logger.error(f"Error in train_models: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@bp.route('/delete-all-sessions', methods=['POST'])
def delete_all_sessions_endpoint():
    """
    Refactored: Business logic moved to session_manager.delete_all_sessions()
    
    Delete ALL sessions and associated data from Supabase database and local storage.
    ⚠️ WARNING: This will permanently delete all data! Use with extreme caution.
    """
    try:
        data = {}
        try:
            if request.is_json:
                data = request.get_json() or {}
            elif request.content_type and 'application/json' in request.content_type:
                data = request.get_json(force=True) or {}
            elif hasattr(request, 'json') and request.json:
                data = request.json
            else:
                data = request.form.to_dict()
                if 'confirm_delete_all' in data:
                    data['confirm_delete_all'] = data['confirm_delete_all'].lower() in ['true', '1', 'yes']
        except Exception as parse_error:
            logger.warning(f"Could not parse request data: {str(parse_error)}")
            try:
                import json
                raw_data = request.get_data(as_text=True)
                if raw_data:
                    data = json.loads(raw_data)
            except:
                data = {}

        confirmation = data.get('confirm_delete_all', False)
        logger.info(f"Parsed confirmation: {confirmation} from data: {data}")

        result = delete_all_sessions(confirm=confirmation)

        response_data = {
            'success': True,
            'message': result['message'],
            'summary': result['summary'],
            'details': result['details']
        }

        if result.get('warnings'):
            response_data['warnings'] = result['warnings']

        return jsonify(response_data)

    except ValueError as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'To delete all sessions, send {"confirm_delete_all": true} in request body'
        }), 400
    except Exception as e:
        logger.error(f"Critical error during delete all sessions: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'Critical error occurred during deletion operation'
        }), 500





@bp.route('/evaluation-tables/<session_id>', methods=['GET'])
def get_evaluation_tables(session_id):
    """
    Get evaluation metrics formatted as tables for display.
    Returns df_eval and df_eval_ts structures matching the original training output.
    """
    try:
        from utils.database import get_supabase_client, create_or_get_session_uuid
        from utils.training_storage import download_training_results
        supabase = get_supabase_client()
        uuid_session_id = create_or_get_session_uuid(session_id)

        response = supabase.table('training_results') \
            .select('id, results_file_path, compressed, results') \
            .eq('session_id', uuid_session_id) \
            .order('created_at.desc') \
            .limit(1) \
            .execute()

        if not response.data:
            return jsonify({
                'success': False,
                'error': 'No training results found for this session',
                'session_id': session_id
            }), 404

        record = response.data[0]

        if record.get('results_file_path'):
            logger.info(f"📥 Downloading evaluation data from storage: {record['results_file_path']}")
            results = download_training_results(
                file_path=record['results_file_path'],
                decompress=record.get('compressed', False)
            )
        else:
            logger.info(f"Using legacy results column for evaluation data")
            results = record.get('results')

        eval_metrics = results.get('evaluation_metrics', {})
        if not eval_metrics or eval_metrics.get('error'):
            eval_metrics = results.get('metrics', {})

        logger.info(f"Found eval_metrics keys: {eval_metrics.keys() if eval_metrics else 'None'}")

        if eval_metrics and eval_metrics.get('test_metrics_scaled'):
            pass
        elif not eval_metrics or (eval_metrics.get('error') and not eval_metrics.get('test_metrics_scaled')):
            return jsonify({
                'success': False,
                'error': f"No valid evaluation metrics found. Metrics: {eval_metrics}",
                'session_id': session_id
            }), 404


        output_features = results.get('output_features', ['Netzlast [kW]'])
        if not output_features:
            output_features = ['Netzlast [kW]']

        df_eval = {}

        df_eval_ts = {}

        time_deltas = [15, 30, 45, 60, 90, 120, 180, 240, 300, 360, 420, 480]

        for feature_name in output_features:
            delt_list = []
            mae_list = []
            mape_list = []
            mse_list = []
            rmse_list = []
            nrmse_list = []
            wape_list = []
            smape_list = []
            mase_list = []

            df_eval_ts[feature_name] = {}

            test_metrics = eval_metrics.get('test_metrics_scaled', {})
            val_metrics = eval_metrics.get('val_metrics_scaled', {})

            for delta in time_deltas:
                delt_list.append(float(delta))

                mae_list.append(float(test_metrics.get('MAE', 0.0)))
                mape_list.append(float(test_metrics.get('MAPE', 0.0)))
                mse_list.append(float(test_metrics.get('MSE', 0.0)))
                rmse_list.append(float(test_metrics.get('RMSE', 0.0)))
                nrmse_list.append(float(test_metrics.get('NRMSE', 0.0)))
                wape_list.append(float(test_metrics.get('WAPE', 0.0)))
                smape_list.append(float(test_metrics.get('sMAPE', 0.0)))
                mase_list.append(float(test_metrics.get('MASE', 0.0)))

                timestep_metrics = []
                n_timesteps = results.get('n_timesteps', 96)

                for ts in range(n_timesteps):
                    timestep_metrics.append({
                        'MAE': float(test_metrics.get('MAE', 0.0)),
                        'MAPE': float(test_metrics.get('MAPE', 0.0)),
                        'MSE': float(test_metrics.get('MSE', 0.0)),
                        'RMSE': float(test_metrics.get('RMSE', 0.0)),
                        'NRMSE': float(test_metrics.get('NRMSE', 0.0)),
                        'WAPE': float(test_metrics.get('WAPE', 0.0)),
                        'sMAPE': float(test_metrics.get('sMAPE', 0.0)),
                        'MASE': float(test_metrics.get('MASE', 0.0))
                    })

                df_eval_ts[feature_name][float(delta)] = timestep_metrics

            df_eval[feature_name] = {
                "delta [min]": delt_list,
                "MAE": mae_list,
                "MAPE": mape_list,
                "MSE": mse_list,
                "RMSE": rmse_list,
                "NRMSE": nrmse_list,
                "WAPE": wape_list,
                "sMAPE": smape_list,
                "MASE": mase_list
            }

        return jsonify({
            'success': True,
            'session_id': session_id,
            'df_eval': df_eval,
            'df_eval_ts': df_eval_ts,
            'model_type': eval_metrics.get('model_type', 'Unknown')
        })

    except Exception as e:
        logger.error(f"Error getting evaluation tables: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e),
            'session_id': session_id
        }), 500


@bp.route('/save-evaluation-tables/<session_id>', methods=['POST'])
def save_evaluation_tables(session_id):
    """
    Save evaluation tables (df_eval and df_eval_ts) to database
    """
    try:
        from utils.database import get_supabase_client, create_or_get_session_uuid
        from datetime import datetime

        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'error': 'No data provided'
            }), 400

        df_eval = data.get('df_eval', {})
        df_eval_ts = data.get('df_eval_ts', {})
        model_type = data.get('model_type', 'Unknown')

        if not df_eval and not df_eval_ts:
            return jsonify({
                'success': False,
                'error': 'No evaluation tables provided'
            }), 400

        supabase = get_supabase_client()
        uuid_session_id = create_or_get_session_uuid(session_id)

        evaluation_data = {
            'session_id': uuid_session_id,
            'df_eval': df_eval,
            'df_eval_ts': df_eval_ts,
            'model_type': model_type,
            'created_at': datetime.now().isoformat(),
            'table_format': 'original_training_format'
        }

        response = supabase.table('evaluation_tables').upsert(evaluation_data).execute()

        if response.data:
            return jsonify({
                'success': True,
                'session_id': session_id,
                'message': 'Evaluation tables saved successfully',
                'saved_at': evaluation_data['created_at']
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to save to database'
            }), 500

    except Exception as e:
        logger.error(f"Error saving evaluation tables: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e),
            'session_id': session_id
        }), 500



@bp.route('/scalers/<session_id>', methods=['GET'])
def get_scalers(session_id):
    """
    Retrieve saved scalers from database for a specific session.
    Refactored: business logic moved to scaler_manager.get_session_scalers()
    """
    try:
        scalers_data = get_session_scalers(session_id)

        return jsonify({
            'success': True,
            'session_id': session_id,
            'scalers': scalers_data
        })

    except ValueError as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 404
    except Exception as e:
        logger.error(f"Error retrieving scalers for session {session_id}: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'Failed to retrieve scalers from database'
        }), 500




@bp.route('/scalers/<session_id>/download', methods=['GET'])
def download_scalers_as_save_files(session_id):
    """
    Download scalers as .save files identical to original training_original.py format.
    Refactored: business logic moved to scaler_manager.create_scaler_download_package()
    """
    try:
        zip_file_path = create_scaler_download_package(session_id)

        from datetime import datetime
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        return send_file(
            zip_file_path,
            as_attachment=True,
            download_name=f'scalers_{session_id}_{timestamp}.zip',
            mimetype='application/zip'
        )

    except ValueError as e:
        return jsonify({'success': False, 'error': str(e)}), 404
    except Exception as e:
        logger.error(f"Error creating scaler download for session {session_id}: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500




@bp.route('/scale-data/<session_id>', methods=['POST'])
def scale_input_data(session_id):
    """
    Scale input data using saved scalers (Skalierung Eingabedaten speichern).
    Takes raw input data and returns scaled data ready for model prediction.

    Refactored: Business logic moved to scaler_manager.scale_new_data()
    """
    try:
        data = request.json
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400

        input_data = data.get('input_data')
        if input_data is None:
            return jsonify({'success': False, 'error': 'input_data field is required'}), 400

        save_scaled = data.get('save_scaled', False)

        result = scale_new_data(session_id, input_data, save_scaled)

        return jsonify({
            'success': True,
            'session_id': session_id,
            'scaled_data': result['scaled_data'],
            'scaling_info': result['scaling_info'],
            'metadata': result['metadata']
        })

    except ValueError as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 404
    except Exception as e:
        logger.error(f"Error scaling data for session {session_id}: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'Failed to scale input data'
        }), 500


def cleanup_incomplete_uploads(upload_base_dir: str, max_age_hours: int = 24) -> int:
    """
    Clean up incomplete or old upload sessions.

    Args:
        upload_base_dir: Base directory for uploads
        max_age_hours: Maximum age for incomplete uploads

    Returns:
        int: Number of cleaned up sessions
    """
    cleaned_count = 0
    current_time = time.time()
    max_age_seconds = max_age_hours * 3600

    try:
        for session_dir in Path(upload_base_dir).iterdir():
            if not session_dir.is_dir():
                continue

            dir_age = current_time - session_dir.stat().st_mtime
            if dir_age > max_age_seconds:
                finalized_marker = session_dir / '.finalized'
                if not finalized_marker.exists():
                    try:
                        shutil.rmtree(session_dir)
                        cleaned_count += 1
                        logger.info(f"Cleaned up old session: {session_dir.name}")
                    except Exception as e:
                        logger.error(f"Error cleaning up {session_dir}: {e}")

    except Exception as e:
        logger.error(f"Error during cleanup: {e}")

    return cleaned_count



@bp.route('/save-model/<session_id>', methods=['POST'])
def save_model(session_id):
    """
    Save trained models to Supabase Storage and return download information.
    Extracts serialized models from training results, deserializes them,
    saves as .h5 (Keras) or .pkl (sklearn) files, and uploads to persistent storage.

    Refactored: Business logic moved to model_manager.save_models_to_storage()

    Args:
        session_id: Training session ID

    Returns:
        JSON response with model information and upload status
    """
    try:
        result = save_models_to_storage(session_id)

        response = {
            'success': True,
            'message': f'Successfully saved {result["total_uploaded"]} model(s) to storage',
            'models': result['uploaded_models'],
            'total_uploaded': result['total_uploaded'],
            'session_id': session_id
        }

        if result['failed_models']:
            response['failed_models'] = result['failed_models']
            response['total_failed'] = result['total_failed']

        return jsonify(response)

    except ValueError as e:
        error_msg = str(e)
        if 'Session' in error_msg and 'not found' in error_msg:
            return jsonify({
                'success': False,
                'error': error_msg
            }), 404
        elif 'No training results' in error_msg:
            return jsonify({
                'success': False,
                'error': error_msg,
                'message': 'Train a model first before attempting to save.'
            }), 404
        elif 'No trained models' in error_msg:
            return jsonify({
                'success': False,
                'error': error_msg,
                'message': 'Training results exist but no models were saved.'
            }), 404
        else:
            return jsonify({
                'success': False,
                'error': error_msg
            }), 400

    except Exception as e:
        logger.error(f"❌ Error saving models: {e}")
        import traceback
        logger.error(traceback.format_exc())

        if 'All model uploads failed' in str(e):
            return jsonify({
                'success': False,
                'error': str(e),
                'failed_models': []
            }), 500

        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'Failed to save models to storage'
        }), 500


@bp.route('/list-models-database/<session_id>', methods=['GET'])
def list_models_database(session_id):
    """
    List all trained models stored in Supabase Storage for a session.

    Refactored: Business logic moved to model_manager.get_models_list()

    Args:
        session_id: Training session ID

    Returns:
        JSON response with list of models in Storage
    """
    try:
        models = get_models_list(session_id)

        return jsonify({
            'success': True,
            'data': {
                'models': models,
                'count': len(models)
            },
            'session_id': session_id
        })

    except ValueError as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 404

    except Exception as e:
        logger.error(f"❌ Error listing models from Storage: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'Failed to list models from Storage'
        }), 500


@bp.route('/download-model-h5/<session_id>', methods=['GET'])
def download_model_h5(session_id):
    """
    Download a trained model file from Supabase Storage.

    Refactored: Business logic moved to model_manager.download_model_file()

    Query Parameters:
        filename (optional): Specific model filename to download

    If no filename specified, downloads the first .h5 model found.

    Args:
        session_id: Training session ID

    Returns:
        Binary model file as attachment
    """
    try:
        from flask import request, send_file
        import io

        filename = request.args.get('filename')

        file_data, file_name = download_model_file(session_id, filename)

        file_obj = io.BytesIO(file_data)

        return send_file(
            file_obj,
            as_attachment=True,
            download_name=file_name,
            mimetype='application/octet-stream'
        )

    except ValueError as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 404

    except Exception as e:
        logger.error(f"❌ Error downloading model: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'Failed to download model from Storage'
        }), 500


@bp.route('/session-name-change', methods=['POST'])
def change_session_name_endpoint():
    """
    Refactored: Business logic moved to session_manager.update_session_name()
    
    Update session name in the database.
    
    Request body:
    {
        "sessionId": "...",
        "sessionName": "novo ime"
    }
    """
    try:
        data = request.get_json()
        if not data:
            return create_error_response('No data provided', 400)

        session_id = data.get('sessionId')
        session_name = data.get('sessionName')

        result = update_session_name(session_id, session_name)

        return create_success_response(
            data={
                'sessionId': result['session_id'],
                'sessionName': result['session_name']
            },
            message=result['message']
        )

    except ValueError as e:
        logger.warning(f"Validation error: {str(e)}")
        return create_error_response(str(e), 400)
    except Exception as e:
        logger.error(f"Error changing session name: {str(e)}")
        return create_error_response(f'Internal server error: {str(e)}', 500)

