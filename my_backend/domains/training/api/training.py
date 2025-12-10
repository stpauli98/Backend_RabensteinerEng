"""
Training API Routes
Flask Blueprint for training domain endpoints

Migrated from api/routes/training.py
Phase 8 of backend refactoring
"""

import os
import json
import logging
from datetime import datetime
from flask import Blueprint, request, jsonify, current_app, send_file, g

from shared.database.operations import (
    save_session_to_supabase, get_string_id_from_uuid,
    create_or_get_session_uuid, get_supabase_client
)
from shared.auth.jwt import require_auth
from shared.auth.subscription import require_subscription, check_processing_limit, check_training_limit
from shared.tracking.usage import increment_processing_count, increment_training_count
from utils.validation import validate_session_id, create_error_response, create_success_response

from domains.training.services.visualization import Visualizer, save_visualization_to_database
from domains.training.ml.scaler import get_session_scalers, create_scaler_download_package, scale_new_data
from domains.training.ml.models import save_models_to_storage, get_models_list, download_model_file

from domains.training.services.session import (
    initialize_session, finalize_session, get_sessions_list,
    get_session_info, get_session_from_database, delete_session,
    delete_all_sessions, update_session_name, save_time_info_data,
    save_zeitschritte_data, get_time_info_data, get_zeitschritte_data,
    get_csv_files_for_session, get_session_status, create_database_session,
    get_session_uuid, get_upload_status
)

from domains.training.services.upload import (
    process_chunk_upload, get_session_metadata_locally,
    create_csv_file_record, update_csv_file_record, delete_csv_file_record
)

from domains.training.data.generator import generate_violin_plots_for_session
from domains.training.services.orchestrator import run_model_training_async

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

bp = Blueprint('training', __name__)

UPLOAD_BASE_DIR = 'uploads/file_uploads'


@bp.route('/upload-chunk', methods=['POST'])
@require_auth
@require_subscription
@check_processing_limit
def upload_chunk():
    """Handle chunk upload from frontend - saving locally."""
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


@bp.route('/finalize-session', methods=['POST'])
@require_auth
def finalize_session_endpoint():
    """Finalize a session after all files have been uploaded."""
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
@require_auth
def list_sessions():
    """List all available training sessions from Supabase database."""
    try:
        limit = request.args.get('limit', 50, type=int)
        user_id = g.user_id
        sessions = get_sessions_list(user_id=user_id, limit=limit)

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
@require_auth
def get_session_endpoint(session_id):
    """Get detailed information about a specific session from local storage."""
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
@require_auth
def get_session_from_database_endpoint(session_id):
    """Get detailed information about a specific session from Supabase database."""
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
            database_session_id = create_or_get_session_uuid(session_id, g.user_id)
            if not database_session_id:
                return jsonify({'success': False, 'error': 'Could not find or create database session'}), 404

        supabase = get_supabase_client(use_service_role=True)
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
            'files': [{
                'id': f['id'],
                'fileName': f['file_name'],
                'bezeichnung': f.get('bezeichnung', ''),
                'utcMin': f.get('utc_min', ''),
                'utcMax': f.get('utc_max', ''),
                'zeitschrittweite': f.get('zeitschrittweite', ''),
                'min': f.get('min', ''),
                'max': f.get('max', ''),
                'offset': f.get('offset', ''),
                'datenpunkte': f.get('datenpunkte', ''),
                'numerischeDatenpunkte': f.get('numerische_datenpunkte', ''),
                'numerischerAnteil': f.get('numerischer_anteil', ''),
                'datenform': f.get('datenform', ''),
                'zeithorizontStart': f.get('zeithorizont_start', ''),
                'zeithorizontEnd': f.get('zeithorizont_end', ''),
                'zeitschrittweiteTransferiertenDaten': f.get('zeitschrittweite_transferierten_daten', ''),
                'offsetTransferiertenDaten': f.get('offset_transferierten_daten', ''),
                'mittelwertbildungÜberDenZeithorizont': f.get('mittelwertbildung_uber_den_zeithorizont', 'nein'),
                'datenanpassung': f.get('datenanpassung', ''),
                'zeitschrittweiteMinValue': f.get('zeitschrittweite_min', ''),
                'zeitschrittweiteAvgValue': f.get('zeitschrittweite_mittelwert', ''),
                'skalierung': f.get('skalierung', 'nein'),
                'skalierungMax': f.get('skalierung_max', ''),
                'skalierungMin': f.get('skalierung_min', ''),
                'storagePath': f.get('storage_path', ''),
                'type': f.get('type', 'input'),
                'createdAt': f.get('created_at', ''),
                'updatedAt': f.get('updated_at', '')
            } for f in (files_response.data or [])],
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
@require_auth
def session_status(session_id):
    """Get the upload status of a session."""
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
@require_auth
def init_session():
    """Initialize a new upload session."""
    try:
        data = request.json
        session_id = data.get('sessionId')
        time_info = data.get('timeInfo', {})
        zeitschritte = data.get('zeitschritte', {})

        result = initialize_session(session_id, time_info, zeitschritte, g.user_id)

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
@require_auth
def save_time_info_endpoint():
    """Save time information via API endpoint."""
    try:
        if not request.is_json:
            return jsonify({'success': False, 'error': 'Request must be JSON'}), 400

        try:
            data = request.get_json(force=True)
        except Exception as e:
            return jsonify({'success': False, 'error': f'Invalid JSON: {str(e)}'}), 400

        if not data or 'sessionId' not in data or 'timeInfo' not in data:
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
@require_auth
def create_database_session_endpoint():
    """Create a new session in Supabase database and return UUID."""
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
@require_auth
def get_session_uuid_endpoint(session_id):
    """Get the UUID session ID for a string session ID."""
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
@require_auth
def save_zeitschritte_endpoint():
    """Save zeitschritte information via API endpoint."""
    try:
        if not request.is_json:
            return jsonify({'success': False, 'error': 'Request must be JSON'}), 400

        try:
            data = request.get_json(force=True)
        except Exception as e:
            return jsonify({'success': False, 'error': f'Invalid JSON: {str(e)}'}), 400

        if not data or 'sessionId' not in data or 'zeitschritte' not in data:
            return jsonify({'success': False, 'error': 'Missing sessionId or zeitschritte'}), 400

        session_id = data['sessionId']
        zeitschritte = data['zeitschritte']

        save_zeitschritte_data(session_id, zeitschritte, user_id=g.user_id)

        return jsonify({'success': True, 'message': 'Zeitschritte saved successfully'})

    except ValueError as e:
        return jsonify({'success': False, 'error': str(e)}), 400
    except Exception as e:
        logger.error(f"Error saving zeitschritte: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@bp.route('/session/<session_id>/delete', methods=['POST'])
@require_auth
def delete_session_endpoint(session_id):
    """Delete a specific session and all its files."""
    try:
        user_id = g.user_id
        result = delete_session(session_id, user_id=user_id)

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

    except PermissionError as e:
        return jsonify({'success': False, 'error': str(e)}), 403
    except ValueError as e:
        return jsonify({'success': False, 'error': str(e)}), 404
    except Exception as e:
        logger.error(f"Error deleting session {session_id}: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@bp.route('/get-zeitschritte/<session_id>', methods=['GET'])
@require_auth
def get_zeitschritte_endpoint(session_id):
    """Get zeitschritte data for a session."""
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
@require_auth
def get_time_info_endpoint(session_id):
    """Get time info data for a session."""
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
        from shared.database.operations import save_file_info, save_csv_file_content

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


@bp.route('/results/<session_id>', methods=['GET'])
@require_auth
def get_training_results(session_id):
    """Get training results for a session."""
    try:
        from utils.training_storage import download_training_results
        supabase = get_supabase_client(use_service_role=True)
        uuid_session_id = create_or_get_session_uuid(session_id, g.user_id)

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
                    logger.info(f"Downloading full results from storage: {record['results_file_path']}")
                    full_results = download_training_results(
                        file_path=record['results_file_path'],
                        decompress=record.get('compressed', False)
                    )
                    record['results'] = full_results
                    logger.info(f"Full results loaded from storage successfully")
                except Exception as download_error:
                    logger.error(f"Failed to download results from storage: {download_error}")
                    record['results'] = record.get('results_metadata', {})
            else:
                legacy_response = supabase.table('training_results')\
                    .select('results')\
                    .eq('id', record['id'])\
                    .single()\
                    .execute()
                if legacy_response.data and legacy_response.data.get('results'):
                    record['results'] = legacy_response.data['results']
                else:
                    record['results'] = record.get('results_metadata', {})

            return jsonify({
                'success': True,
                'results': [record],
                'count': 1
            })
        else:
            return jsonify({
                'success': True,
                'message': 'No training results yet - training may not have been started',
                'results': [],
                'count': 0
            }), 200

    except PermissionError as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 403
    except Exception as e:
        logger.error(f"Error getting training results for {session_id}: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@bp.route('/get-training-results/<session_id>', methods=['GET'])
@require_auth
def get_training_results_details(session_id):
    """Get detailed training results for a session (alias)."""
    return get_training_results(session_id)


@bp.route('/plot-variables/<session_id>', methods=['GET'])
@require_auth
def get_plot_variables(session_id):
    """Get available input and output variables for plotting."""
    try:
        user_id = g.user_id
        visualizer = Visualizer()
        variables = visualizer.get_available_variables(session_id, user_id=user_id)

        return jsonify({
            'success': True,
            'session_id': session_id,
            'input_variables': variables['input_variables'],
            'output_variables': variables['output_variables']
        })

    except PermissionError as e:
        return jsonify({'success': False, 'error': str(e)}), 403
    except Exception as e:
        logger.error(f"Error getting plot variables for {session_id}: {str(e)}")
        return jsonify({
            'success': True,
            'session_id': session_id,
            'input_variables': ['Temperature', 'Load'],
            'output_variables': ['Predicted_Load']
        })


@bp.route('/visualizations/<session_id>', methods=['GET'])
@require_auth
def get_training_visualizations(session_id):
    """Get training visualizations (violin plots) for a session."""
    try:
        user_id = g.user_id
        visualizer = Visualizer()
        viz_data = visualizer.get_session_visualizations(session_id, user_id=user_id)

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

    except PermissionError as e:
        return create_error_response(str(e), 403)
    except Exception as e:
        logger.error(f"Error retrieving visualizations for {session_id}: {str(e)}")
        return create_error_response(f'Failed to retrieve training visualizations: {str(e)}', 500)


@bp.route('/generate-plot', methods=['POST'])
@require_auth
def generate_plot():
    """Generate plot based on user selections."""
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
@require_auth
def get_training_status(session_id: str):
    """Get training status for a session."""
    try:
        supabase = get_supabase_client(use_service_role=True)
        uuid_session_id = create_or_get_session_uuid(session_id, g.user_id)

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
    """Generate datasets and violin plots WITHOUT training models."""
    try:
        data = request.json
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400

        model_parameters = data.get('model_parameters', {})
        training_split = data.get('training_split', {})

        from domains.training.services.violin_tracker import ViolinProgressTracker

        socketio = current_app.extensions.get('socketio')
        progress_tracker = ViolinProgressTracker(socketio, session_id)

        result = generate_violin_plots_for_session(
            session_id=session_id,
            model_parameters=model_parameters,
            training_split=training_split,
            progress_tracker=progress_tracker
        )

        violin_plots = result.get('violin_plots', {})
        if violin_plots:
            progress_tracker.saving_to_database()

            for plot_name, plot_data in violin_plots.items():
                try:
                    if plot_data:
                        save_visualization_to_database(session_id, plot_name, plot_data)
                except Exception as viz_error:
                    logger.error(f"Failed to save visualization {plot_name}: {str(viz_error)}")

        progress_tracker.complete()

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
    """Train models with user-specified parameters."""
    try:
        data = request.json
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400

        original_session_id = session_id
        temporary_session_id = get_string_id_from_uuid(session_id)
        if temporary_session_id:
            session_id = temporary_session_id
            logger.info(f"Converted UUID session {original_session_id} to temporary session {session_id}")

        model_parameters = data.get('model_parameters', {})
        training_split = data.get('training_split', {})

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

        increment_training_count(g.user_id)
        logger.info(f"Tracked training run for user {g.user_id}")

        try:
            supabase = get_supabase_client(use_service_role=True)
            uuid_session_id = create_or_get_session_uuid(session_id, g.user_id)
            delete_response = supabase.table('training_results').delete().eq('session_id', uuid_session_id).execute()
            deleted_count = len(delete_response.data) if delete_response.data else 0
            if deleted_count > 0:
                logger.info(f"Deleted {deleted_count} old training_results for session {uuid_session_id}")
        except Exception as e:
            logger.warning(f"Could not delete old training_results: {e}")

        import threading

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
@require_auth
def delete_all_sessions_endpoint():
    """Delete ALL sessions and associated data."""
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
                raw_data = request.get_data(as_text=True)
                if raw_data:
                    data = json.loads(raw_data)
            except:
                data = {}

        confirmation = data.get('confirm_delete_all', False)

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
@require_auth
def get_evaluation_tables(session_id):
    """Get evaluation metrics formatted as tables for display."""
    try:
        from utils.training_storage import download_training_results
        supabase = get_supabase_client(use_service_role=True)
        uuid_session_id = create_or_get_session_uuid(session_id, g.user_id)

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
            results = download_training_results(
                file_path=record['results_file_path'],
                decompress=record.get('compressed', False)
            )
        else:
            results = record.get('results')

        eval_metrics = results.get('evaluation_metrics', {})
        if not eval_metrics or eval_metrics.get('error'):
            eval_metrics = results.get('metrics', {})

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
@require_auth
def save_evaluation_tables(session_id):
    """Save evaluation tables to database."""
    try:
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

        supabase = get_supabase_client(use_service_role=True)
        uuid_session_id = create_or_get_session_uuid(session_id, g.user_id)

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
@require_auth
def get_scalers(session_id):
    """Retrieve saved scalers from database for a specific session."""
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
@require_auth
def download_scalers_as_save_files(session_id):
    """Download scalers as .save files."""
    try:
        zip_file_path = create_scaler_download_package(session_id)
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
@require_auth
def scale_input_data(session_id):
    """Scale input data using saved scalers."""
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


@bp.route('/save-model/<session_id>', methods=['POST'])
@require_auth
def save_model(session_id):
    """Save trained models to Supabase Storage."""
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
        logger.error(f"Error saving models: {e}")
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
@require_auth
def list_models_database(session_id):
    """List all trained models stored in Supabase Storage for a session."""
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
        logger.error(f"Error listing models from Storage: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'Failed to list models from Storage'
        }), 500


@bp.route('/download-model-h5/<session_id>', methods=['GET'])
@require_auth
def download_model_h5(session_id):
    """Download a trained model file from Supabase Storage."""
    try:
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
        logger.error(f"Error downloading model: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'Failed to download model from Storage'
        }), 500


@bp.route('/session-name-change', methods=['POST'])
@require_auth
def change_session_name_endpoint():
    """Update session name in the database."""
    try:
        data = request.get_json()
        if not data:
            return create_error_response('No data provided', 400)

        session_id = data.get('sessionId')
        session_name = data.get('sessionName')
        user_id = g.user_id

        result = update_session_name(session_id, session_name, user_id=user_id)

        return create_success_response(
            data={
                'sessionId': result['session_id'],
                'sessionName': result['session_name']
            },
            message=result['message']
        )

    except PermissionError as e:
        return create_error_response(str(e), 403)
    except ValueError as e:
        logger.warning(f"Validation error: {str(e)}")
        return create_error_response(str(e), 400)
    except Exception as e:
        logger.error(f"Error changing session name: {str(e)}")
        return create_error_response(f'Internal server error: {str(e)}', 500)
