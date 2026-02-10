"""
Session management routes for training API.

Contains 15 endpoints for session CRUD operations:
- init-session, finalize-session, list-sessions
- session/<id>, session/<id>/database, session-status/<id>
- session/<id>/delete, delete-all-sessions
- create-database-session, get-session-uuid/<id>
- save-time-info, save-zeitschritte
- get-time-info/<id>, get-zeitschritte/<id>
- session-name-change
"""

from flask import Blueprint

from .common import (
    os, json, datetime, request, jsonify, g, logging,
    require_auth, require_subscription,
    get_supabase_client, get_string_id_from_uuid, create_or_get_session_uuid,
    validate_session_id, create_error_response, create_success_response,
    resolve_session_id, is_uuid_format, get_string_session_id, get_uuid_session_id,
    UPLOAD_BASE_DIR, get_logger
)

from domains.training.services.session import (
    initialize_session, finalize_session, get_sessions_list,
    get_session_info, delete_session, delete_all_sessions,
    update_session_name, save_time_info_data, save_zeitschritte_data,
    get_time_info_data, get_zeitschritte_data, get_session_status,
    create_database_session, get_session_uuid, get_upload_status
)

bp = Blueprint('training_sessions', __name__)
logger = get_logger(__name__)


@bp.route('/init-session', methods=['POST'])
@require_auth
@require_subscription
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
        return create_error_response(str(e), 400)
    except Exception as e:
        logger.error(f"Error initializing session: {str(e)}")
        return create_error_response(str(e), 500)


@bp.route('/finalize-session', methods=['POST'])
@require_auth
@require_subscription
def finalize_session_endpoint():
    """Finalize a session after all files have been uploaded."""
    try:
        data = request.json
        if not data or 'sessionId' not in data:
            return create_error_response('No session ID provided', 400)

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
        return create_error_response(str(e), 400)
    except Exception as e:
        logger.error(f"Error finalizing session: {str(e)}")
        return create_error_response(str(e), 500)


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
            return create_error_response('No session ID provided', 400)

        try:
            string_session_id = get_string_session_id(session_id)
        except ValueError as e:
            return create_error_response(str(e), 404)

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
        return create_error_response(str(e), 404)
    except Exception as e:
        logger.error(f"Error getting session {session_id}: {str(e)}")
        return create_error_response(str(e), 500)


@bp.route('/session/<session_id>/database', methods=['GET'])
@require_auth
def get_session_from_database_endpoint(session_id):
    """Get detailed information about a specific session from Supabase database."""
    try:
        if not session_id:
            return create_error_response('No session ID provided', 400)

        try:
            string_session_id, database_session_id = resolve_session_id(session_id, g.user_id)
        except ValueError as e:
            return create_error_response(str(e), 404)

        supabase = get_supabase_client(use_service_role=True)
        if not supabase:
            return create_error_response('Database connection not available', 500)

        session_response = supabase.table('sessions').select('*').eq('id', database_session_id).execute()
        if not session_response.data:
            return create_error_response('Session not found in database', 404)

        session_data = session_response.data[0]
        files_response = supabase.table('files').select('*').eq('session_id', database_session_id).order('color_index').execute()
        time_info_response = supabase.table('time_info').select('*').eq('session_id', database_session_id).execute()
        zeitschritte_response = supabase.table('zeitschritte').select('*').eq('session_id', database_session_id).execute()

        session_info = {
            'sessionId': string_session_id,
            'sessionName': session_data.get('session_name', ''),
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
        return create_error_response(str(e), 500)


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
        return create_error_response(str(e), 403)
    except ValueError as e:
        return create_error_response(str(e), 404)
    except Exception as e:
        logger.error(f"Error deleting session {session_id}: {str(e)}")
        return create_error_response(str(e), 500)


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


@bp.route('/create-database-session', methods=['POST'])
@require_auth
@require_subscription
def create_database_session_endpoint():
    """Create a new session in Supabase database and return UUID."""
    try:
        data = request.json
        session_id = data.get('sessionId') if data else None
        session_name = data.get('sessionName') if data else None

        # Pass user_id from auth context to create_database_session
        session_uuid = create_database_session(session_id, session_name, user_id=g.user_id)

        return jsonify({
            'success': True,
            'sessionUuid': session_uuid,
            'message': f'Database session created with UUID: {session_uuid}'
        })

    except ValueError as e:
        return create_error_response(str(e), 400)
    except Exception as e:
        logger.error(f"Error creating database session: {str(e)}")
        return create_error_response(str(e), 500)


@bp.route('/get-session-uuid/<session_id>', methods=['GET'])
@require_auth
def get_session_uuid_endpoint(session_id):
    """Get the UUID session ID for a string session ID."""
    try:
        if is_uuid_format(session_id):
            return create_success_response(
                data={'sessionUuid': session_id},
                message='Session ID is already in UUID format'
            )

        session_uuid = get_session_uuid(session_id)
        return create_success_response(
            data={'sessionUuid': session_uuid},
            message=f'Found/created UUID for session: {session_uuid}'
        )

    except ValueError as e:
        return create_error_response(str(e), 404)
    except Exception as e:
        logger.error(f"Error getting session UUID: {str(e)}")
        return create_error_response(str(e), 500)


@bp.route('/save-time-info', methods=['POST'])
@require_auth
@require_subscription
def save_time_info_endpoint():
    """Save time information via API endpoint."""
    try:
        if not request.is_json:
            return create_error_response('Request must be JSON', 400)

        try:
            data = request.get_json(force=True)
        except Exception as e:
            return create_error_response(f'Invalid JSON: {str(e)}', 400)

        if not data or 'sessionId' not in data or 'timeInfo' not in data:
            return create_error_response('Missing sessionId or timeInfo', 400)

        session_id = data['sessionId']
        time_info = data['timeInfo']

        save_time_info_data(session_id, time_info)

        return create_success_response(message='Time info saved successfully')

    except ValueError as e:
        return create_error_response(str(e), 400)
    except Exception as e:
        logger.error(f"Error saving time info: {str(e)}")
        return create_error_response(str(e), 500)


@bp.route('/save-zeitschritte', methods=['POST'])
@require_auth
@require_subscription
def save_zeitschritte_endpoint():
    """Save zeitschritte information via API endpoint."""
    try:
        if not request.is_json:
            return create_error_response('Request must be JSON', 400)

        try:
            data = request.get_json(force=True)
        except Exception as e:
            return create_error_response(f'Invalid JSON: {str(e)}', 400)

        if not data or 'sessionId' not in data or 'zeitschritte' not in data:
            return create_error_response('Missing sessionId or zeitschritte', 400)

        session_id = data['sessionId']
        zeitschritte = data['zeitschritte']

        save_zeitschritte_data(session_id, zeitschritte, user_id=g.user_id)

        return create_success_response(message='Zeitschritte saved successfully')

    except ValueError as e:
        return create_error_response(str(e), 400)
    except Exception as e:
        logger.error(f"Error saving zeitschritte: {str(e)}")
        return create_error_response(str(e), 500)


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


@bp.route('/session-name-change', methods=['POST'])
@require_auth
@require_subscription
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
