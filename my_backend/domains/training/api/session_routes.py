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

W11-A T6: hardened with @limiter.limit, UUID guard, and the
standardized error contract (shared.responses.errors.error_response).
"""

from flask import Blueprint

from .common import (
    os, json, datetime, request, jsonify, g, logging,
    require_auth, require_subscription,
    get_supabase_client, get_string_id_from_uuid, create_or_get_session_uuid,
    validate_session_id, create_success_response,
    resolve_session_id, is_uuid_format, get_string_session_id, get_uuid_session_id,
    UPLOAD_BASE_DIR, get_logger
)

from core.rate_limits import limiter, training_limit_string
from shared.responses.errors import error_response as _err
from shared.validators.uuid import validate_training_session_format

from domains.training.services.session import (
    initialize_session, finalize_session, get_sessions_list,
    get_session_info, delete_session, delete_all_sessions,
    update_session_name, save_time_info_data, save_zeitschritte_data,
    get_time_info_data, get_zeitschritte_data, get_session_status,
    create_database_session, get_session_uuid, get_upload_status
)
from domains.training.services.lifecycle import is_training_in_flight

from shared.auth.ownership import assert_session_ownership, SessionOwnershipError

bp = Blueprint('training_sessions', __name__)
logger = get_logger(__name__)


@bp.route('/init-session', methods=['POST'])
@limiter.limit(training_limit_string)
@require_auth
@require_subscription
def init_session():
    """Initialize a new upload session."""
    data = request.get_json(silent=True) or {}
    session_id = data.get('sessionId') or data.get('session_id')
    if not session_id:
        return _err('MISSING_BODY', 'sessionId is required', 400)
    # W11-BE2: validate session_id BEFORE auth-ed work.
    err = validate_training_session_format(session_id)
    if err:
        return err

    try:
        time_info = data.get('timeInfo', {})
        zeitschritte = data.get('zeitschritte', {})

        result = initialize_session(session_id, time_info, zeitschritte, g.user_id)

        return jsonify({
            'success': True,
            'sessionId': result['session_id'],
            'message': result['message']
        })

    except ValueError:
        logger.warning("init-session: invalid input", exc_info=True)
        return _err('BAD_REQUEST', 'Invalid session initialization request', 400)
    except Exception:
        logger.exception("Failed to initialize session")
        return _err(
            'INTERNAL_ERROR',
            'Failed to initialize session',
            500,
            suggestion='Please try again. If the problem persists, contact support.',
        )


@bp.route('/finalize-session', methods=['POST'])
@limiter.limit(training_limit_string)
@require_auth
@require_subscription
def finalize_session_endpoint():
    """Finalize a session after all files have been uploaded."""
    data = request.get_json(silent=True) or {}
    session_id = data.get('sessionId') or data.get('session_id')
    if not session_id:
        return _err('MISSING_BODY', 'No session ID provided', 400)
    # W11-BE2: validate session_id BEFORE auth-ed work.
    err = validate_training_session_format(session_id)
    if err:
        return err

    try:
        result = finalize_session(session_id, data)

        return jsonify({
            'success': True,
            'message': result['message'],
            'sessionId': result['session_id'],
            'n_dat': result['n_dat'],
            'file_count': result['file_count']
        })

    except ValueError:
        logger.warning("finalize-session: invalid input", exc_info=True)
        return _err('BAD_REQUEST', 'Invalid session finalization request', 400)
    except Exception:
        logger.exception("Failed to finalize session")
        return _err(
            'INTERNAL_ERROR',
            'Failed to finalize session',
            500,
            suggestion='Please try again. If the problem persists, contact support.',
        )


@bp.route('/list-sessions', methods=['GET'])
@limiter.limit(training_limit_string)
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

    except Exception:
        logger.exception("Failed to retrieve sessions from database")
        return _err(
            'INTERNAL_ERROR',
            'Failed to retrieve sessions from database',
            500,
            suggestion='Please try again. If the problem persists, contact support.',
        )


@bp.route('/session/<session_id>', methods=['GET'])
@limiter.limit(training_limit_string)
@require_auth
def get_session_endpoint(session_id):
    """Get detailed information about a specific session from local storage."""
    # W11-BE2: validate session_id BEFORE auth-ed work.
    err = validate_training_session_format(session_id)
    if err:
        return err

    try:
        try:
            string_session_id = get_string_session_id(session_id)
        except ValueError:
            logger.warning("get-session: session not found in mappings", exc_info=True)
            return _err('SESSION_NOT_FOUND', 'Session not found', 404)

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

    except FileNotFoundError:
        logger.warning("get-session: file not found", exc_info=True)
        return _err('SESSION_NOT_FOUND', 'Session not found', 404)
    except Exception:
        logger.exception("Failed to get session details")
        return _err(
            'INTERNAL_ERROR',
            'Failed to get session details',
            500,
            suggestion='Please try again. If the problem persists, contact support.',
        )


@bp.route('/session/<session_id>/database', methods=['GET'])
@limiter.limit(training_limit_string)
@require_auth
def get_session_from_database_endpoint(session_id):
    """Get detailed information about a specific session from Supabase database."""
    # W11-BE2: validate session_id BEFORE auth-ed work.
    err = validate_training_session_format(session_id)
    if err:
        return err

    try:
        try:
            string_session_id, database_session_id = resolve_session_id(session_id, g.user_id)
        except ValueError:
            logger.warning("get-session-database: session not found", exc_info=True)
            return _err('SESSION_NOT_FOUND', 'Session not found', 404)

        try:
            assert_session_ownership(database_session_id)
        except SessionOwnershipError:
            return _err('FORBIDDEN', 'You do not have access to this session', 403)

        supabase = get_supabase_client(use_service_role=True)
        if not supabase:
            logger.error("Database connection not available")
            return _err('INTERNAL_ERROR', 'Database connection not available', 500)

        session_response = supabase.table('sessions').select('*').eq('id', database_session_id).execute()
        if not session_response.data:
            return _err('SESSION_NOT_FOUND', 'Session not found', 404)

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

    except Exception:
        logger.exception("Failed to get session from database")
        return _err(
            'INTERNAL_ERROR',
            'Failed to get session from database',
            500,
            suggestion='Please try again. If the problem persists, contact support.',
        )


@bp.route('/session-status/<session_id>', methods=['GET'])
@limiter.limit(training_limit_string)
@require_auth
def session_status(session_id):
    """Get the upload status of a session."""
    # W11-BE2: validate session_id BEFORE auth-ed work.
    err = validate_training_session_format(session_id)
    if err:
        return err

    try:
        status_info = get_upload_status(session_id)
        return jsonify(status_info)

    except ValueError:
        logger.warning("session-status: session not found", exc_info=True)
        return _err('SESSION_NOT_FOUND', 'Session not found', 404)

    except Exception:
        logger.exception("Failed to get session status")
        return _err(
            'INTERNAL_ERROR',
            'Failed to get session status',
            500,
            suggestion='Please try again. If the problem persists, contact support.',
        )


@bp.route('/session/<session_id>/delete', methods=['POST'])
@limiter.limit(training_limit_string)
@require_auth
def delete_session_endpoint(session_id):
    """Delete a specific session and all its files."""
    # W11-BE2: validate session_id BEFORE auth-ed work.
    err = validate_training_session_format(session_id)
    if err:
        return err

    # FIX-5 (Bug 2): reject delete while training is in flight. Otherwise the
    # background thread continues uploading 0.9-12MB orphan storage objects
    # with no FK reference, cascading FK violations for ~30s after delete
    # returns 200 OK. Ownership is enforced inside delete_session() (FIX-1),
    # but we resolve to UUID here for the in-flight lookup — the training
    # thread persists under the UUID, not the string-form session_id.
    #
    # Resolution failure (unknown session) is non-fatal here: we fall through
    # to delete_session() which has its own ownership/missing-session
    # handling. This avoids double-leaking session existence (FIX-1 contract).
    try:
        uuid_session_id = create_or_get_session_uuid(session_id, g.user_id)
        if is_training_in_flight(uuid_session_id):
            return _err(
                'TRAINING_IN_PROGRESS',
                'Cannot delete session while training is running',
                409,
                suggestion='Wait for training to complete or fail (poll /status), then retry delete.',
            )
    except (ValueError, PermissionError):
        # Let delete_session() produce the canonical 404/403; do not branch
        # on resolution errors here.
        pass

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

    except PermissionError:
        logger.warning("delete-session: ownership violation", exc_info=True)
        return _err('FORBIDDEN', 'You do not have access to this session', 403)
    except ValueError:
        logger.warning("delete-session: session not found", exc_info=True)
        return _err('SESSION_NOT_FOUND', 'Session not found', 404)
    except Exception:
        logger.exception("Failed to delete session")
        return _err(
            'INTERNAL_ERROR',
            'Failed to delete session',
            500,
            suggestion='Please try again. If the problem persists, contact support.',
        )


@bp.route('/delete-all-sessions', methods=['POST'])
@limiter.limit(training_limit_string)
@require_auth
def delete_all_sessions_endpoint():
    """Delete ALL sessions and associated data.

    CRITICAL: This is a destructive endpoint. It requires
    ``confirm_delete_all=True`` in the request body. The underlying
    service ``delete_all_sessions(confirm=False, user_id=...)`` raises
    ValueError when the flag is missing — we map that to
    CONFIRMATION_REQUIRED 400. The E2E security contract from
    docs/superpowers/plans/2026-03-30-e2e-delete-all-sessions-security.md
    (user_id scoping) is enforced in the service layer; this route does
    not change it.
    """
    data = {}
    try:
        if request.is_json:
            data = request.get_json(silent=True) or {}
        elif request.content_type and 'application/json' in request.content_type:
            data = request.get_json(force=True, silent=True) or {}
        elif hasattr(request, 'json') and request.json:
            data = request.json
        else:
            data = request.form.to_dict()
            if 'confirm_delete_all' in data:
                data['confirm_delete_all'] = data['confirm_delete_all'].lower() in ['true', '1', 'yes']
    except Exception:
        logger.warning("delete-all-sessions: could not parse request data", exc_info=True)
        try:
            raw_data = request.get_data(as_text=True)
            if raw_data:
                data = json.loads(raw_data)
        except Exception:
            data = {}

    confirmation = data.get('confirm_delete_all', False)
    user_id = g.user_id

    try:
        result = delete_all_sessions(confirm=confirmation, user_id=user_id)

        response_data = {
            'success': True,
            'message': result['message'],
            'summary': result['summary'],
            'details': result['details']
        }

        if result.get('warnings'):
            response_data['warnings'] = result['warnings']

        return jsonify(response_data)

    except ValueError:
        logger.warning("delete-all-sessions: confirmation missing", exc_info=True)
        return _err(
            'CONFIRMATION_REQUIRED',
            'Confirmation required to delete all sessions',
            400,
            suggestion='To delete all sessions, send {"confirm_delete_all": true} in request body',
        )
    except Exception:
        logger.exception("Critical failure during delete all sessions")
        return _err(
            'INTERNAL_ERROR',
            'Critical error occurred during deletion operation',
            500,
            suggestion='Please try again. If the problem persists, contact support.',
        )


@bp.route('/create-database-session', methods=['POST'])
@limiter.limit(training_limit_string)
@require_auth
@require_subscription
def create_database_session_endpoint():
    """Create a new session in Supabase database and return UUID."""
    data = request.get_json(silent=True) or {}
    session_id = data.get('sessionId') or data.get('session_id')
    # session_id is optional on this endpoint, but if provided it must be
    # well-formed (the service will otherwise raise).
    if session_id:
        err = validate_training_session_format(session_id)
        if err:
            return err

    try:
        session_name = data.get('sessionName')

        # Pass user_id from auth context to create_database_session
        session_uuid = create_database_session(session_id, session_name, user_id=g.user_id)

        return jsonify({
            'success': True,
            'sessionUuid': session_uuid,
            'message': f'Database session created with UUID: {session_uuid}'
        })

    except ValueError:
        logger.warning("create-database-session: invalid input", exc_info=True)
        return _err('BAD_REQUEST', 'Invalid session creation request', 400)
    except Exception:
        logger.exception("Failed to create database session")
        return _err(
            'DB_WRITE_ERROR',
            'Failed to create database session',
            500,
            suggestion='Please try again. If the problem persists, contact support.',
        )


@bp.route('/get-session-uuid/<session_id>', methods=['GET'])
@limiter.limit(training_limit_string)
@require_auth
def get_session_uuid_endpoint(session_id):
    """Get the UUID session ID for a string session ID."""
    # W11-BE2: validate session_id BEFORE auth-ed work.
    err = validate_training_session_format(session_id)
    if err:
        return err

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

    except ValueError:
        logger.warning("get-session-uuid: session not found", exc_info=True)
        return _err('SESSION_NOT_FOUND', 'Session not found', 404)
    except Exception:
        logger.exception("Failed to resolve session UUID")
        return _err(
            'INTERNAL_ERROR',
            'Failed to resolve session UUID',
            500,
            suggestion='Please try again. If the problem persists, contact support.',
        )


@bp.route('/save-time-info', methods=['POST'])
@limiter.limit(training_limit_string)
@require_auth
@require_subscription
def save_time_info_endpoint():
    """Save time information via API endpoint."""
    data = request.get_json(silent=True)
    if not data:
        return _err('MISSING_BODY', 'Request body is required', 400)

    session_id = data.get('sessionId') or data.get('session_id')
    if not session_id:
        return _err('MISSING_BODY', 'sessionId is required', 400)
    # W11-BE2: validate session_id BEFORE auth-ed work.
    err = validate_training_session_format(session_id)
    if err:
        return err

    time_info = data.get('timeInfo')
    if time_info is None:
        return _err('INVALID_TIME_INFO', 'timeInfo field is required', 400)
    if not isinstance(time_info, dict):
        return _err('INVALID_TIME_INFO', 'timeInfo must be an object', 400)

    try:
        save_time_info_data(session_id, time_info)

        return create_success_response(message='Time info saved successfully')

    except ValueError:
        logger.warning("save-time-info: invalid input", exc_info=True)
        return _err('INVALID_TIME_INFO', 'Invalid time_info payload', 400)
    except Exception:
        logger.exception("Failed to save time info")
        return _err(
            'DB_WRITE_ERROR',
            'Failed to save time info',
            500,
            suggestion='Please try again. If the problem persists, contact support.',
        )


@bp.route('/save-zeitschritte', methods=['POST'])
@limiter.limit(training_limit_string)
@require_auth
@require_subscription
def save_zeitschritte_endpoint():
    """Save zeitschritte information via API endpoint."""
    data = request.get_json(silent=True)
    if not data:
        return _err('MISSING_BODY', 'Request body is required', 400)

    session_id = data.get('sessionId') or data.get('session_id')
    if not session_id:
        return _err('MISSING_BODY', 'sessionId is required', 400)
    # W11-BE2: validate session_id BEFORE auth-ed work.
    err = validate_training_session_format(session_id)
    if err:
        return err

    zeitschritte = data.get('zeitschritte')
    if zeitschritte is None:
        return _err('INVALID_ZEITSCHRITTE', 'zeitschritte field is required', 400)
    if not isinstance(zeitschritte, dict):
        return _err('INVALID_ZEITSCHRITTE', 'zeitschritte must be an object', 400)

    try:
        save_zeitschritte_data(session_id, zeitschritte, user_id=g.user_id)

        return create_success_response(message='Zeitschritte saved successfully')

    except ValueError:
        logger.warning("save-zeitschritte: invalid input", exc_info=True)
        return _err('INVALID_ZEITSCHRITTE', 'Invalid zeitschritte payload', 400)
    except Exception:
        logger.exception("Failed to save zeitschritte")
        return _err(
            'DB_WRITE_ERROR',
            'Failed to save zeitschritte',
            500,
            suggestion='Please try again. If the problem persists, contact support.',
        )


@bp.route('/get-time-info/<session_id>', methods=['GET'])
@limiter.limit(training_limit_string)
@require_auth
def get_time_info_endpoint(session_id):
    """Get time info data for a session."""
    # W11-BE2: validate session_id BEFORE auth-ed work.
    err = validate_training_session_format(session_id)
    if err:
        return err

    # FIX-1: enforce session ownership before service-layer data access.
    # get_time_info_data does not filter by user_id — without this guard
    # raw-UUID input reaches the data layer (RLS bypassed by service-role).
    try:
        uuid_session_id = create_or_get_session_uuid(session_id, g.user_id)
        assert_session_ownership(uuid_session_id)
    except SessionOwnershipError:
        return _err('SESSION_NOT_FOUND', 'Session not found', 404)
    except (ValueError, PermissionError):
        return _err('SESSION_NOT_FOUND', 'Session not found', 404)

    try:
        time_info = get_time_info_data(session_id)
        return jsonify({
            'success': True,
            'data': time_info
        })

    except ValueError:
        # A session may legitimately have no time-info configured — that is an
        # empty state, not an error. Return 200 with empty data so the frontend
        # (which polls this endpoint) doesn't spam the browser console with 404s
        # and the backend log stays quiet (single debug line, no traceback).
        logger.debug("get-time-info: none for session %s", session_id)
        return jsonify({'success': True, 'data': {}})
    except Exception:
        logger.exception("Failed to get time info")
        return _err(
            'INTERNAL_ERROR',
            'Failed to get time info',
            500,
            suggestion='Please try again. If the problem persists, contact support.',
        )


@bp.route('/get-zeitschritte/<session_id>', methods=['GET'])
@limiter.limit(training_limit_string)
@require_auth
def get_zeitschritte_endpoint(session_id):
    """Get zeitschritte data for a session."""
    # W11-BE2: validate session_id BEFORE auth-ed work.
    err = validate_training_session_format(session_id)
    if err:
        return err

    # FIX-1: enforce session ownership before service-layer data access.
    try:
        uuid_session_id = create_or_get_session_uuid(session_id, g.user_id)
        assert_session_ownership(uuid_session_id)
    except SessionOwnershipError:
        return _err('SESSION_NOT_FOUND', 'Session not found', 404)
    except (ValueError, PermissionError):
        return _err('SESSION_NOT_FOUND', 'Session not found', 404)

    try:
        zeitschritte = get_zeitschritte_data(session_id)
        return create_success_response(zeitschritte)

    except ValueError:
        logger.warning("get-zeitschritte: not found", exc_info=True)
        return _err('SESSION_NOT_FOUND', 'Session or zeitschritte not found', 404)
    except Exception:
        logger.exception("Failed to get zeitschritte")
        return _err(
            'INTERNAL_ERROR',
            'Failed to get zeitschritte',
            500,
            suggestion='Please try again. If the problem persists, contact support.',
        )


@bp.route('/session-name-change', methods=['POST'])
@limiter.limit(training_limit_string)
@require_auth
@require_subscription
def change_session_name_endpoint():
    """Update session name in the database."""
    data = request.get_json(silent=True)
    if not data:
        return _err('MISSING_BODY', 'Request body is required', 400)

    session_id = data.get('sessionId') or data.get('session_id')
    if not session_id:
        return _err('MISSING_BODY', 'sessionId is required', 400)
    # W11-BE2: validate session_id BEFORE auth-ed work.
    err = validate_training_session_format(session_id)
    if err:
        return err

    session_name = data.get('sessionName')

    try:
        user_id = g.user_id

        result = update_session_name(session_id, session_name, user_id=user_id)

        return create_success_response(
            data={
                'sessionId': result['session_id'],
                'sessionName': result['session_name']
            },
            message=result['message']
        )

    except PermissionError:
        logger.warning("session-name-change: ownership violation", exc_info=True)
        return _err('FORBIDDEN', 'You do not have access to this session', 403)
    except ValueError as e:
        # The service distinguishes empty / too-long / missing name from
        # other validation errors via the exception message. We map the
        # name-shape ones to INVALID_SESSION_NAME so the FE can render the
        # right inline hint; everything else is a generic BAD_REQUEST.
        error_msg = str(e)
        logger.warning("session-name-change: validation error", exc_info=True)
        if (
            'sessionName' in error_msg
            or 'session_name' in error_msg
            or 'too long' in error_msg
            or 'empty' in error_msg
        ):
            return _err('INVALID_SESSION_NAME', 'Invalid session name', 400)
        return _err('BAD_REQUEST', 'Invalid session name change request', 400)
    except Exception:
        logger.exception("Failed to change session name")
        return _err(
            'INTERNAL_ERROR',
            'Failed to change session name',
            500,
            suggestion='Please try again. If the problem persists, contact support.',
        )
