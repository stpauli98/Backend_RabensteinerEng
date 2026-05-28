"""
Forecast routes for training API.

Endpoints:
- POST /forecast/<session_id> — run forecast pipeline
- GET /forecast-config/<session_id> — get session config for frontend
- GET /api-parameters — list available API parameters for dropdowns
"""

from .common import (
    request, jsonify, g, logging,
    require_auth, require_subscription,
    get_logger, get_supabase_client,
    create_or_get_session_uuid
)
from flask import Blueprint
import json
import pandas as pd

from werkzeug.exceptions import BadRequest as WerkzeugBadRequest
from shared.auth.api_key import allow_api_key
from shared.auth.ownership import assert_session_ownership, SessionOwnershipError

from domains.training.services.forecast_service import run_forecast

bp = Blueprint('training_forecast', __name__)
logger = get_logger(__name__)


@bp.route('/forecast/<session_id>', methods=['POST'])
@allow_api_key
@require_subscription
def execute_forecast(session_id):
    """Run forecast using saved config + user-provided DataFrame.

    Request body (JSON):
    {
        "user_data": {
            "load_grid": [{"UTC": "...", "value": ...}, ...],
            "temp_out": [{"UTC": "...", "value": ...}, ...]
        }
    }
    """
    # Validate JSON body before any DB work — avoids leaking Flask internals.
    try:
        request_data = request.get_json(force=True, silent=False) or {}
    except (WerkzeugBadRequest, json.JSONDecodeError):
        return jsonify({
            'success': False,
            'code': 'MALFORMED_JSON',
            'error': 'Request body is not valid JSON',
        }), 400

    # W12-F7: Pre-pipeline validation — surface missing/empty user_data before DB work.
    user_data = request_data.get('user_data')
    if not user_data or (isinstance(user_data, dict) and len(user_data) == 0):
        return jsonify({
            'success': False,
            'code': 'MISSING_USER_DATA',
            'error': 'Request body must contain a non-empty user_data object',
        }), 400

    try:
        supabase = get_supabase_client(use_service_role=True)
        uuid_session_id = str(create_or_get_session_uuid(session_id, user_id=g.user_id))

        try:
            assert_session_ownership(uuid_session_id)
        except SessionOwnershipError:
            return jsonify({'success': False, 'error': 'forbidden'}), 403

        files_res = supabase.table('files').select('*').eq('session_id', uuid_session_id).execute()
        time_res = supabase.table('time_info').select('*').eq('session_id', uuid_session_id).execute()
        zeit_res = supabase.table('zeitschritte').select('*').eq('session_id', uuid_session_id).execute()

        if not files_res.data:
            return jsonify({'success': False, 'error': 'No files config found for session', 'code': 'NO_CONFIG'}), 404
        if not zeit_res.data:
            return jsonify({'success': False, 'error': 'No zeitschritte config found', 'code': 'NO_CONFIG'}), 404

        input_files = [f for f in files_res.data if f['type'] == 'input']
        unconfigured = [f['bezeichnung'] for f in input_files if not f.get('data_source')]
        if unconfigured:
            return jsonify({
                'success': False,
                'error': f'Unconfigured features: {", ".join(unconfigured)}. Save forecast config first.',
                'code': 'INCOMPLETE_CONFIG'
            }), 400

        input_features = []
        for f in sorted(input_files, key=lambda x: x.get('feature_index') or 0):
            feat = {
                'bezeichnung': f['bezeichnung'],
                'feature_index': f.get('feature_index') or 0,
                'type': 'input',
                'data_source': f['data_source'],
                'data_type': 'time horizon',
                'horizon_start_h': float(f.get('zeithorizont_start') or 0),
                'horizon_end_h': float(f.get('zeithorizont_end') or 0),
                'data_proc': f.get('datenanpassung') or 'intrpl',
                'storage_path': f.get('storage_path'),
            }
            if f['data_source'] == 'Extern':
                feat['api_source'] = f.get('api_source')
                feat['fcst_var'] = f.get('fcst_var')
                feat['latitude'] = float(f['latitude']) if f.get('latitude') else None
                feat['longitude'] = float(f['longitude']) if f.get('longitude') else None
            input_features.append(feat)

        output_files = [f for f in files_res.data if f['type'] == 'output']
        output_features = []
        for f in sorted(output_files, key=lambda x: x.get('feature_index') or 0):
            output_features.append({
                'bezeichnung': f['bezeichnung'],
                'feature_index': f.get('feature_index') or 0,
                'type': 'output',
                'data_type': 'time horizon',
                'horizon_start_h': float(f.get('zeithorizont_start') or 0),
                'horizon_end_h': float(f.get('zeithorizont_end') or 0),
            })

        time_info = time_res.data[0] if time_res.data else {}
        zeitschritte = zeit_res.data[0]

        # user_data is validated and set before the try block (W12-F7 pre-pipeline check).
        user_csvs = {}
        for name, data in user_data.items():
            if isinstance(data, list):
                # Legacy format: list of dicts [{"UTC": "...", "name": val}, ...]
                df = pd.DataFrame(data)
            elif isinstance(data, dict):
                # New format: dict of arrays {"UTC": [...], "name": [...]}
                df = pd.DataFrame(data)
            else:
                continue
            if 'UTC' in df.columns:
                df['UTC'] = pd.to_datetime(df['UTC'])
            user_csvs[name] = df

        config = {
            'input_features': input_features,
            'output_features': output_features,
            'time_info': time_info,
            'zeitschritte': zeitschritte,
        }

        result = run_forecast(
            session_id=uuid_session_id,
            user_id=g.user_id,
            config=config,
            user_csvs=user_csvs
        )

        return jsonify({
            'success': True,
            **result,
            'session_id': session_id
        })

    except ValueError as e:
        error_msg = str(e)
        if 'Missing' in error_msg or 'No data' in error_msg:
            return jsonify({'success': False, 'error': error_msg, 'code': 'MISSING_USER_DATA'}), 400
        if 'NaN' in error_msg:
            return jsonify({'success': False, 'error': error_msg, 'code': 'INTERPOLATION_ERROR'}), 422
        return jsonify({'success': False, 'error': error_msg, 'code': 'VALIDATION_ERROR'}), 400

    except RuntimeError as e:
        error_msg = str(e)
        if 'API' in error_msg:
            return jsonify({'success': False, 'error': error_msg, 'code': 'EXTERN_API_ERROR'}), 502
        return jsonify({'success': False, 'error': error_msg}), 500

    except FileNotFoundError as e:
        return jsonify({'success': False, 'error': str(e), 'code': 'MODEL_NOT_FOUND'}), 404

    except Exception as e:
        logger.error(f"Forecast error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)}), 500


@bp.route('/forecast-config/<session_id>', methods=['GET'])
@require_auth
def get_forecast_config(session_id):
    """Get session configuration needed for forecast UI."""
    try:
        supabase = get_supabase_client(use_service_role=True)
        uuid_session_id = str(create_or_get_session_uuid(session_id, user_id=g.user_id))

        try:
            assert_session_ownership(uuid_session_id)
        except SessionOwnershipError:
            return jsonify({'success': False, 'error': 'forbidden'}), 403

        files_res = supabase.table('files') \
            .select('*') \
            .eq('session_id', uuid_session_id) \
            .execute()

        time_res = supabase.table('time_info') \
            .select('*') \
            .eq('session_id', uuid_session_id) \
            .execute()

        zeit_res = supabase.table('zeitschritte') \
            .select('*') \
            .eq('session_id', uuid_session_id) \
            .execute()

        from utils.model_storage import list_session_models
        models = list_session_models(uuid_session_id)

        return jsonify({
            'success': True,
            'files': files_res.data,
            'time_info': time_res.data[0] if time_res.data else None,
            'zeitschritte': zeit_res.data[0] if zeit_res.data else None,
            'models': models,
            'session_id': session_id
        })

    except Exception as e:
        logger.error(f"Error getting forecast config: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@bp.route('/save-forecast-config/<session_id>', methods=['POST'])
@require_auth
def save_forecast_config(session_id):
    """Save forecast configuration into files table for a session."""
    try:
        data = request.get_json()
        if not data or 'features' not in data:
            return jsonify({'success': False, 'error': 'Missing features in body'}), 400

        supabase = get_supabase_client(use_service_role=True)
        uuid_session_id = str(create_or_get_session_uuid(session_id, user_id=g.user_id))

        try:
            assert_session_ownership(uuid_session_id)
        except SessionOwnershipError:
            return jsonify({'success': False, 'error': 'forbidden'}), 403

        features = data['features']
        updated = 0

        for feat in features:
            file_id = feat.get('id')
            if not file_id:
                continue

            update_data = {}
            if 'data_source' in feat:
                update_data['data_source'] = feat['data_source']
            if 'api_source' in feat:
                update_data['api_source'] = feat['api_source']
            if 'fcst_var' in feat:
                update_data['fcst_var'] = feat['fcst_var']
            if 'latitude' in feat:
                update_data['latitude'] = feat['latitude']
            if 'longitude' in feat:
                update_data['longitude'] = feat['longitude']
            if 'feature_index' in feat:
                update_data['feature_index'] = feat['feature_index']

            if update_data:
                supabase.table('files') \
                    .update(update_data) \
                    .eq('id', file_id) \
                    .eq('session_id', uuid_session_id) \
                    .execute()
                updated += 1

        if 'zeitzone' in data:
            supabase.table('time_info') \
                .update({'zeitzone': data['zeitzone']}) \
                .eq('session_id', uuid_session_id) \
                .execute()

        return jsonify({
            'success': True,
            'updated_files': updated,
            'session_id': session_id
        })

    except Exception as e:
        logger.error(f"Error saving forecast config: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@bp.route('/api-parameters', methods=['GET'])
@require_auth
def get_api_parameters():
    """List available API parameters for dropdowns."""
    try:
        api_source = request.args.get('api_source')
        supabase = get_supabase_client(use_service_role=True)

        query = supabase.table('api_parameters').select('*')
        if api_source:
            query = query.eq('api_source', api_source)

        result = query.execute()

        return jsonify({
            'success': True,
            'parameters': result.data
        })

    except Exception as e:
        logger.error(f"Error getting API parameters: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500
