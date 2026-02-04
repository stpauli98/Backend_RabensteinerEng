"""
Training routes for training API.

Contains 5 endpoints for ML model training:
- generate-datasets/<session_id>
- train-models/<session_id>
- status/<session_id>
- results/<session_id>
- get-training-results/<session_id>
"""

import threading
import numpy as np
from flask import Blueprint, current_app, Response

from .common import (
    request, jsonify, g, logging,
    require_auth, require_subscription, check_processing_limit, check_training_limit,
    get_supabase_client, create_or_get_session_uuid,
    increment_processing_count, increment_training_count,
    create_error_response,
    get_string_session_id, get_uuid_session_id,
    get_logger
)

from domains.training.services.visualization import save_visualization_to_database, delete_old_violin_plots
from domains.training.data.generator import generate_violin_plots_for_session
from domains.training.services.orchestrator import run_model_training_async

bp = Blueprint('training_training', __name__)
logger = get_logger(__name__)


def convert_numpy_to_native(obj):
    """Recursively convert NumPy arrays and types to native Python types for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {k: convert_numpy_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_to_native(item) for item in obj]
    return obj


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
        from shared.database.lifecycle import finalize_session as db_finalize_session

        # Get UUID for database persistence
        uuid_session_id = create_or_get_session_uuid(session_id, g.user_id)

        socketio = current_app.extensions.get('socketio')
        progress_tracker = ViolinProgressTracker(socketio, session_id)

        result = generate_violin_plots_for_session(
            session_id=session_id,
            model_parameters=model_parameters,
            training_split=training_split,
            progress_tracker=progress_tracker,
            uuid_session_id=uuid_session_id
        )

        violin_plots = result.get('violin_plots', {})
        if violin_plots:
            progress_tracker.saving_to_database()

            # Clean up old violin plots before saving new ones
            # This is necessary because the new structure has different names
            deleted_count = delete_old_violin_plots(session_id)
            if deleted_count > 0:
                logger.info(f"Deleted {deleted_count} old violin plots for session {session_id}")

            for plot_name, plot_data in violin_plots.items():
                try:
                    if plot_data:
                        save_visualization_to_database(session_id, plot_name, plot_data)
                except Exception as viz_error:
                    logger.error(f"Failed to save visualization {plot_name}: {str(viz_error)}")

        # Save n_dat to sessions table
        n_dat = result.get('n_dat', 0)
        if n_dat > 0:
            try:
                db_finalize_session(uuid_session_id, n_dat=n_dat)
            except Exception as e:
                logger.error(f"Failed to save n_dat to database: {e}")

        progress_tracker.complete()

        increment_processing_count(g.user_id)
        logger.info(f"Tracked dataset generation for user {g.user_id}")

        return jsonify({
            'success': True,
            'message': 'Datasets generated successfully',
            'dataset_count': result.get('n_dat', 0),
            'n_dat': result.get('n_dat', 0),
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
            return create_error_response('No data provided', 400)

        # Razriješi session ID u string format za lokalni pristup
        try:
            string_session_id = get_string_session_id(session_id)
            if string_session_id != session_id:
                logger.info(f"Converted UUID session {session_id} to string session {string_session_id}")
            session_id = string_session_id
        except ValueError:
            pass  # Zadrži originalni session_id ako nije UUID

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

        # NOTE: Old results are now cleaned up AFTER successful training in orchestrator.py
        # This prevents orphan storage files when training fails

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


@bp.route('/results/<session_id>', methods=['GET'])
@require_auth
def get_training_results(session_id):
    """Get training results for a session."""
    try:
        from utils.training_storage import download_training_results
        supabase = get_supabase_client(use_service_role=True)
        uuid_session_id = create_or_get_session_uuid(session_id, g.user_id)

        # Fetch n_dat from sessions table
        n_dat = 0
        try:
            session_response = supabase.table('sessions').select('n_dat').eq('id', uuid_session_id).single().execute()
            if session_response.data:
                n_dat = session_response.data.get('n_dat', 0) or 0
        except Exception as e:
            logger.warning(f"Could not fetch n_dat for session {session_id}: {e}")

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

                    full_results = download_training_results(
                        file_path=record['results_file_path'],
                        decompress=record.get('compressed', False)
                    )
                    record['results'] = convert_numpy_to_native(full_results)

                except Exception as download_error:
                    logger.error(f"Failed to download results from storage: {download_error}")
                    record['results'] = convert_numpy_to_native(record.get('results_metadata', {}))
            else:
                legacy_response = supabase.table('training_results')\
                    .select('results')\
                    .eq('id', record['id'])\
                    .single()\
                    .execute()
                if legacy_response.data and legacy_response.data.get('results'):
                    record['results'] = convert_numpy_to_native(legacy_response.data['results'])
                else:
                    record['results'] = convert_numpy_to_native(record.get('results_metadata', {}))

            return jsonify({
                'success': True,
                'results': [record],
                'count': 1,
                'n_dat': n_dat
            })
        else:
            return jsonify({
                'success': True,
                'message': 'No training results yet - training may not have been started',
                'results': [],
                'count': 0,
                'n_dat': n_dat
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


@bp.route('/download-arrays/<session_id>', methods=['GET'])
@require_auth
def download_training_arrays(session_id):
    """Download i_array_3D and o_array_3D as .pkl.gz from Supabase Storage."""
    try:
        from shared.database.client import get_supabase_admin_client

        uuid_session_id = create_or_get_session_uuid(session_id, g.user_id)
        file_path = f"{uuid_session_id}/training_arrays.pkl.gz"

        supabase = get_supabase_admin_client()
        data = supabase.storage.from_('training-results').download(file_path)

        return Response(
            data,
            mimetype='application/gzip',
            headers={
                'Content-Disposition': 'attachment; filename=training_arrays.pkl.gz',
                'Content-Length': str(len(data))
            }
        )

    except Exception as e:
        logger.error(f"Error downloading training arrays for {session_id}: {e}")
        return jsonify({'success': False, 'error': str(e)}), 404
