"""
Model routes for training API.

Contains 6 endpoints for model storage and download:
- scalers/<session_id> GET
- scalers/<session_id>/download GET
- scale-data/<session_id> POST
- save-model/<session_id> POST
- list-models-database/<session_id> GET
- download-model-h5/<session_id> GET
"""

import io
from flask import Blueprint, send_file

from .common import (
    datetime, request, jsonify, g, logging,
    require_auth, require_subscription,
    get_logger
)

from domains.training.ml.scaler import (
    get_session_scalers, create_scaler_download_package, scale_new_data
)
from domains.training.ml.models import (
    save_models_to_storage, get_models_list, download_model_file
)

bp = Blueprint('training_models', __name__)
logger = get_logger(__name__)


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
@require_subscription
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
@require_subscription
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


@bp.route('/predict/<session_id>', methods=['POST'])
@require_auth
@require_subscription
def predict_with_model(session_id):
    """
    Make predictions using a trained model from session.

    Request body:
    {
        "model_filename": "best_model.h5",
        "input_data": [{"feature1": 1.5, "feature2": 2.3}, ...]
    }

    Response:
    {
        "success": true,
        "predictions": [1.234, 2.345, ...],
        "model_used": "best_model.h5",
        "timestamp": "2024-...",
        "input_count": 2
    }
    """
    try:
        data = request.json
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400

        model_filename = data.get('model_filename')
        if not model_filename:
            return jsonify({'success': False, 'error': 'model_filename is required'}), 400

        input_data = data.get('input_data')
        if not input_data:
            return jsonify({'success': False, 'error': 'input_data is required'}), 400

        if not isinstance(input_data, list):
            return jsonify({'success': False, 'error': 'input_data must be a list'}), 400

        # Get user_id from auth context
        user_id = g.user_id if hasattr(g, 'user_id') else 'anonymous'

        # Optional: disable scaling
        apply_scaling = data.get('apply_scaling', True)

        # Import and use prediction service
        from domains.training.services.prediction_service import PredictionService

        service = PredictionService(session_id, user_id)
        result = service.predict(
            model_filename=model_filename,
            input_data=input_data,
            apply_scaling=apply_scaling
        )

        return jsonify({
            'success': True,
            'predictions': result['predictions'],
            'model_used': result['model_used'],
            'timestamp': result['timestamp'],
            'input_count': result['input_count'],
            'scaling_applied': result['scaling_applied']
        })

    except FileNotFoundError as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'code': 'MODEL_NOT_FOUND'
        }), 404

    except ValueError as e:
        error_msg = str(e)
        if 'No training results' in error_msg or 'No scalers' in error_msg:
            return jsonify({
                'success': False,
                'error': error_msg,
                'code': 'SCALERS_NOT_FOUND'
            }), 404
        return jsonify({
            'success': False,
            'error': error_msg,
            'code': 'VALIDATION_ERROR'
        }), 400

    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'Failed to make prediction'
        }), 500
