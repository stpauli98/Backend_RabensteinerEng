"""
Model routes for training API.

Contains 7 endpoints for model storage, download and inference:
- scalers/<session_id> GET
- scalers/<session_id>/download GET
- scale-data/<session_id> POST
- save-model/<session_id> POST
- list-models-database/<session_id> GET
- download-model-h5/<session_id> GET
- predict/<session_id> POST
"""

import io
from flask import Blueprint, send_file

from .common import (
    datetime, request, jsonify, g, logging,
    require_auth, require_subscription,
    get_logger
)

from core.rate_limits import limiter, training_limit_string
from shared.responses.errors import error_response as _err
from shared.storage.errors import is_storage_not_found
from shared.validators.uuid import validate_training_session_format

from domains.training.ml.scaler import (
    get_session_scalers, create_scaler_download_package, scale_new_data
)
from domains.training.ml.models import (
    save_models_to_storage, get_models_list, download_model_file
)

bp = Blueprint('training_models', __name__)
logger = get_logger(__name__)


@bp.route('/scalers/<session_id>', methods=['GET'])
@limiter.limit(training_limit_string)
@require_auth
def get_scalers(session_id):
    """Retrieve saved scalers from database for a specific session."""
    # W11-BE2: validate session_id BEFORE auth-ed work.
    err = validate_training_session_format(session_id)
    if err:
        return err

    try:
        scalers_data = get_session_scalers(session_id)

        return jsonify({
            'success': True,
            'session_id': session_id,
            'scalers': scalers_data
        })

    except ValueError:
        logger.exception("Scalers not found for session")
        return _err('SCALER_NOT_FOUND', 'Scalers not found for this session', 404)
    except Exception:
        logger.exception("Failed to retrieve scalers from database")
        return _err(
            'INTERNAL_ERROR',
            'Failed to retrieve scalers from database',
            500,
            suggestion='Please try again. If the problem persists, contact support.',
        )


@bp.route('/scalers/<session_id>/download', methods=['GET'])
@limiter.limit(training_limit_string)
@require_auth
def download_scalers_as_save_files(session_id):
    """Download scalers as .save files."""
    # W11-BE2: validate session_id BEFORE auth-ed work.
    err = validate_training_session_format(session_id)
    if err:
        return err

    try:
        zip_file_path = create_scaler_download_package(session_id)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        return send_file(
            zip_file_path,
            as_attachment=True,
            download_name=f'scalers_{session_id}_{timestamp}.zip',
            mimetype='application/zip'
        )

    except (ValueError, FileNotFoundError):
        logger.warning("Scaler download: artifacts not found for session", exc_info=True)
        return _err('SCALER_NOT_FOUND', 'Scalers not found for this session', 404)
    except Exception as e:
        # Mirror T4 polish on /download-arrays: distinguish 404 (storage
        # object missing) from real 500 (Supabase outage, network failure).
        if is_storage_not_found(e):
            logger.warning("Scaler download: storage reports not found", exc_info=True)
            return _err('SCALER_NOT_FOUND', 'Scalers not found for this session', 404)
        logger.exception("Scaler download unexpected failure")
        return _err(
            'INTERNAL_ERROR',
            'Failed to create scaler download package',
            500,
            suggestion='Please try again. If the problem persists, contact support.',
        )


@bp.route('/scale-data/<session_id>', methods=['POST'])
@limiter.limit(training_limit_string)
@require_auth
@require_subscription
def scale_input_data(session_id):
    """Scale input data using saved scalers."""
    # W11-BE2: validate session_id BEFORE auth-ed work.
    err = validate_training_session_format(session_id)
    if err:
        return err

    try:
        data = request.get_json(silent=True)
        if not data:
            return _err('MISSING_BODY', 'Request body is required', 400)

        input_data = data.get('input_data')
        if input_data is None:
            return _err(
                'MISSING_PREDICTION_INPUT',
                'input_data field is required',
                400,
            )

        save_scaled = data.get('save_scaled', False)

        result = scale_new_data(session_id, input_data, save_scaled)

        return jsonify({
            'success': True,
            'session_id': session_id,
            'scaled_data': result['scaled_data'],
            'scaling_info': result['scaling_info'],
            'metadata': result['metadata']
        })

    except ValueError:
        logger.exception("Scale-data: scalers not found for session")
        return _err('SCALER_NOT_FOUND', 'Scalers not found for this session', 404)
    except Exception:
        logger.exception("Failed to scale input data")
        return _err(
            'INTERNAL_ERROR',
            'Failed to scale input data',
            500,
            suggestion='Please try again. If the problem persists, contact support.',
        )


@bp.route('/save-model/<session_id>', methods=['POST'])
@limiter.limit(training_limit_string)
@require_auth
@require_subscription
def save_model(session_id):
    """Save trained models to Supabase Storage."""
    # W11-BE2: validate session_id BEFORE auth-ed work.
    err = validate_training_session_format(session_id)
    if err:
        return err

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
        # Classify ValueErrors raised by save_models_to_storage. We do NOT
        # echo error_msg back to the caller — it may contain DB / storage
        # internals. We only use it to pick the right error code.
        if ('Session' in error_msg and 'not found' in error_msg) or \
           'No training results' in error_msg or \
           'No trained models' in error_msg:
            logger.warning("Save model: prerequisite missing", exc_info=True)
            return _err(
                'MODEL_NOT_FOUND',
                'No trained models available to save for this session',
                404,
                suggestion='Train a model first before attempting to save.',
            )
        logger.warning("Save model: invalid request state", exc_info=True)
        return _err('BAD_REQUEST', 'Unable to save models for this session', 400)

    except Exception:
        logger.exception("Failed to save models to storage")
        return _err(
            'MODEL_SAVE_ERROR',
            'Failed to save models to storage',
            500,
            suggestion='Please try again. If the problem persists, contact support.',
        )


@bp.route('/list-models-database/<session_id>', methods=['GET'])
@limiter.limit(training_limit_string)
@require_auth
def list_models_database(session_id):
    """List all trained models stored in Supabase Storage for a session."""
    # W11-BE2: validate session_id BEFORE auth-ed work.
    err = validate_training_session_format(session_id)
    if err:
        return err

    try:
        models = get_models_list(session_id, user_id=g.user_id)

        return jsonify({
            'success': True,
            'data': {
                'models': models,
                'count': len(models)
            },
            'session_id': session_id
        })

    except ValueError:
        logger.warning("List models: no models found for session", exc_info=True)
        return _err('MODEL_NOT_FOUND', 'No models found for this session', 404)

    except Exception:
        logger.exception("Failed to list models from storage")
        return _err(
            'INTERNAL_ERROR',
            'Failed to list models from storage',
            500,
            suggestion='Please try again. If the problem persists, contact support.',
        )


@bp.route('/download-model-h5/<session_id>', methods=['GET'])
@limiter.limit(training_limit_string)
@require_auth
def download_model_h5(session_id):
    """Download a trained model file from Supabase Storage."""
    # W11-BE2: validate session_id BEFORE auth-ed work.
    err = validate_training_session_format(session_id)
    if err:
        return err

    try:
        filename = request.args.get('filename')
        file_data, file_name = download_model_file(session_id, filename)
        file_obj = io.BytesIO(file_data)

        response = send_file(
            file_obj,
            as_attachment=True,
            download_name=file_name,
            mimetype='application/octet-stream'
        )

        import sys
        import tensorflow as tf
        import numpy as np
        import keras
        response.headers['X-Model-Env-Python'] = (
            f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        )
        response.headers['X-Model-Env-TensorFlow'] = tf.__version__
        response.headers['X-Model-Env-Keras'] = keras.__version__
        response.headers['X-Model-Env-Numpy'] = np.__version__

        return response

    except (ValueError, FileNotFoundError):
        logger.warning("Download model: artifact not found", exc_info=True)
        return _err('MODEL_NOT_FOUND', 'Model file not found for this session', 404)

    except Exception as e:
        # Mirror T4 polish on /download-arrays: distinguish 404 (storage
        # object missing) from real 500 (Supabase outage, network failure).
        if is_storage_not_found(e):
            logger.warning("Download model: storage reports not found", exc_info=True)
            return _err('MODEL_NOT_FOUND', 'Model file not found for this session', 404)
        logger.exception("Download model unexpected failure")
        return _err(
            'INTERNAL_ERROR',
            'Failed to download model from storage',
            500,
            suggestion='Please try again. If the problem persists, contact support.',
        )


@bp.route('/predict/<session_id>', methods=['POST'])
@limiter.limit(training_limit_string)
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
    # W11-BE2: validate session_id BEFORE auth-ed work.
    err = validate_training_session_format(session_id)
    if err:
        return err

    try:
        data = request.get_json(silent=True)
        if not data:
            return _err('MISSING_BODY', 'Request body is required', 400)

        model_filename = data.get('model_filename')
        if not model_filename:
            return _err(
                'MISSING_PREDICTION_INPUT',
                'model_filename is required',
                400,
            )

        input_data = data.get('input_data')
        if not input_data:
            return _err(
                'MISSING_PREDICTION_INPUT',
                'input_data is required',
                400,
            )

        if not isinstance(input_data, list):
            return _err(
                'MISSING_PREDICTION_INPUT',
                'input_data must be a list',
                400,
            )

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

    except FileNotFoundError:
        logger.warning("Predict: model artifact not found", exc_info=True)
        return _err('MODEL_NOT_FOUND', 'Model file not found for this session', 404)

    except ValueError as e:
        error_msg = str(e)
        if 'No training results' in error_msg or 'No scalers' in error_msg:
            logger.warning("Predict: scalers not found", exc_info=True)
            return _err('SCALER_NOT_FOUND', 'Scalers not found for this session', 404)
        logger.warning("Predict: validation error", exc_info=True)
        return _err('BAD_REQUEST', 'Invalid prediction request', 400)

    except Exception:
        logger.exception("Failed to make prediction")
        return _err(
            'PREDICTION_ERROR',
            'Failed to make prediction',
            500,
            suggestion='Please try again. If the problem persists, contact support.',
        )
