"""
Model Management Service
Business logic for saving, listing, and downloading trained models

This service handles:
- Extracting serialized models from training results
- Deserializing and saving models to files (.h5, .pkl)
- Uploading models to Supabase Storage
- Listing available models for a session
- Downloading model files from Storage

Created: 2025-10-24
Phase 3 of training.py refactoring
"""

import os
import logging
import tempfile
import base64
import pickle
import io
from typing import Dict, List, Tuple
from tensorflow import keras

logger = logging.getLogger(__name__)


def extract_serialized_models(training_results: Dict) -> List[Dict]:
    """
    Recursively extract serialized models from training results.

    Args:
        training_results: Training results dictionary containing serialized models

    Returns:
        List of dicts with model information:
        [{
            'model_class': str,
            'model_data': str (base64),
            'path': str,
            'data_size': int
        }, ...]
    """
    def extract_models_info(obj, path=""):
        """Recursively extract serialized models"""
        found_models = []
        if isinstance(obj, dict):
            if '_model_type' in obj and obj.get('_model_type') == 'serialized_model':
                model_class = obj.get('_model_class', 'Unknown')
                model_data = obj.get('_model_data', '')

                found_models.append({
                    'model_class': model_class,
                    'model_data': model_data,
                    'path': path,
                    'data_size': len(model_data)
                })

            for key, value in obj.items():
                found_models.extend(extract_models_info(value, f"{path}.{key}" if path else key))
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                found_models.extend(extract_models_info(item, f"{path}[{i}]" if path else f"[{i}]"))

        return found_models

    return extract_models_info(training_results)


def save_models_to_storage(session_id: str, user_id: str = None) -> Dict:
    """
    Save trained models from training results to Supabase Storage.

    Extracts serialized models from training results, deserializes them,
    saves as .h5 (Keras) or .pkl (sklearn) files, and uploads to Storage.

    Args:
        session_id: Training session ID
        user_id: User ID for ownership validation (required for security)

    Returns:
        dict: {
            'uploaded_models': [list of uploaded model info],
            'failed_models': [list of failed models],
            'total_uploaded': int,
            'total_failed': int
        }

    Raises:
        ValueError: If session not found, no training results, or no models
        PermissionError: If session doesn't belong to the user
    """
    from shared.database.operations import create_or_get_session_uuid
    from utils.training_storage import fetch_training_results_with_storage
    from utils.model_storage import upload_trained_model

    logger.info(f"📦 Saving models to storage - session: {session_id}")

    uuid_session_id = create_or_get_session_uuid(session_id, user_id=user_id)
    if not uuid_session_id:
        raise ValueError(f'Session {session_id} not found')

    training_results = fetch_training_results_with_storage(session_id)

    if not training_results:
        raise ValueError('No training results found')

    serialized_models = extract_serialized_models(training_results)

    if not serialized_models:
        raise ValueError('No trained models found in results')

    logger.info(f"Found {len(serialized_models)} serialized model(s) in training results")

    uploaded_models = []
    failed_models = []

    for idx, model_info in enumerate(serialized_models):
        model_class = model_info['model_class']
        model_data = model_info['model_data']
        path = model_info['path']

        try:
            logger.info(f"📥 Deserializing model {idx + 1}/{len(serialized_models)}: {model_class}")

            model_bytes = base64.b64decode(model_data)
            model_obj = pickle.loads(model_bytes)

            logger.info(f"✅ Model deserialized successfully: {model_class}")

            is_keras_model = hasattr(model_obj, 'save') and hasattr(model_obj, 'predict')
            file_extension = '.h5' if is_keras_model else '.pkl'

            with tempfile.NamedTemporaryFile(suffix=file_extension, delete=False, mode='wb') as temp_file:
                temp_file_path = temp_file.name

            try:
                if is_keras_model:
                    model_obj.save(temp_file_path)
                    logger.info(f"💾 Saved Keras model to {temp_file_path}")
                else:
                    with open(temp_file_path, 'wb') as f:
                        pickle.dump(model_obj, f)
                    logger.info(f"💾 Saved sklearn model to {temp_file_path}")

                logger.info(f"📤 Uploading {model_class} model to storage...")

                dataset_name = path.split('.')[0] if '.' in path else 'default'

                storage_result = upload_trained_model(
                    session_id=str(uuid_session_id),
                    model_file_path=temp_file_path,
                    model_type=model_class,
                    dataset_name=dataset_name
                )

                uploaded_models.append({
                    'dataset_name': dataset_name,
                    'model_type': model_class.upper(),
                    'filename': storage_result['original_filename'],
                    'storage_path': storage_result['file_path'],
                    'size_mb': round(storage_result['file_size'] / (1024 * 1024), 2),
                    'path_in_results': path,
                    'file_format': 'h5' if is_keras_model else 'pkl'
                })

                logger.info(f"✅ Uploaded {model_class} model: {storage_result['file_path']}")

            finally:
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)
                    logger.debug(f"🗑️ Cleaned up temporary file: {temp_file_path}")

        except Exception as model_error:
            logger.error(f"❌ Failed to process {model_class}: {model_error}")
            import traceback
            logger.error(traceback.format_exc())
            failed_models.append({
                'model_class': model_class,
                'path': path,
                'error': str(model_error)
            })

    if not uploaded_models and failed_models:
        raise Exception('All model uploads failed')

    logger.info(f"✅ Saved {len(uploaded_models)} models to storage for session {session_id}")

    return {
        'uploaded_models': uploaded_models,
        'failed_models': failed_models,
        'total_uploaded': len(uploaded_models),
        'total_failed': len(failed_models)
    }


def get_models_list(session_id: str, user_id: str = None) -> List[Dict]:
    """
    List all trained models stored in Supabase Storage for a session.

    Args:
        session_id: Training session ID

    Returns:
        List of model info dicts

    Raises:
        ValueError: If session not found
    """
    from shared.database.operations import create_or_get_session_uuid
    from utils.model_storage import list_session_models

    logger.info(f"📋 Listing models from Storage - session: {session_id}")

    uuid_session_id = create_or_get_session_uuid(session_id, user_id=user_id)
    if not uuid_session_id:
        raise ValueError(f'Session {session_id} not found')

    models = list_session_models(str(uuid_session_id))

    logger.info(f"✅ Found {len(models)} model(s) in Storage for session {session_id}")

    return models


def download_model_file(session_id: str, filename: str = None) -> Tuple[bytes, str]:
    """
    Download a trained model file from Supabase Storage.

    Args:
        session_id: Training session ID
        filename: Specific model filename to download (optional)
                 If not specified, downloads first .h5 model found

    Returns:
        Tuple of (file_data: bytes, filename: str)

    Raises:
        ValueError: If session not found, no models, or specific model not found
    """
    from shared.database.operations import create_or_get_session_uuid
    from utils.model_storage import list_session_models, download_trained_model

    logger.info(f"📥 Download request - session: {session_id}, filename: {filename or 'first .h5 model'}")

    uuid_session_id = create_or_get_session_uuid(session_id, user_id=None)
    if not uuid_session_id:
        raise ValueError(f'Session {session_id} not found')

    models = list_session_models(str(uuid_session_id))

    if not models:
        raise ValueError('No models found for this session')

    target_model = None
    if filename:
        target_model = next((m for m in models if m['filename'] == filename), None)
        if not target_model:
            raise ValueError(f'Model {filename} not found')
    else:
        h5_models = [m for m in models if m['format'] == 'h5']
        if not h5_models:
            raise ValueError('No .h5 models found for this session')
        target_model = h5_models[0]

    logger.info(f"📥 Downloading model: {target_model['filename']}")

    file_data = download_trained_model(
        session_id=str(uuid_session_id),
        file_path=target_model['storage_path']
    )

    logger.info(f"✅ Model downloaded successfully: {target_model['filename']}")

    return (file_data, target_model['filename'])
