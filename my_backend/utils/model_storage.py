"""
Model Storage Module
Handles upload/download of trained models (.h5 files) to/from Supabase Storage

This module provides functions to:
- Upload trained model files to Storage bucket
- Download model files from Storage bucket
- List all models for a session

Created: 2025-10-23
Purpose: Enable persistent storage of trained models in Supabase Storage
"""

import os
import logging
from datetime import datetime
from typing import Dict, List, Optional
from shared.database.client import get_supabase_admin_client

logger = logging.getLogger(__name__)


def upload_trained_model(
    session_id: str,
    model_file_path: str,
    model_type: str,
    dataset_name: str
) -> Dict[str, any]:
    """
    Upload a trained model (.h5 file) to Supabase Storage

    Args:
        session_id: UUID session ID
        model_file_path: Local path to the .h5 model file
        model_type: Type of model (e.g., 'dense', 'cnn', 'lstm')
        dataset_name: Name of the dataset used for training

    Returns:
        dict: {
            'file_path': str,       # Path in bucket: "session_id/model_type_dataset_timestamp.h5"
            'file_size': int,       # Size in bytes
            'bucket': str,          # Bucket name
            'model_type': str,      # Model type
            'dataset_name': str     # Dataset name
        }

    Raises:
        Exception: If upload fails

    Example:
        >>> result = upload_trained_model(
        ...     session_id="abc-123",
        ...     model_file_path="/app/uploads/trained_models/dense_dataset1_20251023.h5",
        ...     model_type="dense",
        ...     dataset_name="dataset1"
        ... )
        >>> print(result['file_path'])
        "abc-123/dense_dataset1_20251023_123456.h5"
    """
    try:
        supabase = get_supabase_admin_client()

        if not os.path.exists(model_file_path):
            raise FileNotFoundError(f"Model file not found: {model_file_path}")

        file_size = os.path.getsize(model_file_path)
        filename = os.path.basename(model_file_path)

        file_extension = os.path.splitext(model_file_path)[1]
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        storage_filename = f"{model_type}_{dataset_name}_{timestamp}{file_extension}"
        file_path = f"{session_id}/{storage_filename}"

        with open(model_file_path, 'rb') as f:
            model_data = f.read()

        bucket_name = 'trained-models'

        try:
            response = supabase.storage.from_(bucket_name).upload(
                path=file_path,
                file=model_data,
                file_options={
                    "content-type": "application/octet-stream",
                    "cache-control": "3600",
                    "upsert": "true"
                }
            )

            return {
                'file_path': file_path,
                'file_size': file_size,
                'bucket': bucket_name,
                'model_type': model_type,
                'dataset_name': dataset_name,
                'original_filename': filename
            }

        except Exception as upload_error:
            if "Bucket not found" in str(upload_error):
                try:
                    supabase.storage.create_bucket(
                        bucket_name,
                        options={
                            "public": False,
                            "file_size_limit": 524288000
                        }
                    )

                    response = supabase.storage.from_(bucket_name).upload(
                path=file_path,
                file=model_data,
                file_options={
                    "content-type": "application/octet-stream",
                    "cache-control": "3600",
                    "upsert": "true"
                }
            )

                    return {
                        'file_path': file_path,
                        'file_size': file_size,
                        'bucket': bucket_name,
                        'model_type': model_type,
                        'dataset_name': dataset_name,
                        'original_filename': filename
                    }

                except Exception as create_error:
                    logger.error(f"❌ Failed to create bucket: {create_error}")
                    raise
            else:
                raise upload_error

    except Exception as e:
        logger.error(f"❌ Error uploading model: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise


def download_trained_model(
    session_id: str,
    file_path: str
) -> bytes:
    """
    Download a trained model from Supabase Storage

    Args:
        session_id: UUID session ID
        file_path: Path in bucket (e.g., "session_id/model.h5")

    Returns:
        bytes: Model file data

    Raises:
        Exception: If download fails
    """
    try:
        supabase = get_supabase_admin_client()
        bucket_name = 'trained-models'

        response = supabase.storage.from_(bucket_name).download(file_path)

        return response

    except Exception as e:
        logger.error(f"❌ Error downloading model: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise


def load_model_from_storage(session_id: str, model_filename: str):
    """
    Download and load a model from Supabase Storage.
    
    Args:
        session_id: UUID session ID
        model_filename: Name of model file (e.g., 'best_model.h5')
        
    Returns:
        Loaded model object (Keras or sklearn)
        
    Raises:
        FileNotFoundError: If model not found in storage
        ValueError: If model format not supported
    """
    import tempfile
    import os
    import io
    
    # Construct storage path
    file_path = f"{session_id}/{model_filename}"

    try:
        # Download model bytes
        model_bytes = download_trained_model(session_id, file_path)
        
        # Determine model type by extension
        if model_filename.endswith('.h5') or model_filename.endswith('.keras'):
            # Keras model - requires temp file
            suffix = '.h5' if model_filename.endswith('.h5') else '.keras'
            
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_file:
                tmp_file.write(model_bytes)
                tmp_path = tmp_file.name
            
            try:
                import tensorflow as tf
                from tensorflow import keras
                
                # Suppress TF warnings
                tf.get_logger().setLevel('ERROR')
                
                # Load the model
                model = keras.models.load_model(tmp_path, compile=False)

                return model
                
            finally:
                # Cleanup temp file
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                    
        elif model_filename.endswith('.pkl') or model_filename.endswith('.joblib'):
            # sklearn model
            import joblib
            
            model = joblib.load(io.BytesIO(model_bytes))

            return model
            
        else:
            raise ValueError(f"Unsupported model format: {model_filename}")
            
    except Exception as e:
        logger.error(f"❌ Error loading model {model_filename}: {e}")
        raise


def list_session_models(session_id: str) -> List[Dict]:
    """
    List all trained models for a session

    Args:
        session_id: UUID session ID

    Returns:
        List of model info dicts
    """
    try:
        supabase = get_supabase_admin_client()
        bucket_name = 'trained-models'

        files = supabase.storage.from_(bucket_name).list(session_id)

        models = []
        for file_info in files:
            filename = file_info['name']
            if filename.endswith('.h5') or filename.endswith('.pkl') or filename.endswith('.save'):
                parts = filename.rsplit('.', 1)
                file_extension = parts[1] if len(parts) > 1 else ''

                name_parts = parts[0].split('_')
                model_type = name_parts[0] if len(name_parts) > 0 else 'Unknown'
                dataset_name = name_parts[1] if len(name_parts) > 1 else 'default'

                size_bytes = file_info.get('metadata', {}).get('size', 0)
                size_mb = round(size_bytes / (1024 * 1024), 2) if size_bytes > 0 else 0.0

                models.append({
                    'filename': filename,
                    'model_type': model_type,
                    'dataset_name': dataset_name,
                    'format': file_extension,
                    'storage_path': f"{session_id}/{filename}",
                    'file_size_mb': size_mb,
                    'modified_time': file_info.get('updated_at')
                })

        return models

    except Exception as e:
        logger.error(f"❌ Error listing models: {e}")
        return []


def delete_session_models(session_id: str) -> Dict:
    """
    Delete all trained models for a session from Supabase Storage.

    This should be called before uploading new models to ensure
    old models are removed and don't accumulate.

    Args:
        session_id: UUID session ID

    Returns:
        dict: {
            'deleted_count': int,
            'deleted_files': list of filenames,
            'errors': list of error messages
        }
    """
    try:
        supabase = get_supabase_admin_client()
        bucket_name = 'trained-models'

        # List all files in the session folder
        files = supabase.storage.from_(bucket_name).list(session_id)

        if not files:
            return {
                'deleted_count': 0,
                'deleted_files': [],
                'errors': []
            }

        deleted_files = []
        errors = []

        # Delete each model file (.h5, .pkl, and .save)
        for file_info in files:
            filename = file_info['name']
            if filename.endswith('.h5') or filename.endswith('.pkl') or filename.endswith('.save'):
                file_path = f"{session_id}/{filename}"
                try:
                    supabase.storage.from_(bucket_name).remove([file_path])
                    deleted_files.append(filename)
                except Exception as delete_error:
                    error_msg = f"Failed to delete {file_path}: {str(delete_error)}"
                    errors.append(error_msg)
                    logger.warning(error_msg)

        return {
            'deleted_count': len(deleted_files),
            'deleted_files': deleted_files,
            'errors': errors
        }

    except Exception as e:
        logger.error(f"❌ Error deleting models: {e}")
        return {
            'deleted_count': 0,
            'deleted_files': [],
            'errors': [str(e)]
        }
