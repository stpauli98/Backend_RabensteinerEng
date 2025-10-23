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
from utils.supabase_client import get_supabase_admin_client

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

        # Validate file exists
        if not os.path.exists(model_file_path):
            raise FileNotFoundError(f"Model file not found: {model_file_path}")

        # Get file info
        file_size = os.path.getsize(model_file_path)
        filename = os.path.basename(model_file_path)

        # Generate storage path: session_id/model_type_dataset_timestamp.{ext}
        # Extract file extension from original file
        file_extension = os.path.splitext(model_file_path)[1]  # .h5 or .pkl
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        storage_filename = f"{model_type}_{dataset_name}_{timestamp}{file_extension}"
        file_path = f"{session_id}/{storage_filename}"

        logger.info(f"📤 Uploading model to storage: {file_path} ({file_size / 1024 / 1024:.2f}MB)")

        # Read model file as binary
        with open(model_file_path, 'rb') as f:
            model_data = f.read()

        # Upload to trained-models bucket
        bucket_name = 'trained-models'

        try:
            response = supabase.storage.from_(bucket_name).upload(
                path=file_path,
                file=model_data,
                file_options={
                    "content-type": "application/octet-stream",
                    "cache-control": "3600"
                }
            )

            logger.info(f"✅ Model uploaded successfully: {file_path}")

            return {
                'file_path': file_path,
                'file_size': file_size,
                'bucket': bucket_name,
                'model_type': model_type,
                'dataset_name': dataset_name,
                'original_filename': filename
            }

        except Exception as upload_error:
            # Check if bucket exists
            if "Bucket not found" in str(upload_error):
                logger.error(f"❌ Bucket '{bucket_name}' does not exist. Creating it...")

                # Try to create bucket
                try:
                    supabase.storage.create_bucket(
                        bucket_name,
                        options={
                            "public": False,  # Private bucket
                            "file_size_limit": 524288000  # 500MB limit
                        }
                    )
                    logger.info(f"✅ Bucket '{bucket_name}' created successfully")

                    # Retry upload
                    response = supabase.storage.from_(bucket_name).upload(
                        path=file_path,
                        file=model_data,
                        file_options={
                            "content-type": "application/octet-stream",
                            "cache-control": "3600"
                        }
                    )

                    logger.info(f"✅ Model uploaded successfully after bucket creation: {file_path}")

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

        logger.info(f"📥 Downloading model from storage: {file_path}")

        # Download from storage
        response = supabase.storage.from_(bucket_name).download(file_path)

        logger.info(f"✅ Model downloaded successfully: {file_path}")

        return response

    except Exception as e:
        logger.error(f"❌ Error downloading model: {e}")
        import traceback
        logger.error(traceback.format_exc())
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

        logger.info(f"📋 Listing models for session: {session_id}")

        # List files in session directory
        files = supabase.storage.from_(bucket_name).list(session_id)

        models = []
        for file_info in files:
            filename = file_info['name']
            if filename.endswith('.h5') or filename.endswith('.pkl'):
                # Parse filename: ModelType_dataset_timestamp.extension
                # Example: "Sequential_default_20251023_114355.h5"
                parts = filename.rsplit('.', 1)  # Split extension
                file_extension = parts[1] if len(parts) > 1 else ''

                # Extract model type and dataset from filename
                name_parts = parts[0].split('_')
                model_type = name_parts[0] if len(name_parts) > 0 else 'Unknown'
                dataset_name = name_parts[1] if len(name_parts) > 1 else 'default'

                # Convert size from bytes to MB
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

        logger.info(f"✅ Found {len(models)} models for session {session_id}")

        return models

    except Exception as e:
        logger.error(f"❌ Error listing models: {e}")
        return []
