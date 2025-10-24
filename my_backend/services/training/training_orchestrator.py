"""
Training Orchestration Service
High-level orchestration for async model training, results processing, and storage

This service handles:
- Async model training orchestration with background threading
- Training results JSON serialization (models, scalers, numpy arrays)
- Training results upload to Supabase Storage
- Database persistence of training metadata
- Visualization saving
- SocketIO progress notifications

Created: 2025-10-24
Phase 6b of training.py refactoring
"""

import logging
import threading
import pickle
import base64
import json
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Optional, Any

logger = logging.getLogger(__name__)


def clean_for_json(obj: Any) -> Any:
    """
    Recursively clean Python objects for JSON serialization.

    Handles:
    - Custom MDL class objects
    - NumPy arrays and numeric types
    - Pandas timestamps
    - ML models/scalers (pickle + base64 encoding)
    - Nested dicts/lists
    - NaN values

    Args:
        obj: Object to clean for JSON serialization

    Returns:
        JSON-serializable version of the object
    """
    # Check for custom MDL class
    if hasattr(obj, '__class__') and obj.__class__.__name__ == 'MDL':
        # Convert MDL object to dict with its attributes
        return {
            'MODE': getattr(obj, 'MODE', 'Dense'),
            'LAY': getattr(obj, 'LAY', None),
            'N': getattr(obj, 'N', None),
            'EP': getattr(obj, 'EP', None),
            'ACTF': getattr(obj, 'ACTF', None),
            'K': getattr(obj, 'K', None),
            'KERNEL': getattr(obj, 'KERNEL', None),
            'C': getattr(obj, 'C', None),
            'EPSILON': getattr(obj, 'EPSILON', None)
        }
    # Check numpy array first before other checks
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    elif isinstance(obj, (np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, (np.integer)):
        return int(obj)
    elif isinstance(obj, (np.floating)):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [clean_for_json(item) for item in obj]
    # Check for NaN after numpy types since pd.isna might fail on arrays
    elif obj is None:
        return None
    else:
        try:
            if pd.isna(obj):
                return None
        except (ValueError, TypeError):
            pass

        # Special handling for ML models and scalers - serialize to base64
        try:
            # Check if it's a sklearn scaler or transformer
            if (hasattr(obj, 'fit') and hasattr(obj, 'transform')) or \
               (hasattr(obj, 'predict') and hasattr(obj, 'fit')) or \
               (hasattr(obj, '__class__') and 'sklearn' in str(obj.__class__.__module__)):
                # Serialize model/scaler to bytes
                model_bytes = pickle.dumps(obj)
                # Encode to base64 for JSON storage
                model_b64 = base64.b64encode(model_bytes).decode('utf-8')
                return {
                    '_model_type': 'serialized_model',
                    '_model_class': obj.__class__.__name__,
                    '_model_data': model_b64
                }
        except Exception as e:
            logger.warning(f"Could not serialize model/scaler: {e}")
            pass

        # Try to convert unknown objects to string as last resort
        try:
            json.dumps(obj)
            return obj
        except (TypeError, ValueError):
            return str(obj)


def save_training_results(
    session_id: str,
    uuid_session_id: str,
    training_results: Dict,
    model_config: Dict,
    training_split: Dict,
    result: Dict
) -> bool:
    """
    Save training results to Supabase Storage and database.

    Uploads training results to Storage bucket (supports up to 5GB),
    saves metadata to database, and handles visualization saving.

    Args:
        session_id: String session ID
        uuid_session_id: UUID session ID for database operations
        training_results: Raw training results from runner
        model_config: Model configuration dict
        training_split: Training split configuration
        result: Complete result dict from training

    Returns:
        bool: True if save successful, False otherwise

    Raises:
        Exception: If storage or database save fails
    """
    from utils.database import get_supabase_client, create_or_get_session_uuid
    from utils.supabase_client import get_supabase_admin_client
    from utils.training_storage import upload_training_results

    # Use admin client for training results INSERT to avoid timeout
    supabase = get_supabase_admin_client()

    # Check if we got a valid UUID
    if not uuid_session_id:
        logger.error(f"Failed to get UUID for session {session_id}")
        # Try once more after a short delay
        import time
        time.sleep(1)
        uuid_session_id = create_or_get_session_uuid(session_id)

        if not uuid_session_id:
            logger.error(f"Failed to get UUID for session {session_id} after retry")
            raise ValueError('Failed to save training results - session mapping error')

    # Extract and clean results from training
    # Get metrics from multiple possible locations
    evaluation_metrics = result.get('evaluation_metrics', {})
    if not evaluation_metrics:
        evaluation_metrics = training_results.get('evaluation_metrics', {})
    if not evaluation_metrics:
        evaluation_metrics = training_results.get('metrics', {})

    # IMPORTANT: Save ALL training data for plotting
    cleaned_results = clean_for_json({
        'model_type': model_config.get('MODE', 'Dense'),
        'parameters': model_config,
        'metrics': evaluation_metrics,
        'training_split': training_split,
        'dataset_count': result.get('dataset_count', 0),
        'evaluation_metrics': evaluation_metrics,
        'metadata': training_results.get('metadata', {}),
        # Add full training results for plotting
        'trained_model': training_results.get('trained_model'),
        'train_data': training_results.get('train_data', {}),
        'val_data': training_results.get('val_data', {}),
        'test_data': training_results.get('test_data', {}),
        'scalers': training_results.get('scalers', {}),
        'input_features': training_results.get('metadata', {}).get('input_features', []),
        'output_features': training_results.get('metadata', {}).get('output_features', [])
    })

    try:
        # Upload results to Storage (NO TIMEOUT - can handle up to 5GB)
        logger.info(f"📤 Uploading training results to storage for session {uuid_session_id}...")
        storage_result = upload_training_results(
            session_id=uuid_session_id,
            results=cleaned_results,
            compress=True  # Compress for 70-90% size reduction
        )
        logger.info(f"✅ Storage upload complete: {storage_result['file_size'] / 1024 / 1024:.2f}MB")

        # Save metadata to database (FAST - only ~1KB, no timeout)
        training_data = {
            'session_id': uuid_session_id,
            'status': 'completed',
            'results_file_path': storage_result['file_path'],
            'file_size_bytes': storage_result['file_size'],
            'compressed': storage_result['compressed'],
            'results_metadata': storage_result['metadata'],
            'results': None  # Deprecated field - leave NULL for new results
        }

        supabase.table('training_results').insert(training_data).execute()
        logger.info(f"✅ Training metadata saved to database for session {uuid_session_id}")

    except Exception as storage_error:
        logger.error(f"❌ Failed to save training results: {storage_error}")
        import traceback
        logger.error(traceback.format_exc())

        # Fallback: Try to save at least the metadata without full results
        try:
            logger.warning("Attempting fallback: saving metadata only...")
            fallback_data = {
                'session_id': uuid_session_id,
                'status': 'failed_storage',
                'results_metadata': storage_result.get('metadata', {}) if 'storage_result' in locals() else {},
                'results': None
            }
            supabase.table('training_results').insert(fallback_data).execute()
            logger.warning("Fallback save successful - metadata only")
        except Exception as fallback_error:
            logger.error(f"Fallback save also failed: {fallback_error}")

        raise storage_error  # Re-raise original error

    # Save violin plots if available
    if result.get('violin_plots'):
        try:
            violin_plots = result.get('violin_plots')
            if isinstance(violin_plots, dict):
                # Save each violin plot separately
                for plot_name, plot_data in violin_plots.items():
                    # Extract just the base64 part if it includes data URI prefix
                    if isinstance(plot_data, str) and plot_data.startswith('data:image'):
                        base64_data = plot_data
                    else:
                        base64_data = plot_data

                    viz_data = {
                        'session_id': uuid_session_id,
                        'plot_type': 'violin',
                        'plot_name': plot_name,
                        'dataset_name': plot_name.replace('_distribution', '').replace('_plot', ''),
                        'model_name': model_config.get('MODE', 'Linear'),
                        'plot_data_base64': base64_data,
                        'metadata': {
                            'dataset_count': result.get('dataset_count', 0),
                            'generated_during': 'model_training',
                            'created_at': datetime.now().isoformat()
                        }
                    }
                    supabase.table('training_visualizations').insert(viz_data).execute()
                logger.info(f"Violin plots saved for session {uuid_session_id}")
            else:
                logger.warning(f"Violin plots not in expected format: {type(violin_plots)}")
        except Exception as viz_error:
            logger.error(f"Failed to save violin plots: {str(viz_error)}")
            import traceback
            logger.error(traceback.format_exc())

    return True


def run_model_training_async(
    session_id: str,
    model_config: Dict,
    training_split: Dict,
    socketio_instance: Optional[Any] = None
) -> None:
    """
    Run model training asynchronously in a background thread.

    This function handles:
    - Training orchestration via ModernMiddlemanRunner
    - Results processing and JSON cleaning
    - Storage upload and database persistence
    - SocketIO progress notifications
    - Error handling and fallback

    Args:
        session_id: Training session ID
        model_config: Model configuration parameters
        training_split: Training/validation split configuration
        socketio_instance: SocketIO instance for real-time updates (optional)
    """
    from services.training.middleman_runner import ModernMiddlemanRunner
    from utils.database import create_or_get_session_uuid

    try:
        logger.info(f"🚀 TRAINING THREAD STARTED for session {session_id}")
        logger.info(f"Model config: {model_config}")

        # Create runner
        runner = ModernMiddlemanRunner()
        if socketio_instance:
            runner.set_socketio(socketio_instance)

        # Run training script
        result = runner.run_training_script(session_id, model_config)
        logger.info(f"✅ runner.run_training_script completed with success={result.get('success')}")

        if result['success']:
            # Save training results to database
            try:
                uuid_session_id = create_or_get_session_uuid(session_id)
                training_results = result.get('results', {})

                save_training_results(
                    session_id=session_id,
                    uuid_session_id=uuid_session_id,
                    training_results=training_results,
                    model_config=model_config,
                    training_split=training_split,
                    result=result
                )

            except Exception as e:
                logger.error(f"Failed to save training results: {str(e)}")

            if socketio_instance:
                socketio_instance.emit('training_completed', {
                    'session_id': session_id,
                    'status': 'completed',
                    'message': 'Training completed successfully'
                }, room=session_id)
        else:
            logger.error(f"Training failed: {result.get('error')}")
            if socketio_instance:
                socketio_instance.emit('training_error', {
                    'session_id': session_id,
                    'status': 'failed',
                    'error': result.get('error', 'Unknown error')
                }, room=session_id)

    except Exception as e:
        logger.error(f"Async training error: {str(e)}")
        if socketio_instance:
            socketio_instance.emit('training_error', {
                'session_id': session_id,
                'status': 'failed',
                'error': str(e)
            }, room=session_id)
