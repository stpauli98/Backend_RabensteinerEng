"""
Prediction Service Module
Handles model loading from storage and prediction execution

This module provides:
- Load trained models from Supabase Storage
- Apply scalers to input data
- Execute predictions
- Inverse transform results

Created: 2025-12-15
"""

import os
import io
import tempfile
import logging
import pickle
import base64
import numpy as np
from typing import Dict, List, Optional, Any, Tuple

logger = logging.getLogger(__name__)


class PredictionService:
    """
    Service for making predictions with trained models from Supabase Storage.
    
    Usage:
        service = PredictionService(session_id, user_id)
        result = service.predict(model_filename, input_data)
    """
    
    def __init__(self, session_id: str, user_id: str):
        """
        Initialize prediction service.
        
        Args:
            session_id: UUID session ID
            user_id: User ID for authorization
        """
        self.session_id = session_id
        self.user_id = user_id
        self._model = None
        self._model_filename = None
        self._scalers = None
        self._training_results = None
    
    def load_model(self, model_filename: str) -> Any:
        """
        Download and load model from Supabase Storage.
        
        Args:
            model_filename: Name of model file (e.g., 'best_model.h5')
            
        Returns:
            Loaded model object (Keras or sklearn)
            
        Raises:
            FileNotFoundError: If model not found in storage
            ValueError: If model format not supported
        """
        from utils.model_storage import download_trained_model
        
        # Construct storage path
        file_path = f"{self.session_id}/{model_filename}"
        
        logger.info(f"📥 Loading model from storage: {file_path}")
        
        try:
            # Download model bytes from storage
            model_bytes = download_trained_model(self.session_id, file_path)
            
            # Determine model type by extension
            if model_filename.endswith('.h5') or model_filename.endswith('.keras'):
                # Keras model
                model = self._load_keras_model(model_bytes, model_filename)
            elif model_filename.endswith('.pkl') or model_filename.endswith('.joblib'):
                # sklearn model
                model = self._load_sklearn_model(model_bytes)
            else:
                raise ValueError(f"Unsupported model format: {model_filename}")
            
            self._model = model
            self._model_filename = model_filename
            
            logger.info(f"✅ Model loaded successfully: {model_filename}")
            return model
            
        except Exception as e:
            logger.error(f"❌ Error loading model {model_filename}: {e}")
            raise
    
    def _load_keras_model(self, model_bytes: bytes, filename: str) -> Any:
        """
        Load Keras model from bytes.
        
        Args:
            model_bytes: Model file data
            filename: Original filename for temp file extension
            
        Returns:
            Loaded Keras model
        """
        # Create temp file to load model (Keras requires file path)
        suffix = '.h5' if filename.endswith('.h5') else '.keras'
        
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_file:
            tmp_file.write(model_bytes)
            tmp_path = tmp_file.name
        
        try:
            # Import TensorFlow/Keras
            import tensorflow as tf
            from tensorflow import keras
            
            # Suppress TF warnings
            tf.get_logger().setLevel('ERROR')
            
            # Load the model
            model = keras.models.load_model(tmp_path, compile=False)
            
            logger.info(f"✅ Keras model loaded: {model.name if hasattr(model, 'name') else 'unnamed'}")
            return model
            
        finally:
            # Cleanup temp file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def _load_sklearn_model(self, model_bytes: bytes) -> Any:
        """
        Load sklearn/joblib model from bytes.
        
        Args:
            model_bytes: Model file data
            
        Returns:
            Loaded sklearn model
            
        SECURITY NOTE: pickle.loads() can execute arbitrary code.
        This is safe here because models are only stored by authenticated users
        via our training pipeline and retrieved from trusted Supabase storage.
        """
        import joblib
        
        # Load from bytes using BytesIO
        model = joblib.load(io.BytesIO(model_bytes))
        
        logger.info(f"✅ sklearn model loaded: {type(model).__name__}")
        return model
    
    def load_scalers(self) -> Dict[str, Dict]:
        """
        Load input/output scalers from session training results.
        
        Returns:
            Dict with 'input' and 'output' scalers
            
        Raises:
            ValueError: If no scalers found for session
        """
        from utils.training_storage import fetch_training_results_with_storage
        
        if self._scalers is not None:
            return self._scalers
        
        logger.info(f"📥 Loading scalers for session: {self.session_id}")
        
        # Fetch training results (contains scalers)
        training_results = fetch_training_results_with_storage(self.session_id)
        
        if not training_results:
            raise ValueError(f"No training results found for session {self.session_id}")
        
        self._training_results = training_results
        
        scalers = training_results.get('scalers', {})
        
        if not scalers:
            raise ValueError(f"No scalers found for session {self.session_id}")
        
        # Deserialize scalers
        input_scalers = self._deserialize_scalers(scalers.get('input', {}))
        output_scalers = self._deserialize_scalers(scalers.get('output', {}))
        
        self._scalers = {
            'input': input_scalers,
            'output': output_scalers
        }
        
        logger.info(f"✅ Scalers loaded: {len(input_scalers)} input, {len(output_scalers)} output")
        return self._scalers
    
    def _deserialize_scalers(self, scaler_dict: Dict) -> Dict:
        """
        Deserialize scalers from base64-encoded pickle data.
        
        Args:
            scaler_dict: Dictionary of serialized scalers
            
        Returns:
            Dictionary of deserialized scaler objects
            
        SECURITY NOTE: pickle.loads() can execute arbitrary code.
        This is safe here because scalers are only stored by authenticated users
        via our training pipeline and retrieved from trusted Supabase storage.
        """
        result = {}
        
        for key, scaler_data in scaler_dict.items():
            if scaler_data and isinstance(scaler_data, dict) and '_model_type' in scaler_data:
                try:
                    scaler = pickle.loads(base64.b64decode(scaler_data['_model_data']))
                    result[int(key)] = scaler
                except Exception as e:
                    logger.error(f"Error deserializing scaler {key}: {e}")
                    result[int(key)] = None
            else:
                result[int(key)] = None
        
        return result
    
    def preprocess_input(
        self,
        input_data: List[Dict[str, float]],
        scalers: Optional[Dict] = None
    ) -> np.ndarray:
        """
        Transform input data using scalers.
        
        Args:
            input_data: List of dicts with feature values
            scalers: Optional scalers dict (uses loaded if not provided)
            
        Returns:
            Preprocessed numpy array ready for model
        """
        if scalers is None:
            scalers = self.load_scalers()
        
        input_scalers = scalers.get('input', {})
        
        # Convert input data to numpy array
        # Assuming input_data is list of dicts with numeric values
        if isinstance(input_data, list) and len(input_data) > 0:
            if isinstance(input_data[0], dict):
                # Extract values in consistent order
                keys = sorted(input_data[0].keys())
                input_array = np.array([[row.get(k, 0) for k in keys] for row in input_data])
            else:
                input_array = np.array(input_data)
        else:
            input_array = np.array(input_data)
        
        logger.info(f"Input array shape: {input_array.shape}")
        
        # Apply scaling if scalers available
        scaled_data = input_array.copy()
        
        for i in range(input_array.shape[1] if len(input_array.shape) > 1 else 1):
            if i in input_scalers and input_scalers[i] is not None:
                try:
                    if len(input_array.shape) > 1:
                        column_data = input_array[:, i].reshape(-1, 1)
                    else:
                        column_data = input_array.reshape(-1, 1)
                    
                    scaled_column = input_scalers[i].transform(column_data)
                    
                    if len(scaled_data.shape) > 1:
                        scaled_data[:, i] = scaled_column.flatten()
                    else:
                        scaled_data = scaled_column.flatten()
                        
                except Exception as e:
                    logger.warning(f"Error scaling feature {i}: {e}")
        
        logger.info(f"Preprocessed data shape: {scaled_data.shape}")
        return scaled_data
    
    def predict_raw(self, model: Any, preprocessed_data: np.ndarray) -> np.ndarray:
        """
        Run prediction without post-processing.
        
        Args:
            model: Loaded model object
            preprocessed_data: Scaled input data
            
        Returns:
            Raw prediction output
        """
        logger.info(f"🔮 Running prediction with input shape: {preprocessed_data.shape}")
        
        # Check if Keras model
        if hasattr(model, 'predict'):
            # Keras or sklearn model
            predictions = model.predict(preprocessed_data)
        else:
            raise ValueError(f"Model type {type(model)} does not support predict()")
        
        logger.info(f"✅ Prediction complete, output shape: {predictions.shape}")
        return predictions
    
    def postprocess_output(
        self,
        predictions: np.ndarray,
        scalers: Optional[Dict] = None
    ) -> List[float]:
        """
        Inverse transform predictions using output scalers.
        
        Args:
            predictions: Raw model predictions
            scalers: Optional scalers dict (uses loaded if not provided)
            
        Returns:
            List of unscaled prediction values
        """
        if scalers is None:
            scalers = self.load_scalers()
        
        output_scalers = scalers.get('output', {})
        
        # Ensure predictions is 2D
        if len(predictions.shape) == 1:
            predictions = predictions.reshape(-1, 1)
        
        # Apply inverse scaling if scalers available
        unscaled_predictions = predictions.copy()
        
        for i in range(predictions.shape[1]):
            if i in output_scalers and output_scalers[i] is not None:
                try:
                    column_data = predictions[:, i].reshape(-1, 1)
                    unscaled_column = output_scalers[i].inverse_transform(column_data)
                    unscaled_predictions[:, i] = unscaled_column.flatten()
                except Exception as e:
                    logger.warning(f"Error inverse scaling output {i}: {e}")
        
        # Convert to list
        if unscaled_predictions.shape[1] == 1:
            return unscaled_predictions.flatten().tolist()
        else:
            return unscaled_predictions.tolist()
    
    def predict(
        self,
        model_filename: str,
        input_data: List[Dict[str, float]],
        apply_scaling: bool = True
    ) -> Dict[str, Any]:
        """
        Full prediction pipeline: load model, preprocess, predict, postprocess.
        
        Args:
            model_filename: Name of model file in storage
            input_data: List of input data dicts
            apply_scaling: Whether to apply scalers (default True)
            
        Returns:
            Dict with predictions and metadata
        """
        from datetime import datetime
        
        logger.info(f"🚀 Starting prediction with model: {model_filename}")
        
        # Load model if not already loaded or different filename
        if self._model is None or self._model_filename != model_filename:
            self.load_model(model_filename)
        
        # Load scalers
        if apply_scaling:
            scalers = self.load_scalers()
        else:
            scalers = None
        
        # Preprocess input
        if apply_scaling and scalers:
            preprocessed = self.preprocess_input(input_data, scalers)
        else:
            # Convert directly to numpy
            if isinstance(input_data, list) and len(input_data) > 0:
                if isinstance(input_data[0], dict):
                    keys = sorted(input_data[0].keys())
                    preprocessed = np.array([[row.get(k, 0) for k in keys] for row in input_data])
                else:
                    preprocessed = np.array(input_data)
            else:
                preprocessed = np.array(input_data)
        
        # Run prediction
        raw_predictions = self.predict_raw(self._model, preprocessed)
        
        # Postprocess output
        if apply_scaling and scalers:
            predictions = self.postprocess_output(raw_predictions, scalers)
        else:
            predictions = raw_predictions.flatten().tolist()
        
        result = {
            'predictions': predictions,
            'model_used': model_filename,
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'input_count': len(input_data),
            'scaling_applied': apply_scaling
        }
        
        logger.info(f"✅ Prediction complete: {len(predictions)} predictions generated")
        return result


def load_model_from_storage(session_id: str, model_filename: str) -> Any:
    """
    Utility function to load a model from Supabase Storage.
    
    Args:
        session_id: UUID session ID
        model_filename: Name of model file
        
    Returns:
        Loaded model object
    """
    service = PredictionService(session_id, user_id="system")
    return service.load_model(model_filename)
