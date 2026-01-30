"""
Serialization helpers for backward-compatible pickle/JSON deserialization.

This module provides utility functions for deserializing models and scalers
from both old JSON format (base64-encoded pickle) and new pickle format
(direct objects from pickle.load).

Created: 2026-01-29
Purpose: Support migration from JSON to Pickle for training results storage
"""

import pickle
import base64
import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


def deserialize_model_or_scaler(data: Any) -> Any:
    """
    Deserialize a model or scaler from either:
    - Old format: {'_model_type': 'serialized_model', '_model_data': 'base64...'}
    - New format: Direct object (already deserialized by pickle)

    SECURITY NOTE: pickle.loads() can execute arbitrary code.
    This is safe here because models/scalers are only stored by authenticated users
    via our training pipeline and retrieved from trusted Supabase storage.

    Args:
        data: Either a serialized dict or a direct object

    Returns:
        Deserialized model/scaler object, or None if invalid
    """
    if data is None:
        return None

    # OLD FORMAT: base64-encoded pickle in JSON
    if isinstance(data, dict) and '_model_type' in data:
        if data.get('_model_type') == 'serialized_model':
            try:
                model_b64 = data.get('_model_data', '')
                model_bytes = base64.b64decode(model_b64)
                obj = pickle.loads(model_bytes)
                logger.debug(f"🔍 Deserialized from old JSON format: {type(obj).__name__}")
                return obj
            except Exception as e:
                logger.error(f"Failed to deserialize from old format: {e}")
                return None

    # NEW FORMAT: Direct object (pickle already deserialized it)
    # Check if it's a valid sklearn/keras object
    if hasattr(data, 'fit') or hasattr(data, 'transform') or hasattr(data, 'predict'):
        logger.debug(f"🔍 Using direct object from pickle format: {type(data).__name__}")
        return data

    # If it's a numpy array or other type, return as-is
    logger.debug(f"🔍 Passthrough object type: {type(data).__name__}")
    return data


def deserialize_scalers_dict(scalers_dict: Dict) -> Dict:
    """
    Deserialize a dictionary of scalers (handles both formats).

    Args:
        scalers_dict: Dict like {0: scaler_data, 1: scaler_data, ...}
                     or {'0': scaler_data, '1': scaler_data, ...}

    Returns:
        Dict with deserialized scaler objects, keyed by int
    """
    if not scalers_dict:
        return {}

    result = {}
    for key, scaler_data in scalers_dict.items():
        try:
            int_key = int(key)
        except (ValueError, TypeError):
            int_key = key

        result[int_key] = deserialize_model_or_scaler(scaler_data)

    return result
