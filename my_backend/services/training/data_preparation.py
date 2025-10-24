"""
Data preparation module for training workflow.
Handles data splitting and preparation without model training.
"""

import logging
import numpy as np
from typing import Dict, Any, Optional, Tuple
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

def prepare_data_for_training(
    session_id: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    random_dat: bool = False
) -> Dict[str, Any]:
    """
    Prepare data for training by splitting into train/val/test sets.
    
    Args:
        session_id: Session identifier
        train_ratio: Ratio for training data (default 0.7)
        val_ratio: Ratio for validation data (default 0.2)
        random_dat: If True, use sequential split; if False, use random split
        
    Returns:
        Dictionary containing data splits and metadata
    """
    try:
        from utils.database import get_supabase_client, create_or_get_session_uuid
        
        supabase = get_supabase_client()
        uuid_session_id = create_or_get_session_uuid(session_id)
        
        results_response = supabase.table('training_results').select('*').eq(
            'session_id', uuid_session_id
        ).order('created_at.desc').limit(1).execute()
        
        if results_response.data and len(results_response.data) > 0:
            result = results_response.data[0]
            data_splits = result.get('data_splits', {})
            
            X_data = []
            y_data = []
            
            for split in ['train', 'val', 'test']:
                if split in data_splits:
                    split_data = data_splits[split]
                    if 'X' in split_data and split_data['X']:
                        X_data.extend(split_data['X'])
                    if 'y' in split_data and split_data['y']:
                        y_data.extend(split_data['y'])
            
            if not X_data or not y_data:
                return {
                    'success': False,
                    'error': 'No data found in training results',
                    'message': 'Please process data first'
                }
        else:
            return {
                'success': False,
                'error': 'No processed data found for session',
                'message': 'Please upload and process a file first'
            }
        
        X = np.array(X_data)
        y = np.array(y_data)
        
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        
        test_ratio = 1.0 - train_ratio - val_ratio
        
        if random_dat:
            n_samples = len(X)
            n_train = int(n_samples * train_ratio)
            n_val = int(n_samples * val_ratio)
            
            X_train = X[:n_train]
            y_train = y[:n_train]
            
            X_val = X[n_train:n_train + n_val]
            y_val = y[n_train:n_train + n_val]
            
            X_test = X[n_train + n_val:]
            y_test = y[n_train + n_val:]
            
            logger.info(f"Sequential split: train={n_train}, val={n_val}, test={len(X_test)}")
        else:
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y, test_size=test_ratio, random_state=42, shuffle=True
            )
            
            val_size = val_ratio / (train_ratio + val_ratio)
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=val_size, random_state=42, shuffle=True
            )
            
            logger.info(f"Random split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
        
        if results_response.data and len(results_response.data) > 0:
            model_metadata = result.get('model_metadata', {})
            input_features = model_metadata.get('input_features', [])
            output_features = model_metadata.get('output_features', [])
        else:
            input_features = []
            output_features = []
        
        if not input_features:
            input_features = [f"Feature_{i+1}" for i in range(X.shape[1])]
        if not output_features:
            output_features = [f"Output_{i+1}" for i in range(y.shape[1])]
        
        return {
            'success': True,
            'train': {
                'X': X_train.tolist(),
                'y': y_train.tolist()
            },
            'val': {
                'X': X_val.tolist(),
                'y': y_val.tolist()
            },
            'test': {
                'X': X_test.tolist(),
                'y': y_test.tolist()
            },
            'input_features': input_features,
            'output_features': output_features,
            'shapes': {
                'train': {'X': X_train.shape, 'y': y_train.shape},
                'val': {'X': X_val.shape, 'y': y_val.shape},
                'test': {'X': X_test.shape, 'y': y_test.shape}
            },
            'split_ratios': {
                'train': train_ratio,
                'val': val_ratio,
                'test': test_ratio
            },
            'random_split': not random_dat
        }
        
    except Exception as e:
        logger.error(f"Error preparing data for training: {str(e)}")
        return {
            'success': False,
            'error': str(e)
        }
