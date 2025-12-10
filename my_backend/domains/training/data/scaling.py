"""
Data scaling and transformation module
Extracted from training_backend_test_2.py for data preprocessing
"""

import numpy as np
import pandas as pd
import copy
import logging
from typing import Dict, List, Tuple, Optional, Any
from sklearn.preprocessing import MinMaxScaler, StandardScaler

logger = logging.getLogger(__name__)


class DataScaler:
    """
    Handle data scaling and inverse transformations
    Based on original training_backend_test_2.py implementation
    """
    
    def __init__(self):
        """Initialize DataScaler"""
        self.input_scalers = {}
        self.output_scalers = {}
        self.scaling_config = {}
        
    def prepare_scalers(self, 
                       input_combined_array: np.ndarray,
                       output_combined_array: np.ndarray,
                       input_scaling_config: List[Dict],
                       output_scaling_config: List[Dict],
                       progress_callback=None) -> Dict:
        """
        Prepare scalers for input and output data
        
        Args:
            input_combined_array: Combined input data array (samples × features)
            output_combined_array: Combined output data array (samples × features)
            input_scaling_config: List of scaling configurations for input features
            output_scaling_config: List of scaling configurations for output features
            progress_callback: Function to call for progress updates
            
        Returns:
            Dict containing scaler information
        """
        try:
            
            i_scal_list = [config.get('scale', False) for config in input_scaling_config]
            i_scal_max_list = [config.get('scale_max', 1.0) for config in input_scaling_config]
            i_scal_min_list = [config.get('scale_min', 0.0) for config in input_scaling_config]
            
            o_scal_list = [config.get('scale', False) for config in output_scaling_config]
            o_scal_max_list = [config.get('scale_max', 1.0) for config in output_scaling_config]
            o_scal_min_list = [config.get('scale_min', 0.0) for config in output_scaling_config]
            
            total_scaling_ops = sum(i_scal_list) + sum(o_scal_list)
            current_op = 0
            
            self.input_scalers = {}
            
            for i in range(input_combined_array.shape[1]):
                if i < len(i_scal_list) and i_scal_list[i]:
                    
                    if progress_callback:
                        progress = current_op / total_scaling_ops * 100 if total_scaling_ops > 0 else 0
                        progress_callback(f"Setting up input scaler {i+1}", progress)
                    
                    scaler = MinMaxScaler(feature_range=(i_scal_min_list[i], i_scal_max_list[i]))
                    scaler.fit(input_combined_array[:, i].reshape(-1, 1))
                    self.input_scalers[i] = scaler
                    
                    current_op += 1
                    
                    
                else:
                    self.input_scalers[i] = None
            
            self.output_scalers = {}
            
            for i in range(output_combined_array.shape[1]):
                if i < len(o_scal_list) and o_scal_list[i]:
                    
                    if progress_callback:
                        progress = current_op / total_scaling_ops * 100 if total_scaling_ops > 0 else 0
                        progress_callback(f"Setting up output scaler {i+1}", progress)
                    
                    scaler = MinMaxScaler(feature_range=(o_scal_min_list[i], o_scal_max_list[i]))
                    scaler.fit(output_combined_array[:, i].reshape(-1, 1))
                    self.output_scalers[i] = scaler
                    
                    current_op += 1
                    
                    
                else:
                    self.output_scalers[i] = None
            
            self.scaling_config = {
                'input_config': input_scaling_config,
                'output_config': output_scaling_config,
                'input_scalers_count': len([s for s in self.input_scalers.values() if s is not None]),
                'output_scalers_count': len([s for s in self.output_scalers.values() if s is not None])
            }
            
            if progress_callback:
                progress_callback("Scalers setup completed", 100)
            
            
            return self.scaling_config
            
        except Exception as e:
            logger.error(f"Error preparing scalers: {str(e)}")
            raise
    
    def apply_scaling(self, 
                     input_array_3d: np.ndarray,
                     output_array_3d: np.ndarray,
                     progress_callback=None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply scaling to 3D input and output arrays
        
        Args:
            input_array_3d: Input data array (samples × timesteps × features)
            output_array_3d: Output data array (samples × timesteps × features)
            progress_callback: Function to call for progress updates
            
        Returns:
            Tuple of scaled (input_array_3d, output_array_3d)
        """
        try:
            
            scaled_input = copy.deepcopy(input_array_3d)
            scaled_output = copy.deepcopy(output_array_3d)
            
            n_samples = scaled_input.shape[0]
            n_input_features = scaled_input.shape[2]
            n_output_features = scaled_output.shape[2]
            
            for sample_idx in range(n_samples):
                
                if progress_callback:
                    progress = sample_idx / n_samples * 50
                    progress_callback(f"Scaling input data: sample {sample_idx+1}/{n_samples}", progress)
                
                for feat_idx in range(n_input_features):
                    
                    if self.input_scalers.get(feat_idx) is not None:
                        feature_data = scaled_input[sample_idx, :, feat_idx].reshape(-1, 1)
                        
                        scaled_feature = self.input_scalers[feat_idx].transform(feature_data)
                        
                        scaled_input[sample_idx, :, feat_idx] = scaled_feature.ravel()
            
            for sample_idx in range(n_samples):
                
                if progress_callback:
                    progress = 50 + (sample_idx / n_samples * 50)
                    progress_callback(f"Scaling output data: sample {sample_idx+1}/{n_samples}", progress)
                
                for feat_idx in range(n_output_features):
                    
                    if self.output_scalers.get(feat_idx) is not None:
                        feature_data = scaled_output[sample_idx, :, feat_idx].reshape(-1, 1)
                        
                        scaled_feature = self.output_scalers[feat_idx].transform(feature_data)
                        
                        scaled_output[sample_idx, :, feat_idx] = scaled_feature.ravel()
            
            if progress_callback:
                progress_callback("Data scaling completed", 100)
            
            
            return scaled_input, scaled_output
            
        except Exception as e:
            logger.error(f"Error applying scaling: {str(e)}")
            raise
    
    def inverse_transform_output(self, 
                               scaled_predictions: np.ndarray,
                               progress_callback=None) -> np.ndarray:
        """
        Apply inverse transformation to scaled predictions
        
        Args:
            scaled_predictions: Scaled prediction array (samples × timesteps × features)
            progress_callback: Function to call for progress updates
            
        Returns:
            Unscaled predictions in original units
        """
        try:
            
            unscaled_predictions = copy.deepcopy(scaled_predictions)
            
            n_samples = unscaled_predictions.shape[0]
            n_features = unscaled_predictions.shape[2]
            
            for sample_idx in range(n_samples):
                
                if progress_callback:
                    progress = sample_idx / n_samples * 100
                    progress_callback(f"Inverse scaling: sample {sample_idx+1}/{n_samples}", progress)
                
                for feat_idx in range(n_features):
                    
                    if self.output_scalers.get(feat_idx) is not None:
                        feature_data = unscaled_predictions[sample_idx, :, feat_idx].reshape(-1, 1)
                        
                        unscaled_feature = self.output_scalers[feat_idx].inverse_transform(feature_data)
                        
                        unscaled_predictions[sample_idx, :, feat_idx] = unscaled_feature.ravel()
            
            if progress_callback:
                progress_callback("Inverse scaling completed", 100)
            
            
            return unscaled_predictions
            
        except Exception as e:
            logger.error(f"Error applying inverse transform: {str(e)}")
            raise
    
    def split_train_val_test(self, 
                           input_array_3d: np.ndarray,
                           output_array_3d: np.ndarray,
                           train_ratio: float = 0.7,
                           val_ratio: float = 0.2,
                           test_ratio: float = 0.1,
                           shuffle: bool = False,
                           random_seed: int = None) -> Dict[str, np.ndarray]:
        """
        Split data into train/validation/test sets
        
        Args:
            input_array_3d: Input data array (samples × timesteps × features)
            output_array_3d: Output data array (samples × timesteps × features)
            train_ratio: Ratio for training set (default: 0.7)
            val_ratio: Ratio for validation set (default: 0.2)
            test_ratio: Ratio for test set (default: 0.1)
            shuffle: Whether to shuffle data before splitting
            random_seed: Random seed for reproducibility
            
        Returns:
            Dict containing train/val/test splits
        """
        try:
            
            total_ratio = train_ratio + val_ratio + test_ratio
            if abs(total_ratio - 1.0) > 1e-6:
                logger.warning(f"Ratios don't sum to 1.0: {total_ratio}. Normalizing...")
                train_ratio /= total_ratio
                val_ratio /= total_ratio
                test_ratio /= total_ratio
            
            n_samples = input_array_3d.shape[0]

            # Koristi round() kao u originalu za identično ponašanje
            n_train = round(train_ratio * n_samples)
            n_val = round(val_ratio * n_samples)
            n_test = n_samples - n_train - n_val
            
            
            indices = np.arange(n_samples)
            
            if shuffle:
                if random_seed is not None:
                    np.random.seed(random_seed)
                np.random.shuffle(indices)
            
            train_indices = indices[:n_train]
            val_indices = indices[n_train:n_train + n_val]
            test_indices = indices[n_train + n_val:]
            
            splits = {
                'train_x': input_array_3d[train_indices],
                'val_x': input_array_3d[val_indices],
                'test_x': input_array_3d[test_indices],
                'train_y': output_array_3d[train_indices],
                'val_y': output_array_3d[val_indices],
                'test_y': output_array_3d[test_indices],
                'train_indices': train_indices,
                'val_indices': val_indices,
                'test_indices': test_indices,
                'split_info': {
                    'n_train': n_train,
                    'n_val': n_val,
                    'n_test': n_test,
                    'train_ratio': train_ratio,
                    'val_ratio': val_ratio,
                    'test_ratio': test_ratio,
                    'shuffled': shuffle
                }
            }
            
            
            return splits
            
        except Exception as e:
            logger.error(f"Error splitting data: {str(e)}")
            raise
    
    def get_scaler_statistics(self) -> Dict:
        """
        Get statistics about configured scalers
        
        Returns:
            Dict containing scaler statistics
        """
        try:
            stats = {
                'input_scalers': {},
                'output_scalers': {},
                'summary': {
                    'total_input_scalers': 0,
                    'total_output_scalers': 0,
                    'total_scalers': 0
                }
            }
            
            for idx, scaler in self.input_scalers.items():
                if scaler is not None:
                    stats['input_scalers'][idx] = {
                        'feature_range': scaler.feature_range,
                        'data_min': scaler.data_min_.tolist() if hasattr(scaler, 'data_min_') else None,
                        'data_max': scaler.data_max_.tolist() if hasattr(scaler, 'data_max_') else None,
                        'scale': scaler.scale_.tolist() if hasattr(scaler, 'scale_') else None
                    }
                    stats['summary']['total_input_scalers'] += 1
            
            for idx, scaler in self.output_scalers.items():
                if scaler is not None:
                    stats['output_scalers'][idx] = {
                        'feature_range': scaler.feature_range,
                        'data_min': scaler.data_min_.tolist() if hasattr(scaler, 'data_min_') else None,
                        'data_max': scaler.data_max_.tolist() if hasattr(scaler, 'data_max_') else None,
                        'scale': scaler.scale_.tolist() if hasattr(scaler, 'scale_') else None
                    }
                    stats['summary']['total_output_scalers'] += 1
            
            stats['summary']['total_scalers'] = stats['summary']['total_input_scalers'] + stats['summary']['total_output_scalers']
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting scaler statistics: {str(e)}")
            return {}


def create_default_scaling_config(data_info: Dict, 
                                 default_scale: bool = True,
                                 default_min: float = 0.0,
                                 default_max: float = 1.0) -> List[Dict]:
    """
    Create default scaling configuration for features
    
    Args:
        data_info: Information about data features
        default_scale: Whether to scale by default
        default_min: Default minimum scale value
        default_max: Default maximum scale value
        
    Returns:
        List of scaling configurations
    """
    try:
        config = []
        
        if 'features' in data_info:
            for feature_name in data_info['features']:
                config.append({
                    'feature_name': feature_name,
                    'scale': default_scale,
                    'scale_min': default_min,
                    'scale_max': default_max,
                    'scaler_type': 'MinMaxScaler'
                })
        
        elif 'n_features' in data_info:
            for i in range(data_info['n_features']):
                config.append({
                    'feature_index': i,
                    'scale': default_scale,
                    'scale_min': default_min,
                    'scale_max': default_max,
                    'scaler_type': 'MinMaxScaler'
                })
        
        return config
        
    except Exception as e:
        logger.error(f"Error creating default scaling config: {str(e)}")
        return []


def prepare_data_for_training(input_array_3d: np.ndarray,
                            output_array_3d: np.ndarray,
                            scaling_config: Dict = None,
                            split_config: Dict = None,
                            progress_callback=None) -> Dict:
    """
    Complete data preparation pipeline: scaling + splitting
    
    Args:
        input_array_3d: Input data array (samples × timesteps × features)
        output_array_3d: Output data array (samples × timesteps × features)
        scaling_config: Configuration for data scaling
        split_config: Configuration for train/val/test split
        progress_callback: Function to call for progress updates
        
    Returns:
        Dict containing prepared data and metadata
    """
    try:
        
        if scaling_config is None:
            scaling_config = {
                'input_config': create_default_scaling_config({'n_features': input_array_3d.shape[2]}),
                'output_config': create_default_scaling_config({'n_features': output_array_3d.shape[2]})
            }
        
        if split_config is None:
            split_config = {
                'train_ratio': 0.7,
                'val_ratio': 0.2,
                'test_ratio': 0.1,
                'shuffle': False,
                'random_seed': None
            }
        
        scaler = DataScaler()
        
        if progress_callback:
            progress_callback("Preparing scalers", 10)
        
        input_combined = input_array_3d.reshape(-1, input_array_3d.shape[2])
        output_combined = output_array_3d.reshape(-1, output_array_3d.shape[2])
        
        scaler_info = scaler.prepare_scalers(
            input_combined,
            output_combined,
            scaling_config['input_config'],
            scaling_config['output_config']
        )
        
        if progress_callback:
            progress_callback("Applying scaling", 30)
        
        scaled_input, scaled_output = scaler.apply_scaling(
            input_array_3d,
            output_array_3d
        )
        
        if progress_callback:
            progress_callback("Splitting data", 80)
        
        data_splits = scaler.split_train_val_test(
            scaled_input,
            scaled_output,
            **split_config
        )
        
        original_splits = scaler.split_train_val_test(
            input_array_3d,
            output_array_3d,
            **split_config
        )
        
        if progress_callback:
            progress_callback("Data preparation completed", 100)
        
        results = {
            'scaled_data': data_splits,
            'original_data': original_splits,
            'scaler': scaler,
            'scaler_info': scaler_info,
            'scaling_config': scaling_config,
            'split_config': split_config,
            'data_info': {
                'n_samples': input_array_3d.shape[0],
                'n_timesteps_input': input_array_3d.shape[1],
                'n_timesteps_output': output_array_3d.shape[1],
                'n_input_features': input_array_3d.shape[2],
                'n_output_features': output_array_3d.shape[2]
            }
        }
        
        return results
        
    except Exception as e:
        logger.error(f"Error in data preparation pipeline: {str(e)}")
        raise
