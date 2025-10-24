"""
Scaler manager module that implements EXACT scaling logic
from training_original.py lines 1764-1861 and 2162-2210
"""

import numpy as np
import copy
from sklearn.preprocessing import MinMaxScaler
import logging
from typing import Dict, Tuple, List
import pandas as pd
from .config import T

logger = logging.getLogger(__name__)


def create_scaling_lists(i_dat_inf: pd.DataFrame, o_dat_inf: pd.DataFrame) -> Tuple[List, List, List, List, List, List]:
    """
    Create scaling control lists exactly as in training_original.py lines 1764-1810
    
    Args:
        i_dat_inf: Input data info DataFrame
        o_dat_inf: Output data info DataFrame
        
    Returns:
        Tuple of (i_scal_list, i_scal_max_list, i_scal_min_list, 
                  o_scal_list, o_scal_max_list, o_scal_min_list)
    """
    
    # Get scaling lists from data info (lines 1764-1766)
    i_scal_list = i_dat_inf["scal"].tolist()
    i_scal_max_list = i_dat_inf["scal_max"].tolist()
    i_scal_min_list = i_dat_inf["scal_min"].tolist()
    
    # Time feature import flags (lines 1768-1772)
    imp = [T.Y.IMP, T.M.IMP, T.W.IMP, T.D.IMP, T.H.IMP]
    
    # Time feature scaling flags (lines 1774-1778)
    scal = [T.Y.SCAL, T.M.SCAL, T.W.SCAL, T.D.SCAL, T.H.SCAL]
    
    # Time feature scaling max values (lines 1780-1784)
    scal_max = [T.Y.SCAL_MAX, T.M.SCAL_MAX, T.W.SCAL_MAX, T.D.SCAL_MAX, T.H.SCAL_MAX]
    
    # Time feature scaling min values (lines 1786-1790)
    scal_min = [T.Y.SCAL_MIN, T.M.SCAL_MIN, T.W.SCAL_MIN, T.D.SCAL_MIN, T.H.SCAL_MIN]
    
    # Add time feature scaling info (lines 1792-1806)
    for i in range(len(imp)):
        if imp[i] == True and scal[i] == True:
            # Add for sin and cos components
            i_scal_list.append(True)
            i_scal_list.append(True)
            i_scal_max_list.append(scal_max[i])
            i_scal_max_list.append(scal_max[i])
            i_scal_min_list.append(scal_min[i])
            i_scal_min_list.append(scal_min[i])
        elif imp[i] == True and scal[i] == False:
            # Add for sin and cos components but don't scale
            i_scal_list.append(False)
            i_scal_list.append(False)
            i_scal_max_list.append(scal_max[i])
            i_scal_max_list.append(scal_max[i])
            i_scal_min_list.append(scal_min[i])
            i_scal_min_list.append(scal_min[i])
    
    # Get output scaling lists (lines 1808-1810)
    o_scal_list = o_dat_inf["scal"].tolist()
    o_scal_max_list = o_dat_inf["scal_max"].tolist()
    o_scal_min_list = o_dat_inf["scal_min"].tolist()
    
    return (i_scal_list, i_scal_max_list, i_scal_min_list,
            o_scal_list, o_scal_max_list, o_scal_min_list)


def fit_scalers(i_combined_array: np.ndarray, o_combined_array: np.ndarray,
                i_scal_list: List, i_scal_max_list: List, i_scal_min_list: List,
                o_scal_list: List, o_scal_max_list: List, o_scal_min_list: List) -> Tuple[Dict, Dict]:
    """
    Fit scalers on combined arrays exactly as in training_original.py lines 1814-1861
    
    Args:
        i_combined_array: Combined input array
        o_combined_array: Combined output array
        i_scal_list: Input scaling control list
        i_scal_max_list: Input scaling max values
        i_scal_min_list: Input scaling min values
        o_scal_list: Output scaling control list
        o_scal_max_list: Output scaling max values
        o_scal_min_list: Output scaling min values
        
    Returns:
        Tuple of (i_scalers, o_scalers) dictionaries
    """
    
    # Create empty dictionary for input scalers (line 1814)
    i_scalers = {}
    
    # Calculate total scalers for progress (line 1816)
    scal_all = sum(i_scal_list) + sum(o_scal_list)
    scal_i = 0
    
    # Fit input scalers (lines 1819-1838)
    # Check if arrays have valid data before accessing shape[1]
    if i_combined_array.size == 0 or len(i_combined_array.shape) < 2:
        logger.error("i_combined_array is empty or has invalid shape")
        return {}, {}
        
    for i in range(i_combined_array.shape[1]):
        if i < len(i_scal_list) and i_scal_list[i] == True:
            
            # Progress logging
            prog_2 = scal_i / scal_all * 100 if scal_all > 0 else 0
            
            # Create and fit scaler (lines 1827-1830) with bounds checking
            min_val = i_scal_min_list[i] if i < len(i_scal_min_list) else 0
            max_val = i_scal_max_list[i] if i < len(i_scal_max_list) else 1
            # Ensure min < max
            if min_val >= max_val:
                min_val, max_val = 0, 1
            scaler = MinMaxScaler(feature_range=(min_val, max_val))
            scaler.fit_transform(i_combined_array[:, i].reshape(-1, 1))
            i_scalers[i] = scaler
            
            scal_i += 1
            
            # Progress logging
            prog_2 = scal_i / scal_all * 100 if scal_all > 0 else 0
        else:
            i_scalers[i] = None
    
    # Create empty dictionary for output scalers (line 1842)
    o_scalers = {}
    
    # Fit output scalers (lines 1844-1861)
    # Check if arrays have valid data before accessing shape[1]
    if o_combined_array.size == 0 or len(o_combined_array.shape) < 2:
        logger.error("o_combined_array is empty or has invalid shape")
        return {}, {}
        
    for i in range(o_combined_array.shape[1]):
        if i < len(o_scal_list) and o_scal_list[i] == True:
            
            # Progress logging
            prog_2 = scal_i / scal_all * 100 if scal_all > 0 else 0
            
            # Create and fit scaler (lines 1850-1853) with bounds checking
            min_val = o_scal_min_list[i] if i < len(o_scal_min_list) else 0
            max_val = o_scal_max_list[i] if i < len(o_scal_max_list) else 1
            # Ensure min < max
            if min_val >= max_val:
                min_val, max_val = 0, 1
            scaler = MinMaxScaler(feature_range=(min_val, max_val))
            scaler.fit_transform(o_combined_array[:, i].reshape(-1, 1))
            o_scalers[i] = scaler
            
            scal_i += 1
            
            # Progress logging
            prog_2 = scal_i / scal_all * 100 if scal_all > 0 else 0
        else:
            o_scalers[i] = None
    
    # Scalers prepared
    
    return i_scalers, o_scalers


def apply_scaling(i_array_3D: np.ndarray, o_array_3D: np.ndarray,
                 i_scalers: Dict, o_scalers: Dict,
                 i_dat_inf: pd.DataFrame, o_dat_inf: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply scaling to 3D arrays exactly as in training_original.py lines 2182-2210
    
    Args:
        i_array_3D: Input 3D array
        o_array_3D: Output 3D array
        i_scalers: Input scalers dictionary
        o_scalers: Output scalers dictionary
        i_dat_inf: Input data info
        o_dat_inf: Output data info
        
    Returns:
        Tuple of scaled (i_array_3D, o_array_3D)
    """
    
    # Validate arrays have the correct shape before accessing dimensions
    if i_array_3D.size == 0 or len(i_array_3D.shape) < 3:
        logger.error("i_array_3D is empty or has invalid shape for scaling")
        return i_array_3D, o_array_3D
    
    if o_array_3D.size == 0 or len(o_array_3D.shape) < 3:
        logger.error("o_array_3D is empty or has invalid shape for scaling")
        return i_array_3D, o_array_3D
    
    # Get dimensions
    n_dat = i_array_3D.shape[0]
    n_ft_i = i_array_3D.shape[2]
    n_ft_o = o_array_3D.shape[2]
    
    # Loop through datasets (lines 2182-2209)
    for i in range(n_dat):
        
        # Progress logging
        prog_3 = i / n_dat * 100
        
        # Loop through input features (lines 2188-2196)
        for i1 in range(n_ft_i):
            if i1 in i_scalers and i_scalers[i1] is not None:
                # Apply scaler (line 2193)
                std_i = i_scalers[i1].transform(i_array_3D[i, :, i1].reshape(-1, 1))
                # Overwrite column (line 2196)
                i_array_3D[i, :, i1] = std_i.ravel()
        
        # Loop through output features (lines 2199-2207)
        for i1 in range(len(o_dat_inf)):
            if i1 in o_scalers and o_scalers[i1] is not None:
                # Apply scaler (line 2204)
                std_i = o_scalers[i1].transform(o_array_3D[i, :, i1].reshape(-1, 1))
                # Overwrite column (line 2207)
                o_array_3D[i, :, i1] = std_i.ravel()
    
    # Progress logging
    prog_3 = 100
    
    return i_array_3D, o_array_3D


def process_and_scale_data(i_array_3D: np.ndarray, o_array_3D: np.ndarray,
                          i_combined_array: np.ndarray, o_combined_array: np.ndarray,
                          i_dat_inf: pd.DataFrame, o_dat_inf: pd.DataFrame,
                          random_dat: bool = False,
                          utc_ref_log: List = None) -> Dict:
    """
    Complete processing and scaling pipeline matching training_original.py lines 1764-2210
    
    Args:
        i_array_3D: Input 3D array
        o_array_3D: Output 3D array
        i_combined_array: Combined input array
        o_combined_array: Combined output array
        i_dat_inf: Input data info
        o_dat_inf: Output data info
        random_dat: Whether to shuffle data
        utc_ref_log: UTC reference log
        
    Returns:
        Dictionary with all processed arrays and scalers
    """
    
    # Get number of datasets
    n_dat = i_array_3D.shape[0]
    
    # SHUFFLE DATA BEFORE SCALING (lines 2162-2170)
    if random_dat == True:
        indices = np.random.permutation(n_dat)
        i_array_3D = i_array_3D[indices]
        o_array_3D = o_array_3D[indices]
        
        if utc_ref_log:
            utc_ref_log_int = copy.deepcopy(utc_ref_log)
            utc_ref_log = [utc_ref_log_int[i] for i in indices]
            del utc_ref_log_int
    
    # SAVE UNSCALED DATASETS (lines 2173-2175)
    i_array_3D_orig = copy.deepcopy(i_array_3D)
    o_array_3D_orig = copy.deepcopy(o_array_3D)
    
    # CREATE SCALING LISTS
    (i_scal_list, i_scal_max_list, i_scal_min_list,
     o_scal_list, o_scal_max_list, o_scal_min_list) = create_scaling_lists(i_dat_inf, o_dat_inf)
    
    # FIT SCALERS ON COMBINED ARRAYS
    i_scalers, o_scalers = fit_scalers(
        i_combined_array, o_combined_array,
        i_scal_list, i_scal_max_list, i_scal_min_list,
        o_scal_list, o_scal_max_list, o_scal_min_list
    )
    
    # APPLY SCALING TO 3D ARRAYS
    i_array_3D, o_array_3D = apply_scaling(
        i_array_3D, o_array_3D,
        i_scalers, o_scalers,
        i_dat_inf, o_dat_inf
    )
    
    return {
        'i_array_3D': i_array_3D,
        'o_array_3D': o_array_3D,
        'i_array_3D_orig': i_array_3D_orig,
        'o_array_3D_orig': o_array_3D_orig,
        'i_scalers': i_scalers,
        'o_scalers': o_scalers,
        'i_combined_array': i_combined_array,
        'o_combined_array': o_combined_array,
        'utc_ref_log': utc_ref_log,
        'n_dat': n_dat
    }


def get_session_scalers(session_id: str) -> Dict:
    """
    Retrieve saved scalers from database for a specific session.

    Args:
        session_id: Session identifier

    Returns:
        dict: {
            'input': input_scalers_dict,
            'output': output_scalers_dict,
            'metadata': {...}
        }
    """
    from utils.training_storage import fetch_training_results_with_storage

    # Get training results from Storage or legacy JSONB
    training_results = fetch_training_results_with_storage(session_id)

    if not training_results:
        raise ValueError(f'No training results found for session {session_id}')

    scalers = training_results.get('scalers', {})

    if not scalers:
        raise ValueError(f'No scalers found for session {session_id}')

    # Return scalers in serialized format (JSON-safe)
    input_scalers = scalers.get('input', {})
    output_scalers = scalers.get('output', {})

    return {
        'input': input_scalers,
        'output': output_scalers,
        'metadata': {
            'input_features': len(input_scalers),
            'output_features': len(output_scalers),
            'input_features_scaled': sum(1 for s in input_scalers.values() if s is not None),
            'output_features_scaled': sum(1 for s in output_scalers.values() if s is not None)
        }
    }


def create_scaler_download_package(session_id: str) -> str:
    """
    Create ZIP file with scaler .save files identical to original training_original.py format.

    Args:
        session_id: Session identifier

    Returns:
        str: Path to created ZIP file
    """
    from utils.training_storage import fetch_training_results_with_storage
    import pickle
    import base64
    import os
    import zipfile
    import tempfile
    from datetime import datetime

    # Get training results from Storage or legacy JSONB
    training_results = fetch_training_results_with_storage(session_id)

    if not training_results:
        raise ValueError(f'No training results found for session {session_id}')

    scalers = training_results.get('scalers', {})
    if not scalers:
        raise ValueError(f'No scalers found for session {session_id}')

    # Deserialize scalers back to original format
    def deserialize_scalers_dict(scaler_dict):
        result = {}
        for key, scaler_data in scaler_dict.items():
            if scaler_data and isinstance(scaler_data, dict) and '_model_type' in scaler_data:
                try:
                    scaler = pickle.loads(base64.b64decode(scaler_data['_model_data']))
                    result[int(key)] = scaler  # Convert key to int for original format
                except Exception as e:
                    logger.error(f"Error deserializing scaler {key}: {str(e)}")
                    result[int(key)] = None
            else:
                result[int(key)] = None
        return result

    input_scalers = deserialize_scalers_dict(scalers.get('input', {}))
    output_scalers = deserialize_scalers_dict(scalers.get('output', {}))

    # Create temporary directory for files
    temp_dir = tempfile.mkdtemp()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Create .save files identical to original format
    i_scale_file = os.path.join(temp_dir, f'i_scale_{timestamp}.save')
    o_scale_file = os.path.join(temp_dir, f'o_scale_{timestamp}.save')

    # Save input scalers
    with open(i_scale_file, 'wb') as f:
        pickle.dump(input_scalers, f)

    # Save output scalers
    with open(o_scale_file, 'wb') as f:
        pickle.dump(output_scalers, f)

    # Create ZIP file with both .save files
    zip_file = os.path.join(temp_dir, f'scalers_{session_id}_{timestamp}.zip')
    with zipfile.ZipFile(zip_file, 'w') as zipf:
        zipf.write(i_scale_file, f'i_scale_{timestamp}.save')
        zipf.write(o_scale_file, f'o_scale_{timestamp}.save')

    logger.info(f"Created scaler files for session {session_id}: {zip_file}")

    return zip_file


def scale_new_data(session_id: str, input_data, save_scaled: bool = False) -> Dict:
    """
    Scale input data using saved scalers.

    Args:
        session_id: Session identifier
        input_data: Input data to scale (list, dict, or array-like)
        save_scaled: Whether to save scaled data to file

    Returns:
        dict: {
            'scaled_data': scaled array,
            'scaling_info': {...},
            'metadata': {
                'original_shape': tuple,
                'scaled_shape': tuple,
                'features_scaled': int,
                'total_features': int,
                'saved_file_path': str or None
            }
        }
    """
    from utils.training_storage import fetch_training_results_with_storage
    import pickle
    import base64

    # Convert input_data to numpy array
    try:
        if isinstance(input_data, list):
            input_array = np.array(input_data)
        elif isinstance(input_data, dict):
            # Assume it's a pandas DataFrame-like structure
            input_array = np.array(list(input_data.values())).T
        else:
            input_array = np.array(input_data)
    except Exception as e:
        raise ValueError(f'Failed to convert input_data to array: {str(e)}')

    # Get training results from Storage or legacy JSONB
    training_results = fetch_training_results_with_storage(session_id)

    if not training_results:
        raise ValueError(f'No training results found for session {session_id}')

    scalers = training_results.get('scalers', {})
    input_scalers = scalers.get('input', {})

    if not input_scalers:
        raise ValueError(f'No input scalers found for session {session_id}')

    # Helper function to deserialize scalers
    def deserialize_scaler(scaler_data):
        """Convert serialized scaler back to usable object"""
        if scaler_data is None:
            return None
        elif isinstance(scaler_data, dict) and '_model_type' in scaler_data:
            # Deserialize pickled scaler
            try:
                model_b64 = scaler_data['_model_data']
                model_bytes = base64.b64decode(model_b64)
                scaler = pickle.loads(model_bytes)
                return scaler
            except Exception as e:
                logger.error(f"Error deserializing scaler: {str(e)}")
                return None
        else:
            return scaler_data

    # Scale the input data
    scaled_data = input_array.copy()
    scaling_info = {}

    for i in range(input_array.shape[1]):
        if str(i) in input_scalers:
            scaler = deserialize_scaler(input_scalers[str(i)])
            if scaler is not None:
                try:
                    # Scale the column
                    original_data = input_array[:, i].reshape(-1, 1)
                    scaled_column = scaler.transform(original_data)
                    scaled_data[:, i] = scaled_column.flatten()

                    scaling_info[f'feature_{i}'] = {
                        'scaled': True,
                        'original_range': [float(np.min(original_data)), float(np.max(original_data))],
                        'scaled_range': [float(np.min(scaled_column)), float(np.max(scaled_column))],
                        'feature_range': scaler.feature_range
                    }
                except Exception as e:
                    logger.error(f"Error scaling feature {i}: {str(e)}")
                    scaling_info[f'feature_{i}'] = {'scaled': False, 'error': str(e)}
            else:
                scaling_info[f'feature_{i}'] = {'scaled': False, 'reason': 'no_scaler'}
        else:
            scaling_info[f'feature_{i}'] = {'scaled': False, 'reason': 'scaler_not_found'}

    # Optionally save scaled data
    saved_file_path = None
    if save_scaled:
        try:
            import os
            from datetime import datetime

            # Create scaled data directory if it doesn't exist
            scaled_dir = f"temp_uploads/scaled_data_{session_id}"
            os.makedirs(scaled_dir, exist_ok=True)

            # Save as CSV
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_name = f"scaled_input_data_{timestamp}.csv"
            file_path = os.path.join(scaled_dir, file_name)

            # Create DataFrame and save
            scaled_df = pd.DataFrame(scaled_data, columns=[f'feature_{i}' for i in range(scaled_data.shape[1])])
            scaled_df.to_csv(file_path, index=False)
            saved_file_path = file_path

            logger.info(f"Scaled data saved to: {file_path}")
        except Exception as e:
            logger.error(f"Error saving scaled data: {str(e)}")

    return {
        'scaled_data': scaled_data.tolist(),  # Convert to list for JSON serialization
        'scaling_info': scaling_info,
        'metadata': {
            'original_shape': input_array.shape,
            'scaled_shape': scaled_data.shape,
            'features_scaled': sum(1 for info in scaling_info.values() if info.get('scaled', False)),
            'total_features': len(scaling_info),
            'saved_file_path': saved_file_path
        }
    }