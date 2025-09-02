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
    for i in range(i_combined_array.shape[1]):
        if i < len(i_scal_list) and i_scal_list[i] == True:
            
            # Progress logging
            prog_2 = scal_i / scal_all * 100 if scal_all > 0 else 0
            logger.debug(f"Setting scaler: {prog_2:.2f}%")
            
            # Create and fit scaler (lines 1827-1830)
            scaler = MinMaxScaler(feature_range=(i_scal_min_list[i], i_scal_max_list[i]))
            scaler.fit_transform(i_combined_array[:, i].reshape(-1, 1))
            i_scalers[i] = scaler
            
            scal_i += 1
            
            # Progress logging
            prog_2 = scal_i / scal_all * 100 if scal_all > 0 else 0
            logger.debug(f"Setting scaler: {prog_2:.2f}%")
        else:
            i_scalers[i] = None
    
    # Create empty dictionary for output scalers (line 1842)
    o_scalers = {}
    
    # Fit output scalers (lines 1844-1861)
    for i in range(o_combined_array.shape[1]):
        if i < len(o_scal_list) and o_scal_list[i] == True:
            
            # Progress logging
            prog_2 = scal_i / scal_all * 100 if scal_all > 0 else 0
            logger.debug(f"Setting scaler: {prog_2:.2f}%")
            
            # Create and fit scaler (lines 1850-1853)
            scaler = MinMaxScaler(feature_range=(o_scal_min_list[i], o_scal_max_list[i]))
            scaler.fit_transform(o_combined_array[:, i].reshape(-1, 1))
            o_scalers[i] = scaler
            
            scal_i += 1
            
            # Progress logging
            prog_2 = scal_i / scal_all * 100 if scal_all > 0 else 0
            logger.debug(f"Setting scaler: {prog_2:.2f}%")
        else:
            o_scalers[i] = None
    
    logger.info(f"Created {len([s for s in i_scalers.values() if s is not None])} input scalers "
                f"and {len([s for s in o_scalers.values() if s is not None])} output scalers")
    
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
    
    # Get dimensions
    n_dat = i_array_3D.shape[0]
    n_ft_i = i_array_3D.shape[2]
    
    # Loop through datasets (lines 2182-2209)
    for i in range(n_dat):
        
        # Progress logging
        prog_3 = i / n_dat * 100
        logger.debug(f"Scaling datasets: {prog_3:.2f}%")
        
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
    logger.debug(f"Scaling datasets: {prog_3:.2f}%")
    
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
        'utc_ref_log': utc_ref_log,
        'n_dat': n_dat
    }