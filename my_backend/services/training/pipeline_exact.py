"""
EXACT pipeline implementation matching training_original.py 100%
This module provides the complete training pipeline exactly as in the original script
"""

import numpy as np
import pandas as pd
import copy
import logging
from typing import Dict, Tuple, Optional
from datetime import datetime

# Import all necessary modules
from .data_loader import DataLoader, load, transf
from .data_transformer import create_training_arrays
from .scaler_manager import process_and_scale_data
from .model_trainer import (
    train_dense, train_cnn, train_lstm, train_ar_lstm,
    train_svr_dir, train_svr_mimo, train_linear_model
)
from .config import MDL, MTS, T, HOL

logger = logging.getLogger(__name__)


def run_exact_training_pipeline(
    i_dat: Dict[str, pd.DataFrame],
    o_dat: Dict[str, pd.DataFrame],
    i_dat_inf: pd.DataFrame,
    o_dat_inf: pd.DataFrame,
    utc_strt: datetime,
    utc_end: datetime,
    random_dat: bool = False,
    mdl_config: Optional[MDL] = None
) -> Dict:
    """
    Run the EXACT training pipeline matching training_original.py lines 1068-2260
    
    This function implements the complete flow:
    1. Create training arrays using main time loop (lines 1068-1760)
    2. Process and scale data (lines 1764-2210)
    3. Split into train/val/test (lines 2040-2042, 2217-2234)
    4. Train models (lines 2240-2259)
    
    Args:
        i_dat: Input data dictionary with DataFrames
        o_dat: Output data dictionary with DataFrames
        i_dat_inf: Input data info DataFrame
        o_dat_inf: Output data info DataFrame
        utc_strt: Start UTC timestamp
        utc_end: End UTC timestamp
        random_dat: Whether to shuffle data (default: False)
        mdl_config: Model configuration (if None, uses default MDL())
        
    Returns:
        Dictionary containing:
        - trained_model: The trained model
        - train_data: Training data (scaled and original)
        - val_data: Validation data (scaled and original)
        - test_data: Test data (scaled and original)
        - scalers: Input and output scalers
        - metadata: Additional training metadata
    """
    
    logger.info("Starting EXACT training pipeline matching training_original.py")
    
    # Step 1: Create training arrays using main time loop (lines 1068-1760)
    logger.info("Step 1: Creating training arrays with main time loop...")
    (i_array_3D, o_array_3D, 
     i_combined_array, o_combined_array, 
     utc_ref_log) = create_training_arrays(
        i_dat, o_dat, i_dat_inf, o_dat_inf, utc_strt, utc_end
    )
    
    # Get number of datasets (line 1757)
    n_dat = i_array_3D.shape[0]
    logger.info(f"Created {n_dat} datasets")
    
    # Step 2: Process and scale data (lines 1764-2210)
    logger.info("Step 2: Processing and scaling data...")
    scaling_result = process_and_scale_data(
        i_array_3D, o_array_3D,
        i_combined_array, o_combined_array,
        i_dat_inf, o_dat_inf,
        random_dat=random_dat,
        utc_ref_log=utc_ref_log
    )
    
    # Extract scaled and original arrays
    i_array_3D = scaling_result['i_array_3D']
    o_array_3D = scaling_result['o_array_3D']
    i_array_3D_orig = scaling_result['i_array_3D_orig']
    o_array_3D_orig = scaling_result['o_array_3D_orig']
    i_scalers = scaling_result['i_scalers']
    o_scalers = scaling_result['o_scalers']
    utc_ref_log = scaling_result['utc_ref_log']
    
    # Step 3: Calculate split indices (lines 2040-2042)
    logger.info("Step 3: Splitting data into train/val/test...")
    n_train = round(0.7 * n_dat)  # EXACT as line 2040
    n_val = round(0.2 * n_dat)    # EXACT as line 2041
    n_test = n_dat - n_train - n_val  # EXACT as line 2042
    
    logger.info(f"Split: train={n_train}, val={n_val}, test={n_test}")
    
    # SCALED DATASETS (lines 2217-2224)
    trn_x = i_array_3D[:n_train]
    val_x = i_array_3D[n_train:(n_train+n_val)]
    tst_x = i_array_3D[(n_train+n_val):]
    
    trn_y = o_array_3D[:n_train]
    val_y = o_array_3D[n_train:(n_train+n_val)]
    tst_y = o_array_3D[(n_train+n_val):]
    
    # ORIGINAL (UNSCALED) DATASETS (lines 2228-2234)
    trn_x_orig = i_array_3D_orig[:n_train]
    val_x_orig = i_array_3D_orig[n_train:(n_train+n_val)]
    tst_x_orig = i_array_3D_orig[(n_train+n_val):]
    
    trn_y_orig = o_array_3D_orig[:n_train]
    val_y_orig = o_array_3D_orig[n_train:(n_train+n_val)]
    tst_y_orig = o_array_3D_orig[(n_train+n_val):]
    
    # Step 4: Train model based on MDL.MODE (lines 2240-2259)
    logger.info("Step 4: Training model...")
    
    if mdl_config is None:
        mdl_config = MDL()  # Use default "LIN" mode
    
    mdl = None
    
    if mdl_config.MODE == "Dense":
        logger.info("Training Dense neural network...")
        mdl = train_dense(trn_x, trn_y, val_x, val_y, mdl_config)
        
    elif mdl_config.MODE == "CNN":
        logger.info("Training CNN...")
        mdl = train_cnn(trn_x, trn_y, val_x, val_y, mdl_config)
        
    elif mdl_config.MODE == "LSTM":
        logger.info("Training LSTM...")
        mdl = train_lstm(trn_x, trn_y, val_x, val_y, mdl_config)
        
    elif mdl_config.MODE == "AR LSTM":
        logger.info("Training AR LSTM...")
        mdl = train_ar_lstm(trn_x, trn_y, val_x, val_y, mdl_config)
        
    elif mdl_config.MODE == "SVR_dir":
        logger.info("Training SVR_dir...")
        mdl = train_svr_dir(trn_x, trn_y, mdl_config)
        
    elif mdl_config.MODE == "SVR_MIMO":
        logger.info("Training SVR_MIMO...")
        mdl = train_svr_mimo(trn_x, trn_y, mdl_config)
        
    elif mdl_config.MODE == "LIN":
        logger.info("Training Linear model...")
        mdl = train_linear_model(trn_x, trn_y)
    
    else:
        raise ValueError(f"Unknown model mode: {mdl_config.MODE}")
    
    logger.info("Training complete!")
    
    # Return all results
    return {
        'trained_model': mdl,
        'train_data': {
            'X': trn_x,
            'y': trn_y,
            'X_orig': trn_x_orig,
            'y_orig': trn_y_orig
        },
        'val_data': {
            'X': val_x,
            'y': val_y,
            'X_orig': val_x_orig,
            'y_orig': val_y_orig
        },
        'test_data': {
            'X': tst_x,
            'y': tst_y,
            'X_orig': tst_x_orig,
            'y_orig': tst_y_orig
        },
        'scalers': {
            'input': i_scalers,
            'output': o_scalers
        },
        'metadata': {
            'n_dat': n_dat,
            'n_train': n_train,
            'n_val': n_val,
            'n_test': n_test,
            'utc_ref_log': utc_ref_log,
            'model_config': mdl_config,
            'random_data': random_dat
        }
    }


def prepare_data_for_training(session_data: Dict) -> Tuple[Dict, Dict, pd.DataFrame, pd.DataFrame, datetime, datetime]:
    """
    Prepare data for the exact training pipeline
    
    Args:
        session_data: Session data containing file paths and configuration
        
    Returns:
        Tuple of (i_dat, o_dat, i_dat_inf, o_dat_inf, utc_strt, utc_end)
    """
    
    # Initialize data structures
    i_dat = {}
    o_dat = {}
    
    # Load input data
    if 'input_files' in session_data:
        loader = DataLoader()
        for file_path in session_data['input_files']:
            df = pd.read_csv(file_path)
            # Ensure UTC column is first, value column is second
            if 'UTC' in df.columns:
                # Reorder columns to match original expectation
                cols = df.columns.tolist()
                if 'UTC' != cols[0]:
                    # Move UTC to first position
                    cols.remove('UTC')
                    cols = ['UTC'] + cols
                    df = df[cols]
            i_dat[file_path] = df
    
    # Load output data
    if 'output_files' in session_data:
        for file_path in session_data['output_files']:
            df = pd.read_csv(file_path)
            # Ensure UTC column is first, value column is second
            if 'UTC' in df.columns:
                cols = df.columns.tolist()
                if 'UTC' != cols[0]:
                    cols.remove('UTC')
                    cols = ['UTC'] + cols
                    df = df[cols]
            o_dat[file_path] = df
    
    # Create info DataFrames using load() function
    i_dat_inf = pd.DataFrame()
    o_dat_inf = pd.DataFrame()
    
    # Process each input file
    for key in i_dat:
        i_dat[key], i_dat_inf = load(i_dat, i_dat_inf)
    
    # Process each output file
    for key in o_dat:
        o_dat[key], o_dat_inf = load(o_dat, o_dat_inf)
    
    # Set up required columns for data_transformer (matching training_original.py)
    # These columns must be set after load() but before transf()
    if not i_dat_inf.empty:
        # Convert to object dtype to avoid warnings
        i_dat_inf["spec"] = i_dat_inf.get("spec", "").astype("object") if "spec" in i_dat_inf.columns else ""
        i_dat_inf["meth"] = i_dat_inf.get("meth", "").astype("object") if "meth" in i_dat_inf.columns else ""
        
        # Set values for all input files
        for key in i_dat_inf.index:
            i_dat_inf.loc[key, "spec"] = "Historische Daten"
            i_dat_inf.loc[key, "th_strt"] = -1  # Time horizon start (hours before reference)
            i_dat_inf.loc[key, "th_end"] = 0    # Time horizon end (at reference time)
            i_dat_inf.loc[key, "meth"] = "Lineare Interpolation"
            i_dat_inf.loc[key, "avg"] = False   # No averaging by default
            i_dat_inf.loc[key, "scal"] = True   # Enable scaling
            i_dat_inf.loc[key, "scal_max"] = 1  # Max scaling value
            i_dat_inf.loc[key, "scal_min"] = 0  # Min scaling value
    
    if not o_dat_inf.empty:
        # Convert to object dtype to avoid warnings
        o_dat_inf["spec"] = o_dat_inf.get("spec", "").astype("object") if "spec" in o_dat_inf.columns else ""
        o_dat_inf["meth"] = o_dat_inf.get("meth", "").astype("object") if "meth" in o_dat_inf.columns else ""
        
        # Set values for all output files
        for key in o_dat_inf.index:
            o_dat_inf.loc[key, "spec"] = "Historische Daten"
            o_dat_inf.loc[key, "th_strt"] = 0   # Time horizon start (at reference time)
            o_dat_inf.loc[key, "th_end"] = 1    # Time horizon end (1 hour after reference)
            o_dat_inf.loc[key, "meth"] = "Lineare Interpolation"
            o_dat_inf.loc[key, "avg"] = False   # No averaging by default
            o_dat_inf.loc[key, "scal"] = True   # Enable scaling
            o_dat_inf.loc[key, "scal_max"] = 1  # Max scaling value
            o_dat_inf.loc[key, "scal_min"] = 0  # Min scaling value
    
    # Apply transformations using transf()
    mts_config = MTS()
    i_dat_inf = transf(i_dat_inf, mts_config.I_N, mts_config.OFST)
    o_dat_inf = transf(o_dat_inf, mts_config.O_N, mts_config.OFST)
    
    # Determine time boundaries
    utc_strt = i_dat_inf['utc_min'].min()
    utc_end = i_dat_inf['utc_max'].max()
    
    # Adjust based on output data if available
    if not o_dat_inf.empty:
        utc_strt = max(utc_strt, o_dat_inf['utc_min'].min())
        utc_end = min(utc_end, o_dat_inf['utc_max'].max())
    
    return i_dat, o_dat, i_dat_inf, o_dat_inf, utc_strt, utc_end