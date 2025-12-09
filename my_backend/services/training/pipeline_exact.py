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
from sklearn.metrics import mean_absolute_error, mean_squared_error

from .data_loader import DataLoader, load, transf
from .data_transformer import create_training_arrays
from .scaler_manager import process_and_scale_data
from .model_trainer import (
    train_dense, train_cnn, train_lstm, train_ar_lstm,
    train_svr_dir, train_svr_mimo, train_linear_model
)
from .config import MDL, MTS, T, HOL

logger = logging.getLogger(__name__)


def calculate_evaluation_metrics(y_true, y_pred):
    """
    Calculate comprehensive evaluation metrics for model performance
    
    Args:
        y_true: True values (numpy array)
        y_pred: Predicted values (numpy array)
    
    Returns:
        Dictionary containing all evaluation metrics
    """
    if len(y_true.shape) == 1:
        y_true = y_true.reshape(-1, 1)
    if len(y_pred.shape) == 1:
        y_pred = y_pred.reshape(-1, 1)
    
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    
    mae = mean_absolute_error(y_true_flat, y_pred_flat)
    mse = mean_squared_error(y_true_flat, y_pred_flat)
    rmse = np.sqrt(mse)
    
    mask = y_true_flat != 0
    if np.any(mask):
        mape = np.mean(np.abs((y_true_flat[mask] - y_pred_flat[mask]) / y_true_flat[mask])) * 100
    else:
        mape = 0.0
    
    y_range = np.max(y_true_flat) - np.min(y_true_flat)
    if y_range > 0:
        nrmse = rmse / y_range
    else:
        nrmse = 0.0
    
    sum_abs_true = np.sum(np.abs(y_true_flat))
    if sum_abs_true > 0:
        wape = np.sum(np.abs(y_true_flat - y_pred_flat)) / sum_abs_true * 100
    else:
        wape = 0.0
    
    denominator = np.abs(y_true_flat) + np.abs(y_pred_flat)
    mask = denominator != 0
    if np.any(mask):
        smape = np.mean(2.0 * np.abs(y_true_flat[mask] - y_pred_flat[mask]) / denominator[mask]) * 100
    else:
        smape = 0.0
    
    if len(y_true_flat) > 1:
        naive_errors = np.abs(np.diff(y_true_flat))
        mae_naive = np.mean(naive_errors)
        if mae_naive > 0:
            mase = mae / mae_naive
        else:
            mase = 1.0
    else:
        mase = 1.0
    
    return {
        'MAE': float(mae),
        'MSE': float(mse),
        'RMSE': float(rmse),
        'MAPE': float(mape),
        'NRMSE': float(nrmse),
        'WAPE': float(wape),
        'sMAPE': float(smape),
        'MASE': float(mase)
    }


def run_exact_training_pipeline(
    i_dat: Dict[str, pd.DataFrame],
    o_dat: Dict[str, pd.DataFrame],
    i_dat_inf: pd.DataFrame,
    o_dat_inf: pd.DataFrame,
    utc_strt: datetime,
    utc_end: datetime,
    random_dat: bool = False,
    mdl_config: Optional[MDL] = None,
    socketio=None,
    session_id: str = None
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

    import logging
    logger = logging.getLogger(__name__)

    logger.info(f"   Pipeline Step 1: Creating training arrays from {utc_strt} to {utc_end}")
    try:
        (i_array_3D, o_array_3D,
         i_combined_array, o_combined_array,
         utc_ref_log) = create_training_arrays(
            i_dat, o_dat, i_dat_inf, o_dat_inf, utc_strt, utc_end,
            socketio=socketio,
            session_id=session_id
        )
        logger.info(f"   Pipeline Step 1 complete: Created arrays with shape {i_array_3D.shape if hasattr(i_array_3D, 'shape') else 'unknown'}")
    except Exception as e:
        logger.error(f"   ❌ Pipeline Step 1 FAILED: {str(e)}")
        import traceback
        logger.error(f"   Traceback: {traceback.format_exc()}")
        raise
    
    n_dat = i_array_3D.shape[0]
    
    scaling_result = process_and_scale_data(
        i_array_3D, o_array_3D,
        i_combined_array, o_combined_array,
        i_dat_inf, o_dat_inf,
        random_dat=random_dat,
        utc_ref_log=utc_ref_log
    )
    
    i_array_3D = scaling_result['i_array_3D']
    o_array_3D = scaling_result['o_array_3D']
    i_array_3D_orig = scaling_result['i_array_3D_orig']
    o_array_3D_orig = scaling_result['o_array_3D_orig']
    i_scalers = scaling_result['i_scalers']
    o_scalers = scaling_result['o_scalers']
    i_combined_array = scaling_result.get('i_combined_array')
    o_combined_array = scaling_result.get('o_combined_array')
    utc_ref_log = scaling_result['utc_ref_log']
    
    n_train = round(0.7 * n_dat)
    n_val = round(0.2 * n_dat)
    n_test = n_dat - n_train - n_val
    
    
    trn_x = i_array_3D[:n_train]
    val_x = i_array_3D[n_train:(n_train+n_val)]
    tst_x = i_array_3D[(n_train+n_val):]
    
    trn_y = o_array_3D[:n_train]
    val_y = o_array_3D[n_train:(n_train+n_val)]
    tst_y = o_array_3D[(n_train+n_val):]
    
    trn_x_orig = i_array_3D_orig[:n_train]
    val_x_orig = i_array_3D_orig[n_train:(n_train+n_val)]
    tst_x_orig = i_array_3D_orig[(n_train+n_val):]
    
    trn_y_orig = o_array_3D_orig[:n_train]
    val_y_orig = o_array_3D_orig[n_train:(n_train+n_val)]
    tst_y_orig = o_array_3D_orig[(n_train+n_val):]
    
    
    if mdl_config is None:
        mdl_config = MDL()
    
    # Create SocketIO callback for real-time progress updates
    socketio_callback = None
    if socketio is not None and session_id is not None:
        try:
            from .socketio_callback import SocketIOProgressCallback
            socketio_callback = SocketIOProgressCallback(
                socketio=socketio,
                session_id=session_id,
                total_epochs=mdl_config.EP,
                model_name=mdl_config.MODE
            )
            logger.info(f"   SocketIO callback created for session {session_id}")
        except Exception as e:
            logger.warning(f"   Failed to create SocketIO callback: {e}")
    
    mdl = None
    
    if mdl_config.MODE == "Dense":
        mdl = train_dense(trn_x, trn_y, val_x, val_y, mdl_config, socketio_callback=socketio_callback)
        
    elif mdl_config.MODE == "CNN":
        mdl = train_cnn(trn_x, trn_y, val_x, val_y, mdl_config, socketio_callback=socketio_callback)
        
    elif mdl_config.MODE == "LSTM":
        mdl = train_lstm(trn_x, trn_y, val_x, val_y, mdl_config, socketio_callback=socketio_callback)
        
    elif mdl_config.MODE == "AR LSTM":
        mdl = train_ar_lstm(trn_x, trn_y, val_x, val_y, mdl_config, socketio_callback=socketio_callback)
        
    elif mdl_config.MODE == "SVR_dir":
        mdl = train_svr_dir(trn_x, trn_y, mdl_config)
        
    elif mdl_config.MODE == "SVR_MIMO":
        mdl = train_svr_mimo(trn_x, trn_y, mdl_config)
        
    elif mdl_config.MODE == "LIN":
        mdl = train_linear_model(trn_x, trn_y)
    
    else:
        raise ValueError(f"Unknown model mode: {mdl_config.MODE}")
    
    evaluation_metrics = {}
    
    try:
        if mdl is not None:
            if mdl_config.MODE in ["Dense", "CNN", "LSTM", "AR LSTM"]:
                test_predictions = mdl.predict(tst_x, verbose=0)
            elif mdl_config.MODE == "SVR_dir":
                n_samples, n_timesteps, n_features_in = tst_x.shape
                tst_x_reshaped = tst_x.reshape(n_samples * n_timesteps, n_features_in)
                
                test_predictions = []
                for svr_model in mdl:
                    pred = svr_model.predict(tst_x_reshaped)
                    pred = pred.reshape(n_samples, n_timesteps)
                    test_predictions.append(pred)
                test_predictions = np.stack(test_predictions, axis=-1)
            elif mdl_config.MODE == "SVR_MIMO":
                n_samples, n_timesteps, n_features_in = tst_x.shape
                tst_x_reshaped = tst_x.reshape(n_samples * n_timesteps, n_features_in)
                
                test_predictions = []
                for svr_model in mdl:
                    pred = svr_model.predict(tst_x_reshaped)
                    pred = pred.reshape(n_samples, n_timesteps)
                    test_predictions.append(pred)
                
                test_predictions = np.stack(test_predictions, axis=-1)
            elif mdl_config.MODE == "LIN":
                n_samples, n_timesteps, n_features_in = tst_x.shape
                tst_x_reshaped = tst_x.reshape(n_samples * n_timesteps, n_features_in)
                
                test_predictions = []
                for lin_model in mdl:
                    pred = lin_model.predict(tst_x_reshaped)
                    pred = pred.reshape(n_samples, n_timesteps)
                    test_predictions.append(pred)
                
                test_predictions = np.stack(test_predictions, axis=-1)
            else:
                test_predictions = tst_y
            
            test_metrics = calculate_evaluation_metrics(tst_y, test_predictions)
            
            if tst_y_orig is not None:
                original_metrics = test_metrics
            else:
                original_metrics = test_metrics
            
            evaluation_metrics = {
                'test_metrics_scaled': test_metrics,
                'test_metrics_original': original_metrics,
                'model_type': mdl_config.MODE
            }
            
            if mdl_config.MODE in ["Dense", "CNN", "LSTM", "AR LSTM"]:
                val_predictions = mdl.predict(val_x, verbose=0)
            elif mdl_config.MODE == "SVR_dir":
                n_samples, n_timesteps, n_features_in = val_x.shape
                val_x_reshaped = val_x.reshape(n_samples * n_timesteps, n_features_in)
                
                val_predictions = []
                for svr_model in mdl:
                    pred = svr_model.predict(val_x_reshaped)
                    pred = pred.reshape(n_samples, n_timesteps)
                    val_predictions.append(pred)
                val_predictions = np.stack(val_predictions, axis=-1)
            elif mdl_config.MODE == "SVR_MIMO":
                n_samples, n_timesteps, n_features_in = val_x.shape
                val_x_reshaped = val_x.reshape(n_samples * n_timesteps, n_features_in)
                
                val_predictions = []
                for svr_model in mdl:
                    pred = svr_model.predict(val_x_reshaped)
                    pred = pred.reshape(n_samples, n_timesteps)
                    val_predictions.append(pred)
                
                val_predictions = np.stack(val_predictions, axis=-1)
            elif mdl_config.MODE == "LIN":
                n_samples, n_timesteps, n_features_in = val_x.shape
                val_x_reshaped = val_x.reshape(n_samples * n_timesteps, n_features_in)
                
                val_predictions = []
                for lin_model in mdl:
                    pred = lin_model.predict(val_x_reshaped)
                    pred = pred.reshape(n_samples, n_timesteps)
                    val_predictions.append(pred)
                
                val_predictions = np.stack(val_predictions, axis=-1)
            else:
                val_predictions = val_y
            
            val_metrics = calculate_evaluation_metrics(val_y, val_predictions)
            evaluation_metrics['val_metrics_scaled'] = val_metrics
            
    except Exception as e:
        logger.error(f"Error calculating evaluation metrics: {str(e)}")
        evaluation_metrics = {
            'test_metrics_scaled': {},
            'test_metrics_original': {},
            'val_metrics_scaled': {},
            'model_type': mdl_config.MODE if mdl_config else 'unknown',
            'error': str(e)
        }

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
            'output': o_scalers,
            'i_combined_array': i_combined_array,
            'o_combined_array': o_combined_array
        },
        'metadata': {
            'n_dat': n_dat,
            'n_train': n_train,
            'n_val': n_val,
            'n_test': n_test,
            'utc_ref_log': utc_ref_log,
            'model_config': mdl_config,
            'random_data': random_dat
        },
        'evaluation_metrics': evaluation_metrics,
        'metrics': evaluation_metrics
    }


def prepare_data_for_training(session_data: Dict) -> Tuple[Dict, Dict, pd.DataFrame, pd.DataFrame, datetime, datetime]:
    """
    Prepare data for the exact training pipeline
    
    Args:
        session_data: Session data containing file paths and configuration
        
    Returns:
        Tuple of (i_dat, o_dat, i_dat_inf, o_dat_inf, utc_strt, utc_end)
    """
    
    i_dat = {}
    o_dat = {}
    
    if 'input_files' in session_data:
        loader = DataLoader()
        for file_path in session_data['input_files']:
            df = pd.read_csv(file_path)
            if 'UTC' in df.columns:
                cols = df.columns.tolist()
                if 'UTC' != cols[0]:
                    cols.remove('UTC')
                    cols = ['UTC'] + cols
                    df = df[cols]
            i_dat[file_path] = df
    
    if 'output_files' in session_data:
        for file_path in session_data['output_files']:
            df = pd.read_csv(file_path)
            if 'UTC' in df.columns:
                cols = df.columns.tolist()
                if 'UTC' != cols[0]:
                    cols.remove('UTC')
                    cols = ['UTC'] + cols
                    df = df[cols]
            o_dat[file_path] = df
    
    i_dat_inf = pd.DataFrame()
    o_dat_inf = pd.DataFrame()
    
    for key in i_dat:
        i_dat[key], i_dat_inf = load(i_dat, i_dat_inf)
    
    for key in o_dat:
        o_dat[key], o_dat_inf = load(o_dat, o_dat_inf)
    
    if not i_dat_inf.empty:
        i_dat_inf["spec"] = i_dat_inf.get("spec", "").astype("object") if "spec" in i_dat_inf.columns else ""
        i_dat_inf["meth"] = i_dat_inf.get("meth", "").astype("object") if "meth" in i_dat_inf.columns else ""
        
        for key in i_dat_inf.index:
            i_dat_inf.loc[key, "spec"] = "Historische Daten"
            i_dat_inf.loc[key, "th_strt"] = -1
            i_dat_inf.loc[key, "th_end"] = 0
            i_dat_inf.loc[key, "meth"] = "Lineare Interpolation"
            i_dat_inf.loc[key, "avg"] = False
            i_dat_inf.loc[key, "scal"] = True
            i_dat_inf.loc[key, "scal_max"] = 1
            i_dat_inf.loc[key, "scal_min"] = 0
    
    if not o_dat_inf.empty:
        o_dat_inf["spec"] = o_dat_inf.get("spec", "").astype("object") if "spec" in o_dat_inf.columns else ""
        o_dat_inf["meth"] = o_dat_inf.get("meth", "").astype("object") if "meth" in o_dat_inf.columns else ""
        
        for key in o_dat_inf.index:
            o_dat_inf.loc[key, "spec"] = "Historische Daten"
            o_dat_inf.loc[key, "th_strt"] = 0
            o_dat_inf.loc[key, "th_end"] = 1
            o_dat_inf.loc[key, "meth"] = "Lineare Interpolation"
            o_dat_inf.loc[key, "avg"] = False
            o_dat_inf.loc[key, "scal"] = True
            o_dat_inf.loc[key, "scal_max"] = 1
            o_dat_inf.loc[key, "scal_min"] = 0
    
    mts_config = MTS()
    i_dat_inf = transf(i_dat_inf, mts_config.I_N, mts_config.OFST)
    o_dat_inf = transf(o_dat_inf, mts_config.O_N, mts_config.OFST)
    
    utc_strt = i_dat_inf['utc_min'].min()
    utc_end = i_dat_inf['utc_max'].max()
    
    if not o_dat_inf.empty:
        utc_strt = max(utc_strt, o_dat_inf['utc_min'].min())
        utc_end = min(utc_end, o_dat_inf['utc_max'].max())
    
    return i_dat, o_dat, i_dat_inf, o_dat_inf, utc_strt, utc_end
