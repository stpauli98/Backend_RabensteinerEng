"""
EXACT pipeline implementation matching training_original.py 100%
This module provides the complete training pipeline exactly as in the original script
"""

import os
import random
import numpy as np
import pandas as pd
import copy
import logging
import tensorflow as tf
from typing import Dict, Tuple, Optional
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error

# No fixed seed - matches original training.py which has natural variation

from domains.training.data.loader import DataLoader, load, transf
from domains.training.data.transformer import create_training_arrays
from domains.training.ml.scaler import process_and_scale_data
from domains.training.ml.evaluation import (
    calculate_evaluation_with_averaging,
    convert_df_eval_to_dict,
    convert_df_eval_ts_to_dict
)
from domains.training.ml.trainer import (
    train_dense, train_cnn, train_lstm, train_ar_lstm,
    train_svr_dir, train_svr_mimo, train_linear_model
)
from domains.training.config import MDL, MTS, T, HOL

logger = logging.getLogger(__name__)


def _build_input_feature_names(i_dat_inf, i_dat) -> list:
    """
    Build complete list of input feature names including TIME components.

    This matches how training arrays are built in transformer.py:
    1. File-based features (from i_dat_inf.index or i_dat.keys())
    2. TIME components if enabled in T config (y_sin, y_cos, m_sin, m_cos, etc.)

    Returns:
        List of feature names in the order they appear in training arrays
    """
    # Start with file-based features
    if hasattr(i_dat_inf, 'index'):
        feature_names = i_dat_inf.index.tolist()
    else:
        feature_names = list(i_dat.keys())

    # Add TIME component names based on T config (matches transformer.py order)
    # NOTE: UPPERCASE names to match original training.py
    if T.Y.IMP:
        feature_names.extend(['Y_sin', 'Y_cos'])
    if T.M.IMP:
        feature_names.extend(['M_sin', 'M_cos'])
    if T.W.IMP:
        feature_names.extend(['W_sin', 'W_cos'])
    if T.D.IMP:
        feature_names.extend(['D_sin', 'D_cos'])
    if T.H.IMP:
        feature_names.append('H')

    logger.info(f"Built input feature names: {feature_names}")
    return feature_names


def calculate_evaluation_metrics(y_true, y_pred):
    """
    Calculate comprehensive evaluation metrics for model performance

    Args:
        y_true: True values (numpy array) - shape (samples, timesteps, features) or (samples, timesteps)
        y_pred: Predicted values (numpy array) - same shape as y_true

    Returns:
        Dictionary containing:
        - Ukupne metrike: MAE, MSE, RMSE, MAPE, NRMSE, WAPE, sMAPE, MASE
        - _TS verzije: MAE_TS, MSE_TS, etc. - metrike po svakom vremenskom koraku
    """
    if len(y_true.shape) == 1:
        y_true = y_true.reshape(-1, 1)
    if len(y_pred.shape) == 1:
        y_pred = y_pred.reshape(-1, 1)

    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()

    # =========================================================================
    # UKUPNE METRIKE (GESAMT)
    # =========================================================================
    mae = mean_absolute_error(y_true_flat, y_pred_flat)
    mse = mean_squared_error(y_true_flat, y_pred_flat)
    rmse = np.sqrt(mse)

    mask = y_true_flat != 0
    if np.any(mask):
        mape = np.mean(np.abs((y_true_flat[mask] - y_pred_flat[mask]) / y_true_flat[mask])) * 100
    else:
        mape = 0.0

    # NRMSE normalizacija sa srednjom vrijednosti (kao u originalu)
    y_mean = np.mean(y_true_flat)
    if y_mean != 0:
        nrmse = rmse / y_mean
    else:
        nrmse = 0.0

    # WAPE - EXACT MATCH: returns np.nan when denominator=0 (original lines 564-575)
    numerator = np.sum(np.abs(y_true_flat - y_pred_flat))
    denominator = np.sum(np.abs(y_true_flat))
    if denominator == 0:
        wape = np.nan
    else:
        wape = (numerator / denominator) * 100

    # sMAPE - EXACT MATCH: includes zeros in average (original lines 579-594)
    n = len(y_true_flat)
    smape_values = []
    for yt, yp in zip(y_true_flat, y_pred_flat):
        denom = (abs(yt) + abs(yp)) / 2
        if denom == 0:
            smape_values.append(0)  # Include zero in average
        else:
            smape_values.append(abs(yp - yt) / denom)
    smape = sum(smape_values) / n * 100

    # MASE - EXACT MATCH: raises exceptions (original lines 597-617)
    m = 1  # Saisonalität
    n_mase = len(y_true_flat)
    mae_forecast = sum(abs(yt - yp) for yt, yp in zip(y_true_flat, y_pred_flat)) / n_mase

    if n_mase <= m:
        raise ValueError("Zu wenig Daten für gewählte Saisonalität m.")

    naive_errors = [abs(y_true_flat[t] - y_true_flat[t - m]) for t in range(m, n_mase)]
    mae_naive = sum(naive_errors) / len(naive_errors)

    if mae_naive == 0:
        raise ZeroDivisionError("Naive MAE ist 0 – MASE nicht definiert.")

    mase = mae_forecast / mae_naive

    # =========================================================================
    # _TS VERZIJE - METRIKE PO VREMENSKIM KORACIMA (ZEITSCHRITTE)
    # Kao u originalu: training_original.py linije 82-127
    # =========================================================================
    mae_ts, mape_ts, mse_ts, rmse_ts = [], [], [], []
    nrmse_ts, wape_ts, smape_ts, mase_ts = [], [], [], []

    # Određivanje broja timestepova ovisno o shape-u
    if len(y_true.shape) == 3:
        # Shape: (samples, timesteps, features)
        num_timesteps = y_true.shape[1]

        for i_ts in range(num_timesteps):
            v_true = y_true[:, i_ts, :].flatten()
            v_pred = y_pred[:, i_ts, :].flatten()

            # Izračunaj metrike za ovaj timestep
            ts_metrics = _calculate_single_timestep_metrics(v_true, v_pred)

            mae_ts.append(ts_metrics['mae'])
            mape_ts.append(ts_metrics['mape'])
            mse_ts.append(ts_metrics['mse'])
            rmse_ts.append(ts_metrics['rmse'])
            nrmse_ts.append(ts_metrics['nrmse'])
            wape_ts.append(ts_metrics['wape'])
            smape_ts.append(ts_metrics['smape'])
            mase_ts.append(ts_metrics['mase'])

    elif len(y_true.shape) == 2:
        # Shape: (samples, timesteps)
        num_timesteps = y_true.shape[1]

        for i_ts in range(num_timesteps):
            v_true = y_true[:, i_ts].flatten()
            v_pred = y_pred[:, i_ts].flatten()

            ts_metrics = _calculate_single_timestep_metrics(v_true, v_pred)

            mae_ts.append(ts_metrics['mae'])
            mape_ts.append(ts_metrics['mape'])
            mse_ts.append(ts_metrics['mse'])
            rmse_ts.append(ts_metrics['rmse'])
            nrmse_ts.append(ts_metrics['nrmse'])
            wape_ts.append(ts_metrics['wape'])
            smape_ts.append(ts_metrics['smape'])
            mase_ts.append(ts_metrics['mase'])

    return {
        # Ukupne metrike
        'MAE': float(mae),
        'MSE': float(mse),
        'RMSE': float(rmse),
        'MAPE': float(mape),
        'NRMSE': float(nrmse),
        'WAPE': float(wape),
        'sMAPE': float(smape),
        'MASE': float(mase),
        # _TS verzije (po vremenskim koracima)
        'MAE_TS': mae_ts,
        'MAPE_TS': mape_ts,
        'MSE_TS': mse_ts,
        'RMSE_TS': rmse_ts,
        'NRMSE_TS': nrmse_ts,
        'WAPE_TS': wape_ts,
        'sMAPE_TS': smape_ts,
        'MASE_TS': mase_ts
    }


def _calculate_single_timestep_metrics(v_true, v_pred):
    """
    Pomoćna funkcija za izračun metrika za jedan vremenski korak.
    Koristi se u _TS verzijama metrika.

    Args:
        v_true: True values za jedan timestep (1D array)
        v_pred: Predicted values za jedan timestep (1D array)

    Returns:
        Dict sa svim metrikama za taj timestep
    """
    # MAE
    ts_mae = float(mean_absolute_error(v_true, v_pred))

    # MSE
    ts_mse = float(mean_squared_error(v_true, v_pred))

    # RMSE
    ts_rmse = float(np.sqrt(ts_mse))

    # MAPE
    mask = v_true != 0
    if np.any(mask):
        ts_mape = float(np.mean(np.abs((v_true[mask] - v_pred[mask]) / v_true[mask])) * 100)
    else:
        ts_mape = 0.0

    # NRMSE (normalizacija sa mean, kao u originalu)
    v_mean = np.mean(v_true)
    if v_mean != 0:
        ts_nrmse = float(ts_rmse / v_mean)
    else:
        ts_nrmse = 0.0

    # WAPE - EXACT MATCH: returns np.nan when denominator=0 (original lines 564-575)
    numerator = np.sum(np.abs(v_true - v_pred))
    denominator = np.sum(np.abs(v_true))
    if denominator == 0:
        ts_wape = float('nan')
    else:
        ts_wape = float((numerator / denominator) * 100)

    # sMAPE - EXACT MATCH: includes zeros in average (original lines 579-594)
    n_smape = len(v_true)
    smape_values = []
    for yt, yp in zip(v_true, v_pred):
        denom = (abs(yt) + abs(yp)) / 2
        if denom == 0:
            smape_values.append(0)  # Include zero in average
        else:
            smape_values.append(abs(yp - yt) / denom)
    ts_smape = float(sum(smape_values) / n_smape * 100)

    # MASE - EXACT MATCH: raises exceptions (original lines 597-617)
    m = 1  # Saisonalität
    n_mase = len(v_true)
    mae_forecast = sum(abs(yt - yp) for yt, yp in zip(v_true, v_pred)) / n_mase

    if n_mase <= m:
        raise ValueError("Zu wenig Daten für gewählte Saisonalität m.")

    naive_errors = [abs(v_true[t] - v_true[t - m]) for t in range(m, n_mase)]
    mae_naive_val = sum(naive_errors) / len(naive_errors)

    if mae_naive_val == 0:
        raise ZeroDivisionError("Naive MAE ist 0 – MASE nicht definiert.")

    ts_mase = float(mae_forecast / mae_naive_val)

    return {
        'mae': ts_mae,
        'mape': ts_mape,
        'mse': ts_mse,
        'rmse': ts_rmse,
        'nrmse': ts_nrmse,
        'wape': ts_wape,
        'smape': ts_smape,
        'mase': ts_mase
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
    mts_config: Optional['MTS'] = None,
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
        mts_config: MTS configuration (if None, uses default MTS())
        
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

    # No fixed seeds - matches original training.py which has natural variation each run

    logger.info(f"   Pipeline Step 1: Creating training arrays from {utc_strt} to {utc_end}")
    try:
        (i_array_3D, o_array_3D,
         i_combined_array, o_combined_array,
         utc_ref_log) = create_training_arrays(
            i_dat, o_dat, i_dat_inf, o_dat_inf, utc_strt, utc_end,
            socketio=socketio,
            session_id=session_id,
            mts_config=mts_config
        )
        logger.info(f"   Pipeline Step 1 complete: Created arrays with shape {i_array_3D.shape if hasattr(i_array_3D, 'shape') else 'unknown'}")
    except Exception as e:
        logger.error(f"   ❌ Pipeline Step 1 FAILED: {str(e)}")
        import traceback
        logger.error(f"   Traceback: {traceback.format_exc()}")
        raise
    
    n_dat = i_array_3D.shape[0]
    n_timesteps = i_array_3D.shape[1]
    n_features_in = i_array_3D.shape[2]
    n_features_out = o_array_3D.shape[2]

    logger.info(f"=== ARRAY SHAPES BEFORE SCALING ===")
    logger.info(f"Input:  n_dat={n_dat}, n_timesteps={n_timesteps}, n_features={n_features_in}")
    logger.info(f"Output: n_dat={n_dat}, n_timesteps={o_array_3D.shape[1]}, n_features={n_features_out}")
    logger.info(f"=== END ARRAY SHAPES ===")

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
            from domains.training.services.socketio import SocketIOProgressCallback
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

            # =========================================================================
            # 12-LEVEL AVERAGING EVALUATION - EXACT COPY FROM ORIGINAL training.py
            # Lines 3312-3552 from original
            # =========================================================================
            try:
                # Get shapes for evaluation
                n_tst = tst_y_orig.shape[0] if tst_y_orig is not None else tst_y.shape[0]
                O_N = o_array_3D.shape[1]  # Actual output timesteps from data

                # Prepare tst_y_orig for evaluation - needs shape (n_tst, O_N, num_feat)
                if tst_y_orig is not None:
                    eval_y_orig = tst_y_orig
                else:
                    eval_y_orig = tst_y

                # Prepare predictions for evaluation - same shape as y_orig
                # CRITICAL: Inverse scale predictions to match original scale
                # This matches training_original.py lines 2313-2331 RE-SCALING section
                if o_scalers is not None and len(o_scalers) > 0:
                    test_predictions_orig = np.copy(test_predictions)
                    n_tst_samples = test_predictions.shape[0]
                    n_ft_o = test_predictions.shape[-1] if len(test_predictions.shape) > 2 else 1

                    logger.info(f"Inverse scaling predictions: {n_tst_samples} samples, {n_ft_o} features")

                    for i in range(n_tst_samples):
                        for i1 in range(n_ft_o):
                            if i1 in o_scalers and o_scalers[i1] is not None:
                                if len(test_predictions.shape) == 3:
                                    test_predictions_orig[i, :, i1] = o_scalers[i1].inverse_transform(
                                        test_predictions[i, :, i1].reshape(-1, 1)
                                    ).ravel()
                                elif len(test_predictions.shape) == 2:
                                    test_predictions_orig[i, :] = o_scalers[0].inverse_transform(
                                        test_predictions[i, :].reshape(-1, 1)
                                    ).ravel()

                    eval_fcst = test_predictions_orig
                    logger.info(f"Predictions inverse scaled. Original range: [{np.min(test_predictions):.4f}, {np.max(test_predictions):.4f}] -> [{np.min(eval_fcst):.2f}, {np.max(eval_fcst):.2f}]")
                else:
                    eval_fcst = test_predictions
                    logger.warning("No output scalers available - using scaled predictions for evaluation")

                # Ensure correct shape: (n_tst, O_N, num_feat)
                if len(eval_y_orig.shape) == 2:
                    # Shape is (n_tst, O_N) - add feature dimension
                    eval_y_orig = eval_y_orig[:, :, np.newaxis]
                if len(eval_fcst.shape) == 2:
                    eval_fcst = eval_fcst[:, :, np.newaxis]

                # Call the 12-level averaging evaluation
                dat_eval, df_eval, df_eval_ts = calculate_evaluation_with_averaging(
                    tst_y_orig=eval_y_orig,
                    tst_fcst=eval_fcst,
                    o_dat_inf=o_dat_inf,
                    O_N=O_N,
                    n_max=12
                )

                # Convert to JSON-serializable format
                df_eval_dict = convert_df_eval_to_dict(df_eval)
                df_eval_ts_dict = convert_df_eval_ts_to_dict(df_eval_ts)

                evaluation_metrics['df_eval'] = df_eval_dict
                evaluation_metrics['df_eval_ts'] = df_eval_ts_dict

                logger.info(f"12-level averaging evaluation completed successfully")

            except Exception as eval_error:
                logger.error(f"Error in 12-level averaging evaluation: {str(eval_error)}")
                import traceback
                logger.error(traceback.format_exc())
                evaluation_metrics['df_eval'] = {}
                evaluation_metrics['df_eval_ts'] = {}

    except Exception as e:
        logger.error(f"Error calculating evaluation metrics: {str(e)}")
        evaluation_metrics = {
            'test_metrics_scaled': {},
            'test_metrics_original': {},
            'val_metrics_scaled': {},
            'model_type': mdl_config.MODE if mdl_config else 'unknown',
            'error': str(e),
            'df_eval': {},
            'df_eval_ts': {}
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
            'random_data': random_dat,
            # Build complete input feature names (files + TIME components)
            'input_features': _build_input_feature_names(i_dat_inf, i_dat),
            'output_features': o_dat_inf.index.tolist() if hasattr(o_dat_inf, 'index') else list(o_dat.keys())
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
            # Use just filename without path and extension (matches original training.py)
            feature_name = os.path.splitext(os.path.basename(file_path))[0]
            i_dat[feature_name] = df

    if 'output_files' in session_data:
        for file_path in session_data['output_files']:
            df = pd.read_csv(file_path)
            if 'UTC' in df.columns:
                cols = df.columns.tolist()
                if 'UTC' != cols[0]:
                    cols.remove('UTC')
                    cols = ['UTC'] + cols
                    df = df[cols]
            # Use just filename without path and extension (matches original training.py)
            feature_name = os.path.splitext(os.path.basename(file_path))[0]
            o_dat[feature_name] = df
    
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
    
    # Configure MTS from session_data if zeitschritte is available
    zeitschritte = session_data.get('zeitschritte', {})
    mts_config = MTS()
    mts_config.I_N = int(zeitschritte.get('eingabe', mts_config.I_N))
    mts_config.O_N = int(zeitschritte.get('ausgabe', mts_config.O_N))
    mts_config.DELT = float(zeitschritte.get('zeitschrittweite', mts_config.DELT))
    mts_config.OFST = float(zeitschritte.get('offset', mts_config.OFST))

    i_dat_inf = transf(i_dat_inf, mts_config.I_N, mts_config.OFST)
    o_dat_inf = transf(o_dat_inf, mts_config.O_N, mts_config.OFST)
    
    utc_strt = i_dat_inf['utc_min'].min()
    utc_end = i_dat_inf['utc_max'].max()
    
    if not o_dat_inf.empty:
        utc_strt = max(utc_strt, o_dat_inf['utc_min'].min())
        utc_end = min(utc_end, o_dat_inf['utc_max'].max())
    
    return i_dat, o_dat, i_dat_inf, o_dat_inf, utc_strt, utc_end
