"""
Evaluation module - EXACT COPY from training.py lines 3312-3552
Implements 12-level averaging evaluation system.

This module calculates evaluation metrics with multiple averaging levels,
exactly matching the original training.py implementation.

Created: 2026-01-15
Updated: 2026-01-21 - Fixed to use sklearn functions directly like original
"""

import math
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any

# CRITICAL: Import sklearn functions EXACTLY like the original training.py (lines 18-21)
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import root_mean_squared_error as rmse

logger = logging.getLogger(__name__)


# =============================================================================
# METRIC FUNCTIONS - Using sklearn directly like original training.py
# =============================================================================
# NOTE: mae, mape, mse, rmse are now imported directly from sklearn
# This matches the original training.py lines 18-21:
#   from sklearn.metrics import mean_absolute_error as mae
#   from sklearn.metrics import mean_absolute_percentage_error as mape
#   from sklearn.metrics import mean_squared_error as mse
#   from sklearn.metrics import root_mean_squared_error as rmse


def wape(y_true, y_pred):
    """
    Weighted Absolute Percentage Error
    EXACT COPY from original training.py lines 564-575
    """
    y_true = np.array(y_true, dtype=np.float64)
    y_pred = np.array(y_pred, dtype=np.float64)

    numerator = np.sum(np.abs(y_true - y_pred))
    denominator = np.sum(np.abs(y_true))

    if denominator == 0:
        return np.nan  # Original returns np.nan, not 0.0

    return (numerator / denominator) * 100


def smape(y_true, y_pred):
    """
    Symmetric Mean Absolute Percentage Error
    EXACT COPY from original training.py lines 579-594
    """
    y_true = np.array(y_true, dtype=np.float64)
    y_pred = np.array(y_pred, dtype=np.float64)

    n = len(y_true)
    smape_values = []

    for yt, yp in zip(y_true, y_pred):
        denominator = (abs(yt) + abs(yp)) / 2
        if denominator == 0:
            smape_values.append(0)  # Include zeros in average
        else:
            smape_values.append(abs(yp - yt) / denominator)

    return sum(smape_values) / n * 100


def mase(y_true, y_pred, m=1):
    """
    Mean Absolute Scaled Error
    EXACT COPY from original training.py lines 597-617
    """
    y_true = np.array(y_true, dtype=np.float64)
    y_pred = np.array(y_pred, dtype=np.float64)

    n = len(y_true)

    # Vorhersagefehler (MAE der Prognose)
    mae_forecast = sum(abs(yt - yp) for yt, yp in zip(y_true, y_pred)) / n

    # MAE des Naive-m-Modells (Baseline)
    if n <= m:
        raise ValueError("Zu wenig Daten für gewählte Saisonalität m.")

    naive_errors = [abs(y_true[t] - y_true[t - m]) for t in range(m, n)]
    mae_naive = sum(naive_errors) / len(naive_errors)

    if mae_naive == 0:
        raise ZeroDivisionError("Naive MAE ist 0 – MASE nicht definiert.")

    return mae_forecast / mae_naive


# =============================================================================
# MAIN EVALUATION FUNCTION - EXACT COPY from training.py lines 3312-3552
# =============================================================================

def calculate_evaluation_with_averaging(
    tst_y_orig: np.ndarray,
    tst_fcst: np.ndarray,
    o_dat_inf: pd.DataFrame,
    O_N: int = 13,
    n_max: int = 12
) -> Tuple[Dict, Dict, Dict]:
    """
    Calculate evaluation metrics with 12-level averaging.

    EXACT COPY from original training.py lines 3312-3552.

    This function:
    1. Averages predictions at 12 different levels (n_avg = 1 to 12)
    2. Calculates metrics (MAE, MAPE, MSE, RMSE, NRMSE, WAPE, sMAPE, MASE) for each level
    3. Calculates per-timestep metrics (_TS versions)
    4. Generates df_eval and df_eval_ts structures

    Args:
        tst_y_orig: Original test y values, shape (n_tst, O_N, num_feat)
        tst_fcst: Forecast/predictions, shape (n_tst, O_N, num_feat)
        o_dat_inf: Output data info DataFrame with 'delt_transf' column
        O_N: Number of output timesteps (default 13)
        n_max: Number of averaging levels (default 12)

    Returns:
        Tuple of (dat_eval, df_eval, df_eval_ts)
        - dat_eval: Raw evaluation data structure
        - df_eval: DataFrame per feature with metrics at each averaging level
        - df_eval_ts: Per-timestep metrics for each feature and delta
    """
    logger.info(f"Starting evaluation with averaging: n_max={n_max}, O_N={O_N}")

    n_tst = tst_y_orig.shape[0]
    num_feat = tst_y_orig.shape[2]
    n_ft_o = num_feat

    logger.info(f"Evaluation dimensions: n_tst={n_tst}, num_feat={num_feat}")

    # =========================================================================
    # MITTELWERTBILDUNG (Averaging) - Lines 3323-3359
    # =========================================================================

    dat_eval = {}

    y_all = np.full((n_max, n_tst, O_N, num_feat), np.nan)
    fcst_all = np.full((n_max, n_tst, O_N, num_feat), np.nan)

    for n_avg in range(1, n_max + 1):
        # Anzahl der Zeitschritte der gemittelten Arrays
        n_ts = math.floor(O_N / n_avg)

        # Array vorbereiten
        y = np.zeros((n_tst, n_ts, num_feat))
        fcst = np.zeros((n_tst, n_ts, num_feat))
        dat_eval_int = {}

        # Schleife über jeden Testdatensatz
        for i in range(n_tst):
            # Schleife über jedes Merkmal
            for j in range(num_feat):
                # Schleife über jeden Zeitschritt
                for k in range(n_ts):
                    strt = k * n_avg
                    end = min(strt + n_avg, O_N)
                    y[i, k, j] = np.mean(tst_y_orig[i, strt:end, j])
                    fcst[i, k, j] = np.mean(tst_fcst[i, strt:end, j])

                    y_all[n_avg - 1, i, k, j] = np.mean(tst_y_orig[i, strt:end, j])
                    fcst_all[n_avg - 1, i, k, j] = np.mean(tst_fcst[i, strt:end, j])

        dat_eval_int["y"] = y
        dat_eval_int["fcst"] = fcst

        # Handle delt_transf - calculate delta in minutes
        if isinstance(o_dat_inf, pd.DataFrame) and 'delt_transf' in o_dat_inf.columns:
            dat_eval_int["delt"] = np.array(o_dat_inf["delt_transf"] * n_avg)
        else:
            # Fallback: use 15 min * n_avg as default (standard 15-min intervals)
            dat_eval_int["delt"] = np.array([15 * n_avg] * num_feat)

        dat_eval[n_avg] = dat_eval_int

    logger.info("Averaging complete. Calculating overall metrics...")

    # =========================================================================
    # FEHLERBERECHNUNG - GESAMT (Overall metrics) - Lines 3365-3418
    # =========================================================================

    for i in range(n_max):
        mae_int, mape_int, mse_int, rmse_int = [], [], [], []
        nrmse_int, wape_int, smape_int, mase_int = [], [], [], []

        # Durchlauf aller Merkmale
        for i_feat in range(num_feat):
            v_true = y_all[i, :, :, i_feat]
            v_fcst = fcst_all[i, :, :, i_feat]

            mask = ~np.isnan(v_true) & ~np.isnan(v_fcst)
            mask_1 = ~np.isnan(v_true)

            try:
                mae_int.append(mae(v_true[mask], v_fcst[mask]))
                mape_int.append(100 * mape(v_true[mask], v_fcst[mask]))
                mse_int.append(mse(v_true[mask], v_fcst[mask]))
                rmse_int.append(rmse(v_true[mask], v_fcst[mask]))
                # EXACT COPY from original line 3321 - NO if/else check
                nrmse_int.append(rmse(v_true[mask], v_fcst[mask]) / np.mean(v_true[mask_1]))
                wape_int.append(wape(v_true[mask], v_fcst[mask]))
                smape_int.append(smape(v_true[mask], v_fcst[mask]))
                mase_int.append(mase(v_true[mask], v_fcst[mask]))
            except:
                # Original uses 'except: pass' - does NOT append values on error
                pass

        dat_eval[i + 1]["MAE"] = np.array(mae_int)
        dat_eval[i + 1]["MAPE"] = np.array(mape_int)
        dat_eval[i + 1]["MSE"] = np.array(mse_int)
        dat_eval[i + 1]["RMSE"] = np.array(rmse_int)
        dat_eval[i + 1]["NRMSE"] = np.array(nrmse_int)
        dat_eval[i + 1]["WAPE"] = np.array(wape_int)
        dat_eval[i + 1]["sMAPE"] = np.array(smape_int)
        dat_eval[i + 1]["MASE"] = np.array(mase_int)

    logger.info("Overall metrics complete. Calculating per-timestep metrics...")

    # =========================================================================
    # FEHLERBERECHNUNG - ZEITSCHRITTE (Per-timestep metrics) - Lines 3420-3467
    # =========================================================================

    for i in range(n_max):
        mae_ts, mape_ts, mse_ts, rmse_ts = [], [], [], []
        nrmse_ts, wape_ts, smape_ts, mase_ts = [], [], [], []

        # Durchlauf aller Merkmale
        for i_feat in range(num_feat):
            mae_int, mape_int, mse_int, rmse_int = [], [], [], []
            nrmse_int, wape_int, smape_int, mase_int = [], [], [], []

            # Durchlauf aller Zeitschritte
            # EXACT COPY from original lines 3369-3382 - NO try/except
            for i_ts in range(dat_eval[i + 1]["y"].shape[1]):
                v_true = y_all[i, :, i_ts, i_feat]
                v_fcst = fcst_all[i, :, i_ts, i_feat]

                mae_int.append(mae(v_true, v_fcst))
                mape_int.append(100 * mape(v_true, v_fcst))
                mse_int.append(mse(v_true, v_fcst))
                rmse_int.append(rmse(v_true, v_fcst))
                # EXACT COPY from original line 3379 - NO if/else check
                nrmse_int.append(rmse(v_true, v_fcst) / np.mean(v_true))
                wape_int.append(wape(v_true, v_fcst))
                smape_int.append(smape(v_true, v_fcst))
                mase_int.append(mase(v_true, v_fcst))

            mae_ts.append(mae_int)
            mape_ts.append(mape_int)
            mse_ts.append(mse_int)
            rmse_ts.append(rmse_int)
            nrmse_ts.append(nrmse_int)
            wape_ts.append(wape_int)
            smape_ts.append(smape_int)
            mase_ts.append(mase_int)

        dat_eval[i + 1]["MAE_TS"] = np.array(mae_ts)
        dat_eval[i + 1]["MAPE_TS"] = np.array(mape_ts)
        dat_eval[i + 1]["MSE_TS"] = np.array(mse_ts)
        dat_eval[i + 1]["RMSE_TS"] = np.array(rmse_ts)
        dat_eval[i + 1]["NRMSE_TS"] = np.array(nrmse_ts)
        dat_eval[i + 1]["WAPE_TS"] = np.array(wape_ts)
        dat_eval[i + 1]["sMAPE_TS"] = np.array(smape_ts)
        dat_eval[i + 1]["MASE_TS"] = np.array(mase_ts)

    logger.info("Per-timestep metrics complete. Generating df_eval...")

    # =========================================================================
    # df_eval GENERISANJE - Lines 3479-3514
    # =========================================================================

    df_eval = {}

    # Get feature names from o_dat_inf index
    if isinstance(o_dat_inf, pd.DataFrame) and hasattr(o_dat_inf, 'index'):
        feature_names = list(o_dat_inf.index)
    else:
        feature_names = [f"Feature_{i}" for i in range(n_ft_o)]

    for i_feat in range(n_ft_o):
        delt_int, mae_int, mape_int, mse_int = [], [], [], []
        rmse_int, nrmse_int, wape_int, smape_int, mase_int = [], [], [], [], []

        for i in range(n_max):
            delt_int.append(float(dat_eval[i + 1]["delt"][i_feat]))
            mae_int.append(float(dat_eval[i + 1]["MAE"][i_feat]))
            mape_int.append(float(dat_eval[i + 1]["MAPE"][i_feat]))
            mse_int.append(float(dat_eval[i + 1]["MSE"][i_feat]))
            rmse_int.append(float(dat_eval[i + 1]["RMSE"][i_feat]))
            nrmse_int.append(float(dat_eval[i + 1]["NRMSE"][i_feat]))
            wape_int.append(float(dat_eval[i + 1]["WAPE"][i_feat]))
            smape_int.append(float(dat_eval[i + 1]["sMAPE"][i_feat]))
            mase_int.append(float(dat_eval[i + 1]["MASE"][i_feat]))

        df_eval_int = pd.DataFrame({
            "delta [min]": delt_int,
            "MAE": mae_int,
            "MAPE": mape_int,
            "MSE": mse_int,
            "RMSE": rmse_int,
            "NRMSE": nrmse_int,
            "WAPE": wape_int,
            "sMAPE": smape_int,
            "MASE": mase_int
        })

        feature_name = feature_names[i_feat] if i_feat < len(feature_names) else f"Feature_{i_feat}"
        df_eval[feature_name] = df_eval_int

    logger.info("df_eval complete. Generating df_eval_ts...")

    # =========================================================================
    # df_eval_ts GENERISANJE - Lines 3517-3546
    # =========================================================================

    df_eval_ts = {}

    for i_feat in range(n_ft_o):
        feature_name = feature_names[i_feat] if i_feat < len(feature_names) else f"Feature_{i_feat}"
        df_eval_ts[feature_name] = {}

        for i in range(n_max):
            df_eval_ts_int = pd.DataFrame({
                'MAE': dat_eval[i + 1]["MAE_TS"][i_feat],
                'MAPE': dat_eval[i + 1]["MAPE_TS"][i_feat],
                'MSE': dat_eval[i + 1]["MSE_TS"][i_feat],
                'RMSE': dat_eval[i + 1]["RMSE_TS"][i_feat],
                'NRMSE': dat_eval[i + 1]["NRMSE_TS"][i_feat],
                'WAPE': dat_eval[i + 1]["WAPE_TS"][i_feat],
                'sMAPE': dat_eval[i + 1]["sMAPE_TS"][i_feat],
                'MASE': dat_eval[i + 1]["MASE_TS"][i_feat]
            })

            df_eval_ts[feature_name][float(dat_eval[i + 1]["delt"][i_feat])] = df_eval_ts_int

    # Log key metrics for first averaging level (n_avg=1)
    for feature_name, df in df_eval.items():
        mae_1 = df['MAE'].iloc[0]  # First row = n_avg=1
        mape_1 = df['MAPE'].iloc[0]
        rmse_1 = df['RMSE'].iloc[0]
        logger.info(f"📊 EVALUATION RESULTS [{feature_name}] n_avg=1: MAE={mae_1:.2f}, MAPE={mape_1:.2f}%, RMSE={rmse_1:.2f}")

    logger.info(f"Evaluation complete. Generated {len(df_eval)} features with {n_max} averaging levels each.")

    return dat_eval, df_eval, df_eval_ts


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def convert_df_eval_to_dict(df_eval: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
    """
    Convert df_eval DataFrames to JSON-serializable dicts.

    Args:
        df_eval: Dict of feature_name -> DataFrame

    Returns:
        Dict of feature_name -> dict with lists
    """
    result = {}
    for feature_name, df in df_eval.items():
        result[feature_name] = df.to_dict(orient='list')
    return result


def convert_df_eval_ts_to_dict(df_eval_ts: Dict[str, Dict[float, pd.DataFrame]]) -> Dict[str, Dict[str, Dict]]:
    """
    Convert df_eval_ts nested DataFrames to JSON-serializable dicts.

    Args:
        df_eval_ts: Dict of feature_name -> delta -> DataFrame

    Returns:
        Nested dict structure safe for JSON serialization
    """
    result = {}
    for feature_name, delta_dict in df_eval_ts.items():
        result[feature_name] = {}
        for delta, df in delta_dict.items():
            # Use string key for JSON compatibility
            result[feature_name][str(delta)] = df.to_dict(orient='list')
    return result


def convert_dat_eval_to_serializable(dat_eval: Dict) -> Dict:
    """
    Convert dat_eval numpy arrays to JSON-serializable format.

    Args:
        dat_eval: Raw evaluation dict with numpy arrays

    Returns:
        Dict with lists instead of numpy arrays
    """
    result = {}
    for n_avg, data in dat_eval.items():
        result[n_avg] = {}
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                result[n_avg][key] = value.tolist()
            else:
                result[n_avg][key] = value
    return result
