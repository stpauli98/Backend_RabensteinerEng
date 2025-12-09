"""
Cloud Regression Services
Linear and polynomial regression analysis for cloud data
"""
import logging
import traceback
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from flask import jsonify

from domains.cloud.config import (
    TOLERANCE_ADJUSTMENT_FACTOR,
    MIN_TOLERANCE_THRESHOLD,
    DEFAULT_TOLERANCE_RATIO
)

logger = logging.getLogger(__name__)


def calculate_bounds(predictions, tolerance_type, tol_cnt, tol_dep):
    """Calculate upper and lower bounds based on tolerance type."""
    if tolerance_type == 'cnt':
        upper_bound = predictions + tol_cnt
        lower_bound = predictions - tol_cnt
    else:
        upper_bound = predictions * (1 + tol_dep) + tol_cnt
        lower_bound = predictions * (1 - tol_dep) - tol_cnt

    return upper_bound, lower_bound


def apply_decimal_precision(values, precision):
    """Round values to specified decimal places."""
    if precision == 'full':
        return values
    try:
        precision_int = int(precision)
        if isinstance(values, list):
            return [round(v, precision_int) if v is not None and not (isinstance(v, float) and np.isnan(v)) else v for v in values]
        return values
    except (ValueError, TypeError):
        return values


def validate_and_prepare_data(df1, df2):
    """Validate and prepare CSV data for regression analysis."""
    logger.info(f"Processing dataframes with shapes: {df1.shape}, {df2.shape}")
    logger.info(f"Columns in temperature file: {df1.columns.tolist()}")
    logger.info(f"Columns in load file: {df2.columns.tolist()}")

    temp_cols = [col for col in df1.columns if col != 'UTC']
    load_cols = [col for col in df2.columns if col != 'UTC']

    if not temp_cols:
        raise ValueError(f'No valid temperature column found. Available columns: {df1.columns.tolist()}')
    if not load_cols:
        raise ValueError(f'No valid load column found. Available columns: {df2.columns.tolist()}')

    x = temp_cols[0]
    y = load_cols[0]
    logger.info(f"Using temperature column: {x}")
    logger.info(f"Using load column: {y}")

    df1['UTC'] = pd.to_datetime(df1['UTC'], format="%Y-%m-%d %H:%M:%S")
    df2['UTC'] = pd.to_datetime(df2['UTC'], format="%Y-%m-%d %H:%M:%S")

    df1[x] = pd.to_numeric(df1[x], errors='coerce')
    df2[y] = pd.to_numeric(df2[y], errors='coerce')

    df1 = df1.sort_values('UTC')
    df2 = df2.sort_values('UTC')

    logger.info(f"Data ranges after sorting:")
    logger.info(f"Temperature range: {df1[x].min():.2f} to {df1[x].max():.2f} C")
    logger.info(f"Load range before conversion: {df2[y].min():.2f} to {df2[y].max():.2f} kW")

    df_merged = pd.merge(df1[['UTC', x]], df2[['UTC', y]], on='UTC', how='inner')
    if df_merged.empty:
        raise ValueError('No matching timestamps found between files. Please ensure both files have matching timestamps.')

    logger.info(f"First few timestamps in first file: {df1['UTC'].head().tolist()}")
    logger.info(f"First few timestamps in second file: {df2['UTC'].head().tolist()}")

    df1_duplicates = df1['UTC'].duplicated().sum()
    df2_duplicates = df2['UTC'].duplicated().sum()
    if df1_duplicates > 0 or df2_duplicates > 0:
        raise ValueError('Duplicate timestamps found in data')

    df1 = df1.dropna()
    df2 = df2.dropna()
    if df1.empty or df2.empty:
        raise ValueError('No valid numeric data found after cleaning')

    logger.info(f"Data cleaned. New shapes: {df1.shape}, {df2.shape}")

    cld = pd.DataFrame()
    cld[x] = df1[x]
    cld[y] = df2[y]
    cld = cld.dropna()
    if cld.empty:
        raise ValueError('No valid data points after combining datasets')

    logger.info(f"Combined data shape: {cld.shape}")

    cld_srt = cld.sort_values(by=x).copy()
    if cld_srt[x].isna().any() or cld_srt[y].isna().any():
        raise ValueError('NaN values found in processed data')

    return cld_srt, x, y, df2


def calculate_tolerance_params(data, y_range):
    """Calculate and validate tolerance parameters."""
    default_tol = y_range * DEFAULT_TOLERANCE_RATIO

    try:
        TOL_CNT = float(data.get('TOL_CNT', str(default_tol)))
        TOL_DEP = float(data.get('TOL_DEP', '0.1'))

        TOL_CNT = TOL_CNT / TOLERANCE_ADJUSTMENT_FACTOR
        logger.info(f"Received tolerance values - TOL_CNT: {TOL_CNT}, TOL_DEP: {TOL_DEP}")

        if TOL_CNT <= 0:
            logger.warning(f"Invalid TOL_CNT value: {TOL_CNT}, using default")
            TOL_CNT = default_tol
        if TOL_DEP <= 0:
            logger.warning(f"Invalid TOL_DEP value: {TOL_DEP}, using default")
            TOL_DEP = 0.1
    except (ValueError, TypeError) as e:
        logger.warning(f"Error parsing tolerance values: {str(e)}, using defaults")
        TOL_CNT = default_tol
        TOL_DEP = 0.1

    if TOL_CNT < y_range * MIN_TOLERANCE_THRESHOLD:
        TOL_CNT = y_range * DEFAULT_TOLERANCE_RATIO
        logger.info(f"Adjusted tolerance to {TOL_CNT:.2f} ({DEFAULT_TOLERANCE_RATIO*100}% of data range)")

    return TOL_CNT, TOL_DEP


def perform_linear_regression(cld_srt, x, y, TR, TOL_CNT, TOL_DEP, decimal_precision='full'):
    """Perform linear regression and apply tolerance filtering."""
    lin_mdl = LinearRegression()
    lin_mdl.fit(cld_srt[[x]], cld_srt[y])
    lin_prd = lin_mdl.predict(cld_srt[[x]])
    lin_fcn = f"y = {lin_mdl.coef_[0]:.2f}x + {lin_mdl.intercept_:.2f}"
    logger.info(f"Linear regression: {lin_fcn}")

    upper_bound, lower_bound = calculate_bounds(lin_prd, TR, TOL_CNT, TOL_DEP)

    if TR == "cnt":
        mask = np.abs(cld_srt[y] - lin_prd) <= TOL_CNT
        logger.info(f"Using constant tolerance: {TOL_CNT}")
    elif TR == "dep":
        mask = np.abs(cld_srt[y] - lin_prd) <= (np.abs(lin_prd) * TOL_DEP + TOL_CNT)
        logger.info(f"Using dependent tolerance: {TOL_DEP} + {TOL_CNT}")
    else:
        raise ValueError(f'Unknown tolerance type: {TR}')

    cld_srt_flt = cld_srt[mask]

    if len(cld_srt_flt) == 0:
        logger.warning(f"No points within tolerance bounds. Min y: {cld_srt[y].min()}, Max y: {cld_srt[y].max()}")
        logger.warning(f"Bounds: lower={lower_bound.min()}, upper={upper_bound.max()}")
        raise ValueError('No points within tolerance bounds. Try increasing the tolerance values.')

    return {
        'x_values': apply_decimal_precision(cld_srt[x].tolist(), decimal_precision),
        'y_values': apply_decimal_precision(cld_srt[y].tolist(), decimal_precision),
        'predicted_y': apply_decimal_precision(lin_prd.tolist(), decimal_precision),
        'upper_bound': apply_decimal_precision(upper_bound.tolist(), decimal_precision),
        'lower_bound': apply_decimal_precision(lower_bound.tolist(), decimal_precision),
        'filtered_x': apply_decimal_precision(cld_srt_flt[x].tolist(), decimal_precision),
        'filtered_y': apply_decimal_precision(cld_srt_flt[y].tolist(), decimal_precision),
        'equation': lin_fcn,
        'removed_points': len(cld_srt) - len(cld_srt_flt)
    }


def perform_polynomial_regression(cld_srt, x, y, TR, TOL_CNT, TOL_DEP, decimal_precision='full'):
    """Perform polynomial regression and apply tolerance filtering."""
    poly_features = PolynomialFeatures(degree=2)
    X_poly = poly_features.fit_transform(cld_srt[[x]])
    poly_mdl = LinearRegression()
    poly_mdl.fit(X_poly, cld_srt[y])
    poly_prd = poly_mdl.predict(X_poly)
    coeffs = poly_mdl.coef_
    intercept = poly_mdl.intercept_
    poly_fcn = f"y = {coeffs[2]:.2f}x^2 + {coeffs[1]:.2f}x + {intercept:.2f}"
    logger.info(f"Polynomial regression (degree 2): {poly_fcn}")

    upper_bound, lower_bound = calculate_bounds(poly_prd, TR, TOL_CNT, TOL_DEP)

    if TR == "cnt":
        mask = np.abs(cld_srt[y] - poly_prd) <= TOL_CNT
    elif TR == "dep":
        mask = np.abs(cld_srt[y] - poly_prd) <= (np.abs(poly_prd) * TOL_DEP + TOL_CNT)
    else:
        raise ValueError(f'Unknown tolerance type: {TR}')

    cld_srt_flt = cld_srt[mask]

    if len(cld_srt_flt) == 0:
        logger.warning(f"No points within tolerance bounds. Min y: {cld_srt[y].min()}, Max y: {cld_srt[y].max()}")
        logger.warning(f"Bounds: lower={lower_bound.min()}, upper={upper_bound.max()}")
        raise ValueError('No points within tolerance bounds. Try increasing the tolerance values.')

    return {
        'x_values': apply_decimal_precision(cld_srt[x].tolist(), decimal_precision),
        'y_values': apply_decimal_precision(cld_srt[y].tolist(), decimal_precision),
        'predicted_y': apply_decimal_precision(poly_prd.tolist(), decimal_precision),
        'upper_bound': apply_decimal_precision(upper_bound.tolist(), decimal_precision),
        'lower_bound': apply_decimal_precision(lower_bound.tolist(), decimal_precision),
        'filtered_x': apply_decimal_precision(cld_srt_flt[x].tolist(), decimal_precision),
        'filtered_y': apply_decimal_precision(cld_srt_flt[y].tolist(), decimal_precision),
        'equation': poly_fcn,
        'removed_points': len(cld_srt) - len(cld_srt_flt)
    }


def process_data_frames(df1, df2, data):
    """
    Process data from dataframes for both direct and chunked uploads.
    Refactored to use helper functions for better maintainability.
    """
    try:
        cld_srt, x, y, df2_original = validate_and_prepare_data(df1, df2)

        REG = data.get('REG', 'lin')
        TR = data.get('TR', 'cnt')
        y_range = df2_original[y].max() - df2_original[y].min()
        TOL_CNT, TOL_DEP = calculate_tolerance_params(data, y_range)

        decimal_precision = data.get('decimalPrecision', 'full')
        logger.info(f"Final parameters: REG={REG}, TR={TR}, TOL_CNT={TOL_CNT}, TOL_DEP={TOL_DEP}, decimalPrecision={decimal_precision}")

        if REG == "lin":
            result_data = perform_linear_regression(cld_srt, x, y, TR, TOL_CNT, TOL_DEP, decimal_precision)
        else:
            result_data = perform_polynomial_regression(cld_srt, x, y, TR, TOL_CNT, TOL_DEP, decimal_precision)

        logger.info("Sending response:")
        return jsonify({'success': True, 'data': result_data})

    except ValueError as ve:
        logger.error(f"Validation error: {str(ve)}")
        return jsonify({'success': False, 'data': {'error': str(ve)}}), 400
    except Exception as e:
        logger.error(f"Error in data processing: {str(e)}")
        traceback.print_exc()
        return jsonify({'success': False, 'data': {'error': str(e)}}), 500
