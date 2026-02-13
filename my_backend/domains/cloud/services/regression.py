"""
Cloud Regression Services
Linear and polynomial regression analysis for cloud data
"""
import csv
import json
import math
import random
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
    DEFAULT_TOLERANCE_RATIO,
    REGRESSION_SAMPLE_SIZE,
    REGRESSION_STREAMING_CHUNK_SIZE,
    REGRESSION_MIN_SAMPLE_SIZE
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


# ═══════════════════════════════════════════════════════════════════
# Streaming regression functions (chunked merge + NDJSON response)
# ═══════════════════════════════════════════════════════════════════


def quick_minmax(file_path, sep):
    """
    Read only the value column to determine min/max without loading full file.
    O(N) time, O(1) memory.
    """
    y_min = float('inf')
    y_max = float('-inf')
    count = 0

    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter=sep)
        cols = reader.fieldnames
        y_col = [c for c in cols if c != 'UTC'][0]

        for row in reader:
            try:
                val = float(row[y_col].replace(',', '.'))
                if val < y_min:
                    y_min = val
                if val > y_max:
                    y_max = val
                count += 1
            except (ValueError, TypeError):
                continue

    if y_min == float('inf'):
        raise ValueError("No valid numeric data in load file")

    return y_min, y_max


def chunked_merge_and_sample(temp_file_path, load_file_path, temp_sep, load_sep,
                              sample_size=REGRESSION_SAMPLE_SIZE, tracker=None):
    """
    Memory-efficient merge of two CSV files on UTC column with reservoir sampling.

    Algorithm:
    1. Load temp_file (predictors) into dict {UTC_string: x_value} — typically small
    2. Stream load_file row by row, matching UTCs from the dict
    3. Write ALL matched points to a temp CSV file
    4. Maintain a reservoir sample (Vitter's Algorithm R) for regression fitting

    Returns:
        (matched_file_path, sample_df, x_col, y_col, total_matched)
    """
    # Step 1: Load temp file (predictors) into memory as {UTC: x_value}
    if tracker:
        tracker.emit('processing', 35, 'cloud_loading_predictors', force=True)

    temp_lookup = {}
    x_col = None

    with open(temp_file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter=temp_sep)
        cols = reader.fieldnames
        x_col = [c for c in cols if c != 'UTC'][0]

        for row in reader:
            utc_str = row['UTC'].strip()
            try:
                x_val = float(row[x_col].replace(',', '.'))
                if math.isnan(x_val) or math.isinf(x_val):
                    continue
                temp_lookup[utc_str] = x_val
            except (ValueError, TypeError):
                continue

    if len(temp_lookup) == 0:
        raise ValueError("No valid numeric data in predictor file")

    # Step 2: Stream load file, match UTCs, reservoir sample
    if tracker:
        tracker.emit('processing', 45, 'cloud_matching_timestamps', force=True)

    reservoir = []
    matched_count = 0
    y_col = None

    matched_file_path = temp_file_path.replace('temp_out.csv', 'matched_points.csv')

    with open(load_file_path, 'r', encoding='utf-8') as f_in, \
         open(matched_file_path, 'w', newline='', encoding='utf-8') as f_out:

        reader = csv.DictReader(f_in, delimiter=load_sep)
        cols = reader.fieldnames
        y_col = [c for c in cols if c != 'UTC'][0]

        writer = csv.writer(f_out)
        writer.writerow(['x', 'y'])

        for row in reader:
            utc_str = row['UTC'].strip()
            if utc_str in temp_lookup:
                try:
                    y_val = float(row[y_col].replace(',', '.'))
                except (ValueError, TypeError):
                    continue

                if math.isnan(y_val) or math.isinf(y_val):
                    continue

                x_val = temp_lookup[utc_str]

                # Write to matched file
                writer.writerow([x_val, y_val])
                matched_count += 1

                # Reservoir sampling (Vitter's Algorithm R)
                if len(reservoir) < sample_size:
                    reservoir.append((x_val, y_val))
                else:
                    j = random.randint(0, matched_count - 1)
                    if j < sample_size:
                        reservoir[j] = (x_val, y_val)

                if matched_count % 100_000 == 0 and tracker:
                    progress = 45 + min((matched_count / max(len(temp_lookup), 1)) * 10, 10)
                    tracker.emit('processing', progress, 'cloud_merging_data',
                                 message_params={'matched': matched_count})

    if matched_count == 0:
        raise ValueError('No matching timestamps found between files. '
                         'Please ensure both files have matching UTC timestamps.')

    if matched_count < REGRESSION_MIN_SAMPLE_SIZE:
        raise ValueError(f'Too few matching points: {matched_count} '
                         f'(minimum: {REGRESSION_MIN_SAMPLE_SIZE})')

    # Convert reservoir to DataFrame for regression fitting
    sample_df = pd.DataFrame(reservoir, columns=['x', 'y'])

    return matched_file_path, sample_df, x_col, y_col, matched_count


def _apply_precision(value, precision):
    """Apply decimal precision to a single value."""
    if precision == 'full':
        return value
    try:
        return round(value, int(precision))
    except (ValueError, TypeError):
        return value


def _process_regression_chunk(buffer, predict_fn, tolerance_type, tol_cnt, tol_dep, decimal_precision):
    """
    Apply regression prediction and tolerance to a buffer of (x, y) tuples.
    Uses numpy vectorized operations for performance.

    Returns:
        (chunk_data_list, removed_count)
    """
    x_arr = np.array([p[0] for p in buffer])
    y_arr = np.array([p[1] for p in buffer])

    predictions = predict_fn(x_arr)
    upper, lower = calculate_bounds(predictions, tolerance_type, tol_cnt, tol_dep)

    # Determine which points are inside tolerance
    if tolerance_type == 'cnt':
        inside_mask = np.abs(y_arr - predictions) <= tol_cnt
    else:  # 'dep'
        inside_mask = np.abs(y_arr - predictions) <= (np.abs(predictions) * tol_dep + tol_cnt)

    removed_count = int((~inside_mask).sum())

    chunk_data = []
    for i in range(len(buffer)):
        point = {
            'x': _apply_precision(buffer[i][0], decimal_precision),
            'y': _apply_precision(buffer[i][1], decimal_precision),
            'predicted': _apply_precision(float(predictions[i]), decimal_precision),
            'upperBound': _apply_precision(float(upper[i]), decimal_precision),
            'lowerBound': _apply_precision(float(lower[i]), decimal_precision),
            'inside': bool(inside_mask[i])
        }
        chunk_data.append(point)

    return chunk_data, removed_count


def perform_streaming_regression(matched_file_path, sample_df, x_col, y_col,
                                  total_matched, reg_type, tolerance_type,
                                  tol_cnt, tol_dep, decimal_precision='full',
                                  tracker=None):
    """
    Generator that yields NDJSON lines for streaming regression results.

    1. Fits regression model on the sample DataFrame
    2. Streams ALL matched points with predicted/bounds applied per-chunk
    3. Counts removed points on-the-fly

    Yields NDJSON lines:
      {"type":"meta","total_rows":N,"equation":"...","success":true}
      {"type":"data","chunk_index":I,"data":[{x,y,predicted,upperBound,lowerBound,inside},...]}
      {"type":"complete","removed_points":M,"total_points":N,"equation":"...","success":true}
    """
    if tracker:
        tracker.emit('processing', 60, 'cloud_fitting_model', force=True)

    # Sort sample for model fitting, drop any NaN/Inf values
    sample_sorted = sample_df.sort_values('x').copy()
    sample_sorted = sample_sorted.replace([np.inf, -np.inf], np.nan).dropna(subset=['x', 'y'])

    if len(sample_sorted) < REGRESSION_MIN_SAMPLE_SIZE:
        raise ValueError(f'Too few valid sample points after cleanup: {len(sample_sorted)} '
                         f'(minimum: {REGRESSION_MIN_SAMPLE_SIZE})')

    # Fit model on sample (use .values to avoid sklearn feature name warnings)
    x_train = sample_sorted['x'].values.reshape(-1, 1)
    y_train = sample_sorted['y'].values

    if reg_type == 'lin':
        model = LinearRegression()
        model.fit(x_train, y_train)
        equation = f"y = {model.coef_[0]:.2f}x + {model.intercept_:.2f}"

        def predict_fn(x_vals):
            return model.predict(x_vals.reshape(-1, 1))
    else:
        poly_features = PolynomialFeatures(degree=2)
        X_poly = poly_features.fit_transform(x_train)
        model = LinearRegression()
        model.fit(X_poly, y_train)
        coeffs = model.coef_
        intercept = model.intercept_
        equation = f"y = {coeffs[2]:.2f}x^2 + {coeffs[1]:.2f}x + {intercept:.2f}"

        def predict_fn(x_vals):
            X_p = poly_features.transform(x_vals.reshape(-1, 1))
            return model.predict(X_p)

    if tracker:
        tracker.emit('processing', 70, 'cloud_model_fitted',
                     message_params={'equation': equation}, force=True)

    # Yield meta line
    total_chunks = (total_matched + REGRESSION_STREAMING_CHUNK_SIZE - 1) // REGRESSION_STREAMING_CHUNK_SIZE
    yield json.dumps({
        'type': 'meta',
        'total_rows': total_matched,
        'total_chunks': total_chunks,
        'equation': equation,
        'reg_type': reg_type,
        'x_col': x_col,
        'y_col': y_col,
        'success': True
    }, separators=(',', ':')) + '\n'

    if tracker:
        tracker.emit('streaming', 75, 'cloud_streaming_start', force=True)

    # Stream ALL matched points with predictions applied per-chunk
    chunk_buffer = []
    chunk_index = 0
    removed_count = 0
    streamed_count = 0

    with open(matched_file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)

        for row in reader:
            x_val = float(row['x'])
            y_val = float(row['y'])

            chunk_buffer.append((x_val, y_val))

            if len(chunk_buffer) >= REGRESSION_STREAMING_CHUNK_SIZE:
                chunk_data, chunk_removed = _process_regression_chunk(
                    chunk_buffer, predict_fn, tolerance_type, tol_cnt, tol_dep, decimal_precision
                )
                removed_count += chunk_removed
                streamed_count += len(chunk_buffer)

                yield json.dumps({
                    'type': 'data',
                    'chunk_index': chunk_index,
                    'data': chunk_data
                }, separators=(',', ':')) + '\n'

                chunk_index += 1
                chunk_buffer = []

                # Progress: 75-95% during streaming
                progress = 75 + (streamed_count / total_matched) * 20
                if chunk_index % 5 == 0 and tracker:
                    tracker.emit('streaming', min(progress, 95), 'cloud_streaming_chunk',
                                 message_params={'current': streamed_count, 'total': total_matched})

    # Process remaining buffer
    if chunk_buffer:
        chunk_data, chunk_removed = _process_regression_chunk(
            chunk_buffer, predict_fn, tolerance_type, tol_cnt, tol_dep, decimal_precision
        )
        removed_count += chunk_removed
        streamed_count += len(chunk_buffer)

        yield json.dumps({
            'type': 'data',
            'chunk_index': chunk_index,
            'data': chunk_data
        }, separators=(',', ':')) + '\n'

    if tracker:
        tracker.emit('complete', 100, 'cloud_complete', force=True)

    # Yield completion line
    yield json.dumps({
        'type': 'complete',
        'removed_points': removed_count,
        'total_points': total_matched,
        'equation': equation,
        'success': True
    }, separators=(',', ':')) + '\n'
