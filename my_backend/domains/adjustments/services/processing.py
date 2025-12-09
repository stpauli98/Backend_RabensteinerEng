"""
Data Processing Methods for Adjustments Domain
Vectorized time series processing implementations
"""
import logging
from datetime import timedelta
from typing import Tuple, List, Optional, Dict, Any

import numpy as np
import pandas as pd

from domains.adjustments.config import UTC_FORMAT

logger = logging.getLogger(__name__)


# ============================================================
# VECTORIZED PROCESSING METHODS (identical to original logic)
# ============================================================

def _prepare_data_for_processing(df: pd.DataFrame, col: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare data - common for all methods.
    Filters NaN values and converts to numpy arrays.
    """
    df_clean = df[['UTC', col]].copy()
    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    df_clean = df_clean.dropna(subset=[col]).reset_index(drop=True)

    # Convert UTC to int64 (seconds from epoch)
    utc_values = df_clean['UTC'].values.astype('datetime64[s]').astype(np.int64)
    col_values = df_clean[col].values.astype(np.float64)

    return utc_values, col_values


def _generate_time_list(
    df: pd.DataFrame,
    tss: float,
    ofst: float,
    start_time: Optional[str],
    end_time: Optional[str]
) -> Tuple[List, np.ndarray]:
    """
    Generate continuous timestamp - IDENTICAL TO ORIGINAL
    """
    # Determine boundaries
    if start_time is None:
        t_strt = df['UTC'].min()
    else:
        t_strt = pd.to_datetime(start_time)
        if t_strt.tzinfo is not None:
            t_strt = t_strt.tz_localize(None)

    if end_time is None:
        t_end = df['UTC'].max()
    else:
        t_end = pd.to_datetime(end_time)
        if t_end.tzinfo is not None:
            t_end = t_end.tz_localize(None)

    # Convert to Python datetime if pandas Timestamp
    if hasattr(t_strt, 'to_pydatetime'):
        t_strt = t_strt.to_pydatetime()
    if hasattr(t_end, 'to_pydatetime'):
        t_end = t_end.to_pydatetime()

    # Original logic
    t_ref = t_strt.replace(minute=0, second=0, microsecond=0)
    t_ref += timedelta(minutes=ofst)

    while t_ref < t_strt:
        t_ref += timedelta(minutes=tss)

    t_list = []
    while t_ref <= t_end:
        t_list.append(t_ref)
        t_ref += timedelta(minutes=tss)

    # Convert to numpy int64 (seconds) for fast operations
    t_list_int = np.array([t.timestamp() for t in t_list], dtype=np.int64)

    return t_list, t_list_int


def _method_mean_vectorized(
    utc_values: np.ndarray,
    col_values: np.ndarray,
    t_list_int: np.ndarray,
    tss: float
) -> np.ndarray:
    """
    MEAN method - sliding window +/-tss/2
    """
    half_window = int(tss * 30)  # tss/2 in seconds
    n = len(t_list_int)

    value_list = np.full(n, np.nan, dtype=np.float64)

    # Sort for binary search
    sorted_idx = np.argsort(utc_values)
    utc_sorted = utc_values[sorted_idx]
    col_sorted = col_values[sorted_idx]

    for i in range(n):
        t = t_list_int[i]
        t_min = t - half_window
        t_max = t + half_window

        # Binary search O(log n)
        idx_start = np.searchsorted(utc_sorted, t_min, side='left')
        idx_end = np.searchsorted(utc_sorted, t_max, side='right')

        if idx_start < idx_end:
            window_values = col_sorted[idx_start:idx_end]
            valid = window_values[~np.isnan(window_values)]
            if len(valid) > 0:
                value_list[i] = np.mean(valid)

    return value_list


def _method_intrpl_vectorized(
    utc_values: np.ndarray,
    col_values: np.ndarray,
    t_list_int: np.ndarray,
    intrpl_max: float
) -> np.ndarray:
    """
    INTRPL - bidirectional search + linear interpolation
    """
    intrpl_max_sec = intrpl_max * 60
    n = len(t_list_int)

    value_list = np.full(n, np.nan, dtype=np.float64)

    # Filter only numeric (already filtered in _prepare_data_for_processing)
    valid_mask = ~np.isnan(col_values)
    utc_valid = utc_values[valid_mask]
    col_valid = col_values[valid_mask]

    if len(utc_valid) == 0:
        return value_list

    for i in range(n):
        t = t_list_int[i]

        # Find idx where t would be inserted
        idx = np.searchsorted(utc_valid, t)

        # time_next: first >= t
        if idx >= len(utc_valid):
            continue  # No next -> nan

        idx_next = idx
        if utc_valid[idx_next] < t:
            if idx_next + 1 < len(utc_valid):
                idx_next += 1
            else:
                continue

        time_next = utc_valid[idx_next]
        value_next = col_valid[idx_next]

        # time_prior: last <= t
        idx_prior = idx - 1 if idx > 0 else 0
        if utc_valid[idx_prior] > t:
            continue  # No valid prior

        time_prior = utc_valid[idx_prior]
        value_prior = col_valid[idx_prior]

        # Delta calculation
        delta_time_sec = time_next - time_prior
        delta_value = value_prior - value_next

        if delta_time_sec == 0:
            value_list[i] = value_prior
        elif delta_value == 0 and delta_time_sec <= intrpl_max_sec:
            value_list[i] = value_prior
        elif delta_time_sec > intrpl_max_sec:
            continue  # nan - gap too large
        else:
            # Linear interpolation
            delta_time_prior_sec = t - time_prior
            value_list[i] = value_prior - delta_value / delta_time_sec * delta_time_prior_sec

    return value_list


def _method_nearest_vectorized(
    utc_values: np.ndarray,
    col_values: np.ndarray,
    t_list_int: np.ndarray,
    tss: float
) -> np.ndarray:
    """
    NEAREST - closest value in window +/-tss/2
    """
    half_window = int(tss * 30)  # tss/2 in seconds
    n = len(t_list_int)

    value_list = np.full(n, np.nan, dtype=np.float64)

    valid_mask = ~np.isnan(col_values)
    utc_valid = utc_values[valid_mask]
    col_valid = col_values[valid_mask]

    if len(utc_valid) == 0:
        return value_list

    # Sort for binary search
    sorted_idx = np.argsort(utc_valid)
    utc_sorted = utc_valid[sorted_idx]
    col_sorted = col_valid[sorted_idx]

    for i in range(n):
        t = t_list_int[i]
        t_min = t - half_window
        t_max = t + half_window

        idx_start = np.searchsorted(utc_sorted, t_min, side='left')
        idx_end = np.searchsorted(utc_sorted, t_max, side='right')

        if idx_start < idx_end:
            window_times = utc_sorted[idx_start:idx_end]
            window_values = col_sorted[idx_start:idx_end]

            deltas = np.abs(window_times - t)
            min_idx = np.argmin(deltas)  # First if multiple have same delta
            value_list[i] = window_values[min_idx]

    return value_list


def _method_nearest_mean_vectorized(
    utc_values: np.ndarray,
    col_values: np.ndarray,
    t_list_int: np.ndarray,
    tss: float
) -> np.ndarray:
    """
    NEAREST (MEAN) - average of all values with same minimum delta
    """
    half_window = int(tss * 30)
    n = len(t_list_int)

    value_list = np.full(n, np.nan, dtype=np.float64)

    valid_mask = ~np.isnan(col_values)
    utc_valid = utc_values[valid_mask]
    col_valid = col_values[valid_mask]

    if len(utc_valid) == 0:
        return value_list

    sorted_idx = np.argsort(utc_valid)
    utc_sorted = utc_valid[sorted_idx]
    col_sorted = col_valid[sorted_idx]

    for i in range(n):
        t = t_list_int[i]
        t_min = t - half_window
        t_max = t + half_window

        idx_start = np.searchsorted(utc_sorted, t_min, side='left')
        idx_end = np.searchsorted(utc_sorted, t_max, side='right')

        if idx_start < idx_end:
            window_times = utc_sorted[idx_start:idx_end]
            window_values = col_sorted[idx_start:idx_end]

            deltas = np.abs(window_times - t)
            min_delta = np.min(deltas)

            # All with min delta (original logic - mean of equally distant)
            min_mask = deltas == min_delta
            value_list[i] = np.mean(window_values[min_mask])

    return value_list


def _method_nearest_max_delta_vectorized(
    utc_values: np.ndarray,
    col_values: np.ndarray,
    t_list_int: np.ndarray,
    nearest_max: float
) -> np.ndarray:
    """
    NEAREST (MAX. DELTA) - nearest value if gap <= max
    """
    nearest_max_sec = nearest_max * 60
    n = len(t_list_int)

    value_list = np.full(n, np.nan, dtype=np.float64)

    valid_mask = ~np.isnan(col_values)
    utc_valid = utc_values[valid_mask]
    col_valid = col_values[valid_mask]

    if len(utc_valid) == 0:
        return value_list

    for i in range(n):
        t = t_list_int[i]
        idx = np.searchsorted(utc_valid, t)

        # Prior and Next indices
        idx_next = min(idx, len(utc_valid) - 1)
        idx_prior = max(idx - 1, 0)

        # Correction if next is not actually >= t
        if idx_next < len(utc_valid) and utc_valid[idx_next] < t:
            if idx_next + 1 < len(utc_valid):
                idx_next += 1

        # Index validation
        if idx_prior < 0 or idx_next >= len(utc_valid):
            continue

        time_prior = utc_valid[idx_prior]
        time_next = utc_valid[idx_next]
        value_prior = col_valid[idx_prior]
        value_next = col_valid[idx_next]

        # Additional check that prior <= t and next >= t
        if time_prior > t or time_next < t:
            continue

        delta_time_sec = time_next - time_prior
        delta_value = value_prior - value_next

        if delta_time_sec == 0:
            value_list[i] = value_prior
        elif delta_value == 0 and delta_time_sec <= nearest_max_sec:
            value_list[i] = value_prior
        elif delta_time_sec > nearest_max_sec:
            continue  # nan - gap too large
        else:
            # Take closer value (don't interpolate!)
            delta_time_prior_sec = t - time_prior
            delta_time_next_sec = time_next - t

            if delta_time_prior_sec < delta_time_next_sec:
                value_list[i] = value_prior
            else:
                value_list[i] = value_next

    return value_list


def apply_processing_method(
    df: pd.DataFrame,
    col: str,
    method: str,
    time_step: float,
    offset: float,
    start_time: Optional[str],
    end_time: Optional[str],
    intrpl_max: Optional[float] = None,
    decimal_precision: str = 'full'
) -> pd.DataFrame:
    """
    Apply processing method to dataframe column
    """
    logger.info(f"[apply_processing_method] START - method={method}, col={col}, time_step={time_step}, offset={offset}, decimal_precision={decimal_precision}")

    # 1. Prepare DataFrame
    df = df.copy()
    df['UTC'] = pd.to_datetime(df['UTC']).dt.tz_localize(None)
    df = df.sort_values('UTC').drop_duplicates(subset=['UTC'], keep='first').reset_index(drop=True)

    tss = time_step
    ofst = offset if offset is not None else 0

    logger.info(f"[apply_processing_method] Preparing data for vectorized processing...")

    # 2. Prepare data for vectorized operations
    utc_values, col_values = _prepare_data_for_processing(df, col)

    if len(utc_values) == 0:
        logger.warning(f"[apply_processing_method] No valid numeric data for column {col}")
        return pd.DataFrame({'UTC': [], col: []})

    # 3. Generate continuous timestamp (identical to original)
    t_list, t_list_int = _generate_time_list(df, tss, ofst, start_time, end_time)

    if len(t_list) == 0:
        logger.warning(f"[apply_processing_method] No time points generated")
        return pd.DataFrame({'UTC': [], col: []})

    logger.info(f"[apply_processing_method] Generated {len(t_list)} time points, applying method: {method}")

    # 4. Call appropriate vectorized method
    if method == 'mean':
        values = _method_mean_vectorized(utc_values, col_values, t_list_int, tss)

    elif method == 'intrpl':
        max_gap = intrpl_max if intrpl_max is not None else 60
        values = _method_intrpl_vectorized(utc_values, col_values, t_list_int, max_gap)

    elif method == 'nearest':
        values = _method_nearest_vectorized(utc_values, col_values, t_list_int, tss)

    elif method == 'nearest (mean)':
        values = _method_nearest_mean_vectorized(utc_values, col_values, t_list_int, tss)

    elif method == 'nearest (max. delta)':
        max_gap = intrpl_max if intrpl_max is not None else 60
        values = _method_nearest_max_delta_vectorized(utc_values, col_values, t_list_int, max_gap)

    elif method == 'max':
        # MAX method - uses sliding window like mean, but takes max
        logger.info(f"[apply_processing_method] Applying MAX (sliding window)...")
        half_window = int(tss * 30)
        n = len(t_list_int)
        values = np.full(n, np.nan, dtype=np.float64)

        sorted_idx = np.argsort(utc_values)
        utc_sorted = utc_values[sorted_idx]
        col_sorted = col_values[sorted_idx]

        for i in range(n):
            t = t_list_int[i]
            t_min = t - half_window
            t_max = t + half_window
            idx_start = np.searchsorted(utc_sorted, t_min, side='left')
            idx_end = np.searchsorted(utc_sorted, t_max, side='right')
            if idx_start < idx_end:
                window_values = col_sorted[idx_start:idx_end]
                valid = window_values[~np.isnan(window_values)]
                if len(valid) > 0:
                    values[i] = np.max(valid)

    elif method == 'min':
        # MIN method - uses sliding window like mean, but takes min
        logger.info(f"[apply_processing_method] Applying MIN (sliding window)...")
        half_window = int(tss * 30)
        n = len(t_list_int)
        values = np.full(n, np.nan, dtype=np.float64)

        sorted_idx = np.argsort(utc_values)
        utc_sorted = utc_values[sorted_idx]
        col_sorted = col_values[sorted_idx]

        for i in range(n):
            t = t_list_int[i]
            t_min = t - half_window
            t_max = t + half_window
            idx_start = np.searchsorted(utc_sorted, t_min, side='left')
            idx_end = np.searchsorted(utc_sorted, t_max, side='right')
            if idx_start < idx_end:
                window_values = col_sorted[idx_start:idx_end]
                valid = window_values[~np.isnan(window_values)]
                if len(valid) > 0:
                    values[i] = np.min(valid)
    else:
        logger.warning(f"[apply_processing_method] Unknown method: {method}, returning original data")
        return df[['UTC', col]].copy()

    # 5. Apply decimal precision
    if decimal_precision != 'full':
        try:
            precision = int(decimal_precision)
            values = np.where(
                np.isnan(values),
                values,
                np.round(values, precision)
            )
            logger.info(f"[apply_processing_method] Applied decimal precision: {precision}")
        except (ValueError, TypeError) as e:
            logger.warning(f"[apply_processing_method] Could not apply decimal precision: {e}")

    # 6. Create result DataFrame
    result_df = pd.DataFrame({
        'UTC': t_list,
        col: values
    })

    logger.info(f"[apply_processing_method] COMPLETE - returning {len(result_df)} rows")
    return result_df


def prepare_data(data: pd.DataFrame, filename: str) -> Tuple[pd.DataFrame, List[str]]:
    """Prepare data for processing"""
    df = data.copy()

    if len(df) == 0:
        raise ValueError(f"No data found for file {filename}")

    df['UTC'] = pd.to_datetime(df['UTC'])

    measurement_cols = [col for col in df.columns if col != 'UTC']
    if not measurement_cols:
        raise ValueError(f"No measurement columns found for file {filename}")

    for col in measurement_cols:
        df[f"{col}_original"] = df[col].copy()
        df[col] = pd.to_numeric(df[col], errors='coerce')

    return df, measurement_cols


def filter_by_time_range(
    df: pd.DataFrame,
    start_time: Optional[str],
    end_time: Optional[str]
) -> pd.DataFrame:
    """Filter data by time range"""
    if start_time and end_time:
        start_time = pd.to_datetime(start_time, utc=True)
        end_time = pd.to_datetime(end_time, utc=True)
        return df[(df['UTC'] >= start_time) & (df['UTC'] <= end_time)]
    return df


def get_method_for_file(methods: Dict[str, Any], filename: str) -> Optional[str]:
    """Get processing method for file"""
    method_info = methods.get(filename, {})
    if isinstance(method_info, dict):
        return method_info.get('method', '').strip()
    return None


def create_info_record(
    df: pd.DataFrame,
    col: str,
    filename: str,
    time_step: float,
    offset: float
) -> Dict[str, Any]:
    """Create info record for results"""
    total_points = len(df)
    numeric_points = df[col].count()
    numeric_ratio = (numeric_points / total_points * 100) if total_points > 0 else 0

    def format_utc(val):
        if pd.isnull(val):
            return None
        if hasattr(val, 'strftime'):
            return val.strftime(UTC_FORMAT)
        try:
            dt = pd.to_datetime(val)
            return dt.strftime(UTC_FORMAT)
        except Exception:
            return str(val)

    return {
        'Name der Datei': filename,
        'Name der Messreihe': col,
        'Startzeit (UTC)': format_utc(df['UTC'].iloc[0]) if len(df) > 0 else None,
        'Endzeit (UTC)': format_utc(df['UTC'].iloc[-1]) if len(df) > 0 else None,
        'Zeitschrittweite [min]': time_step,
        'Offset [min]': offset,
        'Anzahl der Datenpunkte': int(total_points),
        'Anzahl der numerischen Datenpunkte': int(numeric_points),
        'Anteil an numerischen Datenpunkten': float(numeric_ratio)
    }


def create_records(
    df: pd.DataFrame,
    col: str,
    filename: str,
    decimal_precision: str = 'full'
) -> List[Dict[str, Any]]:
    """
    Convert DataFrame to records
    """
    original_col = f"{col}_original"

    utc_values = pd.to_datetime(df['UTC']).values
    col_values = df[col].values

    utc_timestamps = (utc_values.astype('datetime64[ms]').astype(np.int64))

    has_original = original_col in df.columns
    original_values = df[original_col].values if has_original else None

    def apply_precision(value):
        if decimal_precision == 'full':
            return value
        try:
            return round(float(value), int(decimal_precision))
        except (ValueError, TypeError):
            return value

    records = []
    for idx in range(len(df)):
        utc_ts = int(utc_timestamps[idx])
        col_val = col_values[idx]

        if pd.notnull(col_val):
            value = apply_precision(float(col_val))
        elif has_original and pd.notnull(original_values[idx]):
            value = str(original_values[idx])
        else:
            value = "None"

        records.append({
            'UTC': utc_ts,
            col: value,
            'filename': filename
        })

    return records


def convert_data_without_processing(
    df: pd.DataFrame,
    filename: str,
    time_step: float,
    offset: float,
    decimal_precision: str = 'full'
) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """
    Direct data conversion without processing when parameters are the same.
    """
    try:
        df = df.copy()
        df['UTC'] = pd.to_datetime(df['UTC'])

        measurement_cols = [col for col in df.columns if col != 'UTC']

        if not measurement_cols:
            logger.warning(f"No measurement columns found for {filename}")
            return [], None

        all_records = []

        for col in measurement_cols:
            records = create_records(df, col, filename, decimal_precision)
            all_records.extend(records)

            if len(all_records) > 0 and not any(r.get('info_created') for r in all_records):
                info_record = create_info_record(df, col, filename, time_step, offset)
                return all_records, info_record

        if not all_records:
            return [], None

        info_record = create_info_record(df, measurement_cols[0], filename, time_step, offset)
        return all_records, info_record

    except Exception as e:
        logger.error(f"Error in convert_data_without_processing: {str(e)}")
        import traceback
        traceback.print_exc()
        return [], None


def process_data_detailed(
    data: pd.DataFrame,
    filename: str,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    time_step: Optional[float] = None,
    offset: Optional[float] = None,
    methods: Optional[Dict[str, Any]] = None,
    intrpl_max: Optional[float] = None,
    decimal_precision: str = 'full'
) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """
    Detailed data processing with method application
    """
    try:
        if methods is None:
            methods = {}

        df, measurement_cols = prepare_data(data, filename)
        df = filter_by_time_range(df, start_time, end_time)

        method = get_method_for_file(methods, filename)

        if not method:
            logger.warning(f"No processing method specified for {filename} but processing is required")
            return [], None

        all_info_records = []

        if len(measurement_cols) == 1:
            measurement_col = measurement_cols[0]

            processed_df = apply_processing_method(
                df, measurement_col, method, time_step, offset, start_time, end_time, intrpl_max, decimal_precision
            )

            records = create_records(processed_df, measurement_col, filename, decimal_precision)
            info_record = create_info_record(processed_df, measurement_col, filename, time_step, offset)

            return records, info_record

        combined_records = []

        for col in measurement_cols:
            processed_df = apply_processing_method(
                df, col, method, time_step, offset, start_time, end_time, intrpl_max, decimal_precision
            )

            records = create_records(processed_df, col, filename, decimal_precision)
            info_record = create_info_record(processed_df, col, filename, time_step, offset)

            combined_records.extend(records)
            all_info_records.append(info_record)

        return combined_records, all_info_records[0] if all_info_records else None

    except Exception as e:
        logger.error(f"Error in process_data_detailed: {str(e)}")
        import traceback
        traceback.print_exc()
        raise
