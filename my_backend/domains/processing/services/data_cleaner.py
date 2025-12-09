"""
Data cleaning service for processing domain.
Handles data cleaning operations with progress tracking.
"""
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def validate_processing_params(params):
    """Validate all numeric processing parameters"""
    validated = {}

    numeric_params = {
        'eqMax': (0, 1000000, "Elimination max duration"),
        'elMax': (-1000000, 1000000, "Upper limit value"),
        'elMin': (-1000000, 1000000, "Lower limit value"),
        'chgMax': (0, 1000000, "Change rate max"),
        'lgMax': (0, 1000000, "Length max"),
        'gapMax': (0, 1000000, "Gap max duration")
    }

    for param, (min_val, max_val, description) in numeric_params.items():
        if param in params and params[param] is not None and str(params[param]).strip() != '':
            try:
                value = float(params[param])
            except (ValueError, TypeError) as e:
                raise ValueError(f"Invalid parameter {param}: must be a valid number")

            if not (min_val <= value <= max_val):
                raise ValueError(f"{description} must be between {min_val} and {max_val}")
            validated[param] = value

    radio_params = ['radioValueNull']
    for param in radio_params:
        if param in params:
            if params[param] in [None, '', 'undefined', 'null']:
                validated[param] = ''
            else:
                validated[param] = params[param]

    return validated


def clean_data(df, value_column, params, tracker=None, decimal_precision='full'):
    """
    Clean data according to specified parameters with per-step ETA tracking.

    Args:
        df: DataFrame with data
        value_column: Name of the value column
        params: Cleaning parameters
        tracker: ProgressTracker instance for progress tracking
        decimal_precision: Decimal precision for rounding ('full' or integer)
    """
    logger.info("Starting data cleaning with parameters: %s", params)
    total_rows = len(df)

    def apply_precision(value):
        """Round value to specified decimal places."""
        if decimal_precision == 'full':
            return value
        try:
            return round(float(value), int(decimal_precision))
        except (ValueError, TypeError):
            return value

    # Calculate which steps will be executed (translation keys)
    active_steps = []
    if params.get("eqMax"):
        active_steps.append(("eqMax", "measurement_failure_removal"))
    if params.get("elMax") is not None:
        active_steps.append(("elMax", "upper_threshold_removal"))
    if params.get("elMin") is not None:
        active_steps.append(("elMin", "lower_threshold_removal"))
    if params.get("radioValueNull") == "ja":
        active_steps.append(("radioValueNull", "zero_value_removal"))
    if params.get("chgMax") and params.get("lgMax"):
        active_steps.append(("chgMax", "outlier_removal"))
    if params.get("gapMax"):
        active_steps.append(("gapMax", "gap_filling"))

    total_active_steps = len(active_steps) if active_steps else 1
    current_step_index = 0

    # Set total_steps in tracker for frontend
    if tracker:
        tracker.total_steps = total_active_steps

    # Emit frequency - every ~2% of steps or min 500 rows
    emit_frequency = max(500, total_rows // 50)

    def start_step(step_key):
        """Start a new step and reset ETA tracking"""
        nonlocal current_step_index
        current_step_index += 1
        if tracker:
            tracker.current_step = current_step_index
            tracker.start_step(total_rows)
            # Cleaning phase: 25-90% (65% range), evenly divided by steps
            progress = 25 + ((current_step_index - 1) / total_active_steps) * 65
            tracker.emit('cleaning', progress, step_key, force=True)

    def update_progress(step_key, iteration_in_step):
        """Update progress within step - with per-step ETA"""
        if tracker:
            tracker.update_step_progress(iteration_in_step)

            # Emit at emit_frequency interval
            if iteration_in_step % emit_frequency == 0 and iteration_in_step > 0:
                # Progress within cleaning phase (25-90%)
                step_progress = iteration_in_step / total_rows
                base_progress = 25 + ((current_step_index - 1) / total_active_steps) * 65
                step_range = 65 / total_active_steps
                progress = base_progress + (step_progress * step_range)
                progress = min(progress, 90)

                tracker.emit('cleaning', progress, step_key)

    def emit_step_complete(step_key, removed_count=None):
        """Emit when step is complete"""
        if tracker:
            progress = 25 + (current_step_index / total_active_steps) * 65
            progress = min(progress, 90)
            # Send key with _complete suffix and count parameter
            complete_key = f"{step_key}_complete"
            params_dict = {'count': removed_count} if removed_count is not None else None
            tracker.emit('cleaning', progress, complete_key, force=True, message_params=params_dict)

    df["UTC"] = pd.to_datetime(df["UTC"], format="%Y-%m-%d %H:%M:%S")

    # === STEP 1: EQ_MAX ===
    if params.get("eqMax"):
        step_key = "measurement_failure_removal"
        start_step(step_key)
        logger.info("Removing measurement failures (identical consecutive values)")
        eq_max = float(params["eqMax"])
        frm = 0
        removed_count = 0
        idx_strt = 0
        for i in range(1, len(df)):
            update_progress(step_key, i)
            if df.iloc[i-1][value_column] == df.iloc[i][value_column] and frm == 0:
                idx_strt = i-1
                frm = 1
            elif df.iloc[i-1][value_column] != df.iloc[i][value_column] and frm == 1:
                idx_end = i-1
                frm_width = (df.iloc[idx_end]["UTC"] - df.iloc[idx_strt]["UTC"]).total_seconds() / 60
                if frm_width >= eq_max:
                    for i_frm in range(idx_strt, idx_end+1):
                        df.iloc[i_frm, df.columns.get_loc(value_column)] = np.nan
                        removed_count += 1
                frm = 0
            elif i == len(df)-1 and frm == 1:
                idx_end = i
                frm_width = (df.iloc[idx_end]["UTC"] - df.iloc[idx_strt]["UTC"]).total_seconds() / 60
                if frm_width >= eq_max:
                    for i_frm in range(idx_strt, idx_end+1):
                        df.iloc[i_frm, df.columns.get_loc(value_column)] = np.nan
                        removed_count += 1
        emit_step_complete(step_key, removed_count)

    # === STEP 2: EL_MAX (vectorized) ===
    if params.get("elMax") is not None:
        step_key = "upper_threshold_removal"
        start_step(step_key)
        logger.info("Removing values above upper threshold")
        el_max = float(params["elMax"])

        mask = df[value_column] > el_max
        removed_count = int(mask.sum())
        df.loc[mask, value_column] = np.nan

        emit_step_complete(step_key, removed_count)

    # === STEP 3: EL_MIN (vectorized) ===
    if params.get("elMin") is not None:
        step_key = "lower_threshold_removal"
        start_step(step_key)
        logger.info("Removing values below lower threshold")
        el_min = float(params["elMin"])

        mask = df[value_column] < el_min
        removed_count = int(mask.sum())
        df.loc[mask, value_column] = np.nan

        emit_step_complete(step_key, removed_count)

    # === STEP 4: RADIO_VALUE_NULL (vectorized) ===
    if params.get("radioValueNull") == "ja":
        step_key = "zero_value_removal"
        start_step(step_key)
        logger.info("Removing zero values")

        mask = df[value_column] == 0
        removed_count = int(mask.sum())
        df.loc[mask, value_column] = np.nan

        emit_step_complete(step_key, removed_count)

    # === STEP 5: CHG_MAX + LG_MAX ===
    if params.get("chgMax") and params.get("lgMax"):
        step_key = "outlier_removal"
        start_step(step_key)
        logger.info("Removing outliers")
        chg_max = float(params["chgMax"])
        lg_max = float(params["lgMax"])
        frm = 0
        removed_count = 0
        idx_strt = 0
        for i in range(1, len(df)):
            update_progress(step_key, i)
            if pd.isna(df.iloc[i][value_column]) and frm == 0:
                pass
            elif pd.isna(df.iloc[i][value_column]) and frm == 1:
                idx_end = i-1
                for i_frm in range(idx_strt, idx_end+1):
                    df.iloc[i_frm, df.columns.get_loc(value_column)] = np.nan
                    removed_count += 1
                frm = 0
            elif pd.isna(df.iloc[i-1][value_column]):
                pass
            else:
                chg = abs(float(df.iloc[i][value_column]) - float(df.iloc[i-1][value_column]))
                t = (df.iloc[i]["UTC"] - df.iloc[i-1]["UTC"]).total_seconds() / 60
                if t > 0 and chg/t > chg_max and frm == 0:
                    idx_strt = i
                    frm = 1
                elif t > 0 and chg/t > chg_max and frm == 1:
                    idx_end = i-1
                    for i_frm in range(idx_strt, idx_end+1):
                        df.iloc[i_frm, df.columns.get_loc(value_column)] = np.nan
                        removed_count += 1
                    frm = 0
                elif frm == 1 and (df.iloc[i]["UTC"] - df.iloc[idx_strt]["UTC"]).total_seconds() / 60 > lg_max:
                    frm = 0
        emit_step_complete(step_key, removed_count)

    # === STEP 6: GAP_MAX ===
    if params.get("gapMax"):
        step_key = "gap_filling"
        start_step(step_key)
        logger.info("Filling measurement gaps")
        gap_max = float(params["gapMax"])
        frm = 0
        filled_count = 0
        idx_strt = 0
        for i in range(1, len(df)):
            update_progress(step_key, i)
            if pd.isna(df.iloc[i][value_column]) and frm == 0:
                idx_strt = i
                frm = 1
            elif not pd.isna(df.iloc[i][value_column]) and frm == 1:
                idx_end = i-1
                frm_width = (df.iloc[idx_end+1]["UTC"] - df.iloc[idx_strt-1]["UTC"]).total_seconds() / 60
                if frm_width <= gap_max and frm_width > 0:
                    dif = float(df.iloc[idx_end+1][value_column]) - float(df.iloc[idx_strt-1][value_column])
                    dif_min = dif/frm_width
                    for i_frm in range(idx_strt, idx_end+1):
                        gap_min = (df.iloc[i_frm]["UTC"] - df.iloc[idx_strt-1]["UTC"]).total_seconds() / 60
                        df.iloc[i_frm, df.columns.get_loc(value_column)] = apply_precision(float(df.iloc[idx_strt-1][value_column]) + gap_min*dif_min)
                        filled_count += 1
                frm = 0
        emit_step_complete(step_key, filled_count)

    # === POST-VALIDATION ===
    if params.get("elMin") is not None:
        el_min = float(params["elMin"])
        final_violations_min = (df[value_column] < el_min).sum()
        zero_values = (df[value_column] == 0).sum()
        logger.info(f"Final validation: Found {final_violations_min} values < {el_min} and {zero_values} zero values")
        if final_violations_min > 0:
            logger.info(f"Removing {final_violations_min} interpolated values below elMin threshold")
            df.loc[df[value_column] < el_min, value_column] = np.nan

    if params.get("elMax") is not None:
        el_max = float(params["elMax"])
        final_violations_max = (df[value_column] > el_max).sum()
        if final_violations_max > 0:
            logger.info(f"Removing {final_violations_max} interpolated values above elMax threshold")
            df.loc[df[value_column] > el_max, value_column] = np.nan

    logger.info("Data cleaning completed")
    return df
