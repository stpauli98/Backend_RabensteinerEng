"""
Cloud Interpolation Services
Time series interpolation for cloud data
"""
import logging
import pandas as pd

logger = logging.getLogger(__name__)


def interpolate_data(df1, df2, x_col, y_col, max_time_span):
    """
    Interpolate missing values in time series data.

    Args:
        df1: DataFrame with UTC column
        df2: DataFrame with values to interpolate
        x_col: Column name for x values (not used directly, kept for API compatibility)
        y_col: Column name for y values to interpolate
        max_time_span: Maximum time gap in minutes to interpolate across

    Returns:
        Tuple of (interpolated DataFrame, count of added points)
    """
    df = pd.DataFrame()
    df['UTC'] = pd.to_datetime(df1['UTC'])
    df['value'] = pd.to_numeric(df2[y_col], errors='coerce')
    df = df.sort_values('UTC').reset_index(drop=True)

    df_final = df.copy()

    # Find first non-NaN value
    i = 0
    while i < len(df_final):
        if pd.isna(df_final.at[i, 'value']):
            i += 1
        else:
            break

    if i == len(df_final):
        logger.warning("No numeric data found for interpolation")
        return df_final, 0

    frame = "non"
    i_start = 0

    i = 0
    while i < len(df_final):
        if frame == "non":
            if pd.isna(df_final.at[i, 'value']):
                i_start = i
                frame = "open"
            i += 1
        elif frame == "open":
            if not pd.isna(df_final.at[i, 'value']):
                frame_width = (df_final.at[i, 'UTC'] - df_final.at[i_start-1, 'UTC']).total_seconds() / 60

                if frame_width <= max_time_span:
                    y0 = df_final.at[i_start-1, 'value']
                    y1 = df_final.at[i, 'value']
                    t0 = df_final.at[i_start-1, 'UTC']
                    t1 = df_final.at[i, 'UTC']

                    y_diff = y1 - y0
                    diff_per_min = y_diff / frame_width

                    for j in range(i_start, i):
                        gap_min = (df_final.at[j, 'UTC'] - t0).total_seconds() / 60
                        df_final.at[j, 'value'] = y0 + (gap_min * diff_per_min)

                frame = "non"
                i += 1
            else:
                i += 1
        else:
            i += 1

        if i >= len(df_final):
            break

    original_nans = df['value'].isna().sum()
    final_nans = df_final['value'].isna().sum()
    added_points = original_nans - final_nans

    return df_final, added_points
