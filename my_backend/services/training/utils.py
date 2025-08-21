"""
Utility functions for training system
Contains helper functions extracted from training_backend_test_2.py
"""

import pandas as pd
import numpy as np
import datetime
import pytz
from typing import Dict, List, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)


def load_and_extract_info(file_path: str) -> Tuple[pd.DataFrame, Dict]:
    """
    Load CSV file and extract information
    Extracted from training_backend_test_2.py load() function around lines 37-168
    
    Args:
        file_path: Path to CSV file
        
    Returns:
        Tuple of (DataFrame, info_dict)
    """
    try:
        # TODO: Extract actual load() function logic from training_backend_test_2.py
        # This is placeholder implementation based on the analysis
        
        # Load CSV
        df = pd.read_csv(file_path)
        
        # Convert UTC to datetime
        if 'UTC' in df.columns:
            df['UTC'] = pd.to_datetime(df['UTC'], format='%Y-%m-%d %H:%M:%S')
        
        # Extract information
        info = {}
        
        if 'UTC' in df.columns:
            # Start time
            info['utc_min'] = df['UTC'].iloc[0]
            
            # End time
            info['utc_max'] = df['UTC'].iloc[-1]
            
            # Time range
            info['time_range'] = info['utc_max'] - info['utc_min']
            
            # Number of rows
            info['num_rows'] = len(df)
            
            # Time step (average)
            if len(df) > 1:
                time_diffs = df['UTC'].diff().dropna()
                info['avg_time_step'] = time_diffs.mean()
            else:
                info['avg_time_step'] = pd.Timedelta(hours=1)
        
        # Data columns info
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        info['numeric_columns'] = numeric_columns
        info['total_columns'] = len(df.columns)
        
        # Basic statistics
        if numeric_columns:
            info['data_stats'] = df[numeric_columns].describe().to_dict()
        
        logger.info(f"Loaded file {file_path}: {df.shape}, time range: {info.get('time_range', 'N/A')}")
        
        return df, info
        
    except Exception as e:
        logger.error(f"Error loading file {file_path}: {str(e)}")
        raise


def utc_idx_pre(df: pd.DataFrame, utc_column: str = 'UTC') -> pd.DataFrame:
    """
    UTC index preprocessing
    Extracted from training_backend_test_2.py utc_idx_pre() function
    
    Args:
        df: Input DataFrame
        utc_column: Name of UTC column
        
    Returns:
        Preprocessed DataFrame
    """
    try:
        # TODO: Extract actual utc_idx_pre() logic from training_backend_test_2.py
        # This is placeholder implementation
        
        df = df.copy()
        
        # Ensure UTC column is datetime
        if utc_column in df.columns:
            df[utc_column] = pd.to_datetime(df[utc_column])
            
            # Set UTC as index
            df.set_index(utc_column, inplace=True)
            
            # Sort by index
            df.sort_index(inplace=True)
            
            # Remove duplicates
            df = df[~df.index.duplicated(keep='first')]
        
        return df
        
    except Exception as e:
        logger.error(f"Error in utc_idx_pre: {str(e)}")
        raise


def utc_idx_post(df: pd.DataFrame, utc_column: str = 'UTC') -> pd.DataFrame:
    """
    UTC index postprocessing
    Extracted from training_backend_test_2.py utc_idx_post() function
    
    Args:
        df: Input DataFrame
        utc_column: Name of UTC column
        
    Returns:
        Postprocessed DataFrame
    """
    try:
        # TODO: Extract actual utc_idx_post() logic from training_backend_test_2.py
        # This is placeholder implementation
        
        df = df.copy()
        
        # Reset index if UTC is the index
        if df.index.name == utc_column or isinstance(df.index, pd.DatetimeIndex):
            df.reset_index(inplace=True)
            
            # Rename index column to UTC if needed
            if df.index.name and df.index.name != utc_column:
                df.rename(columns={df.index.name: utc_column}, inplace=True)
        
        return df
        
    except Exception as e:
        logger.error(f"Error in utc_idx_post: {str(e)}")
        raise


def validate_time_series_data(df: pd.DataFrame, utc_column: str = 'UTC') -> Dict:
    """
    Validate time series data
    
    Args:
        df: DataFrame to validate
        utc_column: Name of UTC column
        
    Returns:
        Dict containing validation results
    """
    try:
        validation_results = {
            'is_valid': True,
            'issues': [],
            'warnings': [],
            'info': {}
        }
        
        # Check if UTC column exists
        if utc_column not in df.columns:
            validation_results['is_valid'] = False
            validation_results['issues'].append(f"UTC column '{utc_column}' not found")
            return validation_results
        
        # Check if UTC column is datetime
        if not pd.api.types.is_datetime64_any_dtype(df[utc_column]):
            validation_results['warnings'].append(f"UTC column '{utc_column}' is not datetime type")
        
        # Check for missing values in UTC column
        utc_missing = df[utc_column].isna().sum()
        if utc_missing > 0:
            validation_results['is_valid'] = False
            validation_results['issues'].append(f"UTC column has {utc_missing} missing values")
        
        # Check for duplicate timestamps
        utc_duplicates = df[utc_column].duplicated().sum()
        if utc_duplicates > 0:
            validation_results['warnings'].append(f"UTC column has {utc_duplicates} duplicate timestamps")
        
        # Check data continuity
        if len(df) > 1:
            df_sorted = df.sort_values(utc_column)
            time_diffs = df_sorted[utc_column].diff().dropna()
            
            # Check for irregular time intervals
            if len(time_diffs) > 0:
                median_diff = time_diffs.median()
                irregular_intervals = (time_diffs > median_diff * 2).sum()
                
                if irregular_intervals > 0:
                    validation_results['warnings'].append(f"Found {irregular_intervals} irregular time intervals")
                
                validation_results['info']['median_time_interval'] = str(median_diff)
                validation_results['info']['time_range'] = str(df_sorted[utc_column].max() - df_sorted[utc_column].min())
        
        # Check for missing values in numeric columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            missing_count = df[col].isna().sum()
            if missing_count > 0:
                validation_results['warnings'].append(f"Column '{col}' has {missing_count} missing values")
        
        validation_results['info']['total_rows'] = len(df)
        validation_results['info']['total_columns'] = len(df.columns)
        validation_results['info']['numeric_columns'] = len(numeric_columns)
        
        return validation_results
        
    except Exception as e:
        logger.error(f"Error validating time series data: {str(e)}")
        raise


def convert_timezone(df: pd.DataFrame, utc_column: str = 'UTC', 
                    target_timezone: str = 'Europe/Vienna') -> pd.DataFrame:
    """
    Convert timezone for UTC column
    
    Args:
        df: Input DataFrame
        utc_column: Name of UTC column
        target_timezone: Target timezone
        
    Returns:
        DataFrame with converted timezone
    """
    try:
        df = df.copy()
        
        if utc_column in df.columns:
            # Ensure UTC column is datetime
            df[utc_column] = pd.to_datetime(df[utc_column])
            
            # If timezone naive, assume UTC
            if df[utc_column].dt.tz is None:
                df[utc_column] = df[utc_column].dt.tz_localize('UTC')
            
            # Convert to target timezone
            df[utc_column] = df[utc_column].dt.tz_convert(target_timezone)
            
            logger.info(f"Converted timezone from UTC to {target_timezone}")
        
        return df
        
    except Exception as e:
        logger.error(f"Error converting timezone: {str(e)}")
        raise


def resample_time_series(df: pd.DataFrame, utc_column: str = 'UTC', 
                        freq: str = '1H', method: str = 'mean') -> pd.DataFrame:
    """
    Resample time series data
    
    Args:
        df: Input DataFrame
        utc_column: Name of UTC column
        freq: Resampling frequency
        method: Resampling method ('mean', 'sum', 'max', 'min')
        
    Returns:
        Resampled DataFrame
    """
    try:
        df = df.copy()
        
        if utc_column in df.columns:
            # Set UTC as index
            df.set_index(utc_column, inplace=True)
            
            # Get numeric columns
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            
            if len(numeric_columns) > 0:
                # Resample based on method
                if method == 'mean':
                    df_resampled = df[numeric_columns].resample(freq).mean()
                elif method == 'sum':
                    df_resampled = df[numeric_columns].resample(freq).sum()
                elif method == 'max':
                    df_resampled = df[numeric_columns].resample(freq).max()
                elif method == 'min':
                    df_resampled = df[numeric_columns].resample(freq).min()
                else:
                    df_resampled = df[numeric_columns].resample(freq).mean()
                
                # Reset index
                df_resampled.reset_index(inplace=True)
                
                logger.info(f"Resampled data from {len(df)} to {len(df_resampled)} rows using {method} method")
                
                return df_resampled
            else:
                logger.warning("No numeric columns found for resampling")
                return df
        else:
            logger.warning(f"UTC column '{utc_column}' not found")
            return df
        
    except Exception as e:
        logger.error(f"Error resampling time series: {str(e)}")
        raise


def detect_outliers(df: pd.DataFrame, columns: List[str] = None, 
                   method: str = 'iqr', threshold: float = 1.5) -> pd.DataFrame:
    """
    Detect outliers in time series data
    
    Args:
        df: Input DataFrame
        columns: Columns to check for outliers (None for all numeric)
        method: Outlier detection method ('iqr', 'zscore')
        threshold: Threshold for outlier detection
        
    Returns:
        DataFrame with outlier flags
    """
    try:
        df = df.copy()
        
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in columns:
            if col in df.columns:
                outlier_col = f'{col}_outlier'
                
                if method == 'iqr':
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    
                    lower_bound = Q1 - threshold * IQR
                    upper_bound = Q3 + threshold * IQR
                    
                    df[outlier_col] = (df[col] < lower_bound) | (df[col] > upper_bound)
                    
                elif method == 'zscore':
                    z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                    df[outlier_col] = z_scores > threshold
                
                outlier_count = df[outlier_col].sum()
                if outlier_count > 0:
                    logger.info(f"Detected {outlier_count} outliers in column '{col}' using {method} method")
        
        return df
        
    except Exception as e:
        logger.error(f"Error detecting outliers: {str(e)}")
        raise


def calculate_time_features(timestamp: pd.Timestamp, timezone: str = 'UTC') -> Dict:
    """
    Calculate time-based features for a timestamp
    
    Args:
        timestamp: Input timestamp
        timezone: Timezone for calculations
        
    Returns:
        Dict containing time features
    """
    try:
        # Convert to specified timezone
        if timestamp.tz is None:
            timestamp = timestamp.tz_localize('UTC')
        
        if timezone != 'UTC':
            timestamp = timestamp.tz_convert(timezone)
        
        features = {
            'year': timestamp.year,
            'month': timestamp.month,
            'day': timestamp.day,
            'hour': timestamp.hour,
            'minute': timestamp.minute,
            'weekday': timestamp.weekday(),
            'week': timestamp.isocalendar().week,
            'quarter': timestamp.quarter,
            'is_weekend': timestamp.weekday() >= 5,
            'is_month_start': timestamp.is_month_start,
            'is_month_end': timestamp.is_month_end,
            'is_quarter_start': timestamp.is_quarter_start,
            'is_quarter_end': timestamp.is_quarter_end,
            'is_year_start': timestamp.is_year_start,
            'is_year_end': timestamp.is_year_end,
            'days_in_month': timestamp.days_in_month,
            'day_of_year': timestamp.dayofyear
        }
        
        # Cyclical features
        features['hour_sin'] = np.sin(2 * np.pi * timestamp.hour / 24)
        features['hour_cos'] = np.cos(2 * np.pi * timestamp.hour / 24)
        features['month_sin'] = np.sin(2 * np.pi * timestamp.month / 12)
        features['month_cos'] = np.cos(2 * np.pi * timestamp.month / 12)
        features['weekday_sin'] = np.sin(2 * np.pi * timestamp.weekday() / 7)
        features['weekday_cos'] = np.cos(2 * np.pi * timestamp.weekday() / 7)
        
        return features
        
    except Exception as e:
        logger.error(f"Error calculating time features: {str(e)}")
        raise


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human readable string
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted duration string
    """
    try:
        if seconds < 60:
            return f"{seconds:.1f} seconds"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f} minutes"
        else:
            hours = seconds / 3600
            return f"{hours:.1f} hours"
            
    except Exception as e:
        logger.error(f"Error formatting duration: {str(e)}")
        return "Unknown duration"


def memory_usage_mb(df: pd.DataFrame) -> float:
    """
    Calculate memory usage of DataFrame in MB
    
    Args:
        df: Input DataFrame
        
    Returns:
        Memory usage in MB
    """
    try:
        memory_usage = df.memory_usage(deep=True).sum()
        return memory_usage / (1024 * 1024)
        
    except Exception as e:
        logger.error(f"Error calculating memory usage: {str(e)}")
        return 0.0