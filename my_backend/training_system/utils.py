"""
Utility functions for training system
Contains helper functions extracted from training_backend_test_2.py
"""

import pandas as pd
import numpy as np
from datetime import datetime
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
        # CSV loading logic implemented with pandas and metadata extraction
        
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
        # UTC index preprocessing implemented with pandas datetime handling
        
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
        # UTC index postprocessing implemented with index reset functionality
        
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


# ============================================================================
# Parameter Validation Functions
# ============================================================================

def validate_frontend_model_parameters(frontend_params: Dict) -> Dict:
    """
    Validate frontend model parameters and return validation results
    
    Args:
        frontend_params: Frontend parameter dictionary
        
    Returns:
        Dict with validation results: {'valid': bool, 'errors': List[str], 'warnings': List[str]}
    """
    validation_result = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'suggestions': []
    }
    
    try:
        if not frontend_params:
            validation_result['valid'] = False
            validation_result['errors'].append("No model parameters provided")
            return validation_result
        
        mode = frontend_params.get('MODE', '').lower()
        if not mode:
            validation_result['valid'] = False
            validation_result['errors'].append("Model MODE is required (Dense, CNN, or SVR)")
            return validation_result
            
        if mode not in ['dense', 'cnn', 'svr']:
            validation_result['valid'] = False
            validation_result['errors'].append(f"Invalid model MODE: {mode}. Must be Dense, CNN, or SVR")
            return validation_result
        
        # Dense model validation
        if mode == 'dense':
            # Validate neurons (N)
            n = frontend_params.get('N')
            if n is not None:
                if not isinstance(n, (int, float)) or n <= 0:
                    validation_result['errors'].append("Neurons (N) must be a positive number")
                elif n < 16:
                    validation_result['warnings'].append("Very small neural network (N < 16) may underperform")
                elif n > 512:
                    validation_result['warnings'].append("Large neural network (N > 512) may overfit or train slowly")
            
            # Validate layers (LAY)
            lay = frontend_params.get('LAY')
            if lay is not None:
                if not isinstance(lay, (int, float)) or lay <= 0:
                    validation_result['errors'].append("Layers (LAY) must be a positive number")
                elif lay > 5:
                    validation_result['warnings'].append("Deep networks (LAY > 5) may have training difficulties")
            
            # Validate epochs (EP)
            ep = frontend_params.get('EP')
            if ep is not None:
                if not isinstance(ep, (int, float)) or ep <= 0:
                    validation_result['errors'].append("Epochs (EP) must be a positive number")
                elif ep < 10:
                    validation_result['warnings'].append("Very few epochs (EP < 10) may lead to underfitting")
                elif ep > 1000:
                    validation_result['warnings'].append("Many epochs (EP > 1000) may cause overfitting")
            
            # Validate activation function
            actf = frontend_params.get('ACTF', '').lower()
            valid_activations = ['relu', 'sigmoid', 'tanh', 'linear']
            if actf and actf not in valid_activations:
                validation_result['errors'].append(f"Invalid activation function: {actf}. Valid options: {', '.join(valid_activations)}")
        
        # CNN model validation
        elif mode == 'cnn':
            # Validate kernel size (K)
            k = frontend_params.get('K')
            if k is not None:
                if not isinstance(k, (int, float)) or k <= 0:
                    validation_result['errors'].append("Kernel size (K) must be a positive number")
                elif k % 2 == 0:
                    validation_result['warnings'].append("Even kernel sizes may cause asymmetric feature detection")
                elif k > 7:
                    validation_result['warnings'].append("Large kernel size (K > 7) may lose fine details")
        
        # SVR model validation  
        elif mode == 'svr':
            # Validate C parameter
            c = frontend_params.get('C')
            if c is not None:
                if not isinstance(c, (int, float)) or c <= 0:
                    validation_result['errors'].append("Regularization parameter (C) must be positive")
                elif c < 0.01:
                    validation_result['warnings'].append("Very small C may cause underfitting")
                elif c > 100:
                    validation_result['warnings'].append("Very large C may cause overfitting")
            
            # Validate epsilon
            epsilon = frontend_params.get('EPSILON')
            if epsilon is not None:
                if not isinstance(epsilon, (int, float)) or epsilon < 0:
                    validation_result['errors'].append("Epsilon must be non-negative")
            
            # Validate kernel
            kernel = frontend_params.get('KERNEL', '').lower()
            valid_kernels = ['rbf', 'linear', 'poly', 'sigmoid']
            if kernel and kernel not in valid_kernels:
                validation_result['errors'].append(f"Invalid kernel: {kernel}. Valid options: {', '.join(valid_kernels)}")
        
        # Add general suggestions
        if not validation_result['errors']:
            validation_result['suggestions'].append("Parameters look good! Consider adjusting based on your dataset size and complexity.")
            
            if mode == 'dense' and frontend_params.get('EP', 100) < 50:
                validation_result['suggestions'].append("For time series data, consider using more epochs (50-200) for better convergence.")
            
            if mode == 'cnn':
                validation_result['suggestions'].append("CNN models work best with spatial or sequential patterns in your data.")
                
            if mode == 'svr':
                validation_result['suggestions'].append("SVR is excellent for non-linear relationships. Consider 'rbf' kernel for complex patterns.")
        
        # Mark as invalid if any errors exist
        validation_result['valid'] = len(validation_result['errors']) == 0
        
        logger.info(f"Parameter validation for {mode} model: {'PASSED' if validation_result['valid'] else 'FAILED'}")
        
        return validation_result
        
    except Exception as e:
        logger.error(f"Error validating model parameters: {str(e)}")
        return {
            'valid': False,
            'errors': [f"Validation system error: {str(e)}"],
            'warnings': [],
            'suggestions': []
        }


def create_user_friendly_error_response(error_type: str, error_details: str, session_id: str = None) -> Dict:
    """
    Create user-friendly error responses with actionable guidance
    
    Args:
        error_type: Type of error (validation, training, database, etc.)
        error_details: Detailed error information
        session_id: Optional session ID for context
        
    Returns:
        Dict with structured error response
    """
    error_response = {
        'success': False,
        'error_type': error_type,
        'user_message': '',
        'technical_details': error_details,
        'recovery_suggestions': [],
        'session_id': session_id
    }
    
    if error_type == 'parameter_validation':
        error_response['user_message'] = "Model parameters need adjustment before training can begin."
        error_response['recovery_suggestions'] = [
            "Review the parameter validation errors above",
            "Adjust the problematic parameters in the model configuration",
            "Ensure all required parameters are provided",
            "Try with default values if unsure about parameter ranges"
        ]
        
    elif error_type == 'training_failed':
        error_response['user_message'] = "Model training encountered an error and could not complete."
        error_response['recovery_suggestions'] = [
            "Check if your dataset has sufficient data points",
            "Try reducing model complexity (fewer layers/neurons)",
            "Verify that your data doesn't contain invalid values",
            "Consider using different model parameters"
        ]
        
    elif error_type == 'data_processing':
        error_response['user_message'] = "There was an issue processing your dataset for training."
        error_response['recovery_suggestions'] = [
            "Ensure your dataset contains numeric columns for prediction",
            "Check for missing or corrupted data in your file",
            "Verify that datetime columns are properly formatted",
            "Try uploading a different dataset or cleaning your current one"
        ]
        
    elif error_type == 'database_error':
        error_response['user_message'] = "Unable to save training results due to a database issue."
        error_response['recovery_suggestions'] = [
            "This is typically a temporary issue - please try again",
            "If the problem persists, contact system administrator",
            "Your training may have completed but results weren't saved"
        ]
        
    elif error_type == 'session_not_found':
        error_response['user_message'] = "The training session could not be found or has expired."
        error_response['recovery_suggestions'] = [
            "Start a new training session by uploading data again",
            "Ensure you're using the correct session ID",
            "Check if the session may have expired (sessions last 24 hours)"
        ]
        
    elif error_type == 'insufficient_data':
        error_response['user_message'] = "Your dataset doesn't have enough data points for reliable model training."
        error_response['recovery_suggestions'] = [
            "Upload a dataset with at least 100 data points",
            "Combine multiple similar datasets if available",
            "Consider collecting more data before training",
            "Use simpler models for small datasets (try SVR instead of Neural Networks)"
        ]
        
    else:
        error_response['user_message'] = "An unexpected error occurred during the operation."
        error_response['recovery_suggestions'] = [
            "Please try the operation again",
            "If the error persists, try with different parameters",
            "Contact support if the issue continues"
        ]
    
    return error_response


def validate_training_split_parameters(split_params: Dict) -> Dict:
    """
    Validate training split parameters
    
    Args:
        split_params: Split parameter dictionary
        
    Returns:
        Dict with validation results
    """
    validation_result = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'suggestions': []
    }
    
    try:
        if not split_params:
            validation_result['suggestions'].append("Using default 80/20 train/test split")
            return validation_result
        
        # Validate train_size
        train_size = split_params.get('train_size')
        if train_size is not None:
            if not isinstance(train_size, (int, float)):
                validation_result['errors'].append("Train size must be a number")
            elif train_size <= 0 or train_size >= 1:
                validation_result['errors'].append("Train size must be between 0 and 1")
            elif train_size < 0.5:
                validation_result['warnings'].append("Very small training set (< 50%) may lead to poor model performance")
            elif train_size > 0.9:
                validation_result['warnings'].append("Very large training set (> 90%) leaves little data for testing")
        
        # Validate test_size
        test_size = split_params.get('test_size')
        if test_size is not None:
            if not isinstance(test_size, (int, float)):
                validation_result['errors'].append("Test size must be a number")
            elif test_size <= 0 or test_size >= 1:
                validation_result['errors'].append("Test size must be between 0 and 1")
        
        # Check if both train_size and test_size are provided
        if train_size is not None and test_size is not None:
            if abs(train_size + test_size - 1.0) > 0.01:
                validation_result['errors'].append("Train size and test size must sum to 1.0")
        
        # Mark as invalid if any errors exist
        validation_result['valid'] = len(validation_result['errors']) == 0
        
        if validation_result['valid'] and not validation_result['warnings']:
            validation_result['suggestions'].append("Split parameters look good for reliable model evaluation")
        
        return validation_result
        
    except Exception as e:
        logger.error(f"Error validating split parameters: {str(e)}")
        return {
            'valid': False,
            'errors': [f"Split validation error: {str(e)}"],
            'warnings': [],
            'suggestions': []
        }


# Training Progress & Real-Time Updates
# ============================================================================

def emit_training_progress(session_id: str, progress_data: Dict):
    """
    Emit training progress updates via SocketIO
    
    Args:
        session_id: Training session ID
        progress_data: Progress information to emit
    """
    try:
        from flask import current_app
        
        if 'socketio' in current_app.extensions:
            socketio = current_app.extensions['socketio']
            
            # Emit to session-specific room
            room = f"training_{session_id}"
            
            # Enhanced progress data with timestamp
            enhanced_data = {
                'timestamp': datetime.now().isoformat(),
                'session_id': session_id,
                **progress_data
            }
            
            socketio.emit('training_progress', enhanced_data, room=room)
            logger.info(f"Emitted training progress to room {room}: {enhanced_data.get('status', 'unknown')}")
            
    except Exception as e:
        logger.error(f"Failed to emit training progress: {str(e)}")


def emit_training_metrics(session_id: str, metrics_data: Dict):
    """
    Emit real-time training metrics (loss, accuracy, etc.)
    
    Args:
        session_id: Training session ID  
        metrics_data: Training metrics to emit
    """
    try:
        from flask import current_app
        
        if 'socketio' in current_app.extensions:
            socketio = current_app.extensions['socketio']
            
            room = f"training_{session_id}"
            
            # Enhanced metrics with metadata
            enhanced_metrics = {
                'timestamp': datetime.now().isoformat(),
                'session_id': session_id,
                'metrics_type': 'real_time',
                **metrics_data
            }
            
            socketio.emit('training_metrics', enhanced_metrics, room=room)
            logger.debug(f"Emitted training metrics to room {room}")
            
    except Exception as e:
        logger.error(f"Failed to emit training metrics: {str(e)}")


def emit_training_error(session_id: str, error_data: Dict):
    """
    Emit training error notifications
    
    Args:
        session_id: Training session ID
        error_data: Error information to emit
    """
    try:
        from flask import current_app
        
        if 'socketio' in current_app.extensions:
            socketio = current_app.extensions['socketio']
            
            room = f"training_{session_id}"
            
            enhanced_error = {
                'timestamp': datetime.now().isoformat(),
                'session_id': session_id,
                'error_type': 'training_error',
                **error_data
            }
            
            socketio.emit('training_error', enhanced_error, room=room)
            logger.warning(f"Emitted training error to room {room}: {error_data.get('message', 'unknown error')}")
            
    except Exception as e:
        logger.error(f"Failed to emit training error: {str(e)}")


def calculate_training_eta(current_epoch: int, total_epochs: int, epoch_start_time: float, epoch_duration: float = None) -> Dict:
    """
    Calculate estimated time of arrival for training completion
    
    Args:
        current_epoch: Current training epoch
        total_epochs: Total number of epochs
        epoch_start_time: Start time of current epoch
        epoch_duration: Average duration per epoch (calculated if not provided)
        
    Returns:
        Dict with ETA information
    """
    try:
        import time
        
        if current_epoch <= 0:
            return {'eta_seconds': None, 'eta_formatted': 'Calculating...', 'progress_percent': 0.0}
        
        # Calculate progress percentage
        progress_percent = (current_epoch / total_epochs) * 100
        
        # Calculate average epoch duration
        if epoch_duration is None:
            current_time = time.time()
            epoch_duration = (current_time - epoch_start_time) / current_epoch
        
        # Calculate remaining time
        remaining_epochs = total_epochs - current_epoch
        eta_seconds = remaining_epochs * epoch_duration
        
        # Format ETA
        if eta_seconds < 60:
            eta_formatted = f"{eta_seconds:.0f} seconds"
        elif eta_seconds < 3600:
            minutes = eta_seconds / 60
            eta_formatted = f"{minutes:.1f} minutes"
        else:
            hours = eta_seconds / 3600
            eta_formatted = f"{hours:.1f} hours"
        
        return {
            'eta_seconds': eta_seconds,
            'eta_formatted': eta_formatted,
            'progress_percent': progress_percent,
            'epochs_remaining': remaining_epochs,
            'avg_epoch_duration': epoch_duration
        }
        
    except Exception as e:
        logger.error(f"Error calculating training ETA: {str(e)}")
        return {'eta_seconds': None, 'eta_formatted': 'Unknown', 'progress_percent': 0.0}


# JSON Sanitization Functions
# ============================================================================

def sanitize_for_json(data):
    """
    Sanitize data for JSON serialization by handling NaN, Inf, and non-serializable objects
    
    Args:
        data: Data structure to sanitize (dict, list, or primitive)
        
    Returns:
        JSON-serializable version of the data
    """
    import json
    import numpy as np
    import math
    
    try:
        if data is None:
            return None
        
        # Handle different data types
        if isinstance(data, dict):
            sanitized = {}
            for key, value in data.items():
                # Skip TensorFlow/Keras model objects
                if hasattr(value, '__class__') and any(
                    model_type in str(type(value)) 
                    for model_type in ['tensorflow', 'keras', 'Sequential', 'Model']
                ):
                    logger.info(f"Skipping non-serializable model object: {key} ({type(value)})")
                    continue
                
                sanitized[key] = sanitize_for_json(value)
            return sanitized
            
        elif isinstance(data, (list, tuple)):
            return [sanitize_for_json(item) for item in data]
            
        elif isinstance(data, np.ndarray):
            # Convert numpy arrays to lists and sanitize
            return sanitize_for_json(data.tolist())
            
        elif isinstance(data, (np.integer, np.floating)):
            # Convert numpy types to Python types
            return sanitize_for_json(data.item())
            
        elif isinstance(data, float):
            # Handle NaN and Inf values
            if math.isnan(data):
                return None  # Convert NaN to null
            elif math.isinf(data):
                return 999999 if data > 0 else -999999  # Convert Inf to large numbers
            else:
                return data
                
        elif isinstance(data, (int, str, bool)):
            return data
            
        else:
            # For other objects, try to convert to string or skip
            try:
                # Test if it's JSON serializable
                json.dumps(data)
                return data
            except (TypeError, ValueError):
                logger.warning(f"Skipping non-serializable object: {type(data)}")
                return str(type(data))  # Return type name as string
                
    except Exception as e:
        logger.error(f"Error sanitizing data for JSON: {str(e)}")
        return None


# ============================================================================
# UI Parameter Conversion Functions
# ============================================================================

def convert_frontend_to_backend_params(frontend_params: Dict) -> Dict:
    """
    Convert frontend flat parameter structure to backend nested structure
    Transforms Training.tsx format to backend-expected format
    
    Args:
        frontend_params: Flat parameters from frontend with MODE indicator
        Expected structure:
        {
            "MODE": "Dense"|"CNN"|"SVR",
            "LAY": 2, "N": 64, "EP": 100, "ACTF": "ReLU",
            "K": 3, "KERNEL": "rbf", "C": 1.0, "EPSILON": 0.1
        }
        
    Returns:
        Dict with nested structure expected by backend:
        {
            "dense": {...} | "cnn": {...} | "svr": {...}
        }
    """
    try:
        logger.info(f"Converting frontend parameters: {list(frontend_params.keys())}")
        
        # Get the model mode
        mode = frontend_params.get('MODE', '').lower()
        
        if not mode:
            logger.error("No MODE specified in frontend parameters")
            return {}
        
        backend_params = {}
        
        if mode == 'dense':
            # Convert Dense model parameters
            dense_config = {}
            
            # Map frontend parameter names to backend names
            if 'N' in frontend_params:
                # Convert single N value to layers array
                layers = [frontend_params['N']]
                if 'LAY' in frontend_params and frontend_params['LAY'] > 1:
                    # Add additional layers if LAY > 1
                    for i in range(frontend_params['LAY'] - 1):
                        layers.append(max(16, frontend_params['N'] // (2 ** (i + 1))))
                dense_config['layers'] = layers
            
            if 'EP' in frontend_params:
                dense_config['epochs'] = int(frontend_params['EP'])
            
            if 'ACTF' in frontend_params:
                # Convert frontend activation names to backend format
                actf_map = {
                    'ReLU': 'relu',
                    'Sigmoid': 'sigmoid', 
                    'Tanh': 'tanh',
                    'Linear': 'linear'
                }
                dense_config['activation'] = actf_map.get(frontend_params['ACTF'], 'relu')
            
            # Add default values for required parameters
            dense_config.setdefault('optimizer', 'adam')
            dense_config.setdefault('loss', 'mse')
            dense_config.setdefault('batch_size', 32)
            dense_config.setdefault('validation_split', 0.2)
            dense_config.setdefault('learning_rate', 0.001)
            dense_config.setdefault('dropout', 0.1)
            
            backend_params['dense'] = dense_config
            logger.info(f"Converted to Dense model config: {dense_config}")
            
        elif mode == 'cnn':
            # Convert CNN model parameters
            cnn_config = {}
            
            if 'K' in frontend_params:
                cnn_config['kernel_size'] = [frontend_params['K'], frontend_params['K']]
            
            if 'EP' in frontend_params:
                cnn_config['epochs'] = int(frontend_params['EP'])
            
            if 'ACTF' in frontend_params:
                actf_map = {
                    'ReLU': 'relu',
                    'Sigmoid': 'sigmoid',
                    'Tanh': 'tanh',
                    'Linear': 'linear'
                }
                cnn_config['activation'] = actf_map.get(frontend_params['ACTF'], 'relu')
            
            # Add default CNN parameters
            cnn_config.setdefault('filters', [32, 64])
            cnn_config.setdefault('pool_size', [2, 2])
            cnn_config.setdefault('dense_layers', [50])
            cnn_config.setdefault('optimizer', 'adam')
            cnn_config.setdefault('loss', 'mse')
            cnn_config.setdefault('batch_size', 32)
            cnn_config.setdefault('validation_split', 0.2)
            cnn_config.setdefault('learning_rate', 0.001)
            cnn_config.setdefault('dropout', 0.1)
            
            backend_params['cnn'] = cnn_config
            logger.info(f"Converted to CNN model config: {cnn_config}")
            
        elif mode == 'svr':
            # Convert SVR model parameters
            svr_config = {}
            
            if 'KERNEL' in frontend_params:
                svr_config['kernel'] = frontend_params['KERNEL'].lower()
            
            if 'C' in frontend_params:
                svr_config['C'] = float(frontend_params['C'])
            
            if 'EPSILON' in frontend_params:
                svr_config['epsilon'] = float(frontend_params['EPSILON'])
            
            # Add default SVR parameters
            svr_config.setdefault('gamma', 'scale')
            svr_config.setdefault('degree', 3)
            svr_config.setdefault('coef0', 0.0)
            svr_config.setdefault('shrinking', True)
            svr_config.setdefault('cache_size', 200)
            svr_config.setdefault('max_iter', 1000)
            
            backend_params['svr'] = svr_config
            logger.info(f"Converted to SVR model config: {svr_config}")
            
        else:
            logger.error(f"Unsupported model MODE: {mode}")
            return {}
        
        logger.info(f"Frontend conversion successful: {list(backend_params.keys())} model(s) configured")
        return backend_params
        
    except Exception as e:
        logger.error(f"Error converting frontend parameters: {str(e)}")
        raise


def convert_ui_to_mdl_config(ui_params: Dict) -> Dict:
    """
    Convert UI model parameters to MDL configuration format
    Maps frontend Training.tsx parameters to backend MDL class
    
    Args:
        ui_params: UI parameters from frontend
        Expected structure:
        {
            "dense": {"layers": [64, 32], "epochs": 100, "batch_size": 32, ...},
            "cnn": {"filters": [32, 64], "kernel_size": [3, 3], ...},
            "lstm": {"units": [50, 50], "dropout": 0.2, ...},
            "svr": {"kernel": "rbf", "C": 1.0, "gamma": "scale", ...},
            "linear": {"fit_intercept": True, "normalize": False, ...}
        }
        
    Returns:
        Dict containing MDL-compatible configuration
    """
    try:
        mdl_config = {}
        
        # Dense Neural Network parameters
        if "dense" in ui_params:
            dense_params = ui_params["dense"]
            mdl_config["dense"] = {
                "layers": dense_params.get("layers", [64, 32, 16]),
                "activation": dense_params.get("activation", "relu"),
                "optimizer": dense_params.get("optimizer", "adam"),
                "loss": dense_params.get("loss", "mse"),
                "epochs": int(dense_params.get("epochs", 100)),
                "batch_size": int(dense_params.get("batch_size", 32)),
                "validation_split": float(dense_params.get("validation_split", 0.2)),
                "learning_rate": float(dense_params.get("learning_rate", 0.001)),
                "dropout": float(dense_params.get("dropout", 0.1))
            }
        
        # CNN parameters
        if "cnn" in ui_params:
            cnn_params = ui_params["cnn"]
            mdl_config["cnn"] = {
                "filters": cnn_params.get("filters", [32, 64]),
                "kernel_size": cnn_params.get("kernel_size", [3, 3]),
                "activation": cnn_params.get("activation", "relu"),
                "pool_size": cnn_params.get("pool_size", [2, 2]),
                "dense_layers": cnn_params.get("dense_layers", [50]),
                "optimizer": cnn_params.get("optimizer", "adam"),
                "loss": cnn_params.get("loss", "mse"),
                "epochs": int(cnn_params.get("epochs", 100)),
                "batch_size": int(cnn_params.get("batch_size", 32)),
                "validation_split": float(cnn_params.get("validation_split", 0.2)),
                "learning_rate": float(cnn_params.get("learning_rate", 0.001)),
                "dropout": float(cnn_params.get("dropout", 0.1))
            }
        
        # LSTM parameters
        if "lstm" in ui_params:
            lstm_params = ui_params["lstm"]
            mdl_config["lstm"] = {
                "units": lstm_params.get("units", [50, 50]),
                "return_sequences": lstm_params.get("return_sequences", True),
                "activation": lstm_params.get("activation", "tanh"),
                "recurrent_activation": lstm_params.get("recurrent_activation", "sigmoid"),
                "dropout": float(lstm_params.get("dropout", 0.0)),
                "recurrent_dropout": float(lstm_params.get("recurrent_dropout", 0.0)),
                "dense_layers": lstm_params.get("dense_layers", [25]),
                "optimizer": lstm_params.get("optimizer", "adam"),
                "loss": lstm_params.get("loss", "mse"),
                "epochs": int(lstm_params.get("epochs", 100)),
                "batch_size": int(lstm_params.get("batch_size", 32)),
                "validation_split": float(lstm_params.get("validation_split", 0.2)),
                "learning_rate": float(lstm_params.get("learning_rate", 0.001))
            }
        
        # SVR parameters
        if "svr" in ui_params:
            svr_params = ui_params["svr"]
            mdl_config["svr"] = {
                "kernel": svr_params.get("kernel", "rbf"),
                "C": float(svr_params.get("C", 1.0)),
                "gamma": svr_params.get("gamma", "scale"),
                "epsilon": float(svr_params.get("epsilon", 0.1)),
                "degree": int(svr_params.get("degree", 3)),
                "coef0": float(svr_params.get("coef0", 0.0)),
                "shrinking": bool(svr_params.get("shrinking", True)),
                "cache_size": int(svr_params.get("cache_size", 200)),
                "max_iter": int(svr_params.get("max_iter", 1000))
            }
        
        # Linear Regression parameters
        if "linear" in ui_params:
            linear_params = ui_params["linear"]
            mdl_config["linear"] = {
                "fit_intercept": bool(linear_params.get("fit_intercept", True)),
                "normalize": bool(linear_params.get("normalize", False)),
                "copy_X": bool(linear_params.get("copy_X", True)),
                "positive": bool(linear_params.get("positive", False))
            }
        
        logger.info(f"Converted UI parameters to MDL config: {len(mdl_config)} models configured")
        return mdl_config
        
    except Exception as e:
        logger.error(f"Error converting UI to MDL config: {str(e)}")
        raise


def convert_ui_to_training_split(ui_split_params: Dict) -> Dict:
    """
    Convert UI training split parameters to backend format
    Maps frontend split configuration to training pipeline format
    
    Args:
        ui_split_params: UI split parameters
        Expected structure:
        {
            "train_ratio": 0.7,
            "validation_ratio": 0.15,
            "test_ratio": 0.15,
            "shuffle": true,
            "random_state": 42,
            "stratify": false
        }
        
    Returns:
        Dict containing training split configuration
    """
    try:
        # Default split ratios if not provided
        train_ratio = float(ui_split_params.get("train_ratio", 0.7))
        validation_ratio = float(ui_split_params.get("validation_ratio", 0.15))
        test_ratio = float(ui_split_params.get("test_ratio", 0.15))
        
        # Validate ratios sum to 1.0
        total_ratio = train_ratio + validation_ratio + test_ratio
        if abs(total_ratio - 1.0) > 0.01:
            logger.warning(f"Split ratios sum to {total_ratio}, normalizing to 1.0")
            train_ratio = train_ratio / total_ratio
            validation_ratio = validation_ratio / total_ratio
            test_ratio = test_ratio / total_ratio
        
        split_config = {
            "train_ratio": train_ratio,
            "validation_ratio": validation_ratio,
            "test_ratio": test_ratio,
            "shuffle": bool(ui_split_params.get("shuffle", True)),
            "random_state": int(ui_split_params.get("random_state", 42)),
            "stratify": bool(ui_split_params.get("stratify", False)),
            "time_series_split": bool(ui_split_params.get("time_series_split", True))  # For time series data
        }
        
        logger.info(f"Converted training split: train={train_ratio:.2f}, val={validation_ratio:.2f}, test={test_ratio:.2f}")
        return split_config
        
    except Exception as e:
        logger.error(f"Error converting UI split parameters: {str(e)}")
        raise


def validate_model_parameters(model_params: Dict) -> Dict:
    """
    Validate model parameters and return validation results
    Ensures all parameters are within acceptable ranges and types
    
    Args:
        model_params: Model parameters to validate
        
    Returns:
        Dict containing validation results:
        {
            "is_valid": bool,
            "errors": List[str],
            "warnings": List[str],
            "corrected_params": Dict
        }
    """
    try:
        validation_result = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "corrected_params": model_params.copy()
        }
        
        # Validate Dense Neural Network parameters
        if "dense" in model_params:
            dense_params = model_params["dense"]
            
            # Validate layers
            if "layers" in dense_params:
                layers = dense_params["layers"]
                if not isinstance(layers, list) or len(layers) == 0:
                    validation_result["errors"].append("Dense layers must be a non-empty list")
                    validation_result["is_valid"] = False
                else:
                    for i, layer_size in enumerate(layers):
                        if not isinstance(layer_size, int) or layer_size <= 0:
                            validation_result["errors"].append(f"Dense layer {i} size must be positive integer")
                            validation_result["is_valid"] = False
            
            # Validate epochs
            if "epochs" in dense_params:
                epochs = dense_params["epochs"]
                if not isinstance(epochs, int) or epochs <= 0 or epochs > 10000:
                    validation_result["warnings"].append("Dense epochs should be between 1 and 10000")
                    validation_result["corrected_params"]["dense"]["epochs"] = max(1, min(epochs, 10000))
            
            # Validate batch_size
            if "batch_size" in dense_params:
                batch_size = dense_params["batch_size"]
                if not isinstance(batch_size, int) or batch_size <= 0 or batch_size > 1024:
                    validation_result["warnings"].append("Dense batch_size should be between 1 and 1024")
                    validation_result["corrected_params"]["dense"]["batch_size"] = max(1, min(batch_size, 1024))
            
            # Validate learning_rate
            if "learning_rate" in dense_params:
                lr = dense_params["learning_rate"]
                if not isinstance(lr, (int, float)) or lr <= 0 or lr > 1:
                    validation_result["warnings"].append("Dense learning_rate should be between 0 and 1")
                    validation_result["corrected_params"]["dense"]["learning_rate"] = max(0.0001, min(lr, 1.0))
        
        # Validate CNN parameters
        if "cnn" in model_params:
            cnn_params = model_params["cnn"]
            
            # Validate filters
            if "filters" in cnn_params:
                filters = cnn_params["filters"]
                if not isinstance(filters, list) or len(filters) == 0:
                    validation_result["errors"].append("CNN filters must be a non-empty list")
                    validation_result["is_valid"] = False
                else:
                    for i, filter_size in enumerate(filters):
                        if not isinstance(filter_size, int) or filter_size <= 0:
                            validation_result["errors"].append(f"CNN filter {i} size must be positive integer")
                            validation_result["is_valid"] = False
            
            # Validate kernel_size
            if "kernel_size" in cnn_params:
                kernel_size = cnn_params["kernel_size"]
                if not isinstance(kernel_size, list) or len(kernel_size) != 2:
                    validation_result["errors"].append("CNN kernel_size must be a list of 2 integers")
                    validation_result["is_valid"] = False
        
        # Validate LSTM parameters
        if "lstm" in model_params:
            lstm_params = model_params["lstm"]
            
            # Validate units
            if "units" in lstm_params:
                units = lstm_params["units"]
                if not isinstance(units, list) or len(units) == 0:
                    validation_result["errors"].append("LSTM units must be a non-empty list")
                    validation_result["is_valid"] = False
                else:
                    for i, unit_size in enumerate(units):
                        if not isinstance(unit_size, int) or unit_size <= 0:
                            validation_result["errors"].append(f"LSTM unit {i} size must be positive integer")
                            validation_result["is_valid"] = False
            
            # Validate dropout
            if "dropout" in lstm_params:
                dropout = lstm_params["dropout"]
                if not isinstance(dropout, (int, float)) or dropout < 0 or dropout >= 1:
                    validation_result["warnings"].append("LSTM dropout should be between 0 and 1")
                    validation_result["corrected_params"]["lstm"]["dropout"] = max(0.0, min(dropout, 0.9))
        
        # Validate SVR parameters
        if "svr" in model_params:
            svr_params = model_params["svr"]
            
            # Validate C parameter
            if "C" in svr_params:
                C = svr_params["C"]
                if not isinstance(C, (int, float)) or C <= 0:
                    validation_result["warnings"].append("SVR C parameter should be positive")
                    validation_result["corrected_params"]["svr"]["C"] = max(0.01, C)
            
            # Validate kernel
            if "kernel" in svr_params:
                kernel = svr_params["kernel"]
                valid_kernels = ["linear", "poly", "rbf", "sigmoid"]
                if kernel not in valid_kernels:
                    validation_result["errors"].append(f"SVR kernel must be one of {valid_kernels}")
                    validation_result["is_valid"] = False
            
            # Validate epsilon
            if "epsilon" in svr_params:
                epsilon = svr_params["epsilon"]
                if not isinstance(epsilon, (int, float)) or epsilon < 0:
                    validation_result["warnings"].append("SVR epsilon should be non-negative")
                    validation_result["corrected_params"]["svr"]["epsilon"] = max(0.0, epsilon)
        
        # Validate Linear Regression parameters
        if "linear" in model_params:
            linear_params = model_params["linear"]
            
            # Validate boolean parameters
            bool_params = ["fit_intercept", "normalize", "copy_X", "positive"]
            for param in bool_params:
                if param in linear_params:
                    if not isinstance(linear_params[param], bool):
                        validation_result["warnings"].append(f"Linear {param} should be boolean")
                        validation_result["corrected_params"]["linear"][param] = bool(linear_params[param])
        
        logger.info(f"Parameter validation completed: valid={validation_result['is_valid']}, errors={len(validation_result['errors'])}, warnings={len(validation_result['warnings'])}")
        return validation_result
        
    except Exception as e:
        logger.error(f"Error validating model parameters: {str(e)}")
        raise


def convert_ui_to_mts_config(ui_params: Dict) -> Dict:
    """
    Convert UI MTS parameters to MTS configuration format
    Maps frontend time series parameters to backend MTS class
    
    Args:
        ui_params: UI MTS parameters
        Expected structure:
        {
            "time_features": {
                "jahr": true,
                "monat": true,
                "woche": true,
                "feiertag": true,
                "zeitzone": "Europe/Vienna"
            },
            "zeitschritte": {
                "eingabe": 24,
                "ausgabe": 1,
                "zeitschrittweite": 1,
                "offset": 0
            },
            "preprocessing": {
                "interpolation": true,
                "outlier_removal": true,
                "scaling": true
            }
        }
        
    Returns:
        Dict containing MTS-compatible configuration
    """
    try:
        mts_config = {}
        
        # Time features configuration
        if "time_features" in ui_params:
            time_features = ui_params["time_features"]
            mts_config["time_features"] = {
                "jahr": bool(time_features.get("jahr", True)),
                "monat": bool(time_features.get("monat", True)),
                "woche": bool(time_features.get("woche", True)),
                "feiertag": bool(time_features.get("feiertag", True)),
                "zeitzone": str(time_features.get("zeitzone", "Europe/Vienna")),
                "use_time_features": bool(time_features.get("use_time_features", True))
            }
        
        # Time steps configuration
        if "zeitschritte" in ui_params:
            zeitschritte = ui_params["zeitschritte"]
            mts_config["zeitschritte"] = {
                "time_steps_in": int(zeitschritte.get("eingabe", 24)),
                "time_steps_out": int(zeitschritte.get("ausgabe", 1)),
                "time_step_size": int(zeitschritte.get("zeitschrittweite", 1)),
                "offset": int(zeitschritte.get("offset", 0))
            }
        
        # Preprocessing configuration
        if "preprocessing" in ui_params:
            preprocessing = ui_params["preprocessing"]
            mts_config["preprocessing"] = {
                "interpolation": bool(preprocessing.get("interpolation", True)),
                "outlier_removal": bool(preprocessing.get("outlier_removal", True)),
                "scaling": bool(preprocessing.get("scaling", True)),
                "normalization_method": str(preprocessing.get("normalization_method", "minmax"))
            }
        
        # Data processing configuration
        mts_config["data_processing"] = {
            "remove_duplicates": bool(ui_params.get("remove_duplicates", True)),
            "handle_missing_values": str(ui_params.get("handle_missing_values", "interpolate")),
            "timezone": str(ui_params.get("timezone", "Europe/Vienna"))
        }
        
        logger.info(f"Converted UI parameters to MTS config: {len(mts_config)} sections configured")
        return mts_config
        
    except Exception as e:
        logger.error(f"Error converting UI to MTS config: {str(e)}")
        raise


def merge_ui_with_session_data(ui_params: Dict, session_data: Dict) -> Dict:
    """
    Merge UI parameters with existing session data
    Combines user-provided parameters with session configuration
    
    Args:
        ui_params: Parameters from UI
        session_data: Existing session data from database
        
    Returns:
        Dict containing merged configuration
    """
    try:
        # Start with existing session data
        merged_data = session_data.copy()
        
        # Convert and merge MTS parameters
        if "mts_params" in ui_params:
            mts_config = convert_ui_to_mts_config(ui_params["mts_params"])
            merged_data.update(mts_config)
        
        # Convert and merge model parameters
        if "model_params" in ui_params:
            # Validate parameters first
            validation_result = validate_model_parameters(ui_params["model_params"])
            
            if not validation_result["is_valid"]:
                raise ValueError(f"Invalid model parameters: {validation_result['errors']}")
            
            # Use corrected parameters
            mdl_config = convert_ui_to_mdl_config(validation_result["corrected_params"])
            merged_data["model_configuration"] = mdl_config
        
        # Convert and merge split parameters
        if "split_params" in ui_params:
            split_config = convert_ui_to_training_split(ui_params["split_params"])
            merged_data["training_split"] = split_config
        
        # Add UI parameter metadata
        merged_data["ui_params_metadata"] = {
            "parameters_provided_by_user": True,
            "ui_timestamp": datetime.datetime.now().isoformat(),
            "parameter_source": "frontend_ui",
            "validation_warnings": validation_result.get("warnings", []) if "model_params" in ui_params else []
        }
        
        logger.info("Successfully merged UI parameters with session data")
        return merged_data
        
    except Exception as e:
        logger.error(f"Error merging UI parameters with session data: {str(e)}")
        raise