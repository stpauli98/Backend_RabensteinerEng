"""
Data processor module for training system
Handles data transformation and preprocessing
Contains functions extracted from training_backend_test_2.py
"""

import pandas as pd
import numpy as np
import datetime
import math
import pytz
import calendar
import copy
from typing import Dict, List, Tuple, Optional
import logging

from .config import MTS, HOL
from .data_loader import utc_idx_pre, utc_idx_post, transf

logger = logging.getLogger(__name__)


class TimeFeatures:
    """
    Time features class (T class from training_backend_test_2.py)
    Extracted from around lines 798-955
    """
    
    def __init__(self, timezone: str = 'UTC'):
        self.timezone = timezone
        self.tz = pytz.timezone(timezone)
    
    def add_time_features(self, df: pd.DataFrame, time_column: str = 'UTC') -> pd.DataFrame:
        """
        Add time-based features to DataFrame
        
        Args:
            df: Input DataFrame
            time_column: Name of the time column
            
        Returns:
            DataFrame with added time features
        """
        try:
            
            df = df.copy()
            
            if not pd.api.types.is_datetime64_any_dtype(df[time_column]):
                df[time_column] = pd.to_datetime(df[time_column])
            
            df['year'] = df[time_column].dt.year
            df['month'] = df[time_column].dt.month
            df['day'] = df[time_column].dt.day
            df['hour'] = df[time_column].dt.hour
            df['minute'] = df[time_column].dt.minute
            df['weekday'] = df[time_column].dt.weekday
            df['week'] = df[time_column].dt.isocalendar().week
            
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
            df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
            df['weekday_sin'] = np.sin(2 * np.pi * df['weekday'] / 7)
            df['weekday_cos'] = np.cos(2 * np.pi * df['weekday'] / 7)
            
            df = self._add_holiday_features(df, time_column)
            
            return df
            
        except Exception as e:
            logger.error(f"Error adding time features: {str(e)}")
            raise
    
    def _add_holiday_features(self, df: pd.DataFrame, time_column: str) -> pd.DataFrame:
        """Add holiday features based on HOL dictionary"""
        try:
            
            df['is_holiday'] = False
            
            for country, holidays in HOL.items():
                for holiday_name, (month, day) in holidays.items():
                    if isinstance(month, int) and isinstance(day, int):
                        holiday_mask = (df[time_column].dt.month == month) & (df[time_column].dt.day == day)
                        df.loc[holiday_mask, 'is_holiday'] = True
            
            return df
            
        except Exception as e:
            logger.error(f"Error adding holiday features: {str(e)}")
            raise


class DataProcessor:
    """
    Main data processor class
    Contains functions extracted from training_backend_test_2.py
    """
    
    def __init__(self, config: MTS):
        self.config = config
        self.time_features = TimeFeatures(config.timezone)
    
    def process_session_data(self, session_data: Dict, input_files: List[str], output_files: List[str]) -> Dict:
        """
        Main processing function that orchestrates all data processing steps
        
        Args:
            session_data: Session configuration data
            input_files: List of input file paths
            output_files: List of output file paths
            
        Returns:
            Dict containing processed data
        """
        try:
            input_data = self._load_data_files(input_files)
            output_data = self._load_data_files(output_files)
            
            processed_input = {}
            processed_output = {}
            
            for file_path, df in input_data.items():
                processed_input[file_path] = self._process_dataframe(df, session_data)
            
            for file_path, df in output_data.items():
                processed_output[file_path] = self._process_dataframe(df, session_data)
            
            train_datasets = self._create_training_datasets(processed_input, processed_output, session_data)
            
            return {
                'input_data': processed_input,
                'output_data': processed_output,
                'train_datasets': train_datasets,
                'metadata': self._extract_metadata(processed_input, processed_output)
            }
            
        except Exception as e:
            logger.error(f"Error processing session data: {str(e)}")
            raise
    
    def _load_data_files(self, file_paths: List[str]) -> Dict[str, pd.DataFrame]:
        """
        Load data from multiple CSV files
        
        Args:
            file_paths: List of file paths
            
        Returns:
            Dict mapping file paths to DataFrames
        """
        try:
            data = {}
            
            for file_path in file_paths:
                df = pd.read_csv(file_path)
                data[file_path] = df
            
            return data
            
        except Exception as e:
            logger.error(f"Error loading data files: {str(e)}")
            raise
    
    def _process_dataframe(self, df: pd.DataFrame, session_data: Dict) -> pd.DataFrame:
        """
        Process a single DataFrame
        
        Args:
            df: Input DataFrame
            session_data: Session configuration
            
        Returns:
            Processed DataFrame
        """
        try:
            
            df = df.copy()
            
            if 'UTC' in df.columns:
                df['UTC'] = pd.to_datetime(df['UTC'])
            
            if self.config.use_time_features:
                df = self.time_features.add_time_features(df)
            
            if self.config.interpolation:
                df = self._apply_interpolation(df)
            
            if self.config.outlier_removal:
                df = self._remove_outliers(df)
            
            if self.config.scaling:
                df = self._apply_scaling(df)
            
            return df
            
        except Exception as e:
            logger.error(f"Error processing DataFrame: {str(e)}")
            raise
    
    def _apply_interpolation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply interpolation to missing values
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with interpolated values
        """
        try:
            
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            df[numeric_columns] = df[numeric_columns].interpolate(method='linear')
            
            return df
            
        except Exception as e:
            logger.error(f"Error applying interpolation: {str(e)}")
            raise
    
    def _remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove outliers from DataFrame
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with outliers removed
        """
        try:
            
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            
            for col in numeric_columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
            
            return df
            
        except Exception as e:
            logger.error(f"Error removing outliers: {str(e)}")
            raise
    
    def _apply_scaling(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply scaling to numerical columns
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with scaled values
        """
        try:
            
            from sklearn.preprocessing import MinMaxScaler
            
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            
            scaler = MinMaxScaler()
            df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
            
            return df
            
        except Exception as e:
            logger.error(f"Error applying scaling: {str(e)}")
            raise
    
    def _create_training_datasets(self, input_data: Dict, output_data: Dict, session_data: Dict) -> Dict:
        """
        Create training datasets from processed data
        
        Args:
            input_data: Processed input data
            output_data: Processed output data
            session_data: Session configuration
            
        Returns:
            Dict containing training datasets
        """
        try:
            
            zeitschritte = session_data.get('zeitschritte', {})
            
            time_steps_in = int(zeitschritte.get('eingabe', 24))
            time_steps_out = int(zeitschritte.get('ausgabe', 1))
            
            datasets = {}
            
            for input_file, input_df in input_data.items():
                for output_file, output_df in output_data.items():
                    X, y = self._create_sequences(input_df, output_df, time_steps_in, time_steps_out)
                    
                    dataset_name = f"{input_file}_{output_file}"
                    datasets[dataset_name] = {
                        'X': X,
                        'y': y,
                        'time_steps_in': time_steps_in,
                        'time_steps_out': time_steps_out
                    }
            
            return datasets
            
        except Exception as e:
            logger.error(f"Error creating training datasets: {str(e)}")
            raise
    
    def _create_sequences(self, input_df: pd.DataFrame, output_df: pd.DataFrame, 
                         time_steps_in: int, time_steps_out: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for time series training
        
        Args:
            input_df: Input DataFrame
            output_df: Output DataFrame
            time_steps_in: Number of input time steps
            time_steps_out: Number of output time steps
            
        Returns:
            Tuple of (X, y) arrays
        """
        try:
            
            input_numeric = input_df.select_dtypes(include=[np.number]).values
            output_numeric = output_df.select_dtypes(include=[np.number]).values
            
            X, y = [], []
            
            for i in range(len(input_numeric) - time_steps_in - time_steps_out + 1):
                X.append(input_numeric[i:(i + time_steps_in)])
                y.append(output_numeric[i + time_steps_in:(i + time_steps_in + time_steps_out)])
            
            return np.array(X), np.array(y)
            
        except Exception as e:
            logger.error(f"Error creating sequences: {str(e)}")
            raise
    
    def transform_data(self, inf: pd.DataFrame, N: int, OFST: float) -> pd.DataFrame:
        """
        Transform data for time steps and offset calculation (extracted from original transf() function)
        
        Args:
            inf: Information DataFrame
            N: Number of time steps
            OFST: Global offset value
            
        Returns:
            Updated information DataFrame
        """
        try:
            for i in range(len(inf)):
                key = inf.index[i]
                
                inf.loc[key, "delt_transf"] = \
                    (inf.loc[key, "th_end"] -
                     inf.loc[key, "th_strt"]) * 60 / (N - 1)
                
                if inf.loc[key, "delt_transf"] != 0 and \
                   round(60 / inf.loc[key, "delt_transf"]) == \
                    60 / inf.loc[key, "delt_transf"]:
                      
                    ofst_transf = OFST - (inf.loc[key, "th_strt"] -
                                        math.floor(inf.loc[key, "th_strt"])) * 60 + 60
                    
                    loop_counter = 0
                    max_iterations = 1000
                    while (ofst_transf - inf.loc[key, "delt_transf"] >= 0 and 
                           inf.loc[key, "delt_transf"] > 0 and 
                           loop_counter < max_iterations):
                       ofst_transf -= inf.loc[key, "delt_transf"]
                       loop_counter += 1
                    
                    inf.loc[key, "ofst_transf"] = ofst_transf
                        
                else: 
                    inf.loc[key, "ofst_transf"] = "var"
                    
            return inf
            
        except Exception as e:
            logger.error(f"Error transforming data: {str(e)}")
            raise
    
    def linear_interpolation(self, data: pd.DataFrame, utc_timestamps: List, 
                            avg: bool = False, utc_th_strt=None, utc_th_end=None) -> List:
        """
        Perform linear interpolation on data
        Exact implementation from training_original.py lines 1303-1344
        
        Args:
            data: DataFrame with UTC and value columns
            utc_timestamps: List of timestamps to interpolate at
            avg: If True, compute average instead of interpolation
            utc_th_strt: Start time for averaging
            utc_th_end: End time for averaging
            
        Returns:
            List of interpolated values
        """
        val_list = []
        
        if avg and utc_th_strt and utc_th_end:
            idx1 = utc_idx_post(data, utc_th_strt)
            idx2 = utc_idx_pre(data, utc_th_end)
            
            val = data.iloc[idx1:idx2, 1].mean()
            
            if math.isnan(float(val)):
                raise ValueError("Cannot calculate mean - no numeric data")
            
            return [val] * len(utc_timestamps)
        
        for utc in utc_timestamps:
            idx1 = utc_idx_pre(data, utc)
            idx2 = utc_idx_post(data, utc)
            
            if idx1 is None or idx2 is None:
                raise ValueError(f"Timestamp {utc} outside data range")
            
            if idx1 == idx2:
                val = data.iloc[idx1, 1]
            else:
                utc1 = data.iloc[idx1, 0]
                utc2 = data.iloc[idx2, 0]
                
                val1 = data.iloc[idx1, 1]
                val2 = data.iloc[idx2, 1]
                
                val = (utc - utc1) / (utc2 - utc1) * (val2 - val1) + val1
            
            if math.isnan(float(val)):
                raise ValueError(f"NaN value at timestamp {utc}")
            
            val_list.append(val)
        
        return val_list
    
    def _extract_metadata(self, input_data: Dict, output_data: Dict) -> Dict:
        """
        Extract metadata from processed data
        
        Args:
            input_data: Processed input data
            output_data: Processed output data
            
        Returns:
            Dict containing metadata
        """
        try:
            metadata = {
                'input_files': {},
                'output_files': {},
                'total_samples': 0,
                'feature_count': 0
            }
            
            for file_path, df in input_data.items():
                metadata['input_files'][file_path] = {
                    'shape': df.shape,
                    'columns': df.columns.tolist(),
                    'dtypes': df.dtypes.to_dict()
                }
            
            for file_path, df in output_data.items():
                metadata['output_files'][file_path] = {
                    'shape': df.shape,
                    'columns': df.columns.tolist(),
                    'dtypes': df.dtypes.to_dict()
                }
            
            return metadata
            
        except Exception as e:
            logger.error(f"Error extracting metadata: {str(e)}")
            raise


def create_data_processor(config: MTS) -> DataProcessor:
    """
    Create and return a DataProcessor instance
    
    Args:
        config: MTS configuration object
        
    Returns:
        DataProcessor instance
    """
    return DataProcessor(config)
