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
from typing import Dict, List, Tuple, Optional
import logging

from .config import MTS, T, HOL
from .time_features import ReferenceTimeFeatures

logger = logging.getLogger(__name__)


class ReferenceTimeProcessor:
    """
    REFERENCE TIME PROCESSOR using exact implementation from training_backend_test_2.py
    Integrates ReferenceTimeFeatures with data processing pipeline
    """
    
    def __init__(self):
        self.reference_time_features = ReferenceTimeFeatures()
    
    def add_reference_time_features(self, i_dat: Dict[str, pd.DataFrame], i_dat_inf: pd.DataFrame, utc_ref: datetime.datetime) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
        """
        Add reference time features to the data using exact reference implementation
        
        Args:
            i_dat: Dictionary of DataFrames (reference format)
            i_dat_inf: Information DataFrame (reference format)  
            utc_ref: Reference UTC timestamp
            
        Returns:
            Tuple of updated (i_dat, i_dat_inf) with time features
        """
        try:
            logger.info(f"Adding reference time features with UTC ref: {utc_ref}")
            
            # Generate time features using reference implementation
            time_array, feature_names = self.reference_time_features.generate_time_features(utc_ref, i_dat_inf)
            
            if len(time_array) > 0:
                logger.info(f"Generated {len(feature_names)} time features: {feature_names}")
                
                # Add time features to i_dat (follow reference pattern)
                for i, feature_name in enumerate(feature_names):
                    # Create DataFrame for each time feature (following reference structure)
                    feature_df = pd.DataFrame({
                        'UTC': pd.date_range(start=utc_ref, periods=len(time_array), freq='1min'),
                        feature_name: time_array[:, i]
                    })
                    
                    # Add to i_dat dictionary with proper naming
                    i_dat[f"Time_Feature_{feature_name}"] = feature_df
                    
                    logger.info(f"Added time feature {feature_name}: shape {feature_df.shape}")
                
                # Update i_dat_inf with time feature metadata
                for feature_name in feature_names:
                    feature_key = f"Time_Feature_{feature_name}"
                    
                    # Add metadata row for time feature (similar to reference)
                    if feature_key not in i_dat_inf.index:
                        i_dat_inf.loc[feature_key] = {
                            "utc_min": utc_ref,
                            "utc_max": utc_ref + pd.Timedelta(minutes=len(time_array)-1),
                            "delt": 1.0,  # 1 minute intervals
                            "ofst": 0.0,
                            "n_all": len(time_array),
                            "n_num": len(time_array),
                            "rate_num": 100.0,
                            "val_min": time_array[:, feature_names.index(feature_name)].min(),
                            "val_max": time_array[:, feature_names.index(feature_name)].max(),
                            "spec": "Time Feature",
                            "th_strt": getattr(T.Y, 'TH_STRT', -24),  # Use appropriate T class values
                            "th_end": getattr(T.Y, 'TH_END', 0),
                            "meth": "Reference Implementation",
                            "avg": False,
                            "delt_transf": 1.0,
                            "ofst_transf": 0.0,
                            "scal": True,
                            "scal_max": 1.0,
                            "scal_min": 0.0
                        }
                
                logger.info(f"Successfully integrated {len(feature_names)} reference time features")
            else:
                logger.warning("No time features generated - all T.*.IMP flags may be False")
            
            return i_dat, i_dat_inf
            
        except Exception as e:
            logger.error(f"Error adding reference time features: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return i_dat, i_dat_inf  # Return original data on error


class DataProcessor:
    """
    Main data processor class
    Contains functions extracted from training_backend_test_2.py
    """
    
    def __init__(self, config: MTS):
        self.config = config
        self.reference_time_processor = ReferenceTimeProcessor()
    
    def process_session_data_with_reference_format(self, i_dat: Dict[str, pd.DataFrame], i_dat_inf: pd.DataFrame, session_data: Dict) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
        """
        Process session data in reference format (i_dat, i_dat_inf) with time features
        
        Args:
            i_dat: Dictionary of DataFrames (reference format)
            i_dat_inf: Information DataFrame (reference format)
            session_data: Session configuration
            
        Returns:
            Tuple of processed (i_dat, i_dat_inf) with time features
        """
        try:
            logger.info("Processing session data with reference format and time features")
            
            # Determine UTC reference from the data
            utc_ref = self._determine_utc_reference(i_dat, i_dat_inf)
            logger.info(f"Using UTC reference: {utc_ref}")
            
            # Add reference time features if any T.*.IMP flags are enabled
            if any([T.Y.IMP, T.M.IMP, T.W.IMP, T.D.IMP, T.H.IMP]):
                logger.info("Adding reference time features (T.*.IMP flags enabled)")
                i_dat, i_dat_inf = self.reference_time_processor.add_reference_time_features(i_dat, i_dat_inf, utc_ref)
            else:
                logger.info("Skipping time features (all T.*.IMP flags disabled)")
            
            return i_dat, i_dat_inf
            
        except Exception as e:
            logger.error(f"Error processing session data with reference format: {str(e)}")
            raise
    
    def _determine_utc_reference(self, i_dat: Dict[str, pd.DataFrame], i_dat_inf: pd.DataFrame) -> datetime.datetime:
        """
        Determine the UTC reference timestamp from the loaded data
        
        Args:
            i_dat: Dictionary of DataFrames
            i_dat_inf: Information DataFrame
            
        Returns:
            UTC reference timestamp
        """
        try:
            # Get the first available UTC timestamp from the data
            for data_name, df in i_dat.items():
                if 'UTC' in df.columns and len(df) > 0:
                    # Get the first timestamp and ensure it's timezone-aware
                    first_utc = df['UTC'].iloc[0]
                    
                    if isinstance(first_utc, pd.Timestamp):
                        # Convert to datetime and ensure it's timezone-naive (as expected by reference)
                        utc_ref = first_utc.to_pydatetime()
                        if utc_ref.tzinfo is not None:
                            utc_ref = utc_ref.replace(tzinfo=None)
                        return utc_ref
                    elif isinstance(first_utc, str):
                        # Parse string timestamp
                        return datetime.datetime.strptime(first_utc, "%Y-%m-%d %H:%M:%S")
                    else:
                        return datetime.datetime.now()
            
            # Fallback to current time if no UTC data found
            logger.warning("No UTC timestamps found in data, using current time as reference")
            return datetime.datetime.now()
            
        except Exception as e:
            logger.error(f"Error determining UTC reference: {str(e)}")
            return datetime.datetime.now()
    
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
            # Load data from files
            input_data = self._load_data_files(input_files)
            output_data = self._load_data_files(output_files)
            
            # Process each dataset
            processed_input = {}
            processed_output = {}
            
            for file_path, df in input_data.items():
                processed_input[file_path] = self._process_dataframe(df, session_data)
            
            for file_path, df in output_data.items():
                processed_output[file_path] = self._process_dataframe(df, session_data)
            
            # Create datasets for training
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
                logger.info(f"Loaded {file_path}: {df.shape}")
            
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
            # Processing logic implemented from training_backend_test_2.py transf() function
            # This includes the transf() function and related preprocessing
            
            df = df.copy()
            
            # Convert UTC column to datetime
            if 'UTC' in df.columns:
                df['UTC'] = pd.to_datetime(df['UTC'])
            
            # Add time features if enabled
            if self.config.use_time_features:
                df = self.time_features.add_time_features(df)
            
            # Apply interpolation if enabled
            if self.config.interpolation:
                df = self._apply_interpolation(df)
            
            # Remove outliers if enabled
            if self.config.outlier_removal:
                df = self._remove_outliers(df)
            
            # Apply scaling if enabled
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
            # Interpolation logic implemented with pandas methods
            # This is placeholder implementation
            
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
            # Outlier removal implemented using IQR method
            # This is placeholder implementation using IQR method
            
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
            # Scaling logic implemented using StandardScaler
            # This should use MinMaxScaler as mentioned in the analysis
            
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
            # Dataset creation logic implemented for time series training
            # This is around lines 1055-1873 in the original file
            
            zeitschritte = session_data.get('zeitschritte', {})
            
            time_steps_in = int(zeitschritte.get('eingabe', 24))
            time_steps_out = int(zeitschritte.get('ausgabe', 1))
            
            # Create sequences for time series
            datasets = {}
            
            # Combine input and output data
            for input_file, input_df in input_data.items():
                for output_file, output_df in output_data.items():
                    # Create sequences
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
        IMPORTANT: Skips samples containing NaN values (matching reference implementation behavior)
        
        Args:
            input_df: Input DataFrame
            output_df: Output DataFrame
            time_steps_in: Number of input time steps
            time_steps_out: Number of output time steps
            
        Returns:
            Tuple of (X, y) arrays
        """
        try:
            # Sequence creation logic implemented for ML model training
            # Matches reference implementation NaN handling
            
            # Get numerical columns only
            input_numeric = input_df.select_dtypes(include=[np.number]).values
            output_numeric = output_df.select_dtypes(include=[np.number]).values
            
            X, y = [], []
            skipped_samples = 0
            
            # Create sequences - skip samples with NaN (matching reference implementation)
            for i in range(len(input_numeric) - time_steps_in - time_steps_out + 1):
                # Extract the candidate sequences
                X_sample = input_numeric[i:(i + time_steps_in)]
                y_sample = output_numeric[i + time_steps_in:(i + time_steps_in + time_steps_out)]
                
                # Check for NaN values in either X or y (matching reference at line 1332)
                # Reference skips entire sample if any NaN is found
                if np.isnan(X_sample).any() or np.isnan(y_sample).any():
                    skipped_samples += 1
                    continue  # Skip this sample entirely
                
                # Only append if no NaN values
                X.append(X_sample)
                y.append(y_sample)
            
            if skipped_samples > 0:
                logger.warning(f"Skipped {skipped_samples} samples containing NaN values (matching reference implementation)")
            
            if len(X) == 0:
                logger.error("No valid samples after removing NaN values")
                raise ValueError("No valid training data after removing NaN values")
            
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
                
                # Prevent division by zero when N = 1
                if N <= 1:
                    inf.loc[key, "delt_transf"] = 0
                else:
                    inf.loc[key, "delt_transf"] = \
                        (inf.loc[key, "th_end"] -
                         inf.loc[key, "th_strt"]) * 60 / (N - 1)
                
                # OFFSET CAN BE CALCULATED (check for division by zero first)
                if inf.loc[key, "delt_transf"] != 0 and \
                   round(60 / inf.loc[key, "delt_transf"]) == \
                    60 / inf.loc[key, "delt_transf"]:
                      
                    # Offset [min]
                    ofst_transf = OFST - (inf.loc[key, "th_strt"] -
                                        math.floor(inf.loc[key, "th_strt"])) * 60 + 60
                    
                    # Prevent infinite loop if delt_transf is 0 or negative
                    loop_counter = 0
                    max_iterations = 1000
                    while (ofst_transf - inf.loc[key, "delt_transf"] >= 0 and 
                           inf.loc[key, "delt_transf"] > 0 and 
                           loop_counter < max_iterations):
                       ofst_transf -= inf.loc[key, "delt_transf"]
                       loop_counter += 1
                    
                    inf.loc[key, "ofst_transf"] = ofst_transf
                        
                # OFFSET CANNOT BE CALCULATED
                else: 
                    inf.loc[key, "ofst_transf"] = str("var")
                    
            return inf
            
        except Exception as e:
            logger.error(f"Error transforming data: {str(e)}")
            raise
    
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


# Factory function to create data processor
def create_data_processor(config: MTS) -> DataProcessor:
    """
    Create and return a DataProcessor instance
    
    Args:
        config: MTS configuration object
        
    Returns:
        DataProcessor instance
    """
    return DataProcessor(config)