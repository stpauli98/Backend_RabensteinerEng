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

from domains.training.config import MTS, HOL
from domains.training.data.loader import utc_idx_pre, utc_idx_post, transf

logger = logging.getLogger(__name__)


class TimeFeatures:
    """
    Time features class (T class from training_original.py)
    Supports LT (local time), SPEC (Zeithorizont/Aktuelle Zeit), and custom time horizons
    Uses exact formulas from original: Year=sec/31557600, Month=sec/2629800, Week=sec/604800, Day=sec/86400
    """

    # Time period constants (seconds)
    YEAR_SECONDS = 31557600    # 60×60×24×365.25 seconds in a year
    MONTH_SECONDS = 2629800    # 60×60×24×365.25/12 seconds in a month
    WEEK_SECONDS = 604800      # 60×60×24×7 seconds in a week
    DAY_SECONDS = 86400        # 60×60×24 seconds in a day

    def __init__(self, timezone: str = 'UTC'):
        self.timezone = timezone
        self.tz = pytz.timezone(timezone)

    def add_time_features(self, df: pd.DataFrame, time_column: str = 'UTC',
                          time_info: Dict = None, category_data: Dict = None) -> pd.DataFrame:
        """
        Add time-based features to DataFrame with optional category configuration

        Args:
            df: Input DataFrame
            time_column: Name of the time column
            time_info: Time configuration (jahr, monat, woche, tag, feiertag flags)
            category_data: Additional category configuration (LT, SPEC, TH_STRT, TH_END, etc.)

        Returns:
            DataFrame with added time features
        """
        try:
            df = df.copy()

            if not pd.api.types.is_datetime64_any_dtype(df[time_column]):
                df[time_column] = pd.to_datetime(df[time_column])

            # Default time_info if not provided
            if time_info is None:
                time_info = {'jahr': True, 'monat': True, 'woche': True, 'tag': True, 'feiertag': False}

            # Get timestamps as seconds for sin/cos calculations
            timestamps = df[time_column]
            sec = timestamps.astype(np.int64) // 10**9  # Convert to Unix timestamp seconds

            # Add Year features (Y)
            if time_info.get('jahr', True):
                df['y_sin'] = np.sin(sec / self.YEAR_SECONDS * 2 * np.pi)
                df['y_cos'] = np.cos(sec / self.YEAR_SECONDS * 2 * np.pi)

            # Add Month features (M)
            if time_info.get('monat', True):
                df['m_sin'] = np.sin(sec / self.MONTH_SECONDS * 2 * np.pi)
                df['m_cos'] = np.cos(sec / self.MONTH_SECONDS * 2 * np.pi)

            # Add Week features (W)
            if time_info.get('woche', True):
                df['w_sin'] = np.sin(sec / self.WEEK_SECONDS * 2 * np.pi)
                df['w_cos'] = np.cos(sec / self.WEEK_SECONDS * 2 * np.pi)

            # Add Day features (D)
            if time_info.get('tag', True):
                df['d_sin'] = np.sin(sec / self.DAY_SECONDS * 2 * np.pi)
                df['d_cos'] = np.cos(sec / self.DAY_SECONDS * 2 * np.pi)

            # Add basic time columns for reference
            df['year'] = timestamps.dt.year
            df['month'] = timestamps.dt.month
            df['day'] = timestamps.dt.day
            df['hour'] = timestamps.dt.hour
            df['minute'] = timestamps.dt.minute
            df['weekday'] = timestamps.dt.weekday
            df['week'] = timestamps.dt.isocalendar().week

            # Add holiday features
            if time_info.get('feiertag', False):
                country = 'Österreich'  # Default country
                if category_data and 'feiertag' in category_data:
                    country = category_data['feiertag'].get('land', 'Österreich')
                df = self._add_holiday_features(df, time_column, country)

            return df

        except Exception as e:
            logger.error(f"Error adding time features: {str(e)}")
            raise

    def add_time_features_with_spec(self, utc_ref, n_timesteps: int, category_config: Dict,
                                     category_type: str = 'Y') -> Dict[str, np.ndarray]:
        """
        Add time features with SPEC (Zeithorizont vs Aktuelle Zeit) and LT logic
        Implementation from OriginalTraining/pipeline/time_features_processing.py

        Args:
            utc_ref: Reference UTC timestamp
            n_timesteps: Number of time steps (MTS.I_N)
            category_config: Config dict with keys: datenform, zeithorizontStart, zeithorizontEnd,
                           detaillierteBerechnung (LT), skalierung, skalierungMax, skalierungMin
            category_type: 'Y' (year), 'M' (month), 'W' (week), 'D' (day), 'H' (holiday)

        Returns:
            Dict with sin/cos arrays for the category
        """
        try:
            spec = category_config.get('datenform', 'Zeithorizont')
            lt = category_config.get('detaillierteBerechnung', False)  # LT flag
            th_strt = float(category_config.get('zeithorizontStart', -24))
            th_end = float(category_config.get('zeithorizontEnd', 0))

            # Calculate time step width for the category
            delt = (th_end - th_strt) * 60 / (n_timesteps - 1) if n_timesteps > 1 else 0

            # Generate timestamps based on SPEC
            if spec == "Zeithorizont":
                utc_th = self._generate_time_horizon(utc_ref, th_strt, th_end, delt, n_timesteps)
            else:  # "Aktuelle Zeit"
                utc_th = [utc_ref] * n_timesteps

            # Calculate sin/cos based on category type and LT flag
            if category_type == 'Y':
                return self._calc_year_features(utc_th, lt)
            elif category_type == 'M':
                return self._calc_month_features(utc_th, lt)
            elif category_type == 'W':
                return self._calc_week_features(utc_th, lt)
            elif category_type == 'D':
                return self._calc_day_features(utc_th, lt)
            elif category_type == 'H':
                country = category_config.get('land', 'Österreich')
                return self._calc_holiday_features(utc_th, lt, country)
            else:
                raise ValueError(f"Unknown category type: {category_type}")

        except Exception as e:
            logger.error(f"Error adding time features with SPEC: {str(e)}")
            raise

    def _generate_time_horizon(self, utc_ref, th_strt: float, th_end: float,
                               delt: float, n_timesteps: int) -> List:
        """Generate timestamp list for time horizon"""
        utc_th_strt = utc_ref + datetime.timedelta(hours=th_strt)
        utc_th_end = utc_ref + datetime.timedelta(hours=th_end)

        try:
            utc_th = pd.date_range(start=utc_th_strt, end=utc_th_end, freq=f'{delt}min').to_list()
        except Exception:
            delt_td = pd.to_timedelta(delt, unit='min')
            utc_th = []
            utc = utc_th_strt
            for _ in range(n_timesteps):
                utc_th.append(utc)
                utc += delt_td

        return utc_th

    def _calc_year_features(self, utc_th: List, lt: bool) -> Dict[str, np.ndarray]:
        """
        Calculate year sin/cos features
        Original: time_features_processing.py lines 44-125
        """
        if lt:
            # Convert to local time
            utc_th = [pytz.utc.localize(dt) if dt.tzinfo is None else dt for dt in utc_th]
            lt_th = [dt.astimezone(self.tz) for dt in utc_th]

            # Calculate seconds from day-of-year (local time)
            sec = np.array([(dt.timetuple().tm_yday - 1) * 86400 +
                            dt.hour * 3600 +
                            dt.minute * 60 +
                            dt.second for dt in lt_th])

            # Get years and check leap years
            years = np.array([dt.year for dt in lt_th])
            is_leap = np.vectorize(calendar.isleap)(years)
            sec_y = np.where(is_leap, 31622400, 31536000)  # 366 or 365 days in seconds

            return {
                'y_sin': np.sin(sec / sec_y * 2 * np.pi),
                'y_cos': np.cos(sec / sec_y * 2 * np.pi)
            }
        else:
            # UTC - use Unix timestamp
            sec = np.array([dt.timestamp() if hasattr(dt, 'timestamp') else pd.Timestamp(dt).timestamp()
                            for dt in utc_th])
            return {
                'y_sin': np.sin(sec / self.YEAR_SECONDS * 2 * np.pi),
                'y_cos': np.cos(sec / self.YEAR_SECONDS * 2 * np.pi)
            }

    def _calc_month_features(self, utc_th: List, lt: bool) -> Dict[str, np.ndarray]:
        """
        Calculate month sin/cos features
        Original: time_features_processing.py lines 127-208
        """
        if lt:
            # Convert to local time
            utc_th = [pytz.utc.localize(dt) if dt.tzinfo is None else dt for dt in utc_th]
            lt_th = [dt.astimezone(self.tz) for dt in utc_th]

            # Calculate seconds from start of month (local time)
            sec = np.array([(dt.day - 1) * 86400 +
                            dt.hour * 3600 +
                            dt.minute * 60 +
                            dt.second for dt in lt_th])

            # Get days in each month
            years = np.array([dt.year for dt in lt_th])
            months = np.array([dt.month for dt in lt_th])
            sec_m = 86400 * np.array([calendar.monthrange(y, m)[1] for y, m in zip(years, months)])

            return {
                'm_sin': np.sin(sec / sec_m * 2 * np.pi),
                'm_cos': np.cos(sec / sec_m * 2 * np.pi)
            }
        else:
            # UTC - use Unix timestamp
            sec = np.array([dt.timestamp() if hasattr(dt, 'timestamp') else pd.Timestamp(dt).timestamp()
                            for dt in utc_th])
            return {
                'm_sin': np.sin(sec / self.MONTH_SECONDS * 2 * np.pi),
                'm_cos': np.cos(sec / self.MONTH_SECONDS * 2 * np.pi)
            }

    def _calc_week_features(self, utc_th: List, lt: bool) -> Dict[str, np.ndarray]:
        """
        Calculate week sin/cos features
        Original: time_features_processing.py lines 210-278
        """
        if lt:
            # Convert to local time
            utc_th = [pytz.utc.localize(dt) if dt.tzinfo is None else dt for dt in utc_th]
            lt_th = [dt.astimezone(self.tz) for dt in utc_th]

            # Calculate seconds from start of week (Monday = 0)
            sec = np.array([dt.weekday() * 86400 +
                            dt.hour * 3600 +
                            dt.minute * 60 +
                            dt.second for dt in lt_th])

            return {
                'w_sin': np.sin(sec / self.WEEK_SECONDS * 2 * np.pi),
                'w_cos': np.cos(sec / self.WEEK_SECONDS * 2 * np.pi)
            }
        else:
            # UTC - use Unix timestamp
            sec = np.array([dt.timestamp() if hasattr(dt, 'timestamp') else pd.Timestamp(dt).timestamp()
                            for dt in utc_th])
            return {
                'w_sin': np.sin(sec / self.WEEK_SECONDS * 2 * np.pi),
                'w_cos': np.cos(sec / self.WEEK_SECONDS * 2 * np.pi)
            }

    def _calc_day_features(self, utc_th: List, lt: bool) -> Dict[str, np.ndarray]:
        """
        Calculate day sin/cos features
        Original: time_features_processing.py lines 280-347
        """
        if lt:
            # Convert to local time
            utc_th = [pytz.utc.localize(dt) if dt.tzinfo is None else dt for dt in utc_th]
            lt_th = [dt.astimezone(self.tz) for dt in utc_th]

            # Calculate seconds from start of day
            sec = np.array([dt.hour * 3600 +
                            dt.minute * 60 +
                            dt.second for dt in lt_th])

            return {
                'd_sin': np.sin(sec / self.DAY_SECONDS * 2 * np.pi),
                'd_cos': np.cos(sec / self.DAY_SECONDS * 2 * np.pi)
            }
        else:
            # UTC - use Unix timestamp
            sec = np.array([dt.timestamp() if hasattr(dt, 'timestamp') else pd.Timestamp(dt).timestamp()
                            for dt in utc_th])
            return {
                'd_sin': np.sin(sec / self.DAY_SECONDS * 2 * np.pi),
                'd_cos': np.cos(sec / self.DAY_SECONDS * 2 * np.pi)
            }

    def _calc_holiday_features(self, utc_th: List, lt: bool, country: str) -> Dict[str, np.ndarray]:
        """
        Calculate holiday features (0 or 1)
        Original: time_features_processing.py lines 350-409
        """
        if country not in HOL:
            return {'h': np.zeros(len(utc_th))}

        hol_dates_set = set(d.date() if hasattr(d, 'date') else d for d in HOL[country])

        if lt:
            # Convert to local time
            utc_th = [pytz.utc.localize(dt) if dt.tzinfo is None else dt for dt in utc_th]
            lt_th = [dt.astimezone(self.tz) for dt in utc_th]
            h_values = np.array([1 if dt.date() in hol_dates_set else 0 for dt in lt_th])
        else:
            h_values = np.array([1 if (dt.date() if hasattr(dt, 'date') else dt) in hol_dates_set else 0
                                 for dt in utc_th])

        return {'h': h_values}

    def generate_all_time_features(self, utc_ref, n_timesteps: int,
                                    time_info: Dict, category_data: Dict = None) -> Dict[str, np.ndarray]:
        """
        Generate all enabled time features for a given reference time
        This should be called during dataset creation loop for each utc_ref

        Args:
            utc_ref: Reference UTC timestamp
            n_timesteps: Number of time steps (MTS.I_N)
            time_info: Dict with keys: jahr, monat, woche, tag, feiertag, zeitzone
            category_data: Dict with category-specific configs (optional)

        Returns:
            Dict mapping feature names to numpy arrays
        """
        features = {}

        # Default category config
        default_config = {
            'datenform': 'Zeithorizont',
            'detaillierteBerechnung': False,
            'zeithorizontStart': -24,
            'zeithorizontEnd': 0
        }

        # Year features (Y)
        if time_info.get('jahr', False):
            config = default_config.copy()
            if category_data and 'jahr' in category_data:
                config.update(category_data['jahr'])
            result = self.add_time_features_with_spec(utc_ref, n_timesteps, config, 'Y')
            features.update(result)

        # Month features (M)
        if time_info.get('monat', False):
            config = default_config.copy()
            if category_data and 'monat' in category_data:
                config.update(category_data['monat'])
            result = self.add_time_features_with_spec(utc_ref, n_timesteps, config, 'M')
            features.update(result)

        # Week features (W)
        if time_info.get('woche', False):
            config = default_config.copy()
            if category_data and 'woche' in category_data:
                config.update(category_data['woche'])
            result = self.add_time_features_with_spec(utc_ref, n_timesteps, config, 'W')
            features.update(result)

        # Day features (D)
        if time_info.get('tag', False):
            config = default_config.copy()
            if category_data and 'tag' in category_data:
                config.update(category_data['tag'])
            result = self.add_time_features_with_spec(utc_ref, n_timesteps, config, 'D')
            features.update(result)

        # Holiday features (H)
        if time_info.get('feiertag', False):
            config = default_config.copy()
            config['land'] = 'Österreich'  # Default country
            if category_data and 'feiertag' in category_data:
                config.update(category_data['feiertag'])
            result = self.add_time_features_with_spec(utc_ref, n_timesteps, config, 'H')
            features.update(result)

        return features

    def _add_holiday_features(self, df: pd.DataFrame, time_column: str,
                               country: str = 'Österreich') -> pd.DataFrame:
        """
        Add holiday features based on HOL dictionary

        Args:
            df: Input DataFrame
            time_column: Name of time column
            country: Country for holiday lookup (Österreich, Deutschland, Schweiz)

        Returns:
            DataFrame with holiday feature added
        """
        try:
            df['h'] = 0  # Holiday flag column

            if country in HOL:
                holiday_dates = HOL[country]
                # HOL contains datetime objects, convert to date for comparison
                hol_dates_set = set(d.date() if hasattr(d, 'date') else d for d in holiday_dates)

                # Check each timestamp
                for idx in df.index:
                    ts = df.loc[idx, time_column]
                    if hasattr(ts, 'date'):
                        if ts.date() in hol_dates_set:
                            df.loc[idx, 'h'] = 1

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
    
    def process_session_data(self, session_data: Dict, input_files: List[str], output_files: List[str],
                              files_info: List[Dict] = None) -> Dict:
        """
        Main processing function that orchestrates all data processing steps

        Args:
            session_data: Session configuration data
            input_files: List of input file paths
            output_files: List of output file paths
            files_info: List of file metadata dicts from loader (contains scaling, zeithorizont, etc.)

        Returns:
            Dict containing processed data
        """
        try:
            input_data = self._load_data_files(input_files)
            output_data = self._load_data_files(output_files)

            # Build file_info lookup by file_name or storage_path
            file_info_map = {}
            if files_info:
                for fi in files_info:
                    # Map by file_name
                    if 'file_name' in fi:
                        file_info_map[fi['file_name']] = fi
                    # Also map by storage_path
                    if 'storage_path' in fi:
                        file_info_map[fi['storage_path']] = fi

            processed_input = {}
            processed_output = {}

            for file_path, df in input_data.items():
                # Find matching file_info
                file_info = self._find_file_info(file_path, file_info_map)
                processed_input[file_path] = self._process_dataframe(df, session_data, file_info)

            for file_path, df in output_data.items():
                # Find matching file_info
                file_info = self._find_file_info(file_path, file_info_map)
                processed_output[file_path] = self._process_dataframe(df, session_data, file_info)

            train_datasets = self._create_training_datasets(
                processed_input, processed_output, session_data, files_info
            )

            return {
                'input_data': processed_input,
                'output_data': processed_output,
                'train_datasets': train_datasets,
                'metadata': self._extract_metadata(processed_input, processed_output),
                'files_info': files_info
            }

        except Exception as e:
            logger.error(f"Error processing session data: {str(e)}")
            raise

    def _find_file_info(self, file_path: str, file_info_map: Dict) -> Optional[Dict]:
        """
        Find file_info for a given file path

        Args:
            file_path: Path to the file
            file_info_map: Map of file names/paths to file_info dicts

        Returns:
            Matching file_info dict or None
        """
        import os

        # Try exact match
        if file_path in file_info_map:
            return file_info_map[file_path]

        # Try matching by filename
        filename = os.path.basename(file_path)
        if filename in file_info_map:
            return file_info_map[filename]

        # Try matching by partial path
        for key, fi in file_info_map.items():
            if key in file_path or file_path.endswith(key):
                return fi

        return None
    
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
    
    def _process_dataframe(self, df: pd.DataFrame, session_data: Dict, file_info: Dict = None) -> pd.DataFrame:
        """
        Process a single DataFrame with file-specific configuration

        Args:
            df: Input DataFrame
            session_data: Session configuration
            file_info: File metadata containing scaling, zeithorizont, datenanpassung settings

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
                df = self._apply_scaling(df, file_info)

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
    
    def _apply_scaling(self, df: pd.DataFrame, file_info: Dict = None) -> pd.DataFrame:
        """
        Apply scaling to numerical columns with custom ranges from file_info

        Args:
            df: Input DataFrame
            file_info: File metadata containing scal, scal_min, scal_max

        Returns:
            DataFrame with scaled values
        """
        try:
            from sklearn.preprocessing import MinMaxScaler

            # Get scaling parameters from file_info or use defaults
            if file_info:
                scal_enabled = file_info.get('scal', True)
                scal_min = file_info.get('scal_min', 0.0)
                scal_max = file_info.get('scal_max', 1.0)
            else:
                scal_enabled = True
                scal_min = 0.0
                scal_max = 1.0

            if not scal_enabled:
                return df

            numeric_columns = df.select_dtypes(include=[np.number]).columns

            # Use custom feature_range from file configuration
            scaler = MinMaxScaler(feature_range=(scal_min, scal_max))
            df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

            # Store scaler for later inverse transform
            self._scalers = getattr(self, '_scalers', {})
            self._scalers['last'] = scaler

            return df

        except Exception as e:
            logger.error(f"Error applying scaling: {str(e)}")
            raise
    
    def _create_training_datasets(self, input_data: Dict, output_data: Dict, session_data: Dict,
                                    files_info: List[Dict] = None) -> Dict:
        """
        Create training datasets from processed data with zeithorizont and datenanpassung support

        Args:
            input_data: Processed input data
            output_data: Processed output data
            session_data: Session configuration
            files_info: List of file metadata for zeithorizont/datenanpassung

        Returns:
            Dict containing training datasets
        """
        try:
            zeitschritte = session_data.get('zeitschritte', {})

            time_steps_in = int(zeitschritte.get('eingabe', 24))
            time_steps_out = int(zeitschritte.get('ausgabe', 1))
            delt = float(zeitschritte.get('zeitschrittweite', 15))  # MTS.DELT
            ofst = float(zeitschritte.get('offset', 0))  # MTS.OFST

            # Build file_info lookup
            file_info_map = {}
            if files_info:
                for fi in files_info:
                    if 'file_name' in fi:
                        file_info_map[fi['file_name']] = fi
                    if 'storage_path' in fi:
                        file_info_map[fi['storage_path']] = fi

            # Get time_info and category_data from session_data
            time_info = session_data.get('time_info', {})
            category_data = time_info.get('category_data', {})

            datasets = {}

            for input_file, input_df in input_data.items():
                for output_file, output_df in output_data.items():
                    # Get file-specific configuration
                    input_file_info = self._find_file_info(input_file, file_info_map)
                    output_file_info = self._find_file_info(output_file, file_info_map)

                    X, y = self._create_sequences_with_zeithorizont(
                        input_df, output_df,
                        time_steps_in, time_steps_out,
                        delt, ofst,
                        input_file_info, output_file_info,
                        time_info, category_data
                    )

                    dataset_name = f"{input_file}_{output_file}"
                    datasets[dataset_name] = {
                        'X': X,
                        'y': y,
                        'time_steps_in': time_steps_in,
                        'time_steps_out': time_steps_out,
                        'delt': delt,
                        'ofst': ofst,
                        'input_file_info': input_file_info,
                        'output_file_info': output_file_info,
                        'time_info': time_info,
                        'category_data': category_data
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

    def _create_sequences_with_zeithorizont(self, input_df: pd.DataFrame, output_df: pd.DataFrame,
                                             time_steps_in: int, time_steps_out: int,
                                             delt: float, ofst: float,
                                             input_file_info: Dict = None,
                                             output_file_info: Dict = None,
                                             time_info: Dict = None,
                                             category_data: Dict = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences with zeithorizont, datenanpassung, and time features support
        Implementation based on OriginalTraining/pipeline/dataset_creation.py and time_features_processing.py

        Args:
            input_df: Input DataFrame with UTC column
            output_df: Output DataFrame with UTC column
            time_steps_in: Number of input time steps (MTS.I_N)
            time_steps_out: Number of output time steps (MTS.O_N)
            delt: Time step width in minutes (MTS.DELT)
            ofst: Offset in minutes (MTS.OFST)
            input_file_info: File-specific config (th_strt, th_end, meth, avg, scal, etc.)
            output_file_info: File-specific config for output
            time_info: Time information config (jahr, monat, woche, tag, feiertag, zeitzone)
            category_data: Category-specific configs for time features (LT, SPEC, TH_STRT, TH_END)

        Returns:
            Tuple of (X, y) arrays
        """
        try:
            # Default file info if not provided
            if input_file_info is None:
                input_file_info = {
                    'th_strt': -24.0,
                    'th_end': 0.0,
                    'meth': 'Lineare Interpolation',
                    'avg': False,
                    'delt_transf': None
                }

            if output_file_info is None:
                output_file_info = {
                    'th_strt': 0.0,
                    'th_end': 24.0,
                    'meth': 'Lineare Interpolation',
                    'avg': False,
                    'delt_transf': None
                }

            # Ensure UTC column is datetime
            if not pd.api.types.is_datetime64_any_dtype(input_df['UTC']):
                input_df = input_df.copy()
                input_df['UTC'] = pd.to_datetime(input_df['UTC'])
            if not pd.api.types.is_datetime64_any_dtype(output_df['UTC']):
                output_df = output_df.copy()
                output_df['UTC'] = pd.to_datetime(output_df['UTC'])

            # Calculate delt_transf for input and output if not set
            input_delt_transf = input_file_info.get('delt_transf')
            if input_delt_transf is None or input_delt_transf == 'var':
                th_strt = input_file_info.get('th_strt', -24.0)
                th_end = input_file_info.get('th_end', 0.0)
                input_delt_transf = (th_end - th_strt) * 60 / (time_steps_in - 1) if time_steps_in > 1 else delt

            output_delt_transf = output_file_info.get('delt_transf')
            if output_delt_transf is None or output_delt_transf == 'var':
                th_strt = output_file_info.get('th_strt', 0.0)
                th_end = output_file_info.get('th_end', 24.0)
                output_delt_transf = (th_end - th_strt) * 60 / (time_steps_out - 1) if time_steps_out > 1 else delt

            # Get data range from input/output
            utc_min_in = input_df['UTC'].min()
            utc_max_in = input_df['UTC'].max()
            utc_min_out = output_df['UTC'].min()
            utc_max_out = output_df['UTC'].max()

            # Determine valid reference time range
            # Input needs data from th_strt hours before reference
            # Output needs data until th_end hours after reference
            input_th_strt = input_file_info.get('th_strt', -24.0)
            input_th_end = input_file_info.get('th_end', 0.0)
            output_th_strt = output_file_info.get('th_strt', 0.0)
            output_th_end = output_file_info.get('th_end', 24.0)

            # Reference time must satisfy:
            # utc_ref + input_th_strt >= utc_min_in
            # utc_ref + output_th_end <= utc_max_out
            utc_ref_start = utc_min_in - datetime.timedelta(hours=input_th_strt)
            utc_ref_end = utc_max_out - datetime.timedelta(hours=output_th_end)

            # Create reference time points using MTS.DELT
            utc_ref_range = pd.date_range(
                start=utc_ref_start,
                end=utc_ref_end,
                freq=f'{delt}min'
            )

            X_list = []
            y_list = []

            input_method = input_file_info.get('meth', 'Lineare Interpolation')
            input_avg = input_file_info.get('avg', False)
            output_method = output_file_info.get('meth', 'Lineare Interpolation')
            output_avg = output_file_info.get('avg', False)

            for utc_ref in utc_ref_range:
                try:
                    # Create input time horizon timestamps
                    input_timestamps = self.create_time_horizon_timestamps(
                        utc_ref, input_th_strt, input_th_end, input_delt_transf, time_steps_in
                    )

                    # Create output time horizon timestamps
                    output_timestamps = self.create_time_horizon_timestamps(
                        utc_ref, output_th_strt, output_th_end, output_delt_transf, time_steps_out
                    )

                    # Calculate input time horizon boundaries
                    input_utc_th_strt = utc_ref + datetime.timedelta(hours=input_th_strt)
                    input_utc_th_end = utc_ref + datetime.timedelta(hours=input_th_end)

                    # Calculate output time horizon boundaries
                    output_utc_th_strt = utc_ref + datetime.timedelta(hours=output_th_strt)
                    output_utc_th_end = utc_ref + datetime.timedelta(hours=output_th_end)

                    # Apply datenanpassung for input
                    input_values = self.apply_datenanpassung(
                        input_df, input_timestamps, input_method,
                        input_utc_th_strt, input_utc_th_end, input_avg, time_steps_in
                    )

                    # Apply datenanpassung for output
                    output_values = self.apply_datenanpassung(
                        output_df, output_timestamps, output_method,
                        output_utc_th_strt, output_utc_th_end, output_avg, time_steps_out
                    )

                    # Add time features to input if time_info is provided
                    # (Original: time_features_processing.py adds y_sin, y_cos, etc. to df_int_i)
                    if time_info:
                        time_features_dict = self.time_features.generate_all_time_features(
                            utc_ref, time_steps_in, time_info, category_data
                        )
                        if time_features_dict:
                            # Convert input_values to array if it's a list
                            input_arr = np.array(input_values).reshape(time_steps_in, -1)

                            # Add each time feature column
                            for feature_name, feature_values in time_features_dict.items():
                                feature_col = feature_values.reshape(time_steps_in, 1)
                                input_arr = np.concatenate([input_arr, feature_col], axis=1)

                            input_values = input_arr.flatten() if input_arr.shape[1] == 1 else input_arr

                    X_list.append(input_values)
                    y_list.append(output_values)

                except Exception as e:
                    # Skip this reference time if data is not available
                    logger.debug(f"Skipping utc_ref {utc_ref}: {str(e)}")
                    continue

            if not X_list:
                raise ValueError("No valid training samples could be created")

            X = np.array(X_list)
            y = np.array(y_list)

            # Reshape if needed
            if len(X.shape) == 2:
                X = X.reshape(X.shape[0], X.shape[1], 1)
            if len(y.shape) == 2:
                y = y.reshape(y.shape[0], y.shape[1], 1)

            return X, y

        except Exception as e:
            logger.error(f"Error creating sequences with zeithorizont: {str(e)}")
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

    def mittelwertbildung(self, data: pd.DataFrame, utc_th_strt, utc_th_end, n_timesteps: int) -> List:
        """
        Mittelwertbildung (averaging) method for data adjustment
        Calculates mean over the time horizon and repeats for all timesteps

        Args:
            data: DataFrame with UTC and value columns
            utc_th_strt: Start time of time horizon
            utc_th_end: End time of time horizon
            n_timesteps: Number of timesteps to return

        Returns:
            List of averaged values (same value repeated n_timesteps times)
        """
        try:
            idx1 = utc_idx_post(data, utc_th_strt)
            idx2 = utc_idx_pre(data, utc_th_end)

            if idx1 is None or idx2 is None:
                raise ValueError(f"Time horizon {utc_th_strt} - {utc_th_end} outside data range")

            val = data.iloc[idx1:idx2, 1].mean()

            if math.isnan(float(val)):
                raise ValueError("Cannot calculate mean - no numeric data in time horizon")

            return [val] * n_timesteps

        except Exception as e:
            logger.error(f"Error in Mittelwertbildung: {str(e)}")
            raise

    def naechster_wert(self, data: pd.DataFrame, utc_timestamps: List) -> List:
        """
        Nächster Wert (nearest value) method for data adjustment
        Returns the nearest value for each timestamp

        Args:
            data: DataFrame with UTC and value columns
            utc_timestamps: List of timestamps to find nearest values for

        Returns:
            List of nearest values
        """
        try:
            val_list = []

            for utc in utc_timestamps:
                # Find nearest index
                idx = utc_idx_post(data, utc)

                if idx is None:
                    # Try previous index
                    idx = utc_idx_pre(data, utc)

                if idx is None:
                    raise ValueError(f"Timestamp {utc} outside data range")

                val = data.iloc[idx, 1]

                if math.isnan(float(val)):
                    raise ValueError(f"NaN value at timestamp {utc}")

                val_list.append(val)

            return val_list

        except Exception as e:
            logger.error(f"Error in Nächster Wert: {str(e)}")
            raise

    def apply_datenanpassung(self, data: pd.DataFrame, utc_timestamps: List,
                             method: str, utc_th_strt=None, utc_th_end=None,
                             avg: bool = False, n_timesteps: int = None) -> List:
        """
        Apply data adjustment method based on configuration
        Supports: Lineare Interpolation, Mittelwertbildung, Nächster Wert

        Args:
            data: DataFrame with UTC and value columns
            utc_timestamps: List of timestamps for interpolation/nearest
            method: Datenanpassung method name
            utc_th_strt: Start time for time horizon
            utc_th_end: End time for time horizon
            avg: If True and method is interpolation, calculate average
            n_timesteps: Number of timesteps (required for Mittelwertbildung)

        Returns:
            List of adjusted values
        """
        try:
            if method == "Lineare Interpolation":
                return self.linear_interpolation(data, utc_timestamps, avg, utc_th_strt, utc_th_end)

            elif method == "Mittelwertbildung":
                if utc_th_strt is None or utc_th_end is None:
                    raise ValueError("Mittelwertbildung requires utc_th_strt and utc_th_end")
                if n_timesteps is None:
                    n_timesteps = len(utc_timestamps)
                return self.mittelwertbildung(data, utc_th_strt, utc_th_end, n_timesteps)

            elif method == "Nächster Wert":
                return self.naechster_wert(data, utc_timestamps)

            else:
                logger.warning(f"Unknown datenanpassung method: {method}, using Lineare Interpolation")
                return self.linear_interpolation(data, utc_timestamps, avg, utc_th_strt, utc_th_end)

        except Exception as e:
            logger.error(f"Error applying datenanpassung ({method}): {str(e)}")
            raise

    def create_time_horizon_timestamps(self, utc_ref, th_strt: float, th_end: float,
                                        delt_transf: float, n_timesteps: int) -> List:
        """
        Create timestamps for time horizon based on zeithorizont parameters

        Args:
            utc_ref: Reference UTC timestamp
            th_strt: Time horizon start in hours (e.g., -24)
            th_end: Time horizon end in hours (e.g., 0)
            delt_transf: Time step width for transformation in minutes
            n_timesteps: Number of timesteps (MTS.I_N or MTS.O_N)

        Returns:
            List of UTC timestamps for the time horizon
        """
        try:
            utc_th_strt = utc_ref + datetime.timedelta(hours=th_strt)
            utc_th_end = utc_ref + datetime.timedelta(hours=th_end)

            # Try to create date_range
            try:
                utc_th = pd.date_range(
                    start=utc_th_strt,
                    end=utc_th_end,
                    freq=f'{delt_transf}min'
                ).to_list()
            except Exception:
                # Manual creation if pandas date_range fails
                delt = pd.to_timedelta(delt_transf, unit='min')
                utc_th = []
                utc = utc_th_strt
                for _ in range(n_timesteps):
                    utc_th.append(utc)
                    utc += delt

            return utc_th

        except Exception as e:
            logger.error(f"Error creating time horizon timestamps: {str(e)}")
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


def create_data_processor(config: MTS) -> DataProcessor:
    """
    Create and return a DataProcessor instance
    
    Args:
        config: MTS configuration object
        
    Returns:
        DataProcessor instance
    """
    return DataProcessor(config)
