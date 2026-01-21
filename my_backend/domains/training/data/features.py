"""
Time-based feature extraction module
Extracted from training_backend_test_2.py for creating cyclical time features
"""

import datetime
import math
import pandas as pd
import numpy as np
import pytz
import calendar
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Time constants - MUST match original training.py exactly
YEAR_SECONDS = 31557600    # 60×60×24×365.25 seconds in a year
MONTH_SECONDS = 2629800    # 60×60×24×365.25/12 seconds in a month  
WEEK_SECONDS = 604800      # 60×60×24×7 seconds in a week
DAY_SECONDS = 86400        # 60×60×24 seconds in a day


class TimeFeatureExtractor:
    """
    Extract time-based cyclical features (Y, M, W, D, H) from timestamps
    Based on original training_backend_test_2.py implementation
    """
    
    def __init__(self, timezone: str = 'UTC'):
        """
        Initialize TimeFeatureExtractor
        
        Args:
            timezone: Timezone for local time calculations (default: 'UTC')
        """
        self.timezone = timezone
        
    def extract_yearly_features(self, timestamps: List[datetime.datetime], 
                               use_local_time: bool = False,
                               mode: str = "Zeithorizont") -> Dict[str, np.ndarray]:
        """
        Extract yearly cyclical features (sin/cos components)
        
        Args:
            timestamps: List of datetime objects
            use_local_time: Whether to use local time or UTC
            mode: "Zeithorizont" or "Aktuelle Zeit"
            
        Returns:
            Dict with y_sin and y_cos arrays
        """
        try:
            features = {}
            
            if mode == "Zeithorizont":
                if not use_local_time:
                    sec = pd.Series(timestamps).map(pd.Timestamp.timestamp)
                    
                    features["Y_sin"] = np.sin(sec / 31557600 * 2 * np.pi)
                    features["Y_cos"] = np.cos(sec / 31557600 * 2 * np.pi)
                    
                else:
                    utc_timestamps = [pytz.utc.localize(dt) if dt.tzinfo is None else dt for dt in timestamps]
                    local_timestamps = [dt.astimezone(pytz.timezone(self.timezone)) for dt in utc_timestamps]
                    
                    sec = np.array([
                        (dt.timetuple().tm_yday - 1) * 86400 +
                        dt.hour * 3600 +
                        dt.minute * 60 +
                        dt.second
                        for dt in local_timestamps
                    ])
                    
                    years = np.array([dt.year for dt in local_timestamps])
                    
                    is_leap = np.vectorize(calendar.isleap)(years)
                    
                    sec_y = np.where(is_leap, 31622400, 31536000)
                    
                    features["Y_sin"] = np.sin(sec / sec_y * 2 * np.pi)
                    features["Y_cos"] = np.cos(sec / sec_y * 2 * np.pi)
                    
            elif mode == "Aktuelle Zeit":
                if len(timestamps) > 0:
                    utc_ref = timestamps[0]
                    
                    if not use_local_time:
                        sec = utc_ref.timestamp()
                        features["Y_sin"] = np.sin(sec / 31557600 * 2 * np.pi)
                        features["Y_cos"] = np.cos(sec / 31557600 * 2 * np.pi)
                        
                    else:
                        if utc_ref.tzinfo is None:
                            utc_ref = pytz.utc.localize(utc_ref)
                        local_time = utc_ref.astimezone(pytz.timezone(self.timezone))
                        
                        sec = ((local_time.timetuple().tm_yday - 1) * 86400 + 
                              local_time.hour * 3600 + 
                              local_time.minute * 60 + 
                              local_time.second)
                        
                        if calendar.isleap(local_time.year):
                            sec_y = 31622400
                        else:
                            sec_y = 31536000
                            
                        features["Y_sin"] = np.sin(sec / sec_y * 2 * np.pi)
                        features["Y_cos"] = np.cos(sec / sec_y * 2 * np.pi)
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting yearly features: {str(e)}")
            return {"Y_sin": np.array([]), "Y_cos": np.array([])}
    
    def extract_monthly_features(self, timestamps: List[datetime.datetime],
                                use_local_time: bool = False,
                                mode: str = "Zeithorizont") -> Dict[str, np.ndarray]:
        """
        Extract monthly cyclical features (sin/cos components)
        
        MATCHES ORIGINAL: Uses constant MONTH_SECONDS (2629800) instead of dynamic month length
        
        Args:
            timestamps: List of datetime objects
            use_local_time: Whether to use local time or UTC
            mode: "Zeithorizont" or "Aktuelle Zeit"
            
        Returns:
            Dict with M_sin and M_cos arrays
        """
        try:
            features = {}
            
            if mode == "Zeithorizont":
                if not use_local_time:
                    # MATCHES ORIGINAL: Use Unix timestamp / constant MONTH_SECONDS
                    sec = pd.Series(timestamps).map(pd.Timestamp.timestamp)
                    
                    features["M_sin"] = np.sin(sec / MONTH_SECONDS * 2 * np.pi)
                    features["M_cos"] = np.cos(sec / MONTH_SECONDS * 2 * np.pi)

                else:
                    utc_timestamps = [pytz.utc.localize(dt) if dt.tzinfo is None else dt for dt in timestamps]
                    local_timestamps = [dt.astimezone(pytz.timezone(self.timezone)) for dt in utc_timestamps]

                    # MATCHES ORIGINAL: Use Unix timestamp of local time / constant
                    sec = np.array([dt.timestamp() for dt in local_timestamps])

                    features["M_sin"] = np.sin(sec / MONTH_SECONDS * 2 * np.pi)
                    features["M_cos"] = np.cos(sec / MONTH_SECONDS * 2 * np.pi)
                    
            elif mode == "Aktuelle Zeit":
                if len(timestamps) > 0:
                    utc_ref = timestamps[0]

                    if not use_local_time:
                        # MATCHES ORIGINAL: Use Unix timestamp / constant MONTH_SECONDS
                        sec = utc_ref.timestamp() if hasattr(utc_ref, 'timestamp') else pd.Timestamp(utc_ref).timestamp()
                        features["M_sin"] = np.sin(sec / MONTH_SECONDS * 2 * np.pi)
                        features["M_cos"] = np.cos(sec / MONTH_SECONDS * 2 * np.pi)
                    else:
                        if utc_ref.tzinfo is None:
                            utc_ref = pytz.utc.localize(utc_ref)
                        local_time = utc_ref.astimezone(pytz.timezone(self.timezone))

                        # MATCHES ORIGINAL: Use Unix timestamp of local time / constant
                        sec = local_time.timestamp()

                        features["M_sin"] = np.sin(sec / MONTH_SECONDS * 2 * np.pi)
                        features["M_cos"] = np.cos(sec / MONTH_SECONDS * 2 * np.pi)
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting monthly features: {str(e)}")
            return {"M_sin": np.array([]), "M_cos": np.array([])}
    
    def extract_weekly_features(self, timestamps: List[datetime.datetime],
                               use_local_time: bool = False) -> Dict[str, np.ndarray]:
        """
        Extract weekly cyclical features (sin/cos components)
        
        Args:
            timestamps: List of datetime objects
            use_local_time: Whether to use local time or UTC
            
        Returns:
            Dict with w_sin and w_cos arrays
        """
        try:
            features = {}
            
            if not use_local_time:
                sec = pd.Series(timestamps).map(pd.Timestamp.timestamp)
                
                features["W_sin"] = np.sin(sec / 604800 * 2 * np.pi)
                features["W_cos"] = np.cos(sec / 604800 * 2 * np.pi)
                
            else:
                utc_timestamps = [pytz.utc.localize(dt) if dt.tzinfo is None else dt for dt in timestamps]
                local_timestamps = [dt.astimezone(pytz.timezone(self.timezone)) for dt in utc_timestamps]
                
                sec = np.array([
                    dt.weekday() * 86400 +
                    dt.hour * 3600 +
                    dt.minute * 60 +
                    dt.second
                    for dt in local_timestamps
                ])
                
                features["W_sin"] = np.sin(sec / 604800 * 2 * np.pi)
                features["W_cos"] = np.cos(sec / 604800 * 2 * np.pi)
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting weekly features: {str(e)}")
            return {"W_sin": np.array([]), "W_cos": np.array([])}
    
    def extract_daily_features(self, timestamps: List[datetime.datetime],
                              use_local_time: bool = False) -> Dict[str, np.ndarray]:
        """
        Extract daily cyclical features (sin/cos components)
        
        Args:
            timestamps: List of datetime objects
            use_local_time: Whether to use local time or UTC
            
        Returns:
            Dict with d_sin and d_cos arrays
        """
        try:
            features = {}
            
            if not use_local_time:
                sec = pd.Series(timestamps).map(pd.Timestamp.timestamp)
                
                features["D_sin"] = np.sin((sec % 86400) / 86400 * 2 * np.pi)
                features["D_cos"] = np.cos((sec % 86400) / 86400 * 2 * np.pi)
                
            else:
                utc_timestamps = [pytz.utc.localize(dt) if dt.tzinfo is None else dt for dt in timestamps]
                local_timestamps = [dt.astimezone(pytz.timezone(self.timezone)) for dt in utc_timestamps]
                
                sec = np.array([
                    dt.hour * 3600 +
                    dt.minute * 60 +
                    dt.second
                    for dt in local_timestamps
                ])
                
                features["D_sin"] = np.sin(sec / 86400 * 2 * np.pi)
                features["D_cos"] = np.cos(sec / 86400 * 2 * np.pi)
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting daily features: {str(e)}")
            return {"D_sin": np.array([]), "D_cos": np.array([])}
    
    def extract_hourly_features(self, timestamps: List[datetime.datetime],
                               use_local_time: bool = False) -> Dict[str, np.ndarray]:
        """
        Extract hourly cyclical features (sin/cos components)
        
        Args:
            timestamps: List of datetime objects
            use_local_time: Whether to use local time or UTC
            
        Returns:
            Dict with h_sin and h_cos arrays
        """
        try:
            features = {}
            
            if not use_local_time:
                sec = pd.Series(timestamps).map(pd.Timestamp.timestamp)
                
                features["H_sin"] = np.sin((sec % 3600) / 3600 * 2 * np.pi)
                features["H_cos"] = np.cos((sec % 3600) / 3600 * 2 * np.pi)
                
            else:
                utc_timestamps = [pytz.utc.localize(dt) if dt.tzinfo is None else dt for dt in timestamps]
                local_timestamps = [dt.astimezone(pytz.timezone(self.timezone)) for dt in utc_timestamps]
                
                sec = np.array([
                    dt.minute * 60 + dt.second
                    for dt in local_timestamps
                ])
                
                features["H_sin"] = np.sin(sec / 3600 * 2 * np.pi)
                features["H_cos"] = np.cos(sec / 3600 * 2 * np.pi)
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting hourly features: {str(e)}")
            return {"H_sin": np.array([]), "H_cos": np.array([])}
    
    def extract_all_time_features(self, timestamps: List[datetime.datetime],
                                 features_config: Dict[str, bool] = None,
                                 use_local_time: bool = False,
                                 mode: str = "Zeithorizont") -> Dict[str, np.ndarray]:
        """
        Extract all time-based features based on configuration
        
        Args:
            timestamps: List of datetime objects
            features_config: Dict specifying which features to extract
                           e.g., {'yearly': True, 'monthly': True, 'weekly': False, ...}
            use_local_time: Whether to use local time or UTC
            mode: "Zeithorizont" or "Aktuelle Zeit"
            
        Returns:
            Dict with all requested time features
        """
        if features_config is None:
            features_config = {
                'yearly': True,
                'monthly': True,
                'weekly': True,
                'daily': True,
                'hourly': False
            }
        
        all_features = {}
        
        try:
            if features_config.get('yearly', False):
                yearly_features = self.extract_yearly_features(timestamps, use_local_time, mode)
                all_features.update(yearly_features)
            
            if features_config.get('monthly', False):
                monthly_features = self.extract_monthly_features(timestamps, use_local_time, mode)
                all_features.update(monthly_features)
            
            if features_config.get('weekly', False):
                weekly_features = self.extract_weekly_features(timestamps, use_local_time)
                all_features.update(weekly_features)
            
            if features_config.get('daily', False):
                daily_features = self.extract_daily_features(timestamps, use_local_time)
                all_features.update(daily_features)
            
            if features_config.get('hourly', False):
                hourly_features = self.extract_hourly_features(timestamps, use_local_time)
                all_features.update(hourly_features)
            
            return all_features
            
        except Exception as e:
            logger.error(f"Error extracting all time features: {str(e)}")
            return {}


def extract_time_features_from_dataframe(df: pd.DataFrame, 
                                        timestamp_column: str = 'UTC',
                                        features_config: Dict[str, bool] = None,
                                        timezone: str = 'UTC') -> pd.DataFrame:
    """
    Convenience function to extract time features from a DataFrame
    
    Args:
        df: DataFrame with timestamp column
        timestamp_column: Name of the timestamp column
        features_config: Configuration for which features to extract
        timezone: Timezone for local time calculations
        
    Returns:
        DataFrame with added time feature columns
    """
    try:
        if timestamp_column in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df[timestamp_column]):
                df[timestamp_column] = pd.to_datetime(df[timestamp_column])
            
            timestamps = df[timestamp_column].tolist()
            
            extractor = TimeFeatureExtractor(timezone)
            time_features = extractor.extract_all_time_features(
                timestamps, features_config, use_local_time=(timezone != 'UTC')
            )
            
            result_df = df.copy()
            for feature_name, feature_values in time_features.items():
                if len(feature_values) == len(df):
                    result_df[feature_name] = feature_values
                else:
                    logger.warning(f"Feature {feature_name} length mismatch: {len(feature_values)} vs {len(df)}")
            
            return result_df
        else:
            logger.error(f"Timestamp column '{timestamp_column}' not found in DataFrame")
            return df
            
    except Exception as e:
        logger.error(f"Error extracting time features from DataFrame: {str(e)}")
        return df


def detect_holidays(timestamps: List[datetime.datetime], 
                   country_code: str = 'DE') -> List[bool]:
    """
    Detect holidays in timestamp list (simplified implementation)
    
    Args:
        timestamps: List of datetime objects
        country_code: Country code for holiday detection
        
    Returns:
        List of boolean values indicating holidays
    """
    try:
        
        holidays = []
        for timestamp in timestamps:
            is_holiday = (
                (timestamp.month == 1 and timestamp.day == 1) or
                (timestamp.month == 12 and timestamp.day == 25) or
                (timestamp.month == 12 and timestamp.day == 26)
            )
            holidays.append(is_holiday)
        
        return holidays
        
    except Exception as e:
        logger.error(f"Error detecting holidays: {str(e)}")
        return [False] * len(timestamps)
