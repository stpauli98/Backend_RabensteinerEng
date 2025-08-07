"""
Time-based feature extraction module
EXACT IMPLEMENTATION from training_backend_test_2.py for creating cyclical time features
Updated to use T class configuration and exact reference logic
"""

import datetime
import math
import pandas as pd
import numpy as np
import pytz
import calendar
import logging
from typing import Dict, List, Optional, Tuple

from .config import T, HOL

logger = logging.getLogger(__name__)


class ReferenceTimeFeatures:
    """
    EXACT IMPLEMENTATION of time feature generation from training_backend_test_2.py
    Uses T class configuration to generate time features identical to reference
    """
    
    def __init__(self):
        """Initialize with T class configuration"""
        self.timezone = T.TZ
        self.holiday_country = T.H.CNTRY if hasattr(T.H, 'CNTRY') else "Österreich"
        
    def generate_time_features(self, utc_ref: datetime.datetime, i_dat_inf: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """
        Generate time features exactly as in reference implementation
        
        Args:
            utc_ref: Reference UTC timestamp
            i_dat_inf: Information DataFrame with metadata
            
        Returns:
            Tuple of (combined_time_array, feature_names)
        """
        try:
            time_arrays = []
            feature_names = []
            
            logger.info(f"Generating time features for UTC reference: {utc_ref}")
            
            # JAHRESZEITLICHE SINUS-/COSINUS-KOMPONENTE (Yearly)
            if T.Y.IMP:
                logger.info("Generating yearly sine/cosine components")
                y_sin, y_cos = self._generate_yearly_features(utc_ref, i_dat_inf)
                time_arrays.extend([y_sin, y_cos])
                feature_names.extend(["Y_sin", "Y_cos"])
                
            # MONATLICHE SINUS-/COSINUS-KOMPONENTE (Monthly)
            if T.M.IMP:
                logger.info("Generating monthly sine/cosine components")
                m_sin, m_cos = self._generate_monthly_features(utc_ref, i_dat_inf)
                time_arrays.extend([m_sin, m_cos])
                feature_names.extend(["M_sin", "M_cos"])
                
            # WÖCHENTLICHE SINUS-/COSINUS-KOMPONENTE (Weekly)
            if T.W.IMP:
                logger.info("Generating weekly sine/cosine components")
                w_sin, w_cos = self._generate_weekly_features(utc_ref, i_dat_inf)
                time_arrays.extend([w_sin, w_cos])
                feature_names.extend(["W_sin", "W_cos"])
                
            # TÄGLICHE SINUS-/COSINUS-KOMPONENTE (Daily)
            if T.D.IMP:
                logger.info("Generating daily sine/cosine components")
                d_sin, d_cos = self._generate_daily_features(utc_ref, i_dat_inf)
                time_arrays.extend([d_sin, d_cos])
                feature_names.extend(["D_sin", "D_cos"])
                
            # FEIERTAGE (Holidays)
            if T.H.IMP:
                logger.info("Generating holiday features")
                holiday_array = self._generate_holiday_features(utc_ref, i_dat_inf)
                time_arrays.append(holiday_array)
                feature_names.append("Holiday")
            
            # Combine all time arrays
            if time_arrays:
                # Filter out empty arrays and ensure consistent shapes
                valid_arrays = []
                valid_names = []
                
                for i, arr in enumerate(time_arrays):
                    if len(arr) > 0:
                        valid_arrays.append(arr)
                        valid_names.append(feature_names[i])
                
                if valid_arrays:
                    # Ensure all arrays have the same first dimension
                    min_length = min(len(arr) for arr in valid_arrays)
                    consistent_arrays = [arr[:min_length] for arr in valid_arrays]
                    
                    # Stack arrays horizontally to create combined array
                    combined_array = np.column_stack(consistent_arrays)
                    logger.info(f"Generated time features: {combined_array.shape} with features {valid_names}")
                    return combined_array, valid_names
                else:
                    logger.warning("No valid time features generated (all arrays empty)")
                    return np.array([]), []
            else:
                logger.warning("No time features were generated (all T.*.IMP flags are False)")
                return np.array([]), []
                
        except Exception as e:
            logger.error(f"Error generating time features: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return np.array([]), []
            
    def _generate_yearly_features(self, utc_ref: datetime.datetime, i_dat_inf: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """EXACT COPY of yearly feature logic from reference lines 1374-1456"""
        try:
            # ZEITHORIZONT (Time Horizon)
            if T.Y.SPEC == "Zeithorizont":
                
                # ZEITGRENZEN (Time boundaries)
                utc_th_strt = utc_ref + datetime.timedelta(hours=T.Y.TH_STRT)  
                utc_th_end = utc_ref + datetime.timedelta(hours=T.Y.TH_END)
                
                # ZEITSTEMPEL (Timestamps)
                try:
                    utc_th = pd.date_range(start=utc_th_strt, end=utc_th_end, freq=f'{T.Y.DELT}min')
                    
                    if T.Y.LT == False:
                        # UTC REFERENCE
                        sec = pd.Series(utc_th).map(pd.Timestamp.timestamp)
                        
                        # 31557600 = 60×60×24×365.25 (seconds in average year)
                        y_sin = np.sin(sec / 31557600 * 2 * np.pi)
                        y_cos = np.cos(sec / 31557600 * 2 * np.pi)
                        
                    else:
                        # LOCAL TIME REFERENCE (LOCAL TIMEZONE)
                        tz = pytz.timezone(self.timezone)
                        utc_th_loc = [tz.normalize(tz.localize(dt.to_pydatetime().replace(tzinfo=None))) for dt in utc_th]
                        
                        # Generate seconds timestamp within year
                        sec = np.array([
                            (dt.timetuple().tm_yday - 1) * 86400 +
                            dt.hour * 3600 +
                            dt.minute * 60 +
                            dt.second
                            for dt in utc_th_loc
                        ])
                        
                        # Years as array
                        years = np.array([dt.year for dt in utc_th_loc])  
                        
                        # Vectorized leap year check
                        is_leap = np.vectorize(calendar.isleap)(years)
                        
                        # Number of seconds in the year
                        sec_y = np.where(is_leap, 31622400, 31536000)  # 366 or 365 days
                        
                        y_sin = np.sin(sec / sec_y * 2 * np.pi)
                        y_cos = np.cos(sec / sec_y * 2 * np.pi)
                    
                    return y_sin.values if hasattr(y_sin, 'values') else y_sin, y_cos.values if hasattr(y_cos, 'values') else y_cos
                    
                except Exception as date_error:
                    logger.error(f"Error generating yearly date range: {str(date_error)}")
                    return np.array([]), np.array([])
                    
            elif T.Y.SPEC == "Aktuelle Zeit":
                # CURRENT TIME MODE
                if T.Y.LT == False:
                    # UTC REFERENCE
                    sec = utc_ref.timestamp()
                    y_sin = np.sin(sec / 31557600 * 2 * np.pi)
                    y_cos = np.cos(sec / 31557600 * 2 * np.pi)
                else:
                    # LOCAL TIME REFERENCE  
                    tz = pytz.timezone(self.timezone)
                    utc_ref_loc = tz.normalize(tz.localize(utc_ref.replace(tzinfo=None)))
                    
                    sec = ((utc_ref_loc.timetuple().tm_yday - 1) * 86400 + 
                          utc_ref_loc.hour * 3600 + 
                          utc_ref_loc.minute * 60 + 
                          utc_ref_loc.second)
                    
                    # Number of seconds in the year
                    if calendar.isleap(utc_ref_loc.year):
                        sec_y = 31622400  # 366 days
                    else:
                        sec_y = 31536000  # 365 days
                        
                    y_sin = np.sin(sec / sec_y * 2 * np.pi)
                    y_cos = np.cos(sec / sec_y * 2 * np.pi)
                
                return np.array([y_sin]), np.array([y_cos])
                
        except Exception as e:
            logger.error(f"Error generating yearly features: {str(e)}")
            return np.array([]), np.array([])
    
    def _generate_monthly_features(self, utc_ref: datetime.datetime, i_dat_inf: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """EXACT COPY of monthly feature logic from reference lines 1460-1539"""
        try:
            # ZEITHORIZONT (Time Horizon)
            if T.M.SPEC == "Zeithorizont":
                
                # ZEITGRENZEN (Time boundaries)
                utc_th_strt = utc_ref + datetime.timedelta(hours=T.M.TH_STRT)
                utc_th_end = utc_ref + datetime.timedelta(hours=T.M.TH_END)
                
                # ZEITSTEMPEL (Timestamps)
                try:
                    utc_th = pd.date_range(start=utc_th_strt, end=utc_th_end, freq=f'{T.M.DELT}min')
                    
                    if T.M.LT == False:
                        # UTC REFERENCE
                        sec = pd.Series(utc_th).map(pd.Timestamp.timestamp)
                        
                        # 2629800 = 60×60×24×365.25/12 (seconds in average month)
                        m_sin = np.sin(sec / 2629800 * 2 * np.pi)
                        m_cos = np.cos(sec / 2629800 * 2 * np.pi)
                        
                    else:
                        # LOCAL TIME REFERENCE
                        tz = pytz.timezone(self.timezone)
                        utc_th_loc = [tz.normalize(tz.localize(dt.to_pydatetime().replace(tzinfo=None))) for dt in utc_th]
                        
                        sec = np.array([
                            (dt.timetuple().tm_yday - 1) * 86400 +
                            dt.hour * 3600 +
                            dt.minute * 60 +
                            dt.second
                            for dt in utc_th_loc
                        ])
                        
                        m_sin = np.sin(sec / 2629800 * 2 * np.pi)
                        m_cos = np.cos(sec / 2629800 * 2 * np.pi)
                    
                    return m_sin.values if hasattr(m_sin, 'values') else m_sin, m_cos.values if hasattr(m_cos, 'values') else m_cos
                    
                except Exception as date_error:
                    logger.error(f"Error generating monthly date range: {str(date_error)}")
                    return np.array([]), np.array([])
                    
            elif T.M.SPEC == "Aktuelle Zeit":
                # CURRENT TIME MODE
                if T.M.LT == False:
                    # UTC REFERENCE
                    sec = utc_ref.timestamp()
                    m_sin = np.sin(sec / 2629800 * 2 * np.pi)
                    m_cos = np.cos(sec / 2629800 * 2 * np.pi)
                else:
                    # LOCAL TIME REFERENCE
                    tz = pytz.timezone(self.timezone)
                    utc_ref_loc = tz.normalize(tz.localize(utc_ref.replace(tzinfo=None)))
                    
                    sec = ((utc_ref_loc.timetuple().tm_yday - 1) * 86400 + 
                          utc_ref_loc.hour * 3600 + 
                          utc_ref_loc.minute * 60 + 
                          utc_ref_loc.second)
                    
                    m_sin = np.sin(sec / 2629800 * 2 * np.pi)
                    m_cos = np.cos(sec / 2629800 * 2 * np.pi)
                
                return np.array([m_sin]), np.array([m_cos])
                
        except Exception as e:
            logger.error(f"Error generating monthly features: {str(e)}")
            return np.array([]), np.array([])
    
    def _generate_weekly_features(self, utc_ref: datetime.datetime, i_dat_inf: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """EXACT COPY of weekly feature logic from reference lines 1543-1609"""
        try:
            # ZEITHORIZONT (Time Horizon)
            if T.W.SPEC == "Zeithorizont":
                
                # ZEITGRENZEN (Time boundaries)
                utc_th_strt = utc_ref + datetime.timedelta(hours=T.W.TH_STRT)
                utc_th_end = utc_ref + datetime.timedelta(hours=T.W.TH_END)
                
                # ZEITSTEMPEL (Timestamps)
                try:
                    utc_th = pd.date_range(start=utc_th_strt, end=utc_th_end, freq=f'{T.W.DELT}min')
                    
                    if T.W.LT == False:
                        # UTC REFERENCE
                        sec = pd.Series(utc_th).map(pd.Timestamp.timestamp)
                        
                        # 604800 = 60×60×24×7 (seconds in week)
                        w_sin = np.sin(sec / 604800 * 2 * np.pi)
                        w_cos = np.cos(sec / 604800 * 2 * np.pi)
                        
                    else:
                        # LOCAL TIME REFERENCE
                        tz = pytz.timezone(self.timezone)
                        utc_th_loc = [tz.normalize(tz.localize(dt.to_pydatetime().replace(tzinfo=None))) for dt in utc_th]
                        
                        sec = np.array([
                            dt.weekday() * 86400 +  # Days from Monday (0)
                            dt.hour * 3600 +
                            dt.minute * 60 +
                            dt.second
                            for dt in utc_th_loc
                        ])
                        
                        w_sin = np.sin(sec / 604800 * 2 * np.pi)
                        w_cos = np.cos(sec / 604800 * 2 * np.pi)
                    
                    return w_sin.values if hasattr(w_sin, 'values') else w_sin, w_cos.values if hasattr(w_cos, 'values') else w_cos
                    
                except Exception as date_error:
                    logger.error(f"Error generating weekly date range: {str(date_error)}")
                    return np.array([]), np.array([])
                    
            elif T.W.SPEC == "Aktuelle Zeit":
                # CURRENT TIME MODE
                if T.W.LT == False:
                    # UTC REFERENCE
                    sec = utc_ref.timestamp()
                    w_sin = np.sin(sec / 604800 * 2 * np.pi)
                    w_cos = np.cos(sec / 604800 * 2 * np.pi)
                else:
                    # LOCAL TIME REFERENCE
                    tz = pytz.timezone(self.timezone)
                    utc_ref_loc = tz.normalize(tz.localize(utc_ref.replace(tzinfo=None)))
                    
                    sec = (utc_ref_loc.weekday() * 86400 + 
                          utc_ref_loc.hour * 3600 + 
                          utc_ref_loc.minute * 60 + 
                          utc_ref_loc.second)
                    
                    w_sin = np.sin(sec / 604800 * 2 * np.pi)
                    w_cos = np.cos(sec / 604800 * 2 * np.pi)
                
                return np.array([w_sin]), np.array([w_cos])
                
        except Exception as e:
            logger.error(f"Error generating weekly features: {str(e)}")
            return np.array([]), np.array([])
    
    def _generate_daily_features(self, utc_ref: datetime.datetime, i_dat_inf: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """EXACT COPY of daily feature logic from reference lines 1613-1678"""
        try:
            # ZEITHORIZONT (Time Horizon)
            if T.D.SPEC == "Zeithorizont":
                
                # ZEITGRENZEN (Time boundaries)
                utc_th_strt = utc_ref + datetime.timedelta(hours=T.D.TH_STRT)
                utc_th_end = utc_ref + datetime.timedelta(hours=T.D.TH_END)
                
                # ZEITSTEMPEL (Timestamps)
                try:
                    utc_th = pd.date_range(start=utc_th_strt, end=utc_th_end, freq=f'{T.D.DELT}min')
                    
                    if T.D.LT == False:
                        # UTC REFERENCE
                        sec = pd.Series(utc_th).map(pd.Timestamp.timestamp)
                        
                        # 86400 = 60×60×24 (seconds in day)
                        d_sin = np.sin((sec % 86400) / 86400 * 2 * np.pi)
                        d_cos = np.cos((sec % 86400) / 86400 * 2 * np.pi)
                        
                    else:
                        # LOCAL TIME REFERENCE
                        tz = pytz.timezone(self.timezone)
                        utc_th_loc = [tz.normalize(tz.localize(dt.to_pydatetime().replace(tzinfo=None))) for dt in utc_th]
                        
                        sec = np.array([
                            dt.hour * 3600 +
                            dt.minute * 60 +
                            dt.second
                            for dt in utc_th_loc
                        ])
                        
                        d_sin = np.sin(sec / 86400 * 2 * np.pi)
                        d_cos = np.cos(sec / 86400 * 2 * np.pi)
                    
                    return d_sin.values if hasattr(d_sin, 'values') else d_sin, d_cos.values if hasattr(d_cos, 'values') else d_cos
                    
                except Exception as date_error:
                    logger.error(f"Error generating daily date range: {str(date_error)}")
                    return np.array([]), np.array([])
                    
            elif T.D.SPEC == "Aktuelle Zeit":
                # CURRENT TIME MODE
                if T.D.LT == False:
                    # UTC REFERENCE
                    sec = utc_ref.timestamp()
                    d_sin = np.sin((sec % 86400) / 86400 * 2 * np.pi)
                    d_cos = np.cos((sec % 86400) / 86400 * 2 * np.pi)
                else:
                    # LOCAL TIME REFERENCE
                    tz = pytz.timezone(self.timezone)
                    utc_ref_loc = tz.normalize(tz.localize(utc_ref.replace(tzinfo=None)))
                    
                    sec = (utc_ref_loc.hour * 3600 + 
                          utc_ref_loc.minute * 60 + 
                          utc_ref_loc.second)
                    
                    d_sin = np.sin(sec / 86400 * 2 * np.pi)
                    d_cos = np.cos(sec / 86400 * 2 * np.pi)
                
                return np.array([d_sin]), np.array([d_cos])
                
        except Exception as e:
            logger.error(f"Error generating daily features: {str(e)}")
            return np.array([]), np.array([])
    
    def _generate_holiday_features(self, utc_ref: datetime.datetime, i_dat_inf: pd.DataFrame) -> np.ndarray:
        """EXACT COPY of holiday feature logic from reference lines 1682-1764"""
        try:
            # Set with date objects of holidays (which are not Sundays)
            hol_d = set(d.date() for d in HOL[self.holiday_country])
            
            # ZEITHORIZONT (Time Horizon)
            if T.H.SPEC == "Zeithorizont":
                
                # ZEITGRENZEN (Time boundaries)
                utc_th_strt = utc_ref + datetime.timedelta(hours=T.H.TH_STRT)
                utc_th_end = utc_ref + datetime.timedelta(hours=T.H.TH_END)
                
                # ZEITSTEMPEL (Timestamps)
                try:
                    utc_th = pd.date_range(start=utc_th_strt, end=utc_th_end, freq=f'{T.H.DELT}min')
                    
                    if T.H.LT == False:
                        # UTC REFERENCE
                        holiday_array = np.array([1.0 if dt.date() in hol_d else 0.0 for dt in utc_th])
                    else:
                        # LOCAL TIME REFERENCE
                        tz = pytz.timezone(self.timezone)
                        utc_th_loc = [tz.normalize(tz.localize(dt.to_pydatetime().replace(tzinfo=None))) for dt in utc_th]
                        holiday_array = np.array([1.0 if dt.date() in hol_d else 0.0 for dt in utc_th_loc])
                    
                    return holiday_array
                    
                except Exception as date_error:
                    logger.error(f"Error generating holiday date range: {str(date_error)}")
                    return np.array([])
                    
            elif T.H.SPEC == "Aktuelle Zeit":
                # CURRENT TIME MODE
                if T.H.LT == False:
                    # UTC REFERENCE
                    is_holiday = 1.0 if utc_ref.date() in hol_d else 0.0
                else:
                    # LOCAL TIME REFERENCE
                    tz = pytz.timezone(self.timezone)
                    utc_ref_loc = tz.normalize(tz.localize(utc_ref.replace(tzinfo=None)))
                    is_holiday = 1.0 if utc_ref_loc.date() in hol_d else 0.0
                
                return np.array([is_holiday])
                
        except Exception as e:
            logger.error(f"Error generating holiday features: {str(e)}")
            return np.array([])


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
                # TIME HORIZON MODE
                if not use_local_time:
                    # UTC REFERENCE
                    sec = pd.Series(timestamps).map(pd.Timestamp.timestamp)
                    
                    # 31557600 = 60×60×24×365.25 (seconds in average year)
                    features["y_sin"] = np.sin(sec / 31557600 * 2 * np.pi)
                    features["y_cos"] = np.cos(sec / 31557600 * 2 * np.pi)
                    
                else:
                    # LOCAL TIME REFERENCE
                    utc_timestamps = [pytz.utc.localize(dt) if dt.tzinfo is None else dt for dt in timestamps]
                    local_timestamps = [dt.astimezone(pytz.timezone(self.timezone)) for dt in utc_timestamps]
                    
                    # Generate seconds timestamp
                    sec = np.array([
                        (dt.timetuple().tm_yday - 1) * 86400 +
                        dt.hour * 3600 +
                        dt.minute * 60 +
                        dt.second
                        for dt in local_timestamps
                    ])
                    
                    # Years as NumPy array
                    years = np.array([dt.year for dt in local_timestamps])
                    
                    # Vectorized leap year check
                    is_leap = np.vectorize(calendar.isleap)(years)
                    
                    # Number of seconds in the year
                    sec_y = np.where(is_leap, 31622400, 31536000)  # 366 or 365 days
                    
                    features["y_sin"] = np.sin(sec / sec_y * 2 * np.pi)
                    features["y_cos"] = np.cos(sec / sec_y * 2 * np.pi)
                    
            elif mode == "Aktuelle Zeit":
                # CURRENT TIME MODE (single timestamp)
                if len(timestamps) > 0:
                    utc_ref = timestamps[0]  # Use first timestamp as reference
                    
                    if not use_local_time:
                        # UTC REFERENCE
                        sec = utc_ref.timestamp()
                        features["y_sin"] = np.sin(sec / 31557600 * 2 * np.pi)
                        features["y_cos"] = np.cos(sec / 31557600 * 2 * np.pi)
                        
                    else:
                        # LOCAL TIME REFERENCE
                        if utc_ref.tzinfo is None:
                            utc_ref = pytz.utc.localize(utc_ref)
                        local_time = utc_ref.astimezone(pytz.timezone(self.timezone))
                        
                        sec = ((local_time.timetuple().tm_yday - 1) * 86400 + 
                              local_time.hour * 3600 + 
                              local_time.minute * 60 + 
                              local_time.second)
                        
                        # Number of seconds in the year
                        if calendar.isleap(local_time.year):
                            sec_y = 31622400  # 366 days
                        else:
                            sec_y = 31536000  # 365 days
                            
                        features["y_sin"] = np.sin(sec / sec_y * 2 * np.pi)
                        features["y_cos"] = np.cos(sec / sec_y * 2 * np.pi)
            
            logger.info(f"Extracted yearly features for {len(timestamps)} timestamps")
            return features
            
        except Exception as e:
            logger.error(f"Error extracting yearly features: {str(e)}")
            return {"y_sin": np.array([]), "y_cos": np.array([])}
    
    def extract_monthly_features(self, timestamps: List[datetime.datetime],
                                use_local_time: bool = False,
                                mode: str = "Zeithorizont") -> Dict[str, np.ndarray]:
        """
        Extract monthly cyclical features (sin/cos components)
        
        Args:
            timestamps: List of datetime objects
            use_local_time: Whether to use local time or UTC
            mode: "Zeithorizont" or "Aktuelle Zeit"
            
        Returns:
            Dict with m_sin and m_cos arrays
        """
        try:
            features = {}
            
            if mode == "Zeithorizont":
                if not use_local_time:
                    # UTC REFERENCE
                    sec = pd.Series(timestamps).map(pd.Timestamp.timestamp)
                    
                    # 2629800 = 60×60×24×365.25/12 (seconds in average month)
                    features["m_sin"] = np.sin(sec / 2629800 * 2 * np.pi)
                    features["m_cos"] = np.cos(sec / 2629800 * 2 * np.pi)
                    
                else:
                    # LOCAL TIME REFERENCE
                    utc_timestamps = [pytz.utc.localize(dt) if dt.tzinfo is None else dt for dt in timestamps]
                    local_timestamps = [dt.astimezone(pytz.timezone(self.timezone)) for dt in utc_timestamps]
                    
                    # Calculate seconds within month cycle
                    sec = np.array([
                        (dt.timetuple().tm_yday - 1) * 86400 +
                        dt.hour * 3600 +
                        dt.minute * 60 +
                        dt.second
                        for dt in local_timestamps
                    ])
                    
                    features["m_sin"] = np.sin(sec / 2629800 * 2 * np.pi)
                    features["m_cos"] = np.cos(sec / 2629800 * 2 * np.pi)
                    
            elif mode == "Aktuelle Zeit":
                # CURRENT TIME MODE
                if len(timestamps) > 0:
                    utc_ref = timestamps[0]
                    
                    if not use_local_time:
                        sec = utc_ref.timestamp()
                        features["m_sin"] = np.sin(sec / 2629800 * 2 * np.pi)
                        features["m_cos"] = np.cos(sec / 2629800 * 2 * np.pi)
                    else:
                        if utc_ref.tzinfo is None:
                            utc_ref = pytz.utc.localize(utc_ref)
                        local_time = utc_ref.astimezone(pytz.timezone(self.timezone))
                        
                        sec = ((local_time.timetuple().tm_yday - 1) * 86400 + 
                              local_time.hour * 3600 + 
                              local_time.minute * 60 + 
                              local_time.second)
                        
                        features["m_sin"] = np.sin(sec / 2629800 * 2 * np.pi)
                        features["m_cos"] = np.cos(sec / 2629800 * 2 * np.pi)
            
            logger.info(f"Extracted monthly features for {len(timestamps)} timestamps")
            return features
            
        except Exception as e:
            logger.error(f"Error extracting monthly features: {str(e)}")
            return {"m_sin": np.array([]), "m_cos": np.array([])}
    
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
                # UTC REFERENCE
                sec = pd.Series(timestamps).map(pd.Timestamp.timestamp)
                
                # 604800 = 60×60×24×7 (seconds in week)
                features["w_sin"] = np.sin(sec / 604800 * 2 * np.pi)
                features["w_cos"] = np.cos(sec / 604800 * 2 * np.pi)
                
            else:
                # LOCAL TIME REFERENCE
                utc_timestamps = [pytz.utc.localize(dt) if dt.tzinfo is None else dt for dt in timestamps]
                local_timestamps = [dt.astimezone(pytz.timezone(self.timezone)) for dt in utc_timestamps]
                
                # Calculate seconds within week
                sec = np.array([
                    dt.weekday() * 86400 +  # Days from Monday (0)
                    dt.hour * 3600 +
                    dt.minute * 60 +
                    dt.second
                    for dt in local_timestamps
                ])
                
                features["w_sin"] = np.sin(sec / 604800 * 2 * np.pi)
                features["w_cos"] = np.cos(sec / 604800 * 2 * np.pi)
            
            logger.info(f"Extracted weekly features for {len(timestamps)} timestamps")
            return features
            
        except Exception as e:
            logger.error(f"Error extracting weekly features: {str(e)}")
            return {"w_sin": np.array([]), "w_cos": np.array([])}
    
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
                # UTC REFERENCE
                sec = pd.Series(timestamps).map(pd.Timestamp.timestamp)
                
                # 86400 = 60×60×24 (seconds in day)
                features["d_sin"] = np.sin((sec % 86400) / 86400 * 2 * np.pi)
                features["d_cos"] = np.cos((sec % 86400) / 86400 * 2 * np.pi)
                
            else:
                # LOCAL TIME REFERENCE
                utc_timestamps = [pytz.utc.localize(dt) if dt.tzinfo is None else dt for dt in timestamps]
                local_timestamps = [dt.astimezone(pytz.timezone(self.timezone)) for dt in utc_timestamps]
                
                # Calculate seconds within day
                sec = np.array([
                    dt.hour * 3600 +
                    dt.minute * 60 +
                    dt.second
                    for dt in local_timestamps
                ])
                
                features["d_sin"] = np.sin(sec / 86400 * 2 * np.pi)
                features["d_cos"] = np.cos(sec / 86400 * 2 * np.pi)
            
            logger.info(f"Extracted daily features for {len(timestamps)} timestamps")
            return features
            
        except Exception as e:
            logger.error(f"Error extracting daily features: {str(e)}")
            return {"d_sin": np.array([]), "d_cos": np.array([])}
    
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
                # UTC REFERENCE
                sec = pd.Series(timestamps).map(pd.Timestamp.timestamp)
                
                # 3600 = 60×60 (seconds in hour)
                features["h_sin"] = np.sin((sec % 3600) / 3600 * 2 * np.pi)
                features["h_cos"] = np.cos((sec % 3600) / 3600 * 2 * np.pi)
                
            else:
                # LOCAL TIME REFERENCE
                utc_timestamps = [pytz.utc.localize(dt) if dt.tzinfo is None else dt for dt in timestamps]
                local_timestamps = [dt.astimezone(pytz.timezone(self.timezone)) for dt in utc_timestamps]
                
                # Calculate seconds within hour
                sec = np.array([
                    dt.minute * 60 + dt.second
                    for dt in local_timestamps
                ])
                
                features["h_sin"] = np.sin(sec / 3600 * 2 * np.pi)
                features["h_cos"] = np.cos(sec / 3600 * 2 * np.pi)
            
            logger.info(f"Extracted hourly features for {len(timestamps)} timestamps")
            return features
            
        except Exception as e:
            logger.error(f"Error extracting hourly features: {str(e)}")
            return {"h_sin": np.array([]), "h_cos": np.array([])}
    
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
                'hourly': False  # Usually not needed for most applications
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
            
            logger.info(f"Extracted {len(all_features)} time features for {len(timestamps)} timestamps")
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
        # Ensure timestamp column is datetime
        if timestamp_column in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df[timestamp_column]):
                df[timestamp_column] = pd.to_datetime(df[timestamp_column])
            
            timestamps = df[timestamp_column].tolist()
            
            # Extract features
            extractor = TimeFeatureExtractor(timezone)
            time_features = extractor.extract_all_time_features(
                timestamps, features_config, use_local_time=(timezone != 'UTC')
            )
            
            # Add features to DataFrame
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


# Holiday detection functionality (simplified version)
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
        # This is a simplified implementation
        # In real implementation, you would use libraries like 'holidays'
        # or implement country-specific holiday logic
        
        holidays = []
        for timestamp in timestamps:
            # Simple check for common holidays (New Year, Christmas)
            is_holiday = (
                (timestamp.month == 1 and timestamp.day == 1) or  # New Year
                (timestamp.month == 12 and timestamp.day == 25) or  # Christmas
                (timestamp.month == 12 and timestamp.day == 26)     # Boxing Day
            )
            holidays.append(is_holiday)
        
        logger.info(f"Detected {sum(holidays)} holidays out of {len(timestamps)} timestamps")
        return holidays
        
    except Exception as e:
        logger.error(f"Error detecting holidays: {str(e)}")
        return [False] * len(timestamps)