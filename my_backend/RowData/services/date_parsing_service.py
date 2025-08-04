"""
Servis za parsiranje datuma sa keširanim format detekcijom
"""
from datetime import datetime
from typing import Optional, List, Tuple, Iterator, Dict
from functools import lru_cache
import pandas as pd
import pytz
import logging
from ..config.settings import SUPPORTED_DATE_FORMATS
from ..utils.exceptions import DateParsingError

logger = logging.getLogger(__name__)


class DateParsingService:
    """Servis za parsiranje i konverziju datuma"""
    
    def __init__(self):
        self.supported_formats = SUPPORTED_DATE_FORMATS
        self._format_cache = {}
        logger.info(f"DateParsingService initialized with {len(self.supported_formats)} supported formats")
    
    @lru_cache(maxsize=1000)
    def detect_format(self, sample_date: str) -> Optional[str]:
        """
        Detektuje format datuma sa keširanim rezultatima
        
        Returns:
            Detektovan format ili None
        """
        if not sample_date:
            return None
        
        sample_date = sample_date.strip()
        
        # Proveri keš prvo
        if sample_date in self._format_cache:
            return self._format_cache[sample_date]
        
        # Probaj sve formate
        for fmt in self.supported_formats:
            try:
                datetime.strptime(sample_date, fmt)
                self._format_cache[sample_date] = fmt
                logger.debug(f"Detected format '{fmt}' for sample '{sample_date}'")
                return fmt
            except ValueError:
                continue
        
        logger.warning(f"Could not detect format for date: '{sample_date}'")
        return None
    
    def parse_single_date(self, date_str: str, format_hint: Optional[str] = None) -> datetime:
        """
        Parsira pojedinačni datum
        
        Args:
            date_str: String reprezentacija datuma
            format_hint: Opcioni hint za format
            
        Returns:
            Parsiran datetime objekat
            
        Raises:
            DateParsingError: Ako datum ne može biti parsiran
        """
        if not date_str:
            raise DateParsingError("Empty date string")
        
        date_str = date_str.strip()
        
        # Prvo probaj sa hint-om ako postoji
        if format_hint:
            try:
                return datetime.strptime(date_str, format_hint)
            except ValueError:
                logger.debug(f"Format hint '{format_hint}' failed for '{date_str}'")
        
        # Zatim auto-detektuj
        detected_format = self.detect_format(date_str)
        if detected_format:
            try:
                return datetime.strptime(date_str, detected_format)
            except ValueError as e:
                raise DateParsingError(
                    f"Failed to parse date with detected format",
                    sample_value=date_str,
                    expected_format=detected_format
                )
        
        # Pokušaj pandas parser kao fallback
        try:
            return pd.to_datetime(date_str).to_pydatetime()
        except Exception:
            pass
        
        raise DateParsingError(
            "Could not parse date with any supported format",
            sample_value=date_str,
            expected_format="See supported formats in settings"
        )
    
    def parse_dates_streaming(self, date_strings: Iterator[str], 
                            format_hint: Optional[str] = None,
                            on_error: str = 'skip') -> Iterator[Optional[datetime]]:
        """
        Streaming parsiranje datuma sa error handling
        
        Args:
            date_strings: Iterator sa date string-ovima
            format_hint: Opcioni format hint
            on_error: 'skip', 'raise', ili 'null'
            
        Yields:
            Parsirani datetime objekti ili None
        """
        detected_format = format_hint
        error_count = 0
        max_errors = 100
        
        for i, date_str in enumerate(date_strings):
            try:
                # Auto-detektuj format na prvom datumu
                if not detected_format and i == 0:
                    detected_format = self.detect_format(date_str)
                    if not detected_format:
                        if on_error == 'raise':
                            raise DateParsingError(
                                "Could not detect date format",
                                sample_value=date_str
                            )
                
                # Parsiraj datum
                if detected_format:
                    yield datetime.strptime(date_str.strip(), detected_format)
                else:
                    # Fallback na pandas parser
                    yield pd.to_datetime(date_str).to_pydatetime()
                    
            except Exception as e:
                error_count += 1
                
                if on_error == 'raise':
                    raise DateParsingError(
                        f"Failed to parse date at position {i}",
                        sample_value=date_str,
                        expected_format=detected_format
                    )
                elif on_error == 'null':
                    yield None
                else:  # skip
                    continue
                
                if error_count > max_errors:
                    raise DateParsingError(
                        f"Too many parsing errors ({error_count}). Stopping.",
                        sample_value=date_str
                    )
    
    def parse_datetime_column(self, dates: List[str], times: Optional[List[str]] = None,
                            custom_format: Optional[str] = None) -> List[datetime]:
        """
        Parsira kolonu sa datumima ili kombinaciju datum+vreme kolona
        
        Args:
            dates: Lista datuma
            times: Opciona lista vremena
            custom_format: Custom format string
            
        Returns:
            Lista parsiranih datetime objekata
        """
        if times and len(dates) != len(times):
            raise DateParsingError("Date and time columns must have same length")
        
        # Kombinuj datum i vreme ako je potrebno
        if times:
            combined = [f"{d.strip()} {t.strip()}" for d, t in zip(dates, times)]
        else:
            combined = dates
        
        # Parsiraj sa custom formatom ako postoji
        if custom_format:
            try:
                return [datetime.strptime(dt.strip(), custom_format) for dt in combined]
            except ValueError as e:
                sample = combined[0] if combined else "N/A"
                raise DateParsingError(
                    f"Failed to parse with custom format",
                    sample_value=sample,
                    expected_format=custom_format
                )
        
        # Auto-detektuj i parsiraj
        result = []
        detected_format = None
        
        for i, dt in enumerate(combined):
            try:
                if not detected_format:
                    detected_format = self.detect_format(dt)
                    if not detected_format:
                        raise DateParsingError(
                            "Could not detect date format",
                            sample_value=dt
                        )
                
                parsed = datetime.strptime(dt.strip(), detected_format)
                result.append(parsed)
                
            except Exception as e:
                logger.error(f"Failed to parse date at index {i}: {dt}")
                raise DateParsingError(
                    f"Failed to parse date at index {i}",
                    sample_value=dt,
                    expected_format=detected_format
                )
        
        return result
    
    def convert_to_utc(self, dt: datetime, timezone: str = 'UTC') -> datetime:
        """
        Konvertuje datetime u UTC
        
        Args:
            dt: Datetime objekat
            timezone: Source timezone
            
        Returns:
            UTC datetime
        """
        try:
            # Ako već ima timezone info, konvertuj direktno
            if dt.tzinfo is not None:
                return dt.astimezone(pytz.UTC)
            
            # Lokalizuj prema zadatom timezone-u
            tz = pytz.timezone(timezone)
            localized = tz.localize(dt)
            
            # Konvertuj u UTC
            return localized.astimezone(pytz.UTC)
            
        except pytz.exceptions.UnknownTimeZoneError:
            raise DateParsingError(f"Unknown timezone: {timezone}")
        except Exception as e:
            raise DateParsingError(f"Failed to convert to UTC: {str(e)}")
    
    def batch_convert_to_utc(self, dates: List[datetime], timezone: str = 'UTC') -> List[datetime]:
        """Batch konverzija u UTC za performanse"""
        if not dates:
            return []
        
        # Pandas je efikasniji za batch operacije
        df = pd.DataFrame({'date': dates})
        
        # Lokalizuj ako nema timezone
        if df['date'].dt.tz is None:
            df['date'] = df['date'].dt.tz_localize(timezone, ambiguous='NaT', nonexistent='NaT')
        
        # Konvertuj u UTC
        if timezone != 'UTC':
            df['date'] = df['date'].dt.tz_convert('UTC')
        
        return df['date'].tolist()
    
    def clean_time_string(self, time_str: str) -> str:
        """
        Čisti time string od nevažećih karaktera
        
        Example: '00:00:00.000Kdd' -> '00:00:00.000'
        """
        if not isinstance(time_str, str):
            return str(time_str)
        
        cleaned = ''
        for c in time_str:
            if c.isdigit() or c in ':-+.T ':
                cleaned += c
        
        return cleaned.strip()
    
    def validate_date_range(self, dates: List[datetime], 
                          min_year: int = 1970, 
                          max_year: Optional[int] = None) -> Tuple[bool, Optional[str]]:
        """
        Validira da li su datumi u razumnom opsegu
        
        Returns:
            Tuple: (da li je valjan, poruka o grešci)
        """
        if not dates:
            return True, None
        
        if max_year is None:
            max_year = datetime.now().year + 10
        
        for i, dt in enumerate(dates):
            if dt.year < min_year:
                return False, f"Date at index {i} is before {min_year}: {dt}"
            if dt.year > max_year:
                return False, f"Date at index {i} is after {max_year}: {dt}"
        
        return True, None
    
    def get_format_statistics(self) -> Dict[str, int]:
        """Vraća statistiku o korišćenim formatima"""
        stats = {}
        for sample, fmt in self._format_cache.items():
            stats[fmt] = stats.get(fmt, 0) + 1
        return stats