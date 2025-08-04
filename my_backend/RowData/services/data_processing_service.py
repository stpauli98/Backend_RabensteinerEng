"""
Servis za procesiranje CSV/TXT podataka sa streaming podrškom
"""
import csv
from io import StringIO
from typing import Iterator, List, Dict, Optional, Tuple, Any
import logging
from datetime import datetime
from .date_parsing_service import DateParsingService
from .file_upload_service import FileUploadService
from ..utils.exceptions import ProcessingError, ValidationError
from ..config.settings import ALLOWED_DELIMITERS

logger = logging.getLogger(__name__)


class DataProcessingService:
    """Servis za procesiranje podataka iz CSV/TXT fajlova"""
    
    def __init__(self, 
                 file_service: Optional[FileUploadService] = None,
                 date_service: Optional[DateParsingService] = None):
        self.file_service = file_service or FileUploadService()
        self.date_service = date_service or DateParsingService()
        logger.info("DataProcessingService initialized")
    
    def detect_delimiter(self, content_sample: str, sample_lines: int = 5) -> str:
        """
        Detektuje delimiter na osnovu uzorka sadržaja
        
        Args:
            content_sample: Uzorak sadržaja
            sample_lines: Broj linija za analizu
            
        Returns:
            Detektovan delimiter
        """
        lines = content_sample.strip().split('\n')[:sample_lines]
        
        if not lines:
            raise ValidationError("No content to analyze for delimiter detection")
        
        # Prebrojava pojavljivanja svakog delimitera
        delimiter_counts = {d: 0 for d in ALLOWED_DELIMITERS}
        
        for line in lines:
            for delimiter in ALLOWED_DELIMITERS:
                delimiter_counts[delimiter] += line.count(delimiter)
        
        # Pronađi delimiter sa najviše pojavljivanja
        best_delimiter = max(delimiter_counts.items(), key=lambda x: x[1])
        
        if best_delimiter[1] == 0:
            logger.warning("No delimiter found, defaulting to comma")
            return ','
        
        logger.info(f"Detected delimiter: '{best_delimiter[0]}' with {best_delimiter[1]} occurrences")
        return best_delimiter[0]
    
    def process_csv_stream(self, 
                          content_stream: Iterator[str],
                          delimiter: str,
                          has_header: bool,
                          encoding: str = 'utf-8') -> Iterator[Dict[str, str]]:
        """
        Procesira CSV podatke u streaming način
        
        Args:
            content_stream: Stream sa sadržajem
            delimiter: CSV delimiter
            has_header: Da li prvi red sadrži header
            encoding: Encoding za čitanje
            
        Yields:
            Dict sa podacima za svaki red
        """
        buffer = ""
        line_count = 0
        headers = None
        csv_reader = None
        
        for chunk in content_stream:
            buffer += chunk
            
            # Podeli na linije
            lines = buffer.split('\n')
            
            # Zadrži poslednju liniju koja može biti nepotpuna
            buffer = lines[-1]
            
            # Procesiraj kompletne linije
            for line in lines[:-1]:
                if not line.strip():
                    continue
                
                try:
                    # Parsiraj liniju kao CSV
                    parsed_line = list(csv.reader([line], delimiter=delimiter))[0]
                    
                    # Prvi red kao header ako je potrebno
                    if has_header and line_count == 0:
                        headers = [self._clean_header(h) for h in parsed_line]
                        line_count += 1
                        continue
                    
                    # Kreiraj dict sa podacima
                    if headers:
                        # Osiguraj da imamo isti broj kolona
                        if len(parsed_line) != len(headers):
                            logger.warning(
                                f"Line {line_count + 1} has {len(parsed_line)} columns, "
                                f"expected {len(headers)}"
                            )
                            # Dopuni ili skrati ako je potrebno
                            if len(parsed_line) < len(headers):
                                parsed_line.extend([''] * (len(headers) - len(parsed_line)))
                            else:
                                parsed_line = parsed_line[:len(headers)]
                        
                        yield dict(zip(headers, parsed_line))
                    else:
                        # Bez header-a, koristi numeričke ključeve
                        yield {f"col_{i}": val for i, val in enumerate(parsed_line)}
                    
                    line_count += 1
                    
                except csv.Error as e:
                    logger.error(f"CSV parsing error at line {line_count + 1}: {str(e)}")
                    raise ProcessingError(
                        f"Failed to parse CSV at line {line_count + 1}",
                        stage="csv_parsing"
                    )
        
        # Procesiraj poslednju liniju ako postoji
        if buffer.strip():
            try:
                parsed_line = list(csv.reader([buffer], delimiter=delimiter))[0]
                if headers:
                    yield dict(zip(headers, parsed_line))
                else:
                    yield {f"col_{i}": val for i, val in enumerate(parsed_line)}
            except csv.Error:
                logger.warning(f"Failed to parse last line: {buffer}")
    
    def _clean_header(self, header: str) -> str:
        """Čisti header naziv"""
        return header.strip().replace('\ufeff', '')  # Ukloni BOM
    
    def process_upload_data(self, 
                          upload_id: str,
                          parameters: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
        """
        Kompletno procesiranje upload-a sa svim transformacijama
        
        Args:
            upload_id: ID upload-a
            parameters: Parametri procesiranja
            
        Yields:
            Procesirani redovi podataka
        """
        # Ekstraktuj parametre
        delimiter = parameters.get('delimiter', ',')
        timezone = parameters.get('timezone', 'UTC')
        has_header = parameters.get('has_header', False)
        selected_columns = parameters.get('selected_columns', {})
        custom_date_format = parameters.get('custom_date_format')
        dropdown_count = parameters.get('dropdown_count', 2)
        value_column_name = parameters.get('value_column_name', '')
        
        # Definiši kolone
        has_separate_date_time = dropdown_count == 3
        date_column = selected_columns.get('column1')
        time_column = selected_columns.get('column2') if has_separate_date_time else None
        value_column = (selected_columns.get('column3') if has_separate_date_time 
                       else selected_columns.get('column2'))
        
        logger.info(
            f"Processing upload {upload_id} with parameters: "
            f"delimiter='{delimiter}', timezone={timezone}, "
            f"date_column={date_column}, value_column={value_column}"
        )
        
        # Stream podataka iz chunk-ova
        content_stream = self.file_service.combine_chunks_streaming(upload_id)
        
        # Procesiraj CSV stream
        row_count = 0
        for row in self.process_csv_stream(content_stream, delimiter, has_header):
            try:
                # Ekstraktuj potrebne vrednosti
                date_str = row.get(date_column, '')
                time_str = row.get(time_column, '') if time_column else ''
                value_str = row.get(value_column, '')
                
                # Parsiraj datum/vreme
                if has_separate_date_time and time_str:
                    # Očisti i kombinuj datum i vreme
                    date_str = self.date_service.clean_time_string(date_str)
                    time_str = self.date_service.clean_time_string(time_str)
                    datetime_str = f"{date_str} {time_str}"
                else:
                    datetime_str = date_str
                
                # Parsiraj datetime
                try:
                    parsed_date = self.date_service.parse_single_date(
                        datetime_str, 
                        format_hint=custom_date_format
                    )
                except Exception as e:
                    logger.warning(f"Failed to parse date at row {row_count + 1}: {datetime_str}")
                    continue  # Preskoči red sa neispravnim datumom
                
                # Konvertuj u UTC
                utc_date = self.date_service.convert_to_utc(parsed_date, timezone)
                
                # Parsiraj vrednost
                try:
                    parsed_value = float(value_str) if value_str else None
                except ValueError:
                    logger.warning(f"Invalid numeric value at row {row_count + 1}: {value_str}")
                    parsed_value = None
                
                # Yield procesirani red
                yield {
                    'UTC': utc_date.strftime('%Y-%m-%d %H:%M:%S'),
                    value_column_name or value_column: str(parsed_value) if parsed_value is not None else '',
                    '_row_number': row_count + 1
                }
                
                row_count += 1
                
            except Exception as e:
                logger.error(f"Error processing row {row_count + 1}: {str(e)}")
                # Odluči da li da prekineš ili samo preskoči red
                if row_count < 10:  # Strožiji na početku
                    raise ProcessingError(
                        f"Failed to process row {row_count + 1}: {str(e)}",
                        stage="row_processing"
                    )
                else:
                    continue  # Preskoči problematične redove posle početka
        
        logger.info(f"Successfully processed {row_count} rows for upload {upload_id}")
    
    def validate_processed_data(self, data: List[Dict[str, Any]]) -> Tuple[bool, Optional[str]]:
        """
        Validira procesirane podatke
        
        Returns:
            Tuple: (da li je validno, poruka o grešci)
        """
        if not data:
            return False, "No data to validate"
        
        # Proveri da li svi redovi imaju potrebne ključeve
        required_keys = {'UTC'}
        first_row = data[0]
        
        for key in required_keys:
            if key not in first_row:
                return False, f"Missing required key: {key}"
        
        # Proveri da li su datumi sortirani (opcionalno)
        dates = [row['UTC'] for row in data]
        if dates != sorted(dates):
            logger.warning("Dates are not sorted")
        
        return True, None
    
    def clean_file_content(self, content: str, delimiter: str) -> str:
        """Čisti sadržaj fajla od viška delimitera i whitespace-a"""
        lines = content.splitlines()
        cleaned_lines = []
        
        for line in lines:
            # Ukloni trailing delimitere
            cleaned = line.rstrip(f"{delimiter};,")
            cleaned_lines.append(cleaned)
        
        return "\n".join(cleaned_lines)
    
    def export_to_csv(self, data: Iterator[Dict[str, Any]], 
                     output_path: str,
                     delimiter: str = ';') -> int:
        """
        Eksportuje procesirane podatke u CSV
        
        Returns:
            Broj eksportovanih redova
        """
        row_count = 0
        headers_written = False
        
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = None
            
            for row in data:
                # Preskoči _row_number iz export-a
                export_row = {k: v for k, v in row.items() if not k.startswith('_')}
                
                # Piši header na prvom redu
                if not headers_written:
                    writer = csv.DictWriter(csvfile, fieldnames=export_row.keys(), 
                                          delimiter=delimiter)
                    writer.writeheader()
                    headers_written = True
                
                writer.writerow(export_row)
                row_count += 1
        
        logger.info(f"Exported {row_count} rows to {output_path}")
        return row_count