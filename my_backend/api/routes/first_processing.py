import os
import pandas as pd
import numpy as np
import json
import gzip
import traceback
import logging
import tempfile
import csv
from io import StringIO
import datetime
import time
from flask import request, jsonify, Response, send_file, Blueprint, g
from core.extensions import socketio
from middleware.auth import require_auth
from middleware.subscription import require_subscription, check_processing_limit
from utils.usage_tracking import increment_processing_count, update_storage_usage

bp = Blueprint('first_processing', __name__)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


temp_files = {}

class ProgressTracker:
    """Real-time progress tracking sa ETA kalkulacijom"""

    def __init__(self, upload_id, total_items=None, file_size_bytes=None, total_chunks=None):
        self.upload_id = upload_id
        self.start_time = time.time()
        self.phase_start_times = {}
        self.phase_durations = {}  # Stvarno vrijeme završenih faza

        # File metadata za inicijalnu ETA procjenu
        self.file_size_bytes = file_size_bytes  # Ukupna veličina fajla
        self.total_chunks = total_chunks  # Broj chunk-ova

        self.total_items = total_items  # Ukupan broj iteracija (len(time_list))
        self.processed_items = 0
        self.last_emit_time = 0
        self.emit_interval = 0.5  # Emit minimum svakih 500ms

        # Kontinuirana rekalibracija sa EWMA
        self.processing_start_time = None
        self.processing_start_items = 0
        self.processing_history = []  # (timestamp, items_processed) za moving average
        self.history_window = 500      # Zadnjih 500 iteracija za prosječnu brzinu
        self.min_calibration_samples = 20  # RANO STARTOVANJE - samo 20 iteracija!

        # EWMA (Exponential Weighted Moving Average)
        self.ewma_time_per_item = None  # Eksponencijalno ponderisan prosjek
        self.ewma_alpha = 0.15  # Faktor glađenja (0.1-0.2 optimal)

        # Trend detection
        self.trend_window = 200  # Koristi zadnjih 200 za trend analizu
        self.detected_trend = 0  # % promjene brzine (+ = ubrzava, - = usporava)

        self.last_eta_sent = None
        self.eta_send_interval = 5.0

        # NE koristimo file-size procjenu jer nije tačna
        # Čekamo stvarne podatke iz processing faze

    def start_phase(self, phase_name):
        """Označi početak nove faze procesiranja"""
        self.phase_start_times[phase_name] = time.time()

    def end_phase(self, phase_name):
        """Završi fazu i snimi stvarno vrijeme trajanja"""
        if phase_name in self.phase_start_times:
            duration = time.time() - self.phase_start_times[phase_name]
            self.phase_durations[phase_name] = duration
            logger.info(f"Phase '{phase_name}' completed in {duration:.2f}s")

    def start_processing_phase(self):
        """Pokreni mjerenje vremena za processing fazu"""
        self.processing_start_time = time.time()
        self.processing_start_items = self.processed_items
        # Reset history za čistu kalibraciju
        self.processing_history = []

    def calculate_eta(self, current_progress=None):
        """
        ADITIVNI ETA SISTEM - UVIJEK sabiramo SVE preostale faze:
        ETA = Processing_Remaining + Finalization_Remaining + Compression_Remaining

        Svaka faza:
        - Ako je u toku: mjeri stvarnu brzinu i ekstrapolira
        - Ako nije počela: procjenjuje na osnovu prethodnih faza
        
        RANO (0-37%): Vraća "calculating" jer nemamo podatke o brzini
        """
        # === 100% ZAVRŠENO ===
        if current_progress and current_progress >= 100:
            return 0

        current_time = time.time()

        # === RANE FAZE (0-37%) - NEMA PODATAKA ZA PROCJENU ===
        # Prije processing faze nemamo nikakve podatke o brzini obrade
        if not self.processing_start_time or not self.total_items:
            # Još nismo u processing fazi - ne možemo procijeniti
            return "calculating"

        # === PROVJERI DA LI SMO U KALIBRACIJI ===
        # Ako procesiranje traje ali još nema dovoljno podataka, vrati "calculating"
        items_processed = self.processed_items - self.processing_start_items
        remaining_items = self.total_items - self.processed_items

        # Processing je aktivan ali još nemamo 50 iteracija za stabilnu procjenu
        if items_processed < 50 and remaining_items > 0:
            return "calculating"

        eta_components = []  # Lista (phase_name, eta_seconds)

        # === 1. PROCESSING FAZA (37-86%) ===
        processing_eta = 0
        processing_complete = False

        if items_processed >= 50:
            processing_elapsed = current_time - self.processing_start_time
            time_per_item = processing_elapsed / items_processed

            if remaining_items > 0:
                # JOŠ UVIJEK PROCESIRAMO
                processing_eta = remaining_items * time_per_item
                eta_components.append(('processing', processing_eta))
            else:
                # SVI ITEMS OBRAĐENI
                processing_complete = True
                # Snimi ukupno vrijeme processing faze
                if 'processing' not in self.phase_durations:
                    self.phase_durations['processing'] = processing_elapsed

        # === 2. FINALIZATION FAZA (86-95%) ===
        finalization_eta = 0
        finalization_start = self.phase_start_times.get('finalizing')

        if current_progress and current_progress >= 95:
            # Finalization završena, ne dodajemo
            pass
        elif finalization_start and current_progress and current_progress >= 86:
            # Finalization u toku - mjeri stvarnu brzinu
            finalization_elapsed = current_time - finalization_start
            finalization_progress = (current_progress - 86) / 9  # 86-95% = 9% range

            if finalization_progress > 0.05:  # Barem 5% progresa
                estimated_total = finalization_elapsed / finalization_progress
                finalization_eta = estimated_total - finalization_elapsed
                eta_components.append(('finalization', finalization_eta))
            else:
                # Prerano za mjerenje, koristi procjenu
                processing_time = self.phase_durations.get('processing', 0)
                finalization_eta = processing_time * 0.15  # 15% od processing vremena
                eta_components.append(('finalization_est', finalization_eta))
        else:
            # Finalization nije počela - procijeni
            processing_time = self.phase_durations.get('processing', 0)
            if processing_time > 0:
                finalization_eta = processing_time * 0.15  # 15% od processing vremena
                eta_components.append(('finalization_est', finalization_eta))

        # === 3. COMPRESSION FAZA (95-100%) ===
        compression_eta = 0
        compression_start = self.phase_start_times.get('compression')

        if current_progress and current_progress >= 100:
            # Compression završena
            pass
        elif compression_start and current_progress and current_progress >= 95:
            # Compression u toku - mjeri stvarnu brzinu
            compression_elapsed = current_time - compression_start
            compression_progress = (current_progress - 95) / 5  # 95-100% = 5% range

            if compression_progress > 0.05:  # Barem 5% progresa
                estimated_total = compression_elapsed / compression_progress
                compression_eta = estimated_total - compression_elapsed
                eta_components.append(('compression', compression_eta))
            else:
                # Prerano za mjerenje, fiksna procjena
                compression_eta = 2  # Fiksno 2 sekunde
                eta_components.append(('compression_est', compression_eta))
        else:
            # Compression nije počela - fiksna procjena
            compression_eta = 2  # Fiksno 2 sekunde
            eta_components.append(('compression_est', compression_eta))

        # === SABERI SVE KOMPONENTE ===
        total_eta = sum(eta for _, eta in eta_components)

        # Detaljno logovanje
        if eta_components and (not hasattr(self, '_last_log_time') or current_time - self._last_log_time > 5):
            self._last_log_time = current_time
            components_str = " + ".join(f"{name}={eta:.1f}s" for name, eta in eta_components)
            logger.info(
                f"ETA [{current_progress:.0f}%]: {components_str} = {total_eta:.1f}s total"
            )

        return max(0, int(total_eta)) if eta_components else None

    def emit(self, step, progress, message, force=False):
        """Šalje progress update sa ETA kalkulacijom od 0% do 100%"""
        current_time = time.time()

        # Rate limiting - ne šalji previše često osim ako nije force
        if not force and (current_time - self.last_emit_time) < self.emit_interval:
            return

        payload = {
            'uploadId': self.upload_id,
            'step': step,
            'progress': int(progress),
            'message': message
        }

        # Izračunaj ETA ZA SVE FAZE (od 0% do 100%)
        eta_seconds = self.calculate_eta(progress)

        # Rukuj sa "calculating" statusom ili numeričkom ETA
        if eta_seconds == "calculating":
            # U fazi kalibracije - šalji "Procjenjujem..."
            payload['eta'] = "calculating"
            payload['etaFormatted'] = "Procjenjujem..."
            self.last_eta_sent = current_time
        elif eta_seconds is not None:
            # Normalna numerička ETA - šalji periodički
            should_send_eta = (
                self.last_eta_sent is None or
                (current_time - self.last_eta_sent) >= self.eta_send_interval or
                force or
                progress >= 100  # Uvijek šalji ETA=0 na 100%
            )

            if should_send_eta:
                payload['eta'] = eta_seconds
                payload['etaFormatted'] = self.format_time(eta_seconds)
                self.last_eta_sent = current_time

        try:
            socketio.emit('processing_progress', payload, room=self.upload_id)
            self.last_emit_time = current_time
            # Loguj sa ETA ako je prisutna
            eta_text = f" (ETA: {payload['etaFormatted']})" if 'etaFormatted' in payload else ""
            logger.info(f"Progress: {progress}% - {message}{eta_text}")
        except Exception as e:
            logger.error(f"Error emitting progress: {e}")

    @staticmethod
    def format_time(seconds):
        """Formatira sekunde u čitljiv format"""
        if seconds < 60:
            return f"{seconds}s"
        elif seconds < 3600:
            minutes = seconds // 60
            secs = seconds % 60
            return f"{minutes}m {secs}s"
        else:
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            return f"{hours}h {minutes}m"

def clean_for_json(obj):
    """Konvertuje numpy i pandas tipove u Python native tipove."""
    if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32,
                         np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, (np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    elif isinstance(obj, (pd.Timestamp)):
        return obj.isoformat()
    elif pd.isna(obj):
        return None
    return obj

UPLOAD_FOLDER = "chunk_uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def process_csv(file_content, tss, offset, mode_input, intrpl_max, upload_id=None, tracker=None):
    """
    Obradjuje CSV sadržaj te vraća rezultat kao gzip-komprimiran JSON odgovor.
    IDENTIČNA LOGIKA KAO U ORIGINALNOM data_prep_1.py FAJLU.

    Args:
        upload_id: Optional upload ID for Socket.IO progress tracking
        tracker: Optional existing ProgressTracker instance (from chunk assembly)
    """
    # Koristi postojeći tracker ili kreiraj novi
    if not tracker and upload_id:
        tracker = ProgressTracker(upload_id)

    def is_numeric(value):
        """Check if value is numeric"""
        try:
            float(value)
            return True
        except (ValueError, TypeError):
            return False

    try:
        # === FAZA 1: PARSING (10-25%) ===
        # Napomena: 0-10% je chunk assembly, zato parsing počinje od 10%
        if tracker:
            tracker.start_phase('parsing')
            tracker.emit('parsing', 10, 'Započinjem parsiranje CSV podataka...', force=True)

        try:
            if tracker:
                tracker.emit('parsing', 12, 'Učitavanje CSV fajla...')

            lines = file_content.strip().split('\n')
            logger.info(f"Total lines in CSV: {len(lines)}")

            if tracker:
                tracker.emit('parsing', 15, f'Učitano {len(lines)} linija iz CSV-a')

            if len(lines) > 0:
                header = lines[0]
                logger.info(f"Header: '{header}'")
                logger.info(f"Header fields: {header.split(';')}")

            if tracker:
                tracker.emit('parsing', 15, 'Parsiranje sa pandas bibliotekom...')

            try:
                df = pd.read_csv(StringIO(file_content), delimiter=';', skipinitialspace=True, on_bad_lines='skip')
                logger.info(f"Successfully parsed CSV with {len(df)} rows after skipping bad lines")

                if tracker:
                    tracker.emit('parsing', 28, f'Parsiranje završeno - {len(df)} redova ✓')
                    tracker.end_phase('parsing')  # Snimi stvarno vrijeme parsing faze
            except Exception as pandas_error:
                logger.error(f"Even with on_bad_lines='skip', pandas failed: {str(pandas_error)}")
                import csv
                try:
                    df = pd.read_csv(StringIO(file_content), delimiter=';', skipinitialspace=True,
                                   quoting=csv.QUOTE_NONE, on_bad_lines='skip')
                    logger.info(f"Successfully parsed CSV with QUOTE_NONE, {len(df)} rows")
                except Exception as final_error:
                    logger.error(f"All parsing attempts failed: {str(final_error)}")
                    raise pandas_error

            df.columns = df.columns.str.strip()

            if len(df.columns) < 2:
                raise ValueError(f"CSV must have at least 2 columns, found {len(df.columns)}: {list(df.columns)}")

            utc_col_name = df.columns[0]
            value_col_name = df.columns[1]
            logger.info(f"Using columns: UTC='{utc_col_name}', Value='{value_col_name}'")

            # === FAZA 2: PREPROCESSING (20-30%) ===
            if tracker:
                tracker.start_phase('preprocessing')
                tracker.emit('preprocessing', 30, 'Konverzija tipova podataka...')

            # Validate that values can be converted to numeric
            non_numeric = df[value_col_name].apply(lambda x: not is_numeric(x))
            if non_numeric.any():
                logger.info(f"Found non-numeric values in {value_col_name}: {df[value_col_name][non_numeric].head()}")

            df[value_col_name] = pd.to_numeric(df[value_col_name], errors='coerce')

            initial_count = len(df)
            df = df.dropna(subset=[utc_col_name, value_col_name])
            final_count = len(df)

            if initial_count != final_count:
                logger.info(f"Removed {initial_count - final_count} rows with invalid data")

        except Exception as e:
            logger.error(f"Error parsing CSV data: {str(e)}")
            return jsonify({"error": f"CSV parsing failed: {str(e)}"}), 400

        # VORBEREITUNG DER ROHDATEN (matching original logic)
        if tracker:
            tracker.emit('preprocessing', 32, 'Uklanjanje duplikata...')

        # Duplikate in den Rohdaten löschen
        df = df.drop_duplicates(subset=[utc_col_name]).reset_index(drop=True)

        if tracker:
            tracker.emit('preprocessing', 34, 'Sortiranje podataka po vremenu...')

        # Rohdaten nach UTC ordnen
        df = df.sort_values(by=[utc_col_name])

        # Reset des Indexes in den Rohdaten
        df = df.reset_index(drop=True)

        if df.empty:
            return jsonify({"error": "Keine Daten gefunden"}), 400

        if tracker:
            tracker.emit('preprocessing', 35, 'Konverzija datum/vrijeme formata...')

        # ZEITGRENZEN (matching original logic)
        # Convert UTC to datetime objects
        df[utc_col_name] = pd.to_datetime(df[utc_col_name], format='%Y-%m-%d %H:%M:%S')

        time_min_raw = df[utc_col_name].iloc[0].to_pydatetime()
        time_max_raw = df[utc_col_name].iloc[-1].to_pydatetime()

        logger.info(f"Time range: {time_min_raw} to {time_max_raw}")

        if tracker:
            tracker.emit('preprocessing', 37, 'Priprema podataka završena ✓')
            tracker.end_phase('preprocessing')  # Snimi stvarno vrijeme preprocessing faze

        # KONTINUIERLICHER ZEITSTEMPEL (matching original logic)
        # Offset der unteren Zeitgrenze in der Rohdaten
        offset_strt = datetime.timedelta(
            minutes=time_min_raw.minute,
            seconds=time_min_raw.second,
            microseconds=time_min_raw.microsecond
        )

        # Realer Offset in den aufbereiteten Daten [min]
        # Ensure positive offset within TSS range
        normalized_offset = abs(offset) % tss if offset >= 0 else 0

        # Untere Zeitgrenze in den aufbereiteten Daten
        time_min = time_min_raw - offset_strt
        if normalized_offset > 0:
            # Add offset to align with the requested time grid
            time_min += datetime.timedelta(minutes=normalized_offset)

        logger.info(f"Applying offset of {normalized_offset} minutes to {time_min_raw}")
        logger.info(f"Resulting start time: {time_min}")

        # Generate continuous timestamp (matching original logic)
        time_list = []
        current_time = time_min

        while current_time <= time_max_raw:
            time_list.append(current_time)
            current_time += datetime.timedelta(minutes=tss)

        if not time_list:
            return jsonify({"error": "Keine gültigen Zeitpunkte generiert"}), 400

        logger.info(f"Generated {len(time_list)} time points")
        logger.info(f"First timestamp: {time_list[0]}")
        logger.info(f"Last timestamp: {time_list[-1]}")

        # === FAZA 3: PROCESSING (37-86%) ===
        if tracker:
            tracker.start_phase('processing')
            tracker.total_items = len(time_list)  # Postavi total za granularni tracking
            tracker.start_processing_phase()  # Pokreni mjerenje brzine
            tracker.emit('processing', 37, f'Započinjem {mode_input} procesiranje...')

        # Convert df to format matching original (for easier index access)
        df_dict = df.to_dict('list')

        # Zähler für den Durchlauf der Rohdaten
        i_raw = 0

        # Initialisierung der Liste mit den aufbereiteten Messwerten
        value_list = []

        # METHODE: MITTELWERTBILDUNG (matching original logic)
        if mode_input == "mean":
            if tracker:
                tracker.emit('processing', 37, 'Računam srednje vrijednosti...', force=True)

            # Schleife durchläuft alle Zeitschritte des kontinuierlichen Zeitstempels
            for i in range(0, len(time_list)):
                # Granularni progress update (30-85% range)
                if tracker and i % max(1, len(time_list) // 50) == 0:  # Update svakih ~2%
                    progress = 37 + (i / len(time_list)) * 49  # Map to 37-86%
                    tracker.processed_items = i
                    tracker.emit('processing', progress, f'Procesiranje: {i}/{len(time_list)} vremenskih tačaka')

                # Zeitgrenzen für die Mittelwertbildung (Untersuchungsraum)
                time_int_min = time_list[i] - datetime.timedelta(minutes=tss/2)
                time_int_max = time_list[i] + datetime.timedelta(minutes=tss/2)

                # Berücksichtigung angrenzender Untersuchungsräume
                if i > 0:
                    i_raw -= 1
                if i > 0 and df[utc_col_name].iloc[i_raw].to_pydatetime() < time_int_min:
                    i_raw += 1

                # Initialisierung der Liste mit den Messwerten im Untersuchungsraum
                value_int_list = []

                # Auflistung numerischer Messwerte im Untersuchungsraum
                while (i_raw < len(df) and
                       df[utc_col_name].iloc[i_raw].to_pydatetime() <= time_int_max and
                       df[utc_col_name].iloc[i_raw].to_pydatetime() >= time_int_min):
                    if is_numeric(df[value_col_name].iloc[i_raw]):
                        value_int_list.append(float(df[value_col_name].iloc[i_raw]))
                    i_raw += 1

                # Mittelwertbildung über die numerischen Messwerte im Untersuchungsraum
                if len(value_int_list) > 0:
                    import statistics
                    value_list.append(statistics.mean(value_int_list))
                else:
                    value_list.append("nan")

        # METHODE: LINEARE INTERPOLATION (matching original logic)
        elif mode_input == "intrpl":
            if tracker:
                tracker.emit('processing', 37, 'Započinjem linearnu interpolaciju...', force=True)

            # Zähler für den Durchlauf der Rohdaten
            i_raw = 0

            # Richtung des Schleifendurchlaufs
            direct = 1

            # Schleife durchläuft alle Zeitschritte des kontinuierlichen Zeitstempels
            for i in range(0, len(time_list)):
                # Granularni progress update (30-85% range)
                if tracker and i % max(1, len(time_list) // 50) == 0:  # Update svakih ~2%
                    progress = 37 + (i / len(time_list)) * 49  # Map to 37-86%
                    tracker.processed_items = i
                    tracker.emit('processing', progress, f'Interpolacija: {i}/{len(time_list)} tačaka')

                # Schleife durchläuft die Rohdaten von vorne bis hinten zur Auffindung des nachfolgenden Wertes
                if direct == 1:

                    loop = True
                    while i_raw < len(df) and loop == True:

                        # Der aktuelle Zeitpunkt in den Rohdaten liegt nach dem aktuellen Zeitpunkt
                        # im kontinuierlichen Zeitstempel oder ist mit diesem identisch.
                        if df[utc_col_name].iloc[i_raw].to_pydatetime() >= time_list[i]:

                            # Der aktuelle Zeitpunkt in den Rohdaten liegt nach dem aktuellen Zeitpunkt
                            # im kontinuierlichen Zeitstempel oder ist mit diesem identisch und der
                            # dazugehörige Messwert ist numerisch
                            if is_numeric(df[value_col_name].iloc[i_raw]):

                                # UTC und Messwert vom nachfolgenden Wert übernehmen
                                time_next = df[utc_col_name].iloc[i_raw].to_pydatetime()
                                value_next = float(df[value_col_name].iloc[i_raw])
                                loop = False

                            else:
                                # Zähler aktuallisieren, wenn Messwert nicht numerisch
                                i_raw += 1

                        else:
                            # Zähler aktuallisieren, wenn der aktuelle Zeitpunkt in den Rohdaten vor dem aktuellen
                            # Zeitpunkt im kontinuierlichen Zeitstempel liegt.
                            i_raw += 1

                    # Die gesamten Rohdaten wurden durchlaufen und es wurde kein gültiger Messwert gefunden
                    if i_raw + 1 > len(df):
                        value_list.append("nan")

                        # Zähler für die Rohdaten auf Null setzen und Schleifenrichtung festlegen
                        i_raw = 0
                        direct = 1
                    else:
                        # Schleifenrichtung umdrehen
                        direct = -1

                # Finden des vorangegangenen Wertes
                if direct == -1:

                    loop = True
                    while i_raw >= 0 and loop == True:

                        # Der aktuelle Zeitpunkt in den Rohdaten liegt vor dem aktuellen Zeitpunkt
                        # im kontinuierlichen Zeitstempel oder ist mit diesem identisch.
                        if df[utc_col_name].iloc[i_raw].to_pydatetime() <= time_list[i]:

                            # Der aktuelle Zeitpunkt in den Rohdaten liegt vor dem aktuellen Zeitpunkt
                            # im kontinuierlichen Zeitstempel oder ist mit diesem identisch und der
                            # dazugehörige Messwert ist numerisch
                            if is_numeric(df[value_col_name].iloc[i_raw]):

                                # UTC und Messwert vom vorangegangenen Wert übernehmen
                                time_prior = df[utc_col_name].iloc[i_raw].to_pydatetime()
                                value_prior = float(df[value_col_name].iloc[i_raw])
                                loop = False
                            else:
                                # Zähler aktuallisieren, wenn Messwert nicht numerisch
                                i_raw -= 1
                        else:
                            # Zähler aktuallisieren, wenn der aktuelle Zeitpunkt in den Rohdaten nach dem aktuellen
                            # Zeitpunkt im kontinuierlichen Zeitstempel liegt.
                            i_raw -= 1

                    # Die gesamten Rohdaten wurden durchlaufen und es wurde kein gültiger Messwert gefunden
                    if i_raw < 0:
                        value_list.append("nan")

                        # Zähler für die Rohdaten auf Null setzen und Schleifenrichtung festlegen
                        i_raw = 0
                        direct = 1

                    # Es wurde ein gültige Messwerte vor dem aktuellen Zeitpunkt im kontinuierlichen Zeitstempel und nach diesem gefunden
                    else:
                        delta_time = time_next - time_prior

                        # Zeitabstand zwischen den entsprechenden Messwerten in den Rohdaten [sec]
                        delta_time_sec = delta_time.total_seconds()
                        delta_value = value_prior - value_next

                        # Zeitpunkte fallen zusammen oder gleichbleibender Messwert - Keine lineare Interpolation notwendig
                        if delta_time_sec == 0 or (delta_value == 0 and delta_time_sec <= intrpl_max*60):
                            value_list.append(value_prior)

                        # Zeitabstand zu groß - Keine lineare Interpolation möglich
                        elif delta_time_sec > intrpl_max*60:
                            value_list.append("nan")

                        # Lineare Interpolation
                        else:
                            delta_time_prior_sec = (time_list[i] - time_prior).total_seconds()
                            value_list.append(value_prior - delta_value/delta_time_sec*delta_time_prior_sec)

                        direct = 1

        # METHODE: ZEITLICH NÄCHSTLIEGENDER MESSWERT (matching original logic)
        elif mode_input == "nearest" or mode_input == "nearest (mean)":
            if tracker:
                tracker.emit('processing', 37, f'Započinjem {mode_input} procesiranje...', force=True)

            i_raw = 0  # Reset index counter

            # Schleife durchläuft alle Zeitschritte des kontinuierlichen Zeitstempels
            for i in range(0, len(time_list)):
                # Granularni progress update (30-85% range)
                if tracker and i % max(1, len(time_list) // 50) == 0:  # Update svakih ~2%
                    progress = 37 + (i / len(time_list)) * 49  # Map to 37-86%
                    tracker.processed_items = i
                    tracker.emit('processing', progress, f'{mode_input}: {i}/{len(time_list)} tačaka')

                try:
                    # Zeitgrenzen für die Untersuchung (Untersuchungsraum)
                    time_int_min = time_list[i] - datetime.timedelta(minutes=tss/2)
                    time_int_max = time_list[i] + datetime.timedelta(minutes=tss/2)

                    # Find values within the time window
                    value_int_list = []
                    delta_time_int_list = []

                    # Scan through data points within the time window
                    while i_raw < len(df):
                        current_time = df[utc_col_name].iloc[i_raw].to_pydatetime()

                        if current_time > time_int_max:
                            break

                        if current_time >= time_int_min:
                            if is_numeric(df[value_col_name].iloc[i_raw]):
                                value_int_list.append(float(df[value_col_name].iloc[i_raw]))
                                delta_time_int_list.append(abs((time_list[i] - current_time).total_seconds()))

                        i_raw += 1

                    # If we've moved past the window, back up to catch potential values in the next window
                    if i_raw > 0:
                        i_raw -= 1

                    # Process values based on mode
                    if value_int_list:
                        if mode_input == "nearest":
                            # Find the value with minimum time difference
                            min_time = min(delta_time_int_list)
                            min_idx = delta_time_int_list.index(min_time)
                            value_list.append(value_int_list[min_idx])
                        else:  # nearest (mean)
                            # Find all values with the minimum time difference
                            import statistics
                            min_time = min(delta_time_int_list)
                            nearest_values = [
                                value_int_list[idx]
                                for idx, delta in enumerate(delta_time_int_list)
                                if abs(delta - min_time) < 0.001  # Small tolerance for float comparison
                            ]
                            value_list.append(statistics.mean(nearest_values))
                    else:
                        value_list.append("nan")

                except Exception as e:
                    logger.error(f"Error processing time step {i}: {str(e)}")
                    value_list.append("nan")

        # DATENRAHMEN MIT DEN AUFBEREITETEN DATEN
        logger.info(f"Length of time_list: {len(time_list)}")
        logger.info(f"Length of value_list: {len(value_list)}")

        if tracker:
            tracker.emit('processing', 86, 'Procesiranje završeno ✓', force=True)
            tracker.end_phase('processing')  # Snimi stvarno vrijeme processing faze

        # === FAZA 4: FINALIZING (85-95%) ===
        if tracker:
            tracker.start_phase('finalizing')
            tracker.emit('finalizing', 87, 'Konverzija rezultata u JSON format...')

        # Create result dataframe
        result_df = pd.DataFrame({"UTC": time_list, value_col_name: value_list})

        # Format UTC column to desired format
        result_df['UTC'] = result_df['UTC'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))

        if tracker:
            tracker.emit('finalizing', 92, 'Formatiranje podataka...')

        # Convert to list of dicts
        result = result_df.apply(
            lambda row: {
                "UTC": row["UTC"],
                value_col_name: clean_for_json(row[value_col_name])
            },
            axis=1
        ).tolist()

        if tracker:
            tracker.emit('finalizing', 95, 'Finalizacija završena ✓')
            tracker.end_phase('finalizing')  # Snimi stvarno vrijeme finalizing faze

        # === FAZA 5: COMPRESSION (95-100%) ===
        if tracker:
            tracker.start_phase('compression')
            tracker.emit('compression', 96, 'Kompresija podataka...')

        result_json = json.dumps(result)
        compressed_data = gzip.compress(result_json.encode('utf-8'))

        if tracker:
            tracker.emit('complete', 100, f'Procesiranje uspješno završeno! 🎉 Generirano {len(result)} vremenskih tačaka.', force=True)

        response = Response(compressed_data)
        response.headers['Content-Encoding'] = 'gzip'
        response.headers['Content-Type'] = 'application/json'
        return response
    except Exception as e:
        error_msg = f"Error processing CSV: {str(e)}"
        logger.error(error_msg)
        return jsonify({"error": str(e)}), 400

@bp.route('/upload_chunk', methods=['POST'])
@require_auth
@require_subscription
@check_processing_limit
def upload_chunk():
    """
    Endpoint za prihvat i obradu CSV podataka u delovima (chunks).
    Očekivani parametri (form data):
      - uploadId: jedinstveni ID za upload (string)
      - chunkIndex: redni broj chunka (int, počinje od 0)
      - totalChunks: ukupan broj chunkova (int)
      - fileChunk: Blob/File sa delom CSV podataka
      - tss: Time step size u minutama (float)
      - offset: Offset u minutama (float)
      - mode: Način obrade ('mean', 'intrpl', 'nearest', 'nearest (mean)')
      - intrplMax: Maksimalno vreme za interpolaciju u minutama (float, default 60)
    """
    try:
        if 'fileChunk' not in request.files:
            return jsonify({"error": "No file chunk found"}), 400

        try:
            upload_id = request.form.get('uploadId')
            chunk_index = int(request.form.get('chunkIndex', 0))
            total_chunks = int(request.form.get('totalChunks', 0))
            tss = float(request.form.get('tss', 0))
            offset = float(request.form.get('offset', 0))
            mode = request.form.get('mode', '')
            intrpl_max = float(request.form.get('intrplMax', 60))
        except (ValueError, TypeError) as e:
            logger.error(f"Error parsing parameters: {e}")
            return jsonify({"error": f"Invalid parameter values: {str(e)}"}), 400

        if not all([upload_id, mode, tss > 0]):
            return jsonify({"error": "Missing required parameters"}), 400

        chunk = request.files['fileChunk']
        if not chunk:
            return jsonify({"error": "Empty chunk received"}), 400

        os.makedirs(UPLOAD_FOLDER, exist_ok=True)

        chunk_filename = os.path.join(UPLOAD_FOLDER, f"{upload_id}_{chunk_index}.chunk")
        chunk.save(chunk_filename)

        logger.info(f"Saved chunk {chunk_index + 1}/{total_chunks} for upload {upload_id}")

        received_chunks = [f for f in os.listdir(UPLOAD_FOLDER) 
                         if f.startswith(upload_id + "_")]

        if len(received_chunks) == total_chunks:
            logger.info(f"All chunks received for upload {upload_id}, processing...")

            # Izračunaj ukupnu veličinu fajla
            total_file_size = sum(
                os.path.getsize(os.path.join(UPLOAD_FOLDER, f))
                for f in received_chunks
            )

            # Inicijaliziraj ProgressTracker SA file size-om
            tracker = ProgressTracker(
                upload_id=upload_id,
                file_size_bytes=total_file_size,
                total_chunks=total_chunks
            )
            tracker.start_phase('chunk_assembly')
            tracker.emit('chunk_assembly', 0, f'Spajam {total_chunks} chunk-ova...', force=True)

            def extract_chunk_index(filename):
                try:
                    parts = filename.split("_")
                    chunk_part = parts[-1].split(".")[0]
                    return int(chunk_part)
                except (ValueError, IndexError) as e:
                    logger.error(f"Error parsing chunk filename {filename}: {e}")
                    return 0

            chunks_sorted = sorted(received_chunks, key=extract_chunk_index)

            try:

                full_content = ""
                logger.info(f"Assembling {len(chunks_sorted)} chunks: {chunks_sorted}")

                for i, chunk_file in enumerate(chunks_sorted):
                    chunk_path = os.path.join(UPLOAD_FOLDER, chunk_file)
                    logger.info(f"Processing chunk {i+1}/{len(chunks_sorted)}: {chunk_file}")

                    # Progress update za chunk assembly (0-10%)
                    chunk_progress = (i / len(chunks_sorted)) * 10
                    if i % max(1, len(chunks_sorted) // 10) == 0:  # Emit svakih 10%
                        tracker.emit('chunk_assembly', chunk_progress,
                                   f'Spajam chunk {i+1}/{len(chunks_sorted)}...')
                    
                    with open(chunk_path, 'rb') as f:
                        chunk_bytes = f.read()
                        logger.info(f"Chunk {i+1} size: {len(chunk_bytes)} bytes")
                        
                        try:
                            chunk_content = chunk_bytes.decode('utf-8')
                            logger.info(f"Chunk {i+1} decoded successfully, content length: {len(chunk_content)}")
                            
                            if i == 0:
                                first_lines = chunk_content.split('\n')[:3]
                                logger.info(f"First chunk first 3 lines: {first_lines}")
                            
                            if i == len(chunks_sorted) - 1:
                                last_lines = chunk_content.split('\n')[-3:]
                                logger.info(f"Last chunk last 3 lines: {last_lines}")
                                
                        except UnicodeDecodeError as decode_error:
                            logger.error(f"Failed to decode chunk {i+1}: {decode_error}")
                            for encoding in ['latin-1', 'cp1252', 'iso-8859-1']:
                                try:
                                    chunk_content = chunk_bytes.decode(encoding)
                                    logger.info(f"Successfully decoded chunk {i+1} with {encoding}")
                                    break
                                except:
                                    continue
                            else:
                                raise decode_error
                        
                        if i < len(chunks_sorted) - 1 and not chunk_content.endswith('\n'):
                            chunk_content += '\n'
                        
                        full_content += chunk_content
                    
                    os.remove(chunk_path)
                
                logger.info(f"Final assembled content length: {len(full_content)}")

                # Završi chunk assembly fazu
                tracker.end_phase('chunk_assembly')
                tracker.emit('chunk_assembly', 10, 'Chunk-ovi uspješno spojeni!', force=True)

                final_lines = full_content.split('\n')
                logger.info(f"Final content total lines: {len(final_lines)}")
                if len(final_lines) > 0:
                    logger.info(f"Final content first line: '{final_lines[0]}'")
                if len(final_lines) > 1:
                    logger.info(f"Final content second line: '{final_lines[1]}'")
                if len(final_lines) > 2:
                    logger.info(f"Final content third line: '{final_lines[2]}'")

                # Proslijedi tracker u process_csv (proces počinje od 10% jer je chunk assembly 0-10%)
                result = process_csv(full_content, tss, offset, mode, intrpl_max, upload_id, tracker)

                # Track processing and storage usage
                try:
                    increment_processing_count(g.user_id)
                    logger.info(f"✅ Tracked processing for user {g.user_id}")

                    # Track storage usage
                    file_size_bytes = len(full_content.encode('utf-8'))
                    file_size_mb = file_size_bytes / (1024 * 1024)
                    update_storage_usage(g.user_id, file_size_mb)
                    logger.info(f"✅ Tracked storage for user {g.user_id}: {file_size_mb:.2f} MB")
                except Exception as e:
                    logger.error(f"⚠️ Failed to track processing usage: {str(e)}")
                    # Don't fail the processing if tracking fails


                return result
                
            except Exception as e:
                for chunk_file in chunks_sorted:
                    try:
                        os.remove(os.path.join(UPLOAD_FOLDER, chunk_file))
                    except:
                        pass
                raise

        return jsonify({
            "message": f"Chunk {chunk_index + 1}/{total_chunks} received",
            "uploadId": upload_id,
            "chunkIndex": chunk_index,
            "totalChunks": total_chunks,
            "remainingChunks": total_chunks - len(received_chunks)
        }), 200

    except Exception as e:
        error_msg = f"Unexpected error in upload_chunk: {str(e)}\nTraceback: {traceback.format_exc()}"
        logger.error(error_msg)
        return jsonify({
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 400

@bp.route('/prepare-save', methods=['POST'])
def prepare_save():
    try:
        data = request.json
        if not data or 'data' not in data:
            return jsonify({"error": "No data received"}), 400
        data_wrapper = data['data']
        save_data = data_wrapper.get('data', [])
        file_name = data_wrapper.get('fileName', '')
        if not save_data:
            return jsonify({"error": "Empty data"}), 400

        temp_file = tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.csv')
        writer = csv.writer(temp_file, delimiter=';')
        for row in save_data:
            writer.writerow(row)
        temp_file.close()

        file_id = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        temp_files[file_id] = {
            'path': temp_file.name,
            'fileName': file_name or f"data_{file_id}.csv",
            'timestamp': time.time()
        }

        return jsonify({"message": "File prepared for download", "fileId": file_id}), 200
    except Exception as e:
        logger.error(f"Error in prepare_save: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@bp.route('/download/<file_id>', methods=['GET'])
def download_file(file_id):
    try:
        if file_id not in temp_files:
            return jsonify({"error": "File not found"}), 404
            
        file_info = temp_files[file_id]
        file_path = file_info['path']
        
        if not os.path.exists(file_path):
            return jsonify({"error": "File not found"}), 404

        download_name = file_info['fileName']
        
        response = send_file(
            file_path,
            as_attachment=True,
            download_name=download_name,
            mimetype='text/csv'
        )
        
        try:
            os.unlink(file_info['path'])
        except Exception as e:
            logger.error(f"Error cleaning up temp file: {e}")
        return response

    except Exception as e:
        logger.error(f"Error in download_file: {str(e)}")
        return jsonify({"error": str(e)}), 500
    finally:
        if file_id in temp_files:
            try:
                os.unlink(temp_files[file_id])
                del temp_files[file_id]
            except Exception as ex:
                logger.error(f"Error cleaning up temp file: {ex}")

