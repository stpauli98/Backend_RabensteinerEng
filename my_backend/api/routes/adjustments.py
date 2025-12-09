import pandas as pd
import numpy as np
import math
import os
import time
import csv
import logging
import traceback
from io import StringIO
from flask import request, jsonify, Blueprint, g
from shared.storage.service import storage_service
import json
from core.app_factory import socketio
from shared.auth.jwt import require_auth
from shared.auth.subscription import require_subscription, check_processing_limit
from shared.tracking.usage import increment_processing_count, update_storage_usage

bp = Blueprint('adjustmentsOfData_bp', __name__)

UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'temp_uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

adjustment_chunks = {}
adjustment_chunks_timestamps = {}
chunk_buffer = {}
chunk_buffer_timestamps = {}
stored_data = {}
stored_data_timestamps = {}
info_df_cache = {}
info_df_cache_timestamps = {}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

UTC_fmt = "%Y-%m-%d %H:%M:%S"

DATAFRAME_CHUNK_SIZE = 10000
CHUNK_BUFFER_TIMEOUT = 30 * 60
ADJUSTMENT_CHUNKS_TIMEOUT = 60 * 60
STORED_DATA_TIMEOUT = 60 * 60
INFO_CACHE_TIMEOUT = 2 * 60 * 60
SOCKETIO_CHUNK_DELAY = 0.1


# ============================================================
# VECTORIZED PROCESSING METHODS (identical to original logic)
# Schritt 4: data_adapt.py - matematički identične implementacije
# ============================================================

def _prepare_data_for_processing(df, col):
    """
    Priprema podataka - zajednička za sve metode.
    Filtrira NaN vrijednosti i konvertuje u numpy nizove.
    """
    df_clean = df[['UTC', col]].copy()
    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    df_clean = df_clean.dropna(subset=[col]).reset_index(drop=True)

    # Konvertuj UTC u int64 (sekunde od epoch-a)
    utc_values = df_clean['UTC'].values.astype('datetime64[s]').astype(np.int64)
    col_values = df_clean[col].values.astype(np.float64)

    return utc_values, col_values


def _generate_time_list(df, tss, ofst, start_time, end_time):
    """
    Generira kontinuirani timestamp - IDENTIČNO ORIGINALU (Schritt 4, linije 177-190)

    Original logika:
        t_ref = t_strt.replace(minute=0, second=0, microsecond=0)
        t_ref += timedelta(minutes=ofst)
        while t_ref < t_strt:
            t_ref += timedelta(minutes=tss)
        t_list = []
        while t_ref <= t_end:
            t_list.append(t_ref)
            t_ref += timedelta(minutes=tss)
    """
    from datetime import timedelta

    # Odredi granice
    if start_time is None:
        t_strt = df['UTC'].min()
    else:
        t_strt = pd.to_datetime(start_time)
        if t_strt.tzinfo is not None:
            t_strt = t_strt.tz_localize(None)

    if end_time is None:
        t_end = df['UTC'].max()
    else:
        t_end = pd.to_datetime(end_time)
        if t_end.tzinfo is not None:
            t_end = t_end.tz_localize(None)

    # Konvertuj u Python datetime ako je pandas Timestamp
    if hasattr(t_strt, 'to_pydatetime'):
        t_strt = t_strt.to_pydatetime()
    if hasattr(t_end, 'to_pydatetime'):
        t_end = t_end.to_pydatetime()

    # Originalna logika (Schritt 4, linije 181-190)
    t_ref = t_strt.replace(minute=0, second=0, microsecond=0)
    t_ref += timedelta(minutes=ofst)

    while t_ref < t_strt:
        t_ref += timedelta(minutes=tss)

    t_list = []
    while t_ref <= t_end:
        t_list.append(t_ref)
        t_ref += timedelta(minutes=tss)

    # Konvertuj u numpy int64 (sekunde) za brze operacije
    t_list_int = np.array([t.timestamp() for t in t_list], dtype=np.int64)

    return t_list, t_list_int


def _method_mean_vectorized(utc_values, col_values, t_list_int, tss):
    """
    MEAN metoda - klizni prozor ±tss/2
    IDENTIČNO ORIGINALU (Schritt 4, linije 269-305)

    Original logika:
        t_int_min = t_list[i1] - timedelta(minutes=tss/2)
        t_int_max = t_list[i1] + timedelta(minutes=tss/2)
        value_int_list = [values in window if not nan]
        result = mean(value_int_list) if len > 0 else nan
    """
    half_window = int(tss * 30)  # tss/2 u sekundama
    n = len(t_list_int)

    value_list = np.full(n, np.nan, dtype=np.float64)

    # Sortiraj za binary search
    sorted_idx = np.argsort(utc_values)
    utc_sorted = utc_values[sorted_idx]
    col_sorted = col_values[sorted_idx]

    for i in range(n):
        t = t_list_int[i]
        t_min = t - half_window
        t_max = t + half_window

        # Binary search O(log n)
        idx_start = np.searchsorted(utc_sorted, t_min, side='left')
        idx_end = np.searchsorted(utc_sorted, t_max, side='right')

        if idx_start < idx_end:
            window_values = col_sorted[idx_start:idx_end]
            valid = window_values[~np.isnan(window_values)]
            if len(valid) > 0:
                value_list[i] = np.mean(valid)

    return value_list


def _method_intrpl_vectorized(utc_values, col_values, t_list_int, intrpl_max):
    """
    INTRPL - bidirekciona pretraga + linearna interpolacija
    IDENTIČNO ORIGINALU (Schritt 4, linije 368-503)

    Original logika:
        1. Pronađi time_next (prvi numerički >= t)
        2. Pronađi time_prior (zadnji numerički <= t)
        3. Ako delta_time > intrpl_max*60 -> nan
        4. Inače linearna interpolacija:
           value = value_prior - delta_value/delta_time_sec * delta_time_prior_sec
    """
    intrpl_max_sec = intrpl_max * 60
    n = len(t_list_int)

    value_list = np.full(n, np.nan, dtype=np.float64)

    # Filtriraj samo numeričke (već filtrirano u _prepare_data_for_processing)
    valid_mask = ~np.isnan(col_values)
    utc_valid = utc_values[valid_mask]
    col_valid = col_values[valid_mask]

    if len(utc_valid) == 0:
        return value_list

    for i in range(n):
        t = t_list_int[i]

        # Pronađi idx gdje bi t bio umetnut
        idx = np.searchsorted(utc_valid, t)

        # time_next: prvi >= t
        if idx >= len(utc_valid):
            continue  # Nema next -> nan

        idx_next = idx
        if utc_valid[idx_next] < t:
            if idx_next + 1 < len(utc_valid):
                idx_next += 1
            else:
                continue

        time_next = utc_valid[idx_next]
        value_next = col_valid[idx_next]

        # time_prior: zadnji <= t
        idx_prior = idx - 1 if idx > 0 else 0
        if utc_valid[idx_prior] > t:
            continue  # Nema valid prior

        time_prior = utc_valid[idx_prior]
        value_prior = col_valid[idx_prior]

        # Delta kalkulacija
        delta_time_sec = time_next - time_prior
        delta_value = value_prior - value_next

        # Originalna logika (linije 470-481)
        if delta_time_sec == 0:
            value_list[i] = value_prior
        elif delta_value == 0 and delta_time_sec <= intrpl_max_sec:
            value_list[i] = value_prior
        elif delta_time_sec > intrpl_max_sec:
            continue  # nan - gap prevelik
        else:
            # Linearna interpolacija
            delta_time_prior_sec = t - time_prior
            value_list[i] = value_prior - delta_value / delta_time_sec * delta_time_prior_sec

    return value_list


def _method_nearest_vectorized(utc_values, col_values, t_list_int, tss):
    """
    NEAREST - najbliža vrijednost u prozoru ±tss/2
    IDENTIČNO ORIGINALU (Schritt 4, linije 312-362)

    Ako više vrijednosti ima istu min deltu, uzmi prvu (np.argmin vraća prvu)
    """
    half_window = int(tss * 30)  # tss/2 u sekundama
    n = len(t_list_int)

    value_list = np.full(n, np.nan, dtype=np.float64)

    valid_mask = ~np.isnan(col_values)
    utc_valid = utc_values[valid_mask]
    col_valid = col_values[valid_mask]

    if len(utc_valid) == 0:
        return value_list

    # Sortiraj za binary search
    sorted_idx = np.argsort(utc_valid)
    utc_sorted = utc_valid[sorted_idx]
    col_sorted = col_valid[sorted_idx]

    for i in range(n):
        t = t_list_int[i]
        t_min = t - half_window
        t_max = t + half_window

        idx_start = np.searchsorted(utc_sorted, t_min, side='left')
        idx_end = np.searchsorted(utc_sorted, t_max, side='right')

        if idx_start < idx_end:
            window_times = utc_sorted[idx_start:idx_end]
            window_values = col_sorted[idx_start:idx_end]

            deltas = np.abs(window_times - t)
            min_idx = np.argmin(deltas)  # Prvi ako ima više sa istom deltom
            value_list[i] = window_values[min_idx]

    return value_list


def _method_nearest_mean_vectorized(utc_values, col_values, t_list_int, tss):
    """
    NEAREST (MEAN) - prosjek svih vrijednosti sa istom minimalnom deltom
    IDENTIČNO ORIGINALU (Schritt 4, linije 351-359)
    """
    half_window = int(tss * 30)
    n = len(t_list_int)

    value_list = np.full(n, np.nan, dtype=np.float64)

    valid_mask = ~np.isnan(col_values)
    utc_valid = utc_values[valid_mask]
    col_valid = col_values[valid_mask]

    if len(utc_valid) == 0:
        return value_list

    sorted_idx = np.argsort(utc_valid)
    utc_sorted = utc_valid[sorted_idx]
    col_sorted = col_valid[sorted_idx]

    for i in range(n):
        t = t_list_int[i]
        t_min = t - half_window
        t_max = t + half_window

        idx_start = np.searchsorted(utc_sorted, t_min, side='left')
        idx_end = np.searchsorted(utc_sorted, t_max, side='right')

        if idx_start < idx_end:
            window_times = utc_sorted[idx_start:idx_end]
            window_values = col_sorted[idx_start:idx_end]

            deltas = np.abs(window_times - t)
            min_delta = np.min(deltas)

            # Svi sa min deltom (originalna logika - mean svih jednako udaljenih)
            min_mask = deltas == min_delta
            value_list[i] = np.mean(window_values[min_mask])

    return value_list


def _method_nearest_max_delta_vectorized(utc_values, col_values, t_list_int, nearest_max):
    """
    NEAREST (MAX. DELTA) - nearest vrijednost ako je gap <= max
    IDENTIČNO ORIGINALU (Schritt 4, linije 483-501)

    Razlika od intrpl: umjesto interpolacije, uzima bližu vrijednost (prior ili next)
    """
    nearest_max_sec = nearest_max * 60
    n = len(t_list_int)

    value_list = np.full(n, np.nan, dtype=np.float64)

    valid_mask = ~np.isnan(col_values)
    utc_valid = utc_values[valid_mask]
    col_valid = col_values[valid_mask]

    if len(utc_valid) == 0:
        return value_list

    for i in range(n):
        t = t_list_int[i]
        idx = np.searchsorted(utc_valid, t)

        # Prior i Next indeksi
        idx_next = min(idx, len(utc_valid) - 1)
        idx_prior = max(idx - 1, 0)

        # Korekcija ako next nije stvarno >= t
        if idx_next < len(utc_valid) and utc_valid[idx_next] < t:
            if idx_next + 1 < len(utc_valid):
                idx_next += 1

        # Validacija indeksa
        if idx_prior < 0 or idx_next >= len(utc_valid):
            continue

        time_prior = utc_valid[idx_prior]
        time_next = utc_valid[idx_next]
        value_prior = col_valid[idx_prior]
        value_next = col_valid[idx_next]

        # Dodatna provjera da prior <= t i next >= t
        if time_prior > t or time_next < t:
            continue

        delta_time_sec = time_next - time_prior
        delta_value = value_prior - value_next

        # Originalna logika (linije 485-501)
        if delta_time_sec == 0:
            value_list[i] = value_prior
        elif delta_value == 0 and delta_time_sec <= nearest_max_sec:
            value_list[i] = value_prior
        elif delta_time_sec > nearest_max_sec:
            continue  # nan - gap prevelik
        else:
            # Uzmi bližu vrijednost (ne interpoliraj!)
            delta_time_prior_sec = t - time_prior
            delta_time_next_sec = time_next - t

            if delta_time_prior_sec < delta_time_next_sec:
                value_list[i] = value_prior
            else:
                value_list[i] = value_next

    return value_list


class ProgressStages:
    """Constants for Socket.IO progress tracking"""
    # Upload faza (koristi se u upload-chunk endpointu)
    FILE_COMBINATION = 25
    FILE_ANALYSIS = 28
    FILE_COMPLETE = 30

    # Processing faza (koristi se u complete_adjustment)
    PARAMETER_PROCESSING = 5
    DATA_PROCESSING_START = 0
    FILE_PROCESSING_START = 10
    FILE_PROCESSING_END = 95
    COMPLETION = 100

    @staticmethod
    def calculate_file_progress(file_index, total_files, start=10, end=95):
        """Calculate progress percentage for file processing"""
        if total_files == 0:
            return start
        return start + (file_index / total_files) * (end - start)


class ProgressTracker:
    """
    Progress tracking sa ETA baziranim na dvije faze:
    1. Faza procesiranja (pandas) - spora, računa se po fajlu
    2. Faza streaminga (chunks) - brza, izračunava se unaprijed
    """

    # Konstante za procjenu streaming vremena
    # Bazirano na: emit vrijeme (~0.02s) + SOCKETIO_CHUNK_DELAY (0.1s) = ~0.12s po chunk-u
    ESTIMATED_TIME_PER_CHUNK = 0.13  # sekundi po chunk-u

    def __init__(self, upload_id, total_files=0, total_rows=0):
        self.upload_id = upload_id
        self.start_time = time.time()
        self.total_files = total_files
        self.total_rows = total_rows

        # File-level tracking
        self.files_processed = 0
        self.rows_processed = 0
        self.file_start_times = {}
        self.file_durations = []
        self.file_rows = {}

        # Current file tracking
        self.current_file = None
        self.current_file_rows = 0

        self.last_emit_time = 0
        self.emit_interval = 0.5

        # Za stabilizaciju ETA - čekaj minimum vremena
        self.min_elapsed_for_eta = 2.0  # sekunde

        # === FAZA 1: Procesiranje (pandas) ===
        self.processing_times = []  # Vrijeme procesiranja po fajlu
        self.current_processing_start = None

        # === FAZA 2: Streaming (chunks) ===
        self.current_streaming_start = None
        self.chunks_sent = 0
        self.total_chunks_for_file = 0

        # Ukupno procijenjeno streaming vrijeme za sve fajlove
        self.total_estimated_streaming_time = 0
        self.completed_streaming_time = 0  # Koliko je streaming vremena već prošlo
        self.current_file_streaming_estimate = 0  # Procjena za trenutni fajl

        # Trenutna faza: 'processing' ili 'streaming'
        self.current_phase = None

        # ETA tracking - čuva zadnju vrijednost da spriječimo rast
        self.last_eta = None

    def estimate_streaming_time(self, row_count):
        """Izračunaj procijenjeno vrijeme streaminga za fajl sa datim brojem redova"""
        total_chunks = (row_count + DATAFRAME_CHUNK_SIZE - 1) // DATAFRAME_CHUNK_SIZE
        return total_chunks * self.ESTIMATED_TIME_PER_CHUNK

    def set_file_rows(self, file_rows_dict):
        """Postavi broj redova po fajlu i izračunaj ukupno streaming vrijeme"""
        self.file_rows = file_rows_dict
        self.total_estimated_streaming_time = sum(
            self.estimate_streaming_time(rows) for rows in file_rows_dict.values()
        )

    def start_file(self, filename, row_count):
        """Započni procesiranje novog fajla"""
        self.current_file = filename
        self.current_file_rows = row_count
        self.file_start_times[filename] = time.time()
        self.file_rows[filename] = row_count
        # Počinje faza procesiranja
        self.current_processing_start = time.time()
        self.current_phase = 'processing'

    def start_streaming(self, filename, total_chunks):
        """Započni streaming fazu za fajl"""
        # Završi procesiranje fazu
        if self.current_processing_start:
            proc_time = time.time() - self.current_processing_start
            self.processing_times.append(proc_time)
            self.current_processing_start = None

        # Počni streaming fazu
        self.current_streaming_start = time.time()
        self.total_chunks_for_file = total_chunks
        self.chunks_sent = 0
        self.current_phase = 'streaming'

        # Izračunaj procijenjeno vrijeme za ovaj streaming
        self.current_file_streaming_estimate = total_chunks * self.ESTIMATED_TIME_PER_CHUNK

    def chunk_sent(self):
        """Zabilježi poslani chunk"""
        self.chunks_sent += 1

    def complete_file(self, filename):
        """Završi procesiranje fajla"""
        # Dodaj završeno streaming vrijeme
        if self.current_streaming_start:
            actual_stream_time = time.time() - self.current_streaming_start
            self.completed_streaming_time += actual_stream_time
            self.current_streaming_start = None

        if filename in self.file_start_times:
            duration = time.time() - self.file_start_times[filename]
            self.file_durations.append(duration)
            self.files_processed += 1
            self.rows_processed += self.file_rows.get(filename, 0)
            self.current_file = None
            self.current_file_rows = 0

        self.current_phase = None
        self.chunks_sent = 0
        self.total_chunks_for_file = 0
        self.current_file_streaming_estimate = 0

    def calculate_eta(self, current_progress):
        """
        ETA kalkulacija bazirana na fazama.

        Streaming vrijeme je poznato unaprijed (bazirano na broju chunk-ova).
        Procesiranje vrijeme se procjenjuje na osnovu prosjeka prethodnih fajlova.

        ETA = preostalo_procesiranje + preostalo_streaming
        """
        elapsed = time.time() - self.start_time

        # Čekaj minimum vremena za stabilnu procjenu
        if elapsed < self.min_elapsed_for_eta:
            return None

        if current_progress <= 0:
            return None

        # === 1. Preostalo STREAMING vrijeme ===
        # Već završeno streaming vrijeme za prethodne fajlove
        remaining_streaming = self.total_estimated_streaming_time - self.completed_streaming_time

        # Ako smo u streaming fazi, izračunaj preostalo za trenutni fajl
        if self.current_phase == 'streaming' and self.total_chunks_for_file > 0:
            # Koliko chunkova je ostalo za trenutni fajl
            chunks_remaining = self.total_chunks_for_file - self.chunks_sent
            # Preostalo vrijeme za trenutni fajl bazirano na procijenjenom vremenu po chunk-u
            current_file_remaining = chunks_remaining * self.ESTIMATED_TIME_PER_CHUNK
            
            # Oduzmi već procijenjeno vrijeme za ovaj fajl (jer je uključeno u total)
            # i dodaj stvarno preostalo
            remaining_streaming -= self.current_file_streaming_estimate
            remaining_streaming += current_file_remaining

        remaining_streaming = max(0, remaining_streaming)

        # === 2. Preostalo PROCESIRANJE vrijeme (procjena) ===
        files_remaining = self.total_files - self.files_processed
        if self.current_file:
            files_remaining -= 1  # Trenutni fajl je u toku

        remaining_processing = 0

        # Za preostale fajlove (ne uključujući trenutni)
        if files_remaining > 0 and self.processing_times:
            avg_processing = sum(self.processing_times) / len(self.processing_times)
            remaining_processing = files_remaining * avg_processing

        # Za trenutni fajl ako je u fazi procesiranja
        if self.current_phase == 'processing' and self.current_processing_start:
            time_in_processing = time.time() - self.current_processing_start
            if self.processing_times:
                avg_processing = sum(self.processing_times) / len(self.processing_times)
                remaining_processing += max(0, avg_processing - time_in_processing)
            else:
                # Nemamo podatke - pretpostavi da je ostalo koliko je već prošlo
                remaining_processing += time_in_processing

        # === 3. Ukupni ETA = preostalo procesiranje + preostalo streaming ===
        raw_eta = int(remaining_processing + remaining_streaming)

        # Ako je progress 100%, ETA je 0
        if current_progress >= 100:
            raw_eta = 0

        # Ako nemamo nikakve podatke, koristi fallback formulu
        if self.total_estimated_streaming_time == 0 and not self.processing_times:
            progress_ratio = current_progress / 100.0
            remaining_ratio = 1.0 - progress_ratio
            if progress_ratio > 0:
                raw_eta = int(elapsed * (remaining_ratio / progress_ratio))
            else:
                return None

        # ETA nikad ne smije rasti
        if self.last_eta is not None:
            raw_eta = min(raw_eta, self.last_eta)

        self.last_eta = raw_eta
        return max(0, raw_eta)

    def emit(self, progress, message_key, step, phase, message_params=None, detail_key=None, detail_params=None, force=False):
        """Šalje progress update sa ETA i i18n podrškom

        Args:
            progress: Progress postotak (0-100)
            message_key: i18n ključ za poruku (npr. 'processing_file')
            step: Trenutni korak procesiranja
            phase: Trenutna faza
            message_params: Parametri za interpolaciju glavne poruke
            detail_key: i18n ključ za detalje (opcionalno)
            detail_params: Parametri za interpolaciju detalja
            force: Forsira emit bez obzira na interval
        """
        current_time = time.time()

        if not force and (current_time - self.last_emit_time) < self.emit_interval:
            return

        payload = {
            'uploadId': self.upload_id,
            'progress': int(progress),
            'messageKey': message_key,
            'step': step,
            'phase': phase,
            'status': 'processing'
        }

        if message_params:
            payload['messageParams'] = message_params

        if detail_key:
            payload['detailKey'] = detail_key
        if detail_params:
            payload['detailParams'] = detail_params

        # Dodaj file tracking info
        if self.total_files > 0:
            # currentFile je fajl koji se trenutno obrađuje (1-based)
            # Ako nemamo current_file, znači da smo završili - prikaži zadnji fajl
            if self.current_file:
                payload['currentFile'] = self.files_processed + 1
                payload['currentFileName'] = self.current_file
            else:
                payload['currentFile'] = min(self.files_processed, self.total_files)

            payload['totalFiles'] = self.total_files

        # Izračunaj i dodaj ETA (preostalo vrijeme do kraja svih fajlova)
        eta = self.calculate_eta(progress)
        if eta is not None:
            payload['eta'] = eta
            payload['etaFormatted'] = self.format_time(eta)

        try:
            socketio.emit('processing_progress', payload, room=self.upload_id)
            self.last_emit_time = current_time

            # Fokusirani log: Fajl X/Y | Progress% | ETA
            file_info = f"Fajl {payload.get('currentFile', '?')}/{payload.get('totalFiles', '?')}"
            eta_info = payload.get('etaFormatted', 'N/A')
            logger.info(f"📊 {file_info} | {int(progress)}% | ETA: {eta_info} | key: {message_key}")
        except Exception as e:
            logger.error(f"Error emitting progress: {e}")

    @staticmethod
    def format_time(seconds):
        """Formatira sekunde u čitljiv format"""
        if seconds is None:
            return "Procjenjujem..."
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


def emit_progress(upload_id, progress, message_key, step, phase, message_params=None, detail_key=None, detail_params=None):
    """
    Emit Socket.IO progress update with error handling and i18n support

    Args:
        upload_id (str): Upload ID for the room
        progress (int/float): Progress percentage (0-100)
        message_key (str): i18n key for the message
        step (str): Current processing step
        phase (str): Current processing phase
        message_params (dict, optional): Parameters for message interpolation
        detail_key (str, optional): i18n key for detail message
        detail_params (dict, optional): Parameters for detail interpolation
    """
    try:
        data = {
            'uploadId': upload_id,
            'progress': progress,
            'messageKey': message_key,
            'step': step,
            'phase': phase,
            'status': 'processing'
        }
        if message_params:
            data['messageParams'] = message_params
        if detail_key:
            data['detailKey'] = detail_key
        if detail_params:
            data['detailParams'] = detail_params
        socketio.emit('processing_progress', data, room=upload_id)
    except Exception as e:
        logger.error(f"Failed to emit progress for {upload_id}: {e}")


def emit_file_result(upload_id, filename, result_data, info_record, file_index, total_files, tracker=None):
    """
    Emit file processing result via SocketIO with chunking for large datasets

    Args:
        upload_id (str): Upload ID for the room
        filename (str): Name of the processed file
        result_data (list): List of data records
        info_record (dict): File information record
        file_index (int): Index of current file
        total_files (int): Total number of files being processed
        tracker (ProgressTracker, optional): Progress tracker for smooth updates
    """
    try:
        # Izračunaj progress opseg za ovaj fajl
        file_start_progress = ProgressStages.FILE_PROCESSING_START + (file_index / total_files) * (ProgressStages.FILE_PROCESSING_END - ProgressStages.FILE_PROCESSING_START)
        file_end_progress = ProgressStages.FILE_PROCESSING_START + ((file_index + 1) / total_files) * (ProgressStages.FILE_PROCESSING_END - ProgressStages.FILE_PROCESSING_START)
        file_progress_range = file_end_progress - file_start_progress

        if len(result_data) <= DATAFRAME_CHUNK_SIZE:
            # Mali fajl - nema streaming faze, samo jedan emit
            if tracker:
                tracker.start_streaming(filename, 1)
                tracker.chunk_sent()
            socketio.emit('file_result', {
                'uploadId': upload_id,
                'filename': filename,
                'info_record': info_record,
                'dataframe_chunk': result_data,
                'fileIndex': file_index,
                'totalFiles': total_files,
                'chunked': False
            }, room=upload_id)
            logger.info(f"Emitted single file_result for {filename} ({len(result_data)} rows)")
        else:
            total_chunks = (len(result_data) + DATAFRAME_CHUNK_SIZE - 1) // DATAFRAME_CHUNK_SIZE

            # Započni streaming fazu u trackeru
            if tracker:
                tracker.start_streaming(filename, total_chunks)

            socketio.emit('file_result', {
                'uploadId': upload_id,
                'filename': filename,
                'info_record': info_record,
                'dataframe_chunk': [],
                'fileIndex': file_index,
                'totalFiles': total_files,
                'chunked': True,
                'totalChunks': total_chunks
            }, room=upload_id)

            for chunk_idx in range(total_chunks):
                start_idx = chunk_idx * DATAFRAME_CHUNK_SIZE
                end_idx = min((chunk_idx + 1) * DATAFRAME_CHUNK_SIZE, len(result_data))
                chunk = result_data[start_idx:end_idx]

                socketio.emit('dataframe_chunk', {
                    'uploadId': upload_id,
                    'filename': filename,
                    'chunk': chunk,
                    'chunkIndex': chunk_idx,
                    'totalChunks': total_chunks,
                    'fileIndex': file_index
                }, room=upload_id)

                # Zabilježi poslani chunk i emituj progress
                if tracker:
                    tracker.chunk_sent()
                    chunk_progress = (chunk_idx + 1) / total_chunks
                    smooth_progress = file_start_progress + (chunk_progress * file_progress_range * 0.8)  # 80% za streaming
                    tracker.emit(
                        smooth_progress,
                        'streaming_chunk',
                        'data_streaming',
                        'data_processing',
                        message_params={'current': chunk_idx + 1, 'total': total_chunks, 'filename': filename},
                        detail_key='detail_streaming_progress',
                        detail_params={'fileNum': file_index + 1, 'totalFiles': total_files, 'percentage': int(chunk_progress * 100)}
                    )

                time.sleep(SOCKETIO_CHUNK_DELAY)

                logger.info(f"Emitted chunk {chunk_idx + 1}/{total_chunks} for {filename} ({len(chunk)} rows)")

            socketio.emit('dataframe_complete', {
                'uploadId': upload_id,
                'filename': filename,
                'totalChunks': total_chunks,
                'totalRows': len(result_data)
            }, room=upload_id)
            logger.info(f"✅ Dataframe streaming complete for {filename} ({total_chunks} chunks, {len(result_data)} rows)")

    except Exception as e:
        logger.error(f"Failed to emit file_result for {filename}: {e}")
        emit_file_error(upload_id, filename, str(e))


def emit_file_error(upload_id, filename, error_message):
    """
    Emit file processing error via SocketIO

    Args:
        upload_id (str): Upload ID for the room
        filename (str): Name of the file that failed
        error_message (str): Error message
    """
    try:
        socketio.emit('file_error', {
            'uploadId': upload_id,
            'filename': filename,
            'error': error_message
        }, room=upload_id)
        logger.error(f"Emitted file_error for {filename}: {error_message}")
    except Exception as e:
        logger.error(f"Failed to emit file_error for {filename}: {e}")


def check_files_need_methods(filenames, time_step, offset, methods, file_info_cache_local=None):
    """
    Fast batch check if files need processing methods

    Args:
        file_info_cache_local: Upload-specific cache (Cloud Run compatible)

    Uses info_df_cache for O(1) lookup instead of pandas filtering O(n)

    Args:
        filenames (list): List of filenames to check
        time_step (float): Requested time step size
        offset (float): Requested offset
        methods (dict): Dictionary of methods per filename

    Returns:
        list: List of files needing methods with their info, or empty list if all OK
    """
    VALID_METHODS = {'mean', 'intrpl', 'nearest', 'nearest (mean)', 'nearest (max. delta)'}
    files_needing_methods = []

    for filename in filenames:
        file_info = None
        if file_info_cache_local:
            file_info = file_info_cache_local.get(filename)
        if not file_info:
            file_info = info_df_cache.get(filename)
        if not file_info:
            logger.warning(f"File {filename} not found in cache")
            continue

        file_time_step = file_info['timestep']
        file_offset = file_info['offset']

        requested_offset = offset
        if file_time_step > 0 and requested_offset >= file_time_step:
            requested_offset = requested_offset % file_time_step

        needs_processing = file_time_step != time_step or file_offset != requested_offset

        if needs_processing:
            method_info = methods.get(filename, {})
            method = method_info.get('method', '').strip() if isinstance(method_info, dict) else ''
            has_valid_method = method and method in VALID_METHODS

            if not has_valid_method:
                files_needing_methods.append({
                    "filename": filename,
                    "current_timestep": file_time_step,
                    "requested_timestep": time_step,
                    "current_offset": file_offset,
                    "requested_offset": requested_offset,
                    "valid_methods": list(VALID_METHODS)
                })

    return files_needing_methods


info_df = pd.DataFrame(columns=['Name der Datei', 'Name der Messreihe', 'Startzeit (UTC)', 'Endzeit (UTC)',
                                'Zeitschrittweite [min]', 'Offset [min]', 'Anzahl der Datenpunkte',
                                'Anzahl der numerischen Datenpunkte', 'Anteil an numerischen Datenpunkten'])


def detect_delimiter(file_content):
    """
    Detect the delimiter used in a CSV file contentt
    """
    delimiters = [';', ',', '\t']
    
    first_line = file_content.split('\n')[0]
    
    counts = {d: first_line.count(d) for d in delimiters}
    
    max_count = max(counts.values())
    if max_count > 0:
        return max(counts.items(), key=lambda x: x[1])[0]
    return ';'

def get_time_column(df):
    """
    Check if DataFrame has exactly 'UTC' column
    """
    if 'UTC' in df.columns:
        return 'UTC'
    return None

@bp.route('/upload-chunk', methods=['POST'])
@require_auth
@require_subscription
@check_processing_limit
def upload_chunk():
    """
    Endpoint za prihvat pojedinačnih chunkova.
    Očekivani parametri (form data):
      - uploadId: jedinstveni ID za upload (string)
      - chunkIndex: redni broj chanka (int, počinje od 0)
      - totalChunks: ukupan broj chunkova (int)
      - filename: originalno ime fajla
      - tss, offset, mode, intrplMax: dodatni parametri za obradu
      - files[]: sadržaj fajla kao file
    Ako su svi chunkovi primljeni, oni se spajaju i obrađuju.
    """
    try:
        upload_id = request.form.get('uploadId')
        chunk_index = int(request.form.get('chunkIndex'))
        total_chunks = int(request.form.get('totalChunks'))
        filename = request.form.get('filename')

        if not all([upload_id, isinstance(chunk_index, int), isinstance(total_chunks, int)]):
            return jsonify({"error": "Missing or invalid required parameters"}), 400

        try:
            filename = sanitize_filename(filename)
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
            
        if 'files[]' not in request.files:
            return jsonify({'error': 'No file part'}), 400
            
        file = request.files['files[]']
        if not file:
            return jsonify({'error': 'No selected file'}), 400
            
        file_content = file.read().decode('utf-8')
        if not file_content:
            return jsonify({'error': 'Empty file content'}), 400

        # Use composite key: upload_id + filename to handle multiple files per upload
        file_key = f"{upload_id}:{filename}"

        if file_key not in chunk_buffer:
            chunk_buffer[file_key] = {}
            chunk_buffer_timestamps[file_key] = time.time()

        chunk_buffer[file_key][chunk_index] = file_content

        if chunk_index == 0:
            cleanup_all_expired_data()

        received_chunks_count = len(chunk_buffer[file_key])

        if received_chunks_count == total_chunks:
            combined_content = ''.join(
                chunk_buffer[file_key][i] for i in range(total_chunks)
            )

            upload_dir = os.path.join(UPLOAD_FOLDER, upload_id)
            os.makedirs(upload_dir, exist_ok=True)

            final_path = os.path.join(upload_dir, filename)
            with open(final_path, 'w', encoding='utf-8') as outfile:
                outfile.write(combined_content)

            if file_key in chunk_buffer:
                del chunk_buffer[file_key]
            if file_key in chunk_buffer_timestamps:
                del chunk_buffer_timestamps[file_key]

            try:
                result = analyse_data(final_path, upload_id)

                # Track storage usage
                try:
                    file_size_bytes = len(combined_content.encode('utf-8'))
                    file_size_mb = file_size_bytes / (1024 * 1024)
                    update_storage_usage(g.user_id, file_size_mb)
                    logger.info(f"✅ Tracked storage for user {g.user_id}: {file_size_mb:.2f} MB")
                except Exception as e:
                    logger.error(f"⚠️ Failed to track storage usage: {str(e)}")
                
                response_data = {
                    'status': 'complete',
                    'message': 'File upload and analysis complete',
                    'success': True,
                    'data': result
                }
                return jsonify(response_data)
            except Exception as e:
                logger.error(f"Error analyzing file {final_path}: {str(e)}")
                return jsonify({"error": str(e)}), 500
            
        return jsonify({
            'status': 'chunk_received',
            'message': f'Received chunk {chunk_index + 1} of {total_chunks}',
            'chunksReceived': received_chunks_count
        })
        
    except Exception as e:
        traceback.print_exc()
        # Clean up file_key if it was created
        try:
            file_key = f"{upload_id}:{filename}"
            if file_key in chunk_buffer:
                del chunk_buffer[file_key]
            if file_key in chunk_buffer_timestamps:
                del chunk_buffer_timestamps[file_key]
        except Exception:
            pass  # upload_id or filename may not be defined yet
        return jsonify({"error": str(e)}), 400


def cleanup_expired_chunk_buffers():
    """Remove chunk buffers older than CHUNK_BUFFER_TIMEOUT"""
    current_time = time.time()
    expired_uploads = []

    for upload_id, timestamp in chunk_buffer_timestamps.items():
        if current_time - timestamp > CHUNK_BUFFER_TIMEOUT:
            expired_uploads.append(upload_id)

    for upload_id in expired_uploads:
        if upload_id in chunk_buffer:
            del chunk_buffer[upload_id]
            # Cleaned up expired chunk buffer
        if upload_id in chunk_buffer_timestamps:
            del chunk_buffer_timestamps[upload_id]

    return len(expired_uploads)


def cleanup_expired_adjustment_chunks():
    """Remove adjustment chunks older than ADJUSTMENT_CHUNKS_TIMEOUT"""
    current_time = time.time()
    expired_uploads = []

    for upload_id, timestamp in adjustment_chunks_timestamps.items():
        if current_time - timestamp > ADJUSTMENT_CHUNKS_TIMEOUT:
            expired_uploads.append(upload_id)

    for upload_id in expired_uploads:
        if upload_id in adjustment_chunks:
            del adjustment_chunks[upload_id]
            # Cleaned up expired adjustment chunks
        if upload_id in adjustment_chunks_timestamps:
            del adjustment_chunks_timestamps[upload_id]

    return len(expired_uploads)


def cleanup_expired_stored_data():
    """Remove stored data older than STORED_DATA_TIMEOUT"""
    current_time = time.time()
    expired_files = []

    for filename, timestamp in stored_data_timestamps.items():
        if current_time - timestamp > STORED_DATA_TIMEOUT:
            expired_files.append(filename)

    for filename in expired_files:
        if filename in stored_data:
            del stored_data[filename]
            # Cleaned up expired stored data
        if filename in stored_data_timestamps:
            del stored_data_timestamps[filename]

    return len(expired_files)


def cleanup_expired_info_cache():
    """Remove info cache entries older than INFO_CACHE_TIMEOUT"""
    current_time = time.time()
    expired_files = []

    for filename, timestamp in info_df_cache_timestamps.items():
        if current_time - timestamp > INFO_CACHE_TIMEOUT:
            expired_files.append(filename)

    for filename in expired_files:
        if filename in info_df_cache:
            del info_df_cache[filename]
            # Cleaned up expired info cache
        if filename in info_df_cache_timestamps:
            del info_df_cache_timestamps[filename]

    return len(expired_files)


def cleanup_all_expired_data():
    """Run all cleanup functions and return total cleaned items"""
    total = 0
    total += cleanup_expired_chunk_buffers()
    total += cleanup_expired_adjustment_chunks()
    total += cleanup_expired_stored_data()
    total += cleanup_expired_info_cache()

    # Cleanup completed silently

    return total


def sanitize_filename(filename):
    """
    Sanitize filename to prevent path traversal attacks

    Args:
        filename (str): User-provided filename

    Returns:
        str: Sanitized filename safe for filesystem operations

    Raises:
        ValueError: If filename is invalid or contains path traversal attempts
    """
    if not filename:
        raise ValueError("Filename cannot be empty")

    safe_filename = os.path.basename(filename)

    if '..' in safe_filename:
        raise ValueError("Invalid filename: path traversal detected")

    return safe_filename


def get_file_info_from_cache(filename, upload_id=None):
    """
    Helper function to retrieve file info from cache with fallback

    Args:
        filename (str): Filename to lookup
        upload_id (str, optional): Upload ID for upload-specific cache

    Returns:
        dict or None: File info dict or None if not found
    """
    if upload_id and upload_id in adjustment_chunks:
        file_info_cache_local = adjustment_chunks[upload_id].get('file_info_cache', {})
        file_info = file_info_cache_local.get(filename)
        if file_info:
            return file_info

    return info_df_cache.get(filename)


def analyse_data(file_path, upload_id=None):
    """
    Analyze CSV file and extract relevant information
    
    Args:
        file_path (str): Path to the CSV file to analyze
        upload_id (str, optional): ID of the upload if this is part of a chunked upload
    """
    try:
        global stored_data, info_df
        
        all_file_info = []
        processed_data = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                file_content = file.read()
        except UnicodeDecodeError as e:
            logger.error(f"UnicodeDecodeError reading {file_path}: {str(e)}")
            raise ValueError(f"Could not decode file {file_path}. Make sure it's a valid UTF-8 encoded CSV file.")
        
        delimiter = detect_delimiter(file_content)

        df = pd.read_csv(
            StringIO(file_content),
            delimiter=delimiter,
            engine='c',
            low_memory=False
        )

        time_col = get_time_column(df)
        if time_col is None:
            raise ValueError(f"No 'UTC' column found in file {os.path.basename(file_path)}. File must have a column named 'UTC'.")

        if len(df.columns) != 2:
            raise ValueError(f"File {os.path.basename(file_path)} must have exactly two columns: 'UTC' and one measurement column.")

        df['UTC'] = pd.to_datetime(df['UTC'], utc=True, cache=True)
                    
        filename = os.path.basename(file_path)
        stored_data[filename] = df
        stored_data_timestamps[filename] = time.time()
        
        if upload_id:
            if upload_id not in adjustment_chunks:
                adjustment_chunks[upload_id] = {'chunks': {}, 'params': {}, 'dataframes': {}}
                adjustment_chunks_timestamps[upload_id] = time.time()
            adjustment_chunks[upload_id]['dataframes'][filename] = df
                    
        time_step = None
        try:
            time_values = df['UTC'].values.astype('datetime64[s]')

            time_diffs_sec = np.diff(time_values.astype(np.int64))

            time_step = round(np.median(time_diffs_sec) / 60)
        except Exception as e:
            logger.error(f"Error calculating time step: {str(e)}")
            traceback.print_exc()
        
        measurement_col = None
        for col in df.columns:
            if col != 'UTC':
                measurement_col = col
                break

        if measurement_col:
            first_time = df['UTC'].iloc[0]
            offset = first_time.minute % time_step if time_step else 0.0
            
            file_info = {
                'Name der Datei': os.path.basename(file_path),
                'Name der Messreihe': str(measurement_col),
                'Startzeit (UTC)': df['UTC'].iloc[0].strftime(UTC_fmt) if 'UTC' in df.columns else None,
                'Endzeit (UTC)': df['UTC'].iloc[-1].strftime(UTC_fmt) if 'UTC' in df.columns else None,
                'Zeitschrittweite [min]': float(time_step) if time_step is not None else None,
                'Offset [min]': float(offset),
                'Anzahl der Datenpunkte': int(len(df)),
                'Anzahl der numerischen Datenpunkte': int(df[measurement_col].count()),
                'Anteil an numerischen Datenpunkten': float(df[measurement_col].count() / len(df) * 100)
            }
            all_file_info.append(file_info)
                    
        df_records = []
        filename = os.path.basename(file_path)
        for record in df.to_dict('records'):
            converted_record = {
                'Name der Datei': filename
            }
            for key, value in record.items():
                if pd.isna(value):
                    converted_record[key] = None
                elif isinstance(value, (pd.Timestamp, np.datetime64)):
                    converted_record[key] = value.strftime(UTC_fmt)
                elif isinstance(value, np.number):
                    converted_record[key] = float(value) if not pd.isna(value) else None
                else:
                    converted_record[key] = value
            df_records.append(converted_record)
        
        processed_data.append(df_records)
        
        if all_file_info:
            new_info_df = pd.DataFrame(all_file_info)
            if info_df.empty:
                info_df = new_info_df
            else:
                existing_files = new_info_df['Name der Datei'].tolist()
                info_df = info_df[~info_df['Name der Datei'].isin(existing_files)]
                info_df = pd.concat([info_df, new_info_df], ignore_index=True)

            if 'file_info_cache' not in adjustment_chunks[upload_id]:
                adjustment_chunks[upload_id]['file_info_cache'] = {}

            for file_info_item in all_file_info:
                filename_key = file_info_item['Name der Datei']
                file_info_data = {
                    'timestep': file_info_item['Zeitschrittweite [min]'],
                    'offset': file_info_item['Offset [min]'],
                    'start_time': file_info_item['Startzeit (UTC)'],
                    'end_time': file_info_item['Endzeit (UTC)'],
                    'measurement_col': file_info_item['Name der Messreihe']
                }
                info_df_cache[filename_key] = file_info_data
                info_df_cache_timestamps[filename_key] = time.time()
                adjustment_chunks[upload_id]['file_info_cache'][filename_key] = file_info_data

        return {
            'info_df': all_file_info,
            'upload_id': upload_id
        }
        
    except Exception as e:
        logger.error(f"Error in analyse_data: {str(e)}\n{traceback.format_exc()}")
        raise

@bp.route('/adjust-data-chunk', methods=['POST'])
@require_auth
@require_subscription
def adjust_data():
    try:
        global adjustment_chunks
        
        data = request.json
        if not data:
            return jsonify({"error": "No JSON data received"}), 400

        upload_id = data.get('upload_id')
        if not upload_id:
            return jsonify({"error": "upload_id is required"}), 400
            
        if upload_id not in adjustment_chunks:
            return jsonify({"error": f"No data found for upload ID: {upload_id}"}), 404
            
        dataframes = adjustment_chunks[upload_id]['dataframes']
        if not dataframes:
            return jsonify({"error": "No dataframes found for this upload"}), 404
        
        start_time = data.get('startTime')
        end_time = data.get('endTime')
        time_step_size = data.get('timeStepSize')
        offset = data.get('offset')
        decimal_precision = data.get('decimalPrecision', 'full')

        methods = data.get('methods', {})
        if not methods:
            methods = adjustment_chunks[upload_id]['params'].get('methods', {})

        intrpl_max_values = {}
        for filename, method_info in methods.items():
            if isinstance(method_info, dict) and 'intrpl_max' in method_info:
                try:
                    intrpl_max_values[filename] = float(method_info['intrpl_max'])
                except (TypeError, ValueError) as e:
                    logger.warning(f"Could not convert intrplMax for {filename}: {e}")
                    intrpl_max_values[filename] = None
        
        if upload_id not in adjustment_chunks:
            adjustment_chunks[upload_id] = {
                'params': {
            'startTime': start_time,
            'endTime': end_time,
            'timeStepSize': time_step_size,
                    'offset': offset,
                    'methods': methods,
                    'intrplMaxValues': intrpl_max_values,
                    'decimalPrecision': decimal_precision
                }
            }
        else:
            params = adjustment_chunks[upload_id]['params']
            if start_time is not None: params['startTime'] = start_time
            if end_time is not None: params['endTime'] = end_time
            if time_step_size is not None: params['timeStepSize'] = time_step_size
            if offset is not None: params['offset'] = offset
            if decimal_precision is not None: params['decimalPrecision'] = decimal_precision
            
            if 'methods' not in params:
                params['methods'] = {}
            if methods:
                params['methods'].update(methods)
            
            if 'intrplMaxValues' not in params:
                params['intrplMaxValues'] = {}
            params['intrplMaxValues'].update(intrpl_max_values)

        filenames = list(dataframes.keys())

        file_info_cache_local = adjustment_chunks[upload_id].get('file_info_cache', {})
        files_needing_methods = check_files_need_methods(
            filenames,
            time_step_size,
            offset,
            methods,
            file_info_cache_local
        )

        if files_needing_methods:
            return jsonify({
                "success": True,
                "methodsRequired": True,
                "hasValidMethod": False,
                "message": f"{len(files_needing_methods)} file(s) require processing method selection",
                "data": {
                    "info_df": files_needing_methods,
                    "dataframe": []
                }
            }), 200

        emit_progress(
            upload_id,
            ProgressStages.PARAMETER_PROCESSING,
            f'Processing parameters for {len(filenames)} files',
            'parameter_processing',
            'data_processing'
        )

        return jsonify({
            "message": "Parameters updated successfully",
            "files": filenames,
            "upload_id": upload_id
        }), 200

    except Exception as e:
        logger.error(f"Error in receive_adjustment_chunk: {str(e)}\n{traceback.format_exc()}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 400

@bp.route('/adjustdata/complete', methods=['POST'])
@require_auth
@require_subscription
@check_processing_limit
def complete_adjustment():
    """
    Ovaj endpoint se poziva kada su svi chunkovi poslani.
    Očekuje JSON payload s:
      - uploadId: jedinstveni ID za upload (string)
      - totalChunks: ukupan broj chunkova (int)
      - startTime: početno vrijeme (opciono)
      - endTime: završno vrijeme (opciono)
      - timeStepSize: veličina vremenskog koraka (opciono)
      - offset: pomak u minutama (opciono, default 0)
      - methods: metode za obradu podataka (opciono)
      - files: lista imena fajlova
    Nakon toga, backend kombinira sve primljene chunkove,
    obrađuje ih i vraća konačni rezultat.
    """
    if request.method == 'OPTIONS':
        response = jsonify({'success': True})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'POST,OPTIONS')
        return response, 200
    
    try:
        global adjustment_chunks
        
        data = request.json
        if not data:
            return jsonify({"error": "No JSON data received"}), 400

        upload_id = data.get('uploadId')
        
        if not upload_id:
            return jsonify({"error": "Missing uploadId"}), 400
            
        if upload_id not in adjustment_chunks:
            return jsonify({"error": "Upload ID not found"}), 404
        
        if 'methods' in data and data['methods']:
            adjustment_chunks[upload_id]['params']['methods'] = data['methods']
        
        params = adjustment_chunks[upload_id]['params']
        
        required_params = ['timeStepSize', 'offset']
        missing_params = [param for param in required_params if param not in params]
        
        if missing_params:
            return jsonify({
                "error": f"Missing required parameters: {', '.join(missing_params)}"
            }), 400
        
        requested_time_step = params['timeStepSize']
        requested_offset = params['offset']
        
        methods = params.get('methods', {})
        start_time = params.get('startTime')
        end_time = params.get('endTime')
        time_step = params.get('timeStepSize')
        offset = params.get('offset')
        decimal_precision = params.get('decimalPrecision', 'full')

        intrpl_max_values = params.get('intrplMaxValues', {})
        
        dataframes = adjustment_chunks[upload_id]['dataframes']
        if not dataframes:
            return jsonify({"error": "No data found for this upload ID"}), 404

        filenames = list(dataframes.keys())

        # Izračunaj ukupan broj redova za ETA procjenu
        total_rows = sum(len(df) for df in dataframes.values())

        # Izračunaj broj redova po fajlu za streaming procjenu
        file_rows_dict = {filename: len(df) for filename, df in dataframes.items()}

        # Inicijaliziraj ProgressTracker sa ETA
        tracker = ProgressTracker(
            upload_id=upload_id,
            total_files=len(filenames),
            total_rows=total_rows
        )

        # Postavi broj redova po fajlu za izračun streaming vremena unaprijed
        tracker.set_file_rows(file_rows_dict)

        tracker.emit(
            ProgressStages.DATA_PROCESSING_START,
            'processing_start',
            'data_processing_start',
            'data_processing',
            message_params={'fileCount': len(filenames), 'rowCount': f'{total_rows:,}'},
            force=True
        )

        if methods:
            methods = {k: v.strip() if isinstance(v, str) else v for k, v in methods.items()}

        VALID_METHODS = {'mean', 'intrpl', 'nearest', 'nearest (mean)', 'nearest (max. delta)'}

        files_needing_methods = []

        for filename in filenames:
            try:
                df = dataframes[filename]

                if 'UTC' not in df.columns:
                    error_msg = f"No UTC column found in file {filename}"
                    logger.error(error_msg)
                    emit_file_error(upload_id, filename, error_msg)
                    continue

                df['UTC'] = pd.to_datetime(df['UTC'])

                file_info = get_file_info_from_cache(filename, upload_id)

                if not file_info:
                    error_msg = f"File {filename} not found in cache"
                    logger.error(error_msg)
                    emit_file_error(upload_id, filename, error_msg)
                    continue

                file_time_step = float(file_info['timestep'])
                file_offset = float(file_info['offset'])

                # Konvertuj u float za sigurnu usporedbu
                time_step_float = float(time_step) if time_step is not None else 0.0
                offset_float = float(offset) if offset is not None else 0.0

                # Prilagodi offset ako je veći od timestep-a
                requested_offset_adjusted = offset_float
                if file_time_step > 0 and requested_offset_adjusted >= file_time_step:
                    requested_offset_adjusted = requested_offset_adjusted % file_time_step

                # Koristi math.isclose za float usporedbu (kao u originalnom kodu)
                timestep_matches = math.isclose(file_time_step, time_step_float, rel_tol=1e-9)
                offset_matches = math.isclose(file_offset, requested_offset_adjusted, rel_tol=1e-9)
                needs_processing = not (timestep_matches and offset_matches)

                if needs_processing:
                    method_info = methods.get(filename, {})
                    method = method_info.get('method', '').strip() if isinstance(method_info, dict) else ''
                    has_valid_method = method and method in VALID_METHODS



                    if not has_valid_method:
                        files_needing_methods.append({
                            "filename": filename,
                            "current_timestep": file_time_step,
                            "requested_timestep": time_step,
                            "current_offset": file_offset,
                            "requested_offset": requested_offset_adjusted,
                            "valid_methods": list(VALID_METHODS)
                        })

            except Exception as e:
                logger.error(f"Phase 1 error checking {filename}: {str(e)}")
                continue

        if files_needing_methods:
            return jsonify({
                "success": True,
                "methodsRequired": True,
                "hasValidMethod": False,
                "message": f"{len(files_needing_methods)} Datei(en) benötigen Verarbeitungsmethoden.",
                "data": {
                    "info_df": files_needing_methods,
                    "dataframe": []
                }
            }), 200



        for file_index, filename in enumerate(filenames):
            try:
                df = dataframes[filename]
                row_count = len(df)

                # Započni tracking za ovaj fajl
                tracker.start_file(filename, row_count)

                file_progress = ProgressStages.calculate_file_progress(file_index, len(filenames))
                tracker.emit(
                    file_progress,
                    'processing_file',
                    'file_analysis',
                    'data_processing',
                    message_params={'current': file_index + 1, 'total': len(filenames), 'filename': filename},
                    detail_key='detail_rows',
                    detail_params={'count': f'{row_count:,}'},
                    force=True
                )

                if 'UTC' not in df.columns:
                    error_msg = f"No UTC column found in file {filename}"
                    logger.error(error_msg)
                    emit_file_error(upload_id, filename, error_msg)
                    tracker.complete_file(filename)
                    continue

                df['UTC'] = pd.to_datetime(df['UTC'])

                file_info = get_file_info_from_cache(filename, upload_id)

                if not file_info:
                    error_msg = f"File {filename} not found in cache"
                    logger.error(error_msg)
                    emit_file_error(upload_id, filename, error_msg)
                    tracker.complete_file(filename)
                    continue

                file_time_step = float(file_info['timestep'])
                file_offset = float(file_info['offset'])

                # Konvertuj u float za sigurnu usporedbu
                time_step_float = float(time_step) if time_step is not None else 0.0
                offset_float = float(offset) if offset is not None else 0.0

                # Prilagodi offset ako je veći od timestep-a
                requested_offset_phase2 = offset_float
                if file_time_step > 0 and requested_offset_phase2 >= file_time_step:
                    requested_offset_phase2 = requested_offset_phase2 % file_time_step

                # Koristi math.isclose za float usporedbu (kao u originalnom kodu)
                timestep_matches = math.isclose(file_time_step, time_step_float, rel_tol=1e-9)
                offset_matches = math.isclose(file_offset, requested_offset_phase2, rel_tol=1e-9)
                needs_processing = not (timestep_matches and offset_matches)



                intrpl_max = intrpl_max_values.get(filename)

                if not needs_processing:
                    conversion_progress = ProgressStages.calculate_file_progress(file_index, len(filenames))
                    tracker.emit(
                        conversion_progress,
                        'conversion_direct',
                        'data_conversion',
                        'data_processing',
                        message_params={'filename': filename},
                        detail_key='detail_no_processing'
                    )


                    result_data, info_record = convert_data_without_processing(
                        dataframes[filename],
                        filename,
                        file_time_step,
                        file_offset,
                        decimal_precision
                    )
                else:
                    method_name = methods.get(filename, {}).get('method', 'default') if isinstance(methods.get(filename), dict) else 'default'
                    adjustment_progress = ProgressStages.calculate_file_progress(file_index, len(filenames))
                    tracker.emit(
                        adjustment_progress,
                        'processing_method',
                        'data_adjustment',
                        'data_processing',
                        message_params={'filename': filename, 'method': method_name},
                        detail_key='detail_timestep_change',
                        detail_params={'fromStep': file_time_step, 'toStep': time_step_float, 'fromOffset': file_offset, 'toOffset': offset_float}
                    )

                    # Koristi float verzije za procesiranje
                    process_time_step = time_step_float if needs_processing else file_time_step
                    process_offset = offset_float if needs_processing else file_offset

                    result_data, info_record = process_data_detailed(
                        dataframes[filename],
                        filename,
                        start_time,
                        end_time,
                        process_time_step,
                        process_offset,
                        methods,
                        intrpl_max,
                        decimal_precision
                    )

                if result_data is not None and info_record is not None:
                    emit_file_result(
                        upload_id,
                        filename,
                        result_data,
                        info_record,
                        file_index,
                        len(filenames),
                        tracker=tracker
                    )

                    # Završi tracking za ovaj fajl
                    tracker.complete_file(filename)

                    file_complete_progress = ProgressStages.calculate_file_progress(file_index + 1, len(filenames))

                    quality_percentage = 0
                    if info_record and 'Anteil an numerischen Datenpunkten' in info_record:
                        quality_percentage = info_record['Anteil an numerischen Datenpunkten']

                    # Poruka za završetak fajla
                    file_complete_message_params = {'filename': filename}
                    if needs_processing:
                        file_complete_message_params['fromStep'] = file_time_step
                        file_complete_message_params['toStep'] = time_step

                    # Detail za kvalitetu
                    detail_params = {'count': f'{len(result_data):,}'}
                    if quality_percentage > 0:
                        detail_params['percentage'] = f'{quality_percentage:.1f}'

                    tracker.emit(
                        file_complete_progress,
                        'file_complete' if not needs_processing else 'file_complete_with_change',
                        'file_complete',
                        'data_processing',
                        message_params=file_complete_message_params,
                        detail_key='detail_quality' if quality_percentage > 0 else 'detail_rows_generated',
                        detail_params=detail_params,
                        force=True
                    )

                    del result_data
                    del info_record
                    if filename in adjustment_chunks[upload_id]['dataframes']:
                        del adjustment_chunks[upload_id]['dataframes'][filename]

                    logger.info(f"Memory cleaned up for {filename}")

            except Exception as file_error:
                error_msg = f"Error processing {filename}: {str(file_error)}"
                logger.error(f"{error_msg}\n{traceback.format_exc()}")
                emit_file_error(upload_id, filename, error_msg)
                tracker.complete_file(filename)
                continue

        # Izračunaj ukupno vrijeme procesiranja
        total_processing_time = time.time() - tracker.start_time
        tracker.emit(
            ProgressStages.COMPLETION,
            'complete',
            'completion',
            'finalization',
            message_params={'duration': tracker.format_time(int(total_processing_time))},
            detail_key='detail_files_processed',
            detail_params={'fileCount': len(filenames)},
            force=True
        )

        # Track processing usage
        try:
            increment_processing_count(g.user_id)
            logger.info(f"✅ Tracked processing for user {g.user_id}")
            
            # Track storage usage - calculate total size from dataframes
            total_size_bytes = sum(
                df.memory_usage(deep=True).sum() 
                for df in dataframes.values()
            )
            file_size_mb = total_size_bytes / (1024 * 1024)
            update_storage_usage(g.user_id, file_size_mb)
            logger.info(f"✅ Tracked storage for user {g.user_id}: {file_size_mb:.2f} MB")
        except Exception as e:
            logger.error(f"⚠️ Failed to track processing usage: {str(e)}")
            # Don't fail the processing if tracking fails

        return jsonify({
            "success": True,
            "streaming": True,
            "totalFiles": len(filenames),
            "message": "Results sent via SocketIO streaming"
        }), 200

    except Exception as e:
        logger.error(f"Error in complete_adjustment: {str(e)}\n{traceback.format_exc()}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 400

def prepare_data(data, filename):
    """Priprema podataka za obradu"""
    df = data.copy()
    
    if len(df) == 0:
        raise ValueError(f"No data found for file {filename}")
    
    df['UTC'] = pd.to_datetime(df['UTC'])
    
    measurement_cols = [col for col in df.columns if col != 'UTC']
    if not measurement_cols:
        raise ValueError(f"No measurement columns found for file {filename}")
    
    for col in measurement_cols:
        df[f"{col}_original"] = df[col].copy()
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df, measurement_cols

def filter_by_time_range(df, start_time, end_time):
    """Filtriranje podataka po vremenskom rasponu"""
    if start_time and end_time:
        start_time = pd.to_datetime(start_time, utc=True)
        end_time = pd.to_datetime(end_time, utc=True)
        return df[(df['UTC'] >= start_time) & (df['UTC'] <= end_time)]
    return df

def get_method_for_file(methods, filename):
    """Dobijanje metode obrade za fajl"""
    method_info = methods.get(filename, {})
    if isinstance(method_info, dict):
        return method_info.get('method', '').strip()
    return None

def apply_processing_method(df, col, method, time_step, offset, start_time, end_time, intrpl_max=None, decimal_precision='full'):
    """
    REFACTORED: Koristi vektorizirane metode identične originalu (data_adapt Schritt 4)

    Metode:
    - mean: Klizni prozor ±tss/2, prosjek svih vrijednosti u prozoru
    - intrpl: Bidirekciona pretraga + linearna interpolacija sa intrpl_max limitom
    - nearest: Najbliža vrijednost u prozoru ±tss/2 (prva ako ima više sa istom deltom)
    - nearest (mean): Prosjek svih vrijednosti sa minimalnom deltom u prozoru
    - nearest (max. delta): Nearest sa max delta limitom (koristi intrpl_max parametar)

    Args:
        decimal_precision: Broj decimala za zaokruživanje ('full', '1', '2', '3', '4', '5', '6')
    """
    logger.info(f"[apply_processing_method] START - method={method}, col={col}, time_step={time_step}, offset={offset}, decimal_precision={decimal_precision}")
    
    # 1. Priprema DataFrame-a
    df = df.copy()
    df['UTC'] = pd.to_datetime(df['UTC']).dt.tz_localize(None)
    df = df.sort_values('UTC').drop_duplicates(subset=['UTC'], keep='first').reset_index(drop=True)
    
    # Get time parameters
    tss = time_step
    ofst = offset if offset is not None else 0
    
    logger.info(f"[apply_processing_method] Preparing data for vectorized processing...")
    
    # 2. Priprema podataka za vektorizirane operacije
    utc_values, col_values = _prepare_data_for_processing(df, col)
    
    if len(utc_values) == 0:
        logger.warning(f"[apply_processing_method] No valid numeric data for column {col}")
        return pd.DataFrame({'UTC': [], col: []})
    
    # 3. Generiraj kontinuirani timestamp (identično originalu)
    t_list, t_list_int = _generate_time_list(df, tss, ofst, start_time, end_time)
    
    if len(t_list) == 0:
        logger.warning(f"[apply_processing_method] No time points generated")
        return pd.DataFrame({'UTC': [], col: []})
    
    logger.info(f"[apply_processing_method] Generated {len(t_list)} time points, applying method: {method}")
    
    # 4. Pozovi odgovarajuću vektoriziranu metodu
    if method == 'mean':
        values = _method_mean_vectorized(utc_values, col_values, t_list_int, tss)
        
    elif method == 'intrpl':
        max_gap = intrpl_max if intrpl_max is not None else 60
        values = _method_intrpl_vectorized(utc_values, col_values, t_list_int, max_gap)
        
    elif method == 'nearest':
        values = _method_nearest_vectorized(utc_values, col_values, t_list_int, tss)
        
    elif method == 'nearest (mean)':
        values = _method_nearest_mean_vectorized(utc_values, col_values, t_list_int, tss)
        
    elif method == 'nearest (max. delta)':
        max_gap = intrpl_max if intrpl_max is not None else 60
        values = _method_nearest_max_delta_vectorized(utc_values, col_values, t_list_int, max_gap)
        
    elif method == 'max':
        # MAX metoda - koristi sliding window kao mean, ali uzima max
        logger.info(f"[apply_processing_method] Applying MAX (sliding window)...")
        half_window = int(tss * 30)
        n = len(t_list_int)
        values = np.full(n, np.nan, dtype=np.float64)
        
        sorted_idx = np.argsort(utc_values)
        utc_sorted = utc_values[sorted_idx]
        col_sorted = col_values[sorted_idx]
        
        for i in range(n):
            t = t_list_int[i]
            t_min = t - half_window
            t_max = t + half_window
            idx_start = np.searchsorted(utc_sorted, t_min, side='left')
            idx_end = np.searchsorted(utc_sorted, t_max, side='right')
            if idx_start < idx_end:
                window_values = col_sorted[idx_start:idx_end]
                valid = window_values[~np.isnan(window_values)]
                if len(valid) > 0:
                    values[i] = np.max(valid)
                    
    elif method == 'min':
        # MIN metoda - koristi sliding window kao mean, ali uzima min
        logger.info(f"[apply_processing_method] Applying MIN (sliding window)...")
        half_window = int(tss * 30)
        n = len(t_list_int)
        values = np.full(n, np.nan, dtype=np.float64)
        
        sorted_idx = np.argsort(utc_values)
        utc_sorted = utc_values[sorted_idx]
        col_sorted = col_values[sorted_idx]
        
        for i in range(n):
            t = t_list_int[i]
            t_min = t - half_window
            t_max = t + half_window
            idx_start = np.searchsorted(utc_sorted, t_min, side='left')
            idx_end = np.searchsorted(utc_sorted, t_max, side='right')
            if idx_start < idx_end:
                window_values = col_sorted[idx_start:idx_end]
                valid = window_values[~np.isnan(window_values)]
                if len(valid) > 0:
                    values[i] = np.min(valid)
    else:
        logger.warning(f"[apply_processing_method] Unknown method: {method}, returning original data")
        return df[['UTC', col]].copy()

    # 5. Apliciraj preciznost decimala
    if decimal_precision != 'full':
        try:
            precision = int(decimal_precision)
            # Zaokruži samo numeričke vrijednosti, ostavi NaN
            values = np.where(
                np.isnan(values),
                values,
                np.round(values, precision)
            )
            logger.info(f"[apply_processing_method] Applied decimal precision: {precision}")
        except (ValueError, TypeError) as e:
            logger.warning(f"[apply_processing_method] Could not apply decimal precision: {e}")

    # 6. Kreiraj rezultat DataFrame
    result_df = pd.DataFrame({
        'UTC': t_list,
        col: values
    })

    logger.info(f"[apply_processing_method] COMPLETE - returning {len(result_df)} rows")
    return result_df
def create_info_record(df, col, filename, time_step, offset):
    """Kreiranje info zapisa za rezultate"""
    total_points = len(df)
    numeric_points = df[col].count()
    numeric_ratio = (numeric_points / total_points * 100) if total_points > 0 else 0
    
    def format_utc(val):
        if pd.isnull(val):
            return None
        if hasattr(val, 'strftime'):
            return val.strftime(UTC_fmt)
        try:
            dt = pd.to_datetime(val)
            return dt.strftime(UTC_fmt)
        except Exception:
            return str(val)

    return {
        'Name der Datei': filename,
        'Name der Messreihe': col,
        'Startzeit (UTC)': format_utc(df['UTC'].iloc[0]) if len(df) > 0 else None,
        'Endzeit (UTC)': format_utc(df['UTC'].iloc[-1]) if len(df) > 0 else None,
        'Zeitschrittweite [min]': time_step,
        'Offset [min]': offset,
        'Anzahl der Datenpunkte': int(total_points),
        'Anzahl der numerischen Datenpunkte': int(numeric_points),
        'Anteil an numerischen Datenpunkten': float(numeric_ratio)
    }
def create_records(df, col, filename, decimal_precision='full'):
    """
    Konverzija DataFrame-a u zapise
    OPTIMIZATION #4: Vectorized numpy operations instead of iterrows (50-100x faster)

    Args:
        decimal_precision: Broj decimala za zaokruživanje ('full', '1', '2', '3', '4', '5', '6')
    """
    original_col = f"{col}_original"

    utc_values = pd.to_datetime(df['UTC']).values
    col_values = df[col].values

    utc_timestamps = (utc_values.astype('datetime64[ms]').astype(np.int64))

    has_original = original_col in df.columns
    original_values = df[original_col].values if has_original else None

    # Helper za zaokruživanje
    def apply_precision(value):
        if decimal_precision == 'full':
            return value
        try:
            return round(float(value), int(decimal_precision))
        except (ValueError, TypeError):
            return value

    records = []
    for idx in range(len(df)):
        utc_ts = int(utc_timestamps[idx])
        col_val = col_values[idx]

        if pd.notnull(col_val):
            value = apply_precision(float(col_val))
        elif has_original and pd.notnull(original_values[idx]):
            value = str(original_values[idx])
        else:
            value = "None"

        records.append({
            'UTC': utc_ts,
            col: value,
            'filename': filename
        })

    return records

def convert_data_without_processing(df, filename, time_step, offset, decimal_precision='full'):
    """
    Direktna konverzija podataka bez obrade kada su parametri isti.
    Ova funkcija preskače kompletan proces obrade i samo konvertuje podatke u format
    koji frontend očekuje, što značajno ubrzava proces kada nema potrebe za transformacijom.

    Args:
        decimal_precision: Broj decimala za zaokruživanje ('full', '1', '2', '3', '4', '5', '6')
    """
    try:
        df = df.copy()

        df['UTC'] = pd.to_datetime(df['UTC'])

        measurement_cols = [col for col in df.columns if col != 'UTC']

        if not measurement_cols:
            logger.warning(f"No measurement columns found for {filename}")
            return [], None

        all_records = []

        for col in measurement_cols:
            records = create_records(df, col, filename, decimal_precision)
            all_records.extend(records)

            if len(all_records) > 0 and not any(r.get('info_created') for r in all_records):
                info_record = create_info_record(df, col, filename, time_step, offset)
                return all_records, info_record

        if not all_records:
            return [], None

        info_record = create_info_record(df, measurement_cols[0], filename, time_step, offset)
        return all_records, info_record

    except Exception as e:
        logger.error(f"Error in convert_data_without_processing: {str(e)}")
        traceback.print_exc()
        return [], None

def process_data_detailed(data, filename, start_time=None, end_time=None, time_step=None, offset=None, methods={}, intrpl_max=None, decimal_precision='full'):
    """
    Args:
        decimal_precision: Broj decimala za zaokruživanje ('full', '1', '2', '3', '4', '5', '6')
    """
    try:
        df, measurement_cols = prepare_data(data, filename)

        df = filter_by_time_range(df, start_time, end_time)

        method = get_method_for_file(methods, filename)

        if not method:
            logger.warning(f"No processing method specified for {filename} but processing is required")
            return [], None

        all_info_records = []

        if len(measurement_cols) == 1:
            measurement_col = measurement_cols[0]

            processed_df = apply_processing_method(
                df, measurement_col, method, time_step, offset, start_time, end_time, intrpl_max, decimal_precision
            )

            records = create_records(processed_df, measurement_col, filename, decimal_precision)
            info_record = create_info_record(processed_df, measurement_col, filename, time_step, offset)

            return records, info_record

        combined_records = []

        for col in measurement_cols:
            processed_df = apply_processing_method(
                df, col, method, time_step, offset, start_time, end_time, intrpl_max, decimal_precision
            )

            records = create_records(processed_df, col, filename, decimal_precision)
            info_record = create_info_record(processed_df, col, filename, time_step, offset)

            combined_records.extend(records)
            all_info_records.append(info_record)

        return combined_records, all_info_records[0] if all_info_records else None

    except Exception as e:
        logger.error(f"Error in process_data_detailed: {str(e)}")
        traceback.print_exc()
        raise

