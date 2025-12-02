import csv
import json
import logging
import math
import os
import re
import tempfile
import time
from io import StringIO

import numpy as np
import pandas as pd
from flask import Blueprint, request, Response, jsonify, send_file, g, redirect
from core.extensions import socketio
from middleware.auth import require_auth
from middleware.subscription import require_subscription, check_processing_limit
from utils.usage_tracking import increment_processing_count, update_storage_usage
from utils.storage_service import storage_service

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

bp = Blueprint('data_processing', __name__)
UPLOAD_FOLDER = os.path.join(tempfile.gettempdir(), "upload_chunks")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

MAX_FILE_SIZE = 100 * 1024 * 1024
ALLOWED_EXTENSIONS = {'.csv', '.txt'}
MAX_CHUNK_SIZE = 10 * 1024 * 1024

# Progress faze za data processing
# 0-10%   → chunk_assembly
# 10-15%  → parsing
# 15-25%  → preprocessing
# 25-85%  → cleaning (GLAVNA FAZA - 7 koraka)
# 85-95%  → finalizing
# 95-100% → streaming

STREAMING_CHUNK_SIZE = 50000
TOTAL_CLEANING_STEPS = 7


class ProgressTracker:
    """
    Progress tracking sa ETA po koraku (Opcija 3).
    ETA se računa samo za trenutni korak - precizno i stabilno.
    """

    def __init__(self, upload_id, total_items=None, file_size_bytes=None, total_chunks=None):
        self.upload_id = upload_id
        self.start_time = time.time()
        self.phase_start_times = {}
        self.phase_durations = {}

        self.file_size_bytes = file_size_bytes
        self.total_chunks = total_chunks
        self.total_items = total_items

        self.last_emit_time = 0
        self.emit_interval = 0.5  # Emit svakih 500ms

        # Per-step ETA tracking
        self.current_step_start = None
        self.current_step_rows = 0
        self.current_step_processed = 0
        self.min_calibration_rows = 1000  # Čekaj 1000 redova za stabilnu procjenu

        # Step tracking za frontend
        self.current_step = 0
        self.total_steps = 0

    def start_phase(self, phase_name):
        """Označi početak nove faze procesiranja"""
        self.phase_start_times[phase_name] = time.time()

    def end_phase(self, phase_name):
        """Završi fazu i snimi stvarno vrijeme trajanja"""
        if phase_name in self.phase_start_times:
            duration = time.time() - self.phase_start_times[phase_name]
            self.phase_durations[phase_name] = duration
            logger.info(f"Phase '{phase_name}' completed in {duration:.2f}s")

    def start_step(self, total_rows):
        """Započni novi cleaning korak"""
        self.current_step_start = time.time()
        self.current_step_rows = total_rows
        self.current_step_processed = 0

    def update_step_progress(self, processed_rows):
        """Ažuriraj napredak trenutnog koraka"""
        self.current_step_processed = processed_rows

    def calculate_step_eta(self):
        """
        Izračunaj ETA samo za TRENUTNI korak.
        Vraća broj sekundi ili None ako još nema dovoljno podataka.
        """
        if not self.current_step_start or self.current_step_rows == 0:
            return None

        elapsed = time.time() - self.current_step_start
        remaining_rows = self.current_step_rows - self.current_step_processed

        # Čekaj minimalni broj redova za kalibraciju
        if self.current_step_processed < self.min_calibration_rows:
            return None

        if self.current_step_processed > 0:
            time_per_row = elapsed / self.current_step_processed
            eta = remaining_rows * time_per_row
            return max(0, int(eta))

        return None

    def emit(self, step, progress, message_key, eta_seconds=None, force=False, message_params=None):
        """
        Šalje progress update sa ključem za prevod.

        Args:
            step: Faza procesiranja (chunk_assembly, parsing, cleaning, etc.)
            progress: Procenat napretka (0-100)
            message_key: Ključ za prevod na frontendu (npr. 'gap_filling', 'outlier_removal')
            eta_seconds: ETA u sekundama (opciono)
            force: Ignoriši rate limiting
            message_params: Dodatni parametri za poruku (npr. {'count': 150, 'total': 1000})
        """
        current_time = time.time()

        if not force and (current_time - self.last_emit_time) < self.emit_interval:
            return

        # Odredi status na osnovu step-a
        if step == 'complete':
            status = 'completed'
        elif step == 'error':
            status = 'error'
        else:
            status = 'processing'

        payload = {
            'uploadId': self.upload_id,
            'step': step,
            'progress': int(progress),
            'messageKey': message_key,
            'status': status
        }

        # Dodaj parametre za poruku ako postoje
        if message_params:
            payload['messageParams'] = message_params

        # Dodaj currentStep i totalSteps za cleaning fazu
        if step == 'cleaning' and self.total_steps > 0:
            payload['currentStep'] = self.current_step
            payload['totalSteps'] = self.total_steps

        # Dodaj ETA ako je proslijeđen ili izračunaj za trenutni korak
        if eta_seconds is not None:
            payload['eta'] = eta_seconds
            payload['etaFormatted'] = self.format_time(eta_seconds)
        elif step == 'cleaning':
            step_eta = self.calculate_step_eta()
            if step_eta is not None:
                payload['eta'] = step_eta
                payload['etaFormatted'] = f"~{self.format_time(step_eta)}"

        try:
            socketio.emit('processing_progress', payload, room=self.upload_id)
            self.last_emit_time = current_time
            eta_text = f" (ETA: {payload.get('etaFormatted', '-')})" if 'etaFormatted' in payload else ""
            step_info = f" [{self.current_step}/{self.total_steps}]" if self.total_steps > 0 else ""
            params_text = f" {message_params}" if message_params else ""
            logger.info(f"Progress: {progress}%{step_info} - {message_key}{params_text}{eta_text}")
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

def secure_path_join(base_dir, user_input):
    """Safely join paths preventing directory traversal attacks"""
    if not user_input:
        raise ValueError("Empty path component")

    if '..' in user_input or '/' in user_input or '\\' in user_input:
        raise ValueError("Path traversal attempt detected")

    if '%2e%2e' in user_input.lower() or '%2f' in user_input.lower() or '%5c' in user_input.lower():
        raise ValueError("Encoded path traversal attempt detected")

    clean_input = os.path.basename(user_input)

    dangerous_chars = ['<', '>', ':', '"', '|', '?', '*', '\x00']
    for char in dangerous_chars:
        clean_input = clean_input.replace(char, '')

    if not clean_input or clean_input in ['.', '..', ''] or len(clean_input.strip()) == 0:
        raise ValueError("Invalid path component")

    full_path = os.path.join(base_dir, clean_input)

    base_real = os.path.realpath(base_dir)
    full_real = os.path.realpath(full_path)

    if not full_real.startswith(base_real + os.sep) and full_real != base_real:
        raise ValueError("Path traversal attempt detected")

    return full_path

def validate_processing_params(params):
    """Validate all numeric processing parameters"""
    validated = {}

    numeric_params = {
        'eqMax': (0, 1000000, "Elimination max duration"),
        'elMax': (-1000000, 1000000, "Upper limit value"),
        'elMin': (-1000000, 1000000, "Lower limit value"),
        'chgMax': (0, 1000000, "Change rate max"),
        'lgMax': (0, 1000000, "Length max"),
        'gapMax': (0, 1000000, "Gap max duration")
    }

    for param, (min_val, max_val, description) in numeric_params.items():
        if param in params and params[param] is not None and str(params[param]).strip() != '':
            try:
                value = float(params[param])
            except (ValueError, TypeError) as e:
                raise ValueError(f"Invalid parameter {param}: must be a valid number")

            if not (min_val <= value <= max_val):
                raise ValueError(f"{description} must be between {min_val} and {max_val}")
            validated[param] = value

    radio_params = ['radioValueNull', 'radioValueNotNull']
    for param in radio_params:
        if param in params:
            if params[param] in [None, '', 'undefined', 'null']:
                validated[param] = ''
            else:
                validated[param] = params[param]

    return validated

def validate_file_upload(file_chunk, filename):
    """Validate uploaded file security and format"""
    if not filename:
        raise ValueError("Empty filename")

    _, ext = os.path.splitext(filename.lower())
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {ext}. Only CSV and TXT files allowed.")

    chunk_data = file_chunk.read()
    if len(chunk_data) > MAX_CHUNK_SIZE:
        raise ValueError("Chunk size too large")

    file_chunk.seek(0)
    return chunk_data

def safe_error_response(error_msg, status_code=500, error_type=None):
    """Sanitize error messages to prevent information disclosure while preserving specific error types"""
    sanitized = re.sub(r'/[^\s]+', '[PATH]', str(error_msg))
    sanitized = sanitized.split('\n')[0]

    logger.error(f"Error occurred: {error_msg}")

    if error_type == 'validation' or 'validation' in str(error_msg).lower():
        if 'parameter' in str(error_msg).lower():
            return jsonify({"error": f"Parameter validation failed: {sanitized}"}), status_code
        elif 'csv' in str(error_msg).lower() or 'file format' in str(error_msg).lower():
            return jsonify({"error": f"CSV validation failed: {sanitized}"}), status_code
        elif 'file size' in str(error_msg).lower() or 'too large' in str(error_msg).lower():
            return jsonify({"error": f"File size validation failed: {sanitized}"}), status_code
        else:
            return jsonify({"error": f"Data validation failed: {sanitized}"}), status_code
    elif error_type == 'security' or 'security' in str(error_msg).lower():
        return jsonify({"error": f"Security validation failed: {sanitized}"}), status_code
    elif error_type == 'file' or status_code in [400, 413]:
        return jsonify({"error": sanitized}), status_code
    else:
        return jsonify({"error": "A processing error occurred. Please check your input and try again."}), status_code

def clean_data(df, value_column, params, tracker=None):
    """
    Čisti podatke prema zadanim parametrima sa per-step ETA praćenjem.

    Args:
        df: DataFrame sa podacima
        value_column: Naziv kolone sa vrijednostima
        params: Parametri čišćenja
        tracker: ProgressTracker instanca za praćenje napretka
    """
    logger.info("Starting data cleaning with parameters: %s", params)
    total_rows = len(df)

    # Izračunaj koje korake ćemo izvršiti (ključevi za prevod)
    active_steps = []
    if params.get("eqMax"):
        active_steps.append(("eqMax", "measurement_failure_removal"))
    if params.get("elMax") is not None:
        active_steps.append(("elMax", "upper_threshold_removal"))
    if params.get("elMin") is not None:
        active_steps.append(("elMin", "lower_threshold_removal"))
    if params.get("radioValueNull") == "ja":
        active_steps.append(("radioValueNull", "zero_value_removal"))
    if params.get("radioValueNotNull") == "ja":
        active_steps.append(("radioValueNotNull", "non_numeric_removal"))
    if params.get("chgMax") and params.get("lgMax"):
        active_steps.append(("chgMax", "outlier_removal"))
    if params.get("gapMax"):
        active_steps.append(("gapMax", "gap_filling"))

    total_active_steps = len(active_steps) if active_steps else 1
    current_step_index = 0

    # Postavi total_steps u tracker za frontend
    if tracker:
        tracker.total_steps = total_active_steps

    # Emit frekvencija - svakih ~2% koraka ili min 500 redova
    emit_frequency = max(500, total_rows // 50)

    def start_step(step_key):
        """Započni novi korak i resetiraj ETA tracking"""
        nonlocal current_step_index
        current_step_index += 1
        if tracker:
            tracker.current_step = current_step_index
            tracker.start_step(total_rows)
            # Cleaning faza: 25-90% (65% range), ravnomjerno podijeljeno po koracima
            progress = 25 + ((current_step_index - 1) / total_active_steps) * 65
            tracker.emit('cleaning', progress, step_key, force=True)

    def update_progress(step_key, iteration_in_step):
        """Ažuriraj progress unutar koraka - sa per-step ETA"""
        if tracker:
            tracker.update_step_progress(iteration_in_step)

            # Emit na emit_frequency interval
            if iteration_in_step % emit_frequency == 0 and iteration_in_step > 0:
                # Progress unutar cleaning faze (25-90%)
                step_progress = iteration_in_step / total_rows
                base_progress = 25 + ((current_step_index - 1) / total_active_steps) * 65
                step_range = 65 / total_active_steps
                progress = base_progress + (step_progress * step_range)
                progress = min(progress, 90)

                tracker.emit('cleaning', progress, step_key)

    def emit_step_complete(step_key, removed_count=None):
        """Emit kada je korak završen"""
        if tracker:
            progress = 25 + (current_step_index / total_active_steps) * 65
            progress = min(progress, 90)
            # Šalje ključ sa _complete sufiksom i count parametrom
            complete_key = f"{step_key}_complete"
            params = {'count': removed_count} if removed_count is not None else None
            tracker.emit('cleaning', progress, complete_key, force=True, message_params=params)

    df["UTC"] = pd.to_datetime(df["UTC"], format="%Y-%m-%d %H:%M:%S")

    # === KORAK 1: EQ_MAX ===
    if params.get("eqMax"):
        step_key = "measurement_failure_removal"
        start_step(step_key)
        logger.info("Removing measurement failures (identical consecutive values)")
        eq_max = float(params["eqMax"])
        frm = 0
        removed_count = 0
        for i in range(1, len(df)):
            update_progress(step_key, i)
            if df.iloc[i-1][value_column] == df.iloc[i][value_column] and frm == 0:
                idx_strt = i-1
                frm = 1
            elif df.iloc[i-1][value_column] != df.iloc[i][value_column] and frm == 1:
                idx_end = i-1
                frm_width = (df.iloc[idx_end]["UTC"] - df.iloc[idx_strt]["UTC"]).total_seconds() / 60
                if frm_width >= eq_max:
                    for i_frm in range(idx_strt, idx_end+1):
                        df.iloc[i_frm, df.columns.get_loc(value_column)] = np.nan
                        removed_count += 1
                frm = 0
            elif i == len(df)-1 and frm == 1:
                idx_end = i
                frm_width = (df.iloc[idx_end]["UTC"] - df.iloc[idx_strt]["UTC"]).total_seconds() / 60
                if frm_width >= eq_max:
                    for i_frm in range(idx_strt, idx_end+1):
                        df.iloc[i_frm, df.columns.get_loc(value_column)] = np.nan
                        removed_count += 1
        emit_step_complete(step_key, removed_count)

    # === KORAK 2: EL_MAX ===
    if params.get("elMax") is not None:
        step_key = "upper_threshold_removal"
        start_step(step_key)
        logger.info("Removing values above upper threshold")
        el_max = float(params["elMax"])
        removed_count = 0
        for i in range(len(df)):
            update_progress(step_key, i)
            try:
                if float(df.iloc[i][value_column]) > el_max:
                    df.iloc[i, df.columns.get_loc(value_column)] = np.nan
                    removed_count += 1
            except:
                df.iloc[i, df.columns.get_loc(value_column)] = np.nan
                removed_count += 1
        emit_step_complete(step_key, removed_count)

    # === KORAK 3: EL_MIN ===
    if params.get("elMin") is not None:
        step_key = "lower_threshold_removal"
        start_step(step_key)
        logger.info("Removing values below lower threshold")
        el_min = float(params["elMin"])
        removed_count = 0
        for i in range(len(df)):
            update_progress(step_key, i)
            try:
                if float(df.iloc[i][value_column]) < el_min:
                    df.iloc[i, df.columns.get_loc(value_column)] = np.nan
                    removed_count += 1
            except:
                df.iloc[i, df.columns.get_loc(value_column)] = np.nan
                removed_count += 1
        emit_step_complete(step_key, removed_count)

    # === KORAK 4: RADIO_VALUE_NULL ===
    if params.get("radioValueNull") == "ja":
        step_key = "zero_value_removal"
        start_step(step_key)
        logger.info("Removing null values")
        removed_count = 0
        for i in range(len(df)):
            update_progress(step_key, i)
            if df.iloc[i][value_column] == 0:
                df.iloc[i, df.columns.get_loc(value_column)] = np.nan
                removed_count += 1
        emit_step_complete(step_key, removed_count)

    # === KORAK 5: RADIO_VALUE_NOT_NULL ===
    if params.get("radioValueNotNull") == "ja":
        step_key = "non_numeric_removal"
        start_step(step_key)
        logger.info("Removing non-numeric values")
        removed_count = 0
        for i in range(len(df)):
            update_progress(step_key, i)
            try:
                float(df.iloc[i][value_column])
                if math.isnan(float(df.iloc[i][value_column])) == True:
                    df.iloc[i, df.columns.get_loc(value_column)] = np.nan
                    removed_count += 1
            except:
                df.iloc[i, df.columns.get_loc(value_column)] = np.nan
                removed_count += 1
        emit_step_complete(step_key, removed_count)

    # === KORAK 6: CHG_MAX + LG_MAX ===
    if params.get("chgMax") and params.get("lgMax"):
        step_key = "outlier_removal"
        start_step(step_key)
        logger.info("Removing outliers")
        chg_max = float(params["chgMax"])
        lg_max = float(params["lgMax"])
        frm = 0
        removed_count = 0
        for i in range(1, len(df)):
            update_progress(step_key, i)
            if pd.isna(df.iloc[i][value_column]) and frm == 0:
                pass
            elif pd.isna(df.iloc[i][value_column]) and frm == 1:
                idx_end = i-1
                for i_frm in range(idx_strt, idx_end+1):
                    df.iloc[i_frm, df.columns.get_loc(value_column)] = np.nan
                    removed_count += 1
                frm = 0
            elif pd.isna(df.iloc[i-1][value_column]):
                pass
            else:
                chg = abs(float(df.iloc[i][value_column]) - float(df.iloc[i-1][value_column]))
                t = (df.iloc[i]["UTC"] - df.iloc[i-1]["UTC"]).total_seconds() / 60
                if t > 0 and chg/t > chg_max and frm == 0:
                    idx_strt = i
                    frm = 1
                elif t > 0 and chg/t > chg_max and frm == 1:
                    idx_end = i-1
                    for i_frm in range(idx_strt, idx_end+1):
                        df.iloc[i_frm, df.columns.get_loc(value_column)] = np.nan
                        removed_count += 1
                    frm = 0
                elif frm == 1 and (df.iloc[i]["UTC"] - df.iloc[idx_strt]["UTC"]).total_seconds() / 60 > lg_max:
                    frm = 0
        emit_step_complete(step_key, removed_count)

    # === KORAK 7: GAP_MAX ===
    if params.get("gapMax"):
        step_key = "gap_filling"
        start_step(step_key)
        logger.info("Filling measurement gaps")
        gap_max = float(params["gapMax"])
        frm = 0
        filled_count = 0
        for i in range(1, len(df)):
            update_progress(step_key, i)
            if pd.isna(df.iloc[i][value_column]) and frm == 0:
                idx_strt = i
                frm = 1
            elif not pd.isna(df.iloc[i][value_column]) and frm == 1:
                idx_end = i-1
                frm_width = (df.iloc[idx_end+1]["UTC"] - df.iloc[idx_strt-1]["UTC"]).total_seconds() / 60
                if frm_width <= gap_max and frm_width > 0:
                    dif = float(df.iloc[idx_end+1][value_column]) - float(df.iloc[idx_strt-1][value_column])
                    dif_min = dif/frm_width
                    for i_frm in range(idx_strt, idx_end+1):
                        gap_min = (df.iloc[i_frm]["UTC"] - df.iloc[idx_strt-1]["UTC"]).total_seconds() / 60
                        df.iloc[i_frm, df.columns.get_loc(value_column)] = float(df.iloc[idx_strt-1][value_column]) + gap_min*dif_min
                        filled_count += 1
                frm = 0
        emit_step_complete(step_key, filled_count)

    # === POST-VALIDACIJA ===
    if params.get("elMin") is not None:
        el_min = float(params["elMin"])
        final_violations_min = (df[value_column] < el_min).sum()
        zero_values = (df[value_column] == 0).sum()
        logger.info(f"Final validation: Found {final_violations_min} values < {el_min} and {zero_values} zero values")
        if final_violations_min > 0:
            logger.info(f"Removing {final_violations_min} interpolated values below elMin threshold")
            df.loc[df[value_column] < el_min, value_column] = np.nan

    if params.get("elMax") is not None:
        el_max = float(params["elMax"])
        final_violations_max = (df[value_column] > el_max).sum()
        if final_violations_max > 0:
            logger.info(f"Removing {final_violations_max} interpolated values above elMax threshold")
            df.loc[df[value_column] > el_max, value_column] = np.nan

    logger.info("Data cleaning completed")
    return df

def _combine_chunks_efficiently(upload_dir, total_chunks):
    """Memory-efficient chunk combination using temporary file streaming"""
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    total_size = 0

    try:
        for i in range(total_chunks):
            chunk_path = os.path.join(upload_dir, f"chunk_{i:04d}.part")
            if not os.path.exists(chunk_path):
                temp_file.close()
                os.unlink(temp_file.name)
                raise FileNotFoundError(f"Missing chunk file: {i}")

            with open(chunk_path, "rb") as chunk_file:
                while True:
                    block = chunk_file.read(8192)
                    if not block:
                        break
                    temp_file.write(block)
                    total_size += len(block)

                    if total_size > MAX_FILE_SIZE:
                        temp_file.close()
                        os.unlink(temp_file.name)
                        raise ValueError(f"Combined file size exceeds {MAX_FILE_SIZE} bytes")

        temp_file.close()
        return temp_file.name, total_size

    except Exception:
        if not temp_file.closed:
            temp_file.close()
        if os.path.exists(temp_file.name):
            os.unlink(temp_file.name)
        raise


@bp.route("/api/dataProcessingMain/upload-chunk", methods=["POST"])
@require_auth
@require_subscription
@check_processing_limit
def upload_chunk():
    """
    Endpoint za prihvat i obradu CSV podataka u dijelovima (chunks).
    Koristi ProgressTracker za real-time praćenje napretka sa ETA kalkulacijom.
    """
    tracker = None

    try:
        logger.info(f"Processing chunk {request.form.get('chunkIndex')}/{request.form.get('totalChunks')}")

        if not all(key in request.form for key in ["uploadId", "chunkIndex", "totalChunks"]):
            missing_fields = [key for key in ["uploadId", "chunkIndex", "totalChunks"] if key not in request.form]
            logger.error(f"Missing required fields: {missing_fields}")
            return jsonify({"error": f"Missing required fields: {missing_fields}"}), 400

        upload_id = request.form["uploadId"]
        chunk_index = int(request.form["chunkIndex"])
        total_chunks = int(request.form["totalChunks"])

        if 'fileChunk' not in request.files:
            logger.error("No fileChunk in request.files")
            return safe_error_response("No file chunk provided", 400)

        file_chunk = request.files['fileChunk']
        if file_chunk.filename == '':
            logger.error("Empty filename in fileChunk")
            return safe_error_response("Empty filename", 400)

        try:
            chunk = validate_file_upload(file_chunk, file_chunk.filename)
            chunk_size = len(chunk)
        except ValueError as e:
            logger.error(f"File validation failed: {e}")
            return safe_error_response(str(e), 400, 'validation')

        if chunk_size == 0:
            logger.error("Received empty chunk")
            return jsonify({"error": "Empty chunk received"}), 400

        try:
            upload_dir = secure_path_join(UPLOAD_FOLDER, upload_id)
            os.makedirs(upload_dir, exist_ok=True)
        except ValueError as e:
            logger.error(f"Invalid upload_id: {upload_id}")
            return safe_error_response("Invalid upload identifier", 400, 'security')

        chunk_path = os.path.join(upload_dir, f"chunk_{chunk_index:04d}.part")

        with open(chunk_path, "wb") as f:
            f.write(chunk)

        if chunk_index < total_chunks - 1:
            return jsonify({"status": "chunk received", "chunkIndex": chunk_index})

        # === ZADNJI CHUNK PRIMLJEN - PROVJERI DA LI SU SVI CHUNKOVI NA DISKU ===
        # Kod paralelnog uploada, zadnji chunk može stići prije ostalih
        missing_chunks = []
        for i in range(total_chunks):
            part_path = os.path.join(upload_dir, f"chunk_{i:04d}.part")
            if not os.path.exists(part_path):
                missing_chunks.append(i)

        if missing_chunks:
            # Kod paralelnog uploada, zadnji chunk može stići prije ostalih
            # Čekaj kratko da ostali chunkovi stignu
            import time as time_module
            max_wait = 5  # sekundi
            wait_interval = 0.2  # sekundi
            waited = 0

            while waited < max_wait and missing_chunks:
                time_module.sleep(wait_interval)
                waited += wait_interval
                # Provjeri ponovo
                missing_chunks = [
                    i for i in range(total_chunks)
                    if not os.path.exists(os.path.join(upload_dir, f"chunk_{i:04d}.part"))
                ]

            if missing_chunks:
                logger.error(f"Timeout waiting for chunks: {missing_chunks}")
                return jsonify({
                    "error": "Missing chunks after timeout",
                    "missingChunks": missing_chunks
                }), 400

            logger.info(f"All chunks arrived after {waited:.1f}s wait")

        # === SVI CHUNK-OVI PRIMLJENI - POČINJE PROCESIRANJE ===
        logger.info("All chunks received, starting processing...")

        # Izračunaj ukupnu veličinu fajla
        total_file_size = sum(
            os.path.getsize(os.path.join(upload_dir, f"chunk_{i:04d}.part"))
            for i in range(total_chunks)
            if os.path.exists(os.path.join(upload_dir, f"chunk_{i:04d}.part"))
        )

        # Inicijaliziraj ProgressTracker
        tracker = ProgressTracker(
            upload_id=upload_id,
            file_size_bytes=total_file_size,
            total_chunks=total_chunks
        )

        # === FAZA 1: CHUNK ASSEMBLY (0-10%) ===
        tracker.start_phase('chunk_assembly')
        tracker.emit('chunk_assembly', 0, 'chunk_assembly_start', force=True, message_params={'totalChunks': total_chunks})

        try:
            combined_file_path, total_size = _combine_chunks_efficiently(upload_dir, total_chunks)
            tracker.end_phase('chunk_assembly')
            tracker.emit('chunk_assembly', 10, 'chunk_assembly_complete', force=True)
        except (FileNotFoundError, ValueError) as e:
            logger.error(f"Chunk combination failed: {e}")
            for i in range(total_chunks):
                part_path = os.path.join(upload_dir, f"chunk_{i:04d}.part")
                try:
                    if os.path.exists(part_path):
                        os.remove(part_path)
                except OSError:
                    pass

            if "exceeds" in str(e):
                return safe_error_response("File too large", 413, 'validation')
            else:
                return jsonify({"error": "Missing chunk file"}), 400

        if total_size == 0:
            logger.error("No data in combined chunks")
            os.unlink(combined_file_path)
            return jsonify({"error": "No data in combined chunks"}), 400

        try:
            # === FAZA 2: PARSING (10-15%) ===
            tracker.start_phase('parsing')
            tracker.emit('parsing', 10, 'parsing_start', force=True)

            with open(combined_file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            os.unlink(combined_file_path)
            lines = content.splitlines()

            tracker.emit('parsing', 12, 'parsing_lines_loaded', message_params={'lineCount': len(lines)})

            if len(lines) < 2:
                logger.error("File has less than 2 lines")
                return jsonify({"error": "Invalid file format"}), 400

            separator = ";" if ";" in lines[0] else ","
            header = lines[0].split(separator)

            if len(header) < 2:
                logger.error("Invalid header format")
                return jsonify({"error": "Invalid header format"}), 400

            value_column = header[1].strip()
            data = [line.split(separator) for line in lines[1:] if line.strip()]

            if not data:
                logger.error("No data rows found")
                return jsonify({"error": "No data rows found"}), 400

            tracker.end_phase('parsing')
            tracker.emit('parsing', 15, 'parsing_complete', force=True, message_params={'rowCount': len(data)})

            # === FAZA 3: PREPROCESSING (15-25%) ===
            tracker.start_phase('preprocessing')
            tracker.emit('preprocessing', 15, 'preprocessing_type_conversion')

            df = pd.DataFrame(data, columns=["UTC", value_column])
            df[value_column] = df[value_column].replace('', np.nan)
            df[value_column] = pd.to_numeric(df[value_column].str.replace(",", "."), errors='coerce')

            tracker.emit('preprocessing', 20, 'preprocessing_param_validation')

            raw_params = {
                "eqMax": request.form.get("eqMax"),
                "elMax": request.form.get("elMax"),
                "elMin": request.form.get("elMin"),
                "chgMax": request.form.get("chgMax"),
                "lgMax": request.form.get("lgMax"),
                "gapMax": request.form.get("gapMax"),
                "radioValueNull": request.form.get("radioValueNull"),
                "radioValueNotNull": request.form.get("radioValueNotNull")
            }

            try:
                params = validate_processing_params(raw_params)
            except ValueError as e:
                logger.error(f"Parameter validation failed: {e}")
                return safe_error_response(str(e), 400, 'validation')

            tracker.end_phase('preprocessing')
            tracker.emit('preprocessing', 25, 'preprocessing_complete', force=True, message_params={'rowCount': len(df)})

            # === FAZA 4: CLEANING (25-85%) ===
            tracker.start_phase('cleaning')
            tracker.emit('cleaning', 25, 'cleaning_start', force=True, message_params={'rowCount': len(df)})

            df_clean = clean_data(df, value_column, params, tracker)

            tracker.end_phase('cleaning')
            tracker.emit('cleaning', 90, 'cleaning_complete', force=True, message_params={'rowCount': len(df_clean)})

            # Resetiraj step tracking za ostale faze (finalizing, streaming)
            tracker.current_step = 0
            tracker.total_steps = 0

            # Track processing usage
            try:
                increment_processing_count(g.user_id)
                logger.info(f"✅ Tracked processing for user {g.user_id}")

                file_size_mb = total_size / (1024 * 1024)
                update_storage_usage(g.user_id, file_size_mb)
                logger.info(f"✅ Tracked storage for user {g.user_id}: {file_size_mb:.2f} MB")
            except Exception as e:
                logger.error(f"⚠️ Failed to track processing usage: {str(e)}")

            # Kreiraj closure za tracker da ga generator može koristiti
            tracker_ref = tracker

            def generate():
                """Generator za streaming NDJSON sa progress trackingom - OPTIMIZIRANO"""
                import math
                BACKPRESSURE_DELAY = 0.01  # 10ms između chunkova

                def clean_value_for_json(val):
                    """Čisti vrijednost za JSON - NaN/Inf -> None"""
                    if val is None:
                        return None
                    # pd.isna() hvata sve tipove NaN (numpy, pandas, python)
                    try:
                        if pd.isna(val):
                            return None
                    except (TypeError, ValueError):
                        pass
                    # Dodatna provjera za float i numpy float
                    try:
                        if isinstance(val, (float, np.floating)):
                            if math.isnan(val) or math.isinf(val):
                                return None
                    except (TypeError, ValueError):
                        pass
                    return val

                class CustomJSONEncoder(json.JSONEncoder):
                    def default(self, obj):
                        if isinstance(obj, pd.Timestamp):
                            return obj.strftime('%Y-%m-%d %H:%M:%S')
                        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
                            return None
                        return super().default(obj)

                try:
                    # === FAZA 5: STREAMING (90-100%) ===
                    tracker_ref.start_phase('streaming')
                    tracker_ref.emit('streaming', 90, 'streaming_start', force=True, message_params={'rowCount': len(df_clean)})

                    yield json.dumps({"total_rows": len(df_clean)}, cls=CustomJSONEncoder) + "\n"

                    chunk_size = STREAMING_CHUNK_SIZE
                    total_chunks_to_stream = (len(df_clean) // chunk_size) + 1
                    streaming_start_time = time.time()

                    for i in range(0, len(df_clean), chunk_size):
                        try:
                            # Progress 90-99%
                            chunk_progress = 90 + ((i / len(df_clean)) * 9)
                            current_chunk = (i // chunk_size) + 1

                            # Izračunaj ETA za streaming
                            streaming_eta = None
                            if current_chunk > 1:
                                elapsed = time.time() - streaming_start_time
                                chunks_done = current_chunk - 1
                                chunks_remaining = total_chunks_to_stream - current_chunk + 1
                                time_per_chunk = elapsed / chunks_done
                                streaming_eta = int(chunks_remaining * time_per_chunk)

                            tracker_ref.emit('streaming', chunk_progress, 'streaming_chunk',
                                            eta_seconds=streaming_eta,
                                            message_params={'currentChunk': current_chunk, 'totalChunks': total_chunks_to_stream})

                            # Dohvati chunk podataka
                            chunk = df_clean.iloc[i:i + chunk_size]

                            if i == 0:
                                logger.info("First 10 rows of processed data:")
                                logger.info(chunk.head(10).to_string())

                            # OPTIMIZIRANO: Selektuj SAMO potrebne kolone (30-50x brže od iterrows)
                            chunk_subset = chunk[['UTC', value_column]].copy()
                            chunk_subset['UTC'] = chunk_subset['UTC'].dt.strftime('%Y-%m-%d %H:%M:%S')

                            # KRITIČNO: Konvertuj u object dtype PRIJE apply da pandas ne vrati NaN
                            chunk_subset[value_column] = chunk_subset[value_column].astype(object)
                            # Zamijeni NaN sa None (object dtype čuva None)
                            chunk_subset[value_column] = chunk_subset[value_column].where(
                                pd.notna(chunk_subset[value_column]), None
                            )

                            # Brza konverzija u records - sada NaN postaje None
                            for record in chunk_subset.to_dict('records'):
                                yield json.dumps(record) + "\n"

                            # Backpressure: pauza između chunkova za stabilnost
                            time.sleep(BACKPRESSURE_DELAY)

                        except Exception as chunk_error:
                            logger.error(f"Streaming error at chunk {i}: {chunk_error}")
                            yield json.dumps({"error": f"Chunk {i} failed: {str(chunk_error)}", "partial": True}, cls=CustomJSONEncoder) + "\n"
                            break

                    tracker_ref.end_phase('streaming')
                    tracker_ref.emit('complete', 100, 'processing_complete', force=True, message_params={'rowCount': len(df_clean)})
                    yield json.dumps({"status": "complete"}, cls=CustomJSONEncoder) + "\n"

                except GeneratorExit:
                    logger.info("Client disconnected during streaming")
                except BrokenPipeError:
                    logger.warning("Broken pipe - client forcefully disconnected")
                except Exception as e:
                    logger.error(f"Generator error: {e}")
                    try:
                        yield json.dumps({"error": str(e)}, cls=CustomJSONEncoder) + "\n"
                    except:
                        pass  # Client već disconnected

            return Response(generate(), mimetype="application/x-ndjson")

        except Exception as e:
            logger.error(f"Error processing data: {str(e)}")
            return safe_error_response("Error processing data", 500, 'processing')

    except Exception as e:
        logger.error(f"Error in upload_chunk: {str(e)}")
        return safe_error_response("Error in upload", 500)

@bp.route("/api/dataProcessingMain/prepare-save", methods=["POST"])
@require_auth
def prepare_save():
    """
    Prepare processed data for download.

    Saves CSV data to Supabase Storage for persistent access on Cloud Run.

    Expected JSON body:
        - data: Dict containing:
            - data: Array of rows to save
            - fileName: Optional filename

    Returns:
        JSON response with file ID for download
    """
    try:
        data = request.json
        if not data or 'data' not in data:
            return jsonify({"error": "No data received"}), 400

        data_wrapper = data['data']
        save_data = data_wrapper.get('data', [])
        file_name = data_wrapper.get('fileName', '')

        if not save_data:
            return jsonify({"error": "Empty data"}), 400

        logger.info(f"Preparing to save file: {file_name} with {len(save_data)} rows")

        # Convert data array to CSV string
        output = StringIO()
        writer = csv.writer(output, delimiter=';')
        for row in save_data:
            writer.writerow(row)
        csv_content = output.getvalue()

        # Upload to Supabase Storage
        user_id = g.user_id

        file_id = storage_service.upload_csv(
            user_id=user_id,
            csv_content=csv_content,
            original_filename=file_name or "processed_data.csv",
            metadata={
                'totalRows': len(save_data) - 1,  # Exclude header
                'source': 'data-processing-prepare-save'
            }
        )

        if not file_id:
            return jsonify({"error": "Failed to save file to storage"}), 500

        logger.info(f"File prepared for download: {file_id}")

        return jsonify({
            "message": "File prepared for download",
            "fileId": file_id,
            "totalRows": len(save_data) - 1
        }), 200

    except Exception as e:
        logger.error(f"Error in prepare_save: {str(e)}")
        return safe_error_response("Error preparing file", 500)


@bp.route("/api/dataProcessingMain/download/<path:file_id>", methods=["GET"])
@require_auth
def download_file(file_id):
    """
    Get download URL for prepared CSV file from Supabase Storage.

    Returns JSON with signed URL for frontend to use directly.

    Args:
        file_id: File identifier from prepare_save (format: user_id/file_id)

    Returns:
        JSON with downloadUrl for frontend to open directly
    """
    try:
        logger.info(f"Download request for file: {file_id}")

        # Get signed URL from Supabase Storage (valid for 1 hour)
        signed_url = storage_service.get_download_url(file_id, expires_in=3600)

        if signed_url:
            logger.info(f"Generated signed URL for: {file_id}")

            # Return URL as JSON for frontend to use
            return jsonify({
                "success": True,
                "downloadUrl": signed_url,
                "fileId": file_id
            }), 200

        logger.warning(f"Signed URL failed for: {file_id}")
        return jsonify({"error": "Failed to generate download URL"}), 500

    except Exception as e:
        logger.error(f"Error in download_file: {str(e)}")
        return safe_error_response("Error downloading file", 500)


@bp.route("/api/dataProcessingMain/cleanup-files", methods=["POST"])
@require_auth
def cleanup_files():
    """
    Delete files from Supabase Storage after successful download.

    Expected JSON body:
        - fileIds: Array of file IDs to delete (format: user_id/timestamp_uuid)

    Returns:
        JSON response with deletion results
    """
    try:
        data = request.json

        if not data:
            return jsonify({"error": "No data received"}), 400

        file_ids = data.get('fileIds', [])

        if not file_ids:
            return jsonify({"message": "No files to delete", "deletedCount": 0}), 200

        deleted_count = 0
        failed_ids = []

        for file_id in file_ids:
            try:
                if storage_service.delete_file(file_id):
                    deleted_count += 1
                    logger.info(f"Cleaned up file: {file_id}")
                else:
                    failed_ids.append(file_id)
            except Exception as del_error:
                logger.error(f"Failed to delete file {file_id}: {del_error}")
                failed_ids.append(file_id)

        return jsonify({
            "message": "Cleanup complete",
            "deletedCount": deleted_count,
            "totalRequested": len(file_ids),
            "failedIds": failed_ids
        }), 200

    except Exception as e:
        logger.error(f"Error in cleanup_files: {str(e)}")
        return jsonify({"error": str(e)}), 500
