"""
Adjustments API Routes
Handles data adjustment operations including chunked uploads and data processing
"""
import os
import time
import math
import logging
import traceback
from typing import Any, Dict, Optional, Tuple

from flask import Blueprint, request, jsonify, g, Response
import pandas as pd

from shared.auth.jwt import require_auth
from shared.auth.subscription import require_subscription, check_processing_limit
from shared.tracking.usage import increment_processing_count, update_storage_usage, log_compute_duration
from shared.exceptions.errors import AnomalyException, ThresholdOutOfRangeError
from shared.responses.gzip import gzip_json_response

from domains.adjustments.config import UPLOAD_FOLDER, VALID_METHODS
from domains.adjustments.services.state_manager import (
    adjustment_chunks,
    adjustment_chunks_timestamps,
    chunk_buffer,
    chunk_buffer_timestamps,
    cleanup_all_expired_data,
    get_file_info_from_cache,
    check_files_need_methods,
    is_adjustment_owner
)
from domains.adjustments.services.progress import (
    ProgressStages,
    ProgressTracker,
    emit_progress,
    emit_file_result,
    emit_file_error
)
from domains.adjustments.services.processing import (
    convert_data_without_processing,
    process_data_detailed
)
from domains.adjustments.services.utils import (
    sanitize_filename,
    analyse_data
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

bp = Blueprint('adjustmentsOfData_bp', __name__)

# Anomaly plot payloads carry every datapoint at full resolution (no
# decimation), so large CSVs produce multi-MB JSON. Cloud Run rejects any
# non-streamed response > 32 MiB with a bare 500 (no CORS header → misleading
# "CORS" error in the browser). Gzip the JSON on the wire: time-series JSON
# compresses ~5-15x, which keeps realistic payloads well under the cap while
# preserving every point. The browser decompresses transparently (Content-
# Encoding: gzip), so no frontend change is needed. The actual logic lives in
# shared/responses/gzip.py so the cloud blueprint can reuse the same guard.
@bp.after_request
def _gzip_json(response: Response) -> Response:
    """Gzip large JSON responses when the client accepts it."""
    return gzip_json_response(response)


def _coerce_upload_id(data: Dict[str, Any]) -> Tuple[Optional[str], Optional[Tuple[Response, int]]]:
    """
    Pull `uploadId`/`upload_id` from the body as a string.

    uploadId may legitimately be absent (some endpoints treat it as optional),
    but if it IS present it must be a non-empty string. A JSON object/array/
    number would otherwise reach a filesystem/state lookup and crash with a 500
    (e.g. ``Path(...) / {dict}``). Returns ``(upload_id_or_None, error_or_None)``.
    """
    uid = data.get("uploadId") or data.get("upload_id")
    if uid is not None and (not isinstance(uid, str) or not uid):
        return None, (jsonify({
            "success": False,
            "code": "BAD_UPLOAD_ID",
            "error": "uploadId must be a non-empty string",
        }), 400)
    return uid, None


def _internal_error_message(e: Exception) -> str:
    """Return the exception message when EXPOSE_INTERNAL_ERRORS=true (dev/staging);
    otherwise return the sanitized generic string for production safety."""
    if os.environ.get('EXPOSE_INTERNAL_ERRORS', '').lower() in ('1', 'true', 'yes'):
        return f'Internal error: {e}'
    return 'Internal server error'


@bp.route('/upload-chunk', methods=['POST'])
@require_auth
@require_subscription
@check_processing_limit
def upload_chunk() -> Tuple[Response, int]:
    """
    Endpoint for receiving individual chunks.
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
                result = analyse_data(final_path, upload_id, g.user_id)

                try:
                    file_size_bytes = len(combined_content.encode('utf-8'))
                    file_size_mb = file_size_bytes / (1024 * 1024)
                    update_storage_usage(g.user_id, file_size_mb)
                    logger.info(f"Tracked storage for user {g.user_id}: {file_size_mb:.2f} MB")
                except Exception as e:
                    logger.error(f"Failed to track storage usage: {str(e)}")

                response_data = {
                    'status': 'complete',
                    'message': 'File upload and analysis complete',
                    'success': True,
                    'data': result
                }
                return jsonify(response_data), 200
            except Exception as e:
                logger.error(f"Error analyzing file {final_path}: {str(e)}")
                return jsonify({"error": str(e)}), 500

        return jsonify({
            'status': 'chunk_received',
            'message': f'Received chunk {chunk_index + 1} of {total_chunks}',
            'chunksReceived': received_chunks_count
        }), 200

    except Exception as e:
        traceback.print_exc()
        try:
            file_key = f"{upload_id}:{filename}"
            if file_key in chunk_buffer:
                del chunk_buffer[file_key]
            if file_key in chunk_buffer_timestamps:
                del chunk_buffer_timestamps[file_key]
        except Exception:
            pass
        return jsonify({"error": str(e)}), 400


@bp.route('/adjust-data-chunk', methods=['POST'])
@require_auth
@require_subscription
def adjust_data() -> Tuple[Response, int]:
    """
    Endpoint for adjusting data parameters before complete processing.
    """
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No JSON data received"}), 400

        upload_id = data.get('upload_id')
        if not upload_id:
            return jsonify({"error": "upload_id is required"}), 400

        if not is_adjustment_owner(upload_id, g.user_id):
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
        decimal_precision_values = {}
        for filename, method_info in methods.items():
            if isinstance(method_info, dict):
                if 'intrpl_max' in method_info:
                    try:
                        intrpl_max_values[filename] = float(method_info['intrpl_max'])
                    except (TypeError, ValueError) as e:
                        logger.warning(f"Could not convert intrplMax for {filename}: {e}")
                        intrpl_max_values[filename] = None
                if 'decimal_precision' in method_info:
                    decimal_precision_values[filename] = method_info['decimal_precision']

        if upload_id not in adjustment_chunks:
            adjustment_chunks[upload_id] = {
                'user_id': g.user_id,
                'params': {
                    'startTime': start_time,
                    'endTime': end_time,
                    'timeStepSize': time_step_size,
                    'offset': offset,
                    'methods': methods,
                    'intrplMaxValues': intrpl_max_values,
                    'decimalPrecisionValues': decimal_precision_values,
                    'decimalPrecision': decimal_precision
                }
            }
        else:
            params = adjustment_chunks[upload_id]['params']
            if start_time is not None:
                params['startTime'] = start_time
            if end_time is not None:
                params['endTime'] = end_time
            if time_step_size is not None:
                params['timeStepSize'] = time_step_size
            if offset is not None:
                params['offset'] = offset
            if decimal_precision is not None:
                params['decimalPrecision'] = decimal_precision

            if 'methods' not in params:
                params['methods'] = {}
            if methods:
                params['methods'].update(methods)

            if 'intrplMaxValues' not in params:
                params['intrplMaxValues'] = {}
            params['intrplMaxValues'].update(intrpl_max_values)

            if 'decimalPrecisionValues' not in params:
                params['decimalPrecisionValues'] = {}
            params['decimalPrecisionValues'].update(decimal_precision_values)

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
def complete_adjustment() -> Tuple[Response, int]:
    """
    Complete the adjustment processing.
    """

    try:
        data = request.json
        if not data:
            return jsonify({"error": "No JSON data received"}), 400

        upload_id = data.get('uploadId')

        if not upload_id:
            return jsonify({"error": "Missing uploadId"}), 400

        if not is_adjustment_owner(upload_id, g.user_id):
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

        methods = params.get('methods', {})
        start_time = params.get('startTime')
        end_time = params.get('endTime')
        time_step = params.get('timeStepSize')
        offset = params.get('offset')
        decimal_precision = params.get('decimalPrecision', 'full')

        intrpl_max_values = params.get('intrplMaxValues', {})
        decimal_precision_values = params.get('decimalPrecisionValues', {})

        dataframes = adjustment_chunks[upload_id]['dataframes']
        if not dataframes:
            return jsonify({"error": "No data found for this upload ID"}), 404

        filenames = list(dataframes.keys())
        total_rows = sum(len(df) for df in dataframes.values())
        file_rows_dict = {filename: len(df) for filename, df in dataframes.items()}

        tracker = ProgressTracker(
            upload_id=upload_id,
            total_files=len(filenames),
            total_rows=total_rows
        )
        tracker.set_file_rows(file_rows_dict)

        _compute_start = time.time()

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

                time_step_float = float(time_step) if time_step is not None else 0.0
                offset_float = float(offset) if offset is not None else 0.0

                requested_offset_adjusted = offset_float
                if file_time_step > 0 and requested_offset_adjusted >= file_time_step:
                    requested_offset_adjusted = requested_offset_adjusted % file_time_step

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
                "message": f"{len(files_needing_methods)} Datei(en) benotigen Verarbeitungsmethoden.",
                "data": {
                    "info_df": files_needing_methods,
                    "dataframe": []
                }
            }), 200

        for file_index, filename in enumerate(filenames):
            try:
                df = dataframes[filename]
                row_count = len(df)

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

                time_step_float = float(time_step) if time_step is not None else 0.0
                offset_float = float(offset) if offset is not None else 0.0

                requested_offset_phase2 = offset_float
                if file_time_step > 0 and requested_offset_phase2 >= file_time_step:
                    requested_offset_phase2 = requested_offset_phase2 % file_time_step

                timestep_matches = math.isclose(file_time_step, time_step_float, rel_tol=1e-9)
                offset_matches = math.isclose(file_offset, requested_offset_phase2, rel_tol=1e-9)
                needs_processing = not (timestep_matches and offset_matches)

                intrpl_max = intrpl_max_values.get(filename)
                file_decimal_precision = decimal_precision_values.get(filename, decimal_precision)

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
                        file_decimal_precision,
                        start_time,
                        end_time
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
                        file_decimal_precision
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

                    tracker.complete_file(filename)

                    file_complete_progress = ProgressStages.calculate_file_progress(file_index + 1, len(filenames))

                    quality_percentage = 0
                    if info_record and 'Anteil an numerischen Datenpunkten' in info_record:
                        quality_percentage = info_record['Anteil an numerischen Datenpunkten']

                    file_complete_message_params = {'filename': filename}
                    if needs_processing:
                        file_complete_message_params['fromStep'] = file_time_step
                        file_complete_message_params['toStep'] = time_step

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

        try:
            increment_processing_count(g.user_id)
            logger.info(f"Tracked processing for user {g.user_id}")

            total_size_bytes = sum(
                df.memory_usage(deep=True).sum()
                for df in dataframes.values()
            )
            file_size_mb = total_size_bytes / (1024 * 1024)
            update_storage_usage(g.user_id, file_size_mb)
            logger.info(f"Tracked storage for user {g.user_id}: {file_size_mb:.2f} MB")

            log_compute_duration(g.user_id, time.time() - _compute_start, 'dritte-bearbeitung', {'upload_id': upload_id})
        except Exception as e:
            logger.error(f"Failed to track processing usage: {str(e)}")

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


# ===========================================================================
# Anomaly Detection (Anomalieerkennung) — `/zweite-bearbeitung` flow
# ===========================================================================
# Ports the Python reference script `test2/anomaly_detection_1.py` into a
# REST + SocketIO pipeline. State is held in `adjustment_chunks[upload_id]`
# under the `anomaly` key (see state_manager.init_anomaly_state).
# ---------------------------------------------------------------------------

from pathlib import Path as _Path  # local alias to avoid clobbering existing refs

from core.app_factory import socketio as _socketio
from domains.adjustments.data.loader import load_and_validate_csv as _load_csv
from domains.adjustments.services.timestep_detection import detect_timestep_offset_minutes
from domains.adjustments.services.state_manager import (
    PipelineStatus as _PipelineStatus,
    init_anomaly_state as _init_anomaly,
    get_anomaly_state as _get_anomaly,
    set_pipeline_status as _set_pipeline_status,
    try_acquire_pipeline as _try_acquire_pipeline,
)
from domains.adjustments.services.plot_builder import (
    build_anomaly_overlay as _build_anomaly_overlay,
    build_lstm_error as _build_lstm_error,
    build_original_plot as _build_original_plot,
    build_processed_plot as _build_processed_plot,
    build_slope_plot as _build_slope_plot,
    build_stl_decomposition as _build_stl_decomposition,
)
from domains.adjustments.services.anomaly_helpers import (
    slope_calc as _slope_calc,
    tr as _tr,
)
from domains.adjustments.services.anomaly_pipeline import (
    apply_lstm_threshold as _apply_lstm_threshold,
    apply_stl_threshold as _apply_stl_threshold,
    build_par_dict as _build_par_dict,
    prepare_lstm as _prepare_lstm,
    prepare_stl as _prepare_stl,
    process_constants as _process_constants,
    process_range as _process_range,
    process_sbad as _process_sbad,
    process_short_ranges as _process_short_ranges,
    process_zeros as _process_zeros,
)
from domains.adjustments.services.anomaly_validators import (
    validate_par_dict as _validate_par_dict,
    validate_param_single as _validate_param_single,
)
from domains.adjustments.debug_log import log_request, log_phase, dlog, _short


def _normalize_lang(value) -> str:
    """Coerce arbitrary input to 'en' or 'de' (default 'en')."""
    if isinstance(value, str) and value.lower() == "de":
        return "de"
    return "en"


def _compute_phase_keys(par: Dict[str, Any], current_endpoint: str) -> list:
    """Return ordered list of messageKeys the pipeline will emit for this run.

    `current_endpoint` is one of "start", "stl_threshold", "lstm_threshold".
    Used to size `totalSteps` so the FE counter matches reality.

    NOTE the finalize step uses `anomaly_phase_short_ranges` (matches Task 1's
    actual emission key from `process_short_ranges`).
    """
    keys = []
    if current_endpoint == "start":
        if par["EQ_MAX"]["value"] is not None:
            keys.append("anomaly_phase_constants")
        if par["EL0"]["value"]:
            keys.append("anomaly_phase_zeros")
        if par["V_MAX"]["value"] is not None or par["V_MIN"]["value"] is not None:
            keys.append("anomaly_phase_range")
        sbad = par["SBAD"]["var"]
        if sbad["CHG_MAX"]["value"] is not None and sbad["LG_MAX"]["value"] is not None:
            keys.append("anomaly_phase_sbad")
        if par["STL"]["run"]:
            keys.append("anomaly_phase_stl_prep")
        elif par["LSTM"]["run"]:
            keys.append("anomaly_phase_lstm_prep")
        else:
            keys.append("anomaly_phase_short_ranges")
    elif current_endpoint == "stl_threshold":
        keys.append("anomaly_phase_stl_apply")
        if par["LSTM"]["run"]:
            keys.append("anomaly_phase_lstm_prep")
        else:
            keys.append("anomaly_phase_short_ranges")
    elif current_endpoint == "lstm_threshold":
        keys.append("anomaly_phase_lstm_apply")
        keys.append("anomaly_phase_short_ranges")
    return keys


def _make_progress_callback(
    upload_id: str,
    started_at: Optional[float] = None,
    phase_keys: Optional[list] = None,
):
    """Build a callable that emits SocketIO `anomaly_progress` events.

    Pipeline phases call `cb(label, fraction, message_key=..., message_params=...)`.
    The callback emits a payload that matches the contract used by other pages
    (upload, processing) — `messageKey` for i18n, `currentStep/totalSteps`
    derived from the ordered `phase_keys` list, `etaFormatted` always (or
    `None` until enough samples accrue, so FE i18n fallback renders the
    localized "Calculating..."), and `status`.

    Args:
        upload_id: SocketIO room key.
        started_at: Pipeline-wide start timestamp; if None, uses time.time() now.
        phase_keys: Ordered list of message_keys this pipeline run will emit.
            Used to compute currentStep/totalSteps. If None or empty,
            currentStep/totalSteps are omitted from the payload.
    """
    state_holder = {
        "last_label": None,
        "last_message_key": None,
        "last_fraction": -1.0,
        "started_at": started_at if started_at is not None else time.time(),
    }
    phase_keys = phase_keys or []
    total_steps = len(phase_keys)

    def _format_eta(seconds: float) -> str:
        if seconds < 1:
            return "<1s"
        if seconds < 60:
            return f"{int(round(seconds))}s"
        m = int(seconds // 60)
        s = int(round(seconds - m * 60))
        return f"{m}m {s}s" if s > 0 else f"{m}m"

    def cb(label: str, fraction: float, message_key: Optional[str] = None,
           message_params: Optional[dict] = None) -> None:
        try:
            f = max(0.0, min(1.0, float(fraction)))
            label_changed = state_holder["last_label"] != label
            key_changed = state_holder["last_message_key"] != message_key
            will_emit = (label_changed or key_changed
                         or f - state_holder["last_fraction"] >= 0.05
                         or f >= 0.999)
            dlog("EMIT_DECISION",
                 upload=_short(upload_id), step=label, key=message_key,
                 fraction=f"{f:.3f}", will_emit=will_emit,
                 last_step=state_holder["last_label"],
                 last_fraction=f"{state_holder['last_fraction']:.3f}")
            if not will_emit:
                return

            elapsed = time.time() - state_holder["started_at"]
            payload = {
                "uploadId": upload_id,
                "step": label,
                "progress": int(round(f * 100)),
                "status": "processing",
            }
            if message_key:
                payload["messageKey"] = message_key
            if message_params:
                payload["messageParams"] = message_params
            if total_steps > 0 and message_key in phase_keys:
                payload["currentStep"] = phase_keys.index(message_key) + 1
                payload["totalSteps"] = total_steps

            # Always include etaFormatted; None means "not yet computable" —
            # frontend renders the localized "Calculating..." via i18n fallback
            # (loading.calculating). Sending the literal English string here
            # would bypass per-language translation.
            if f >= 0.05 and elapsed > 1.0:
                remaining = elapsed * (1 - f) / f
                payload["etaFormatted"] = _format_eta(remaining)
            else:
                payload["etaFormatted"] = None

            dlog("EMIT", room=_short(upload_id), payload=payload)
            _socketio.emit("anomaly_progress", payload, room=upload_id)
            state_holder["last_label"] = label
            state_holder["last_message_key"] = message_key
            state_holder["last_fraction"] = f
        except Exception:
            logger.debug("progress emit failed", exc_info=True)

    return cb


def _run_preprocess_and_sbad(state, par, lang, progress_cb):
    """
    Execute the deterministic phases (constants → zeros → range → sbad) on the
    *current* original_df, and stash the result into state["processed_df"].
    """
    df = state["original_df"].copy()

    df = _process_constants(
        df,
        par["EQ_MAX"]["value"],
        par["GAP_MAX"]["value"],
        par["DEC"]["value"],
        lang=lang,
        progress_callback=progress_cb,
    )
    df = _process_zeros(
        df,
        par["EL0"]["value"],
        par["GAP_MAX"]["value"],
        par["DEC"]["value"],
        lang=lang,
        progress_callback=progress_cb,
    )
    df = _process_range(
        df,
        par["V_MAX"]["value"],
        par["V_MIN"]["value"],
        par["GAP_MAX"]["value"],
        par["DEC"]["value"],
        lang=lang,
        progress_callback=progress_cb,
    )

    sbad_chg_max = par["SBAD"]["var"]["CHG_MAX"]["value"]
    sbad_lg_max = par["SBAD"]["var"]["LG_MAX"]["value"]
    if sbad_chg_max is not None and sbad_lg_max is not None:
        df, _count = _process_sbad(
            df,
            sbad_chg_max,
            sbad_lg_max,
            par["GAP_MAX"]["value"],
            par["DEC"]["value"],
            lang=lang,
            progress_callback=progress_cb,
        )

    state["processed_df"] = df
    return df


def _finalize_complete(state, par, lang, progress_cb):
    """Apply LG_MIN trimming and build the final 'complete' response payload."""
    df = state["processed_df"]
    df = _process_short_ranges(
        df, par["LG_MIN"]["value"], lang=lang, progress_callback=progress_cb
    )
    state["processed_df"] = df
    state["pipeline_status"] = _PipelineStatus.COMPLETE

    plots = state.get("plots", {})
    plots["processed"] = _build_processed_plot(df, lang=lang)
    state["plots"] = plots
    return plots


@bp.route('/load', methods=['POST'])
@require_auth
@require_subscription
@check_processing_limit
@log_request
def anomaly_load() -> Tuple[Response, int]:
    """
    Validate the uploaded CSV and return original + slope plots.

    Body: {"uploadId": str, "filename": str | None, "lang": "en"|"de"}
    Returns: {plots: {original, slope}, columnName, dtAvgH} OR {error}
    """
    cleanup_all_expired_data()

    try:
        data = request.get_json(silent=True) or {}
        upload_id, _uid_err = _coerce_upload_id(data)
        if _uid_err:
            return _uid_err
        filename = data.get("filename")
        lang = _normalize_lang(data.get("lang"))

        if not upload_id:
            return jsonify({"error": "uploadId is required"}), 400

        # Locate the assembled file. The chunk-upload flow already stored it at
        # UPLOAD_FOLDER/<upload_id>/<filename>.
        upload_dir = _Path(UPLOAD_FOLDER) / upload_id
        if not upload_dir.exists() or not upload_dir.is_dir():
            return jsonify({"error": f"No upload found for ID '{upload_id}'"}), 404

        if filename:
            try:
                filename = sanitize_filename(filename)
            except ValueError as e:
                return jsonify({"error": str(e)}), 400
            file_path = upload_dir / filename
        else:
            csv_files = sorted(upload_dir.glob("*.csv"))
            if not csv_files:
                return jsonify({"error": "No CSV file present in upload directory"}), 404
            file_path = csv_files[0]
            filename = file_path.name

        dlog("CSV_RESOLVED", path=str(file_path))

        try:
            df, dt_avg = _load_csv(
                file_path,
                lang=lang,
                allowed_root=_Path(UPLOAD_FOLDER),
            )
        except ValueError as e:
            return jsonify({"error": str(e)}), 400

        dlog("CSV_PARSED", rows=int(len(df)), cols=list(df.columns))

        # Persist anomaly state — owner-aware. Reject if upload_id is owned
        # by someone else (do not leak existence: 404).
        state = _init_anomaly(upload_id, g.user_id, lang)
        if state is None:
            return jsonify({"error": f"No upload found for ID '{upload_id}'"}), 404

        dlog("STATE_INIT", upload=_short(upload_id), state_present=state is not None)

        state["filename"] = filename
        state["file_path"] = str(file_path)
        state["original_df"] = df
        state["processed_df"] = None  # allocated on first pipeline mutation
        state["dt_avg"] = dt_avg
        state["pipeline_status"] = _PipelineStatus.LOADED
        state["plots"] = {}
        state["intermediate"] = {"stl_result": None, "lstm_results_df": None}

        # Build response plots
        original_plot = _build_original_plot(df, lang=lang)
        df_slope = _slope_calc(df)
        slope_plot = _build_slope_plot(df_slope, lang=lang)

        # Detected time step + offset (minutes) for the read-only info line the
        # frontend shows above the original-data plot once a file is loaded.
        timestep_min, offset_min = detect_timestep_offset_minutes(df)

        return jsonify({
            "plots": {"original": original_plot, "slope": slope_plot},
            "columnName": str(df.columns[1]),
            "dtAvgH": dt_avg.total_seconds() / 3600.0,
            "timestepMin": timestep_min,
            "offsetMin": offset_min,
            "rowCount": int(len(df)),
            "status": _PipelineStatus.LOADED,
        }), 200

    except AnomalyException as exc:
        logger.warning(f"Anomaly validation rejected in /load: {exc.error_code} — {exc.message}")
        return jsonify({
            "ok": False,
            "error": exc.message,
            "error_code": exc.error_code,
            "details": exc.details,
            "suggestions": exc.suggestions,
        }), 400
    except Exception as e:
        logger.error(f"Error in /load: {str(e)}\n{traceback.format_exc()}")
        return jsonify({"error": _internal_error_message(e)}), 500


@bp.route('/validate-param', methods=['POST'])
@require_auth
@require_subscription
@log_request
def anomaly_validate_param() -> Tuple[Response, int]:
    """
    Validate a single parameter on input change.

    Body: {"name": "EQ_MAX"|"GAP_MAX"|...|"SBAD.CHG_MAX"|"STL.PERIOD_H"|...,
           "value": Any, "currentParams": {...}, "lang": "en"|"de",
           "uploadId": str | None}
    Returns: {ok: true} or {error: localized message}

    Always returns HTTP 200 with `{ok: bool, error?: str}` payload, even on
    invalid input — clients use this on every keystroke and 4xx would noise
    the network panel.
    """
    cleanup_all_expired_data()

    try:
        data = request.get_json(silent=True) or {}
        name = data.get("name")
        value = data.get("value")
        current = data.get("currentParams") or {}
        lang = _normalize_lang(data.get("lang"))
        upload_id, _uid_err = _coerce_upload_id(data)
        if _uid_err:
            return _uid_err

        if not name:
            return jsonify({"error": "Parameter 'name' is required"}), 400

        # dt_avg comes from the loaded session (needed for STL period check)
        dt_avg = None
        if upload_id:
            state = _get_anomaly(upload_id, g.user_id)
            if state is not None:
                dt_avg = state.get("dt_avg")

        par = _build_par_dict(current)
        # Validation errors propagate to the outer except AnomalyException handler.
        _validate_param_single(name, value, par, dt_avg, lang)

        return jsonify({"ok": True}), 200

    except AnomalyException as exc:
        logger.warning(f"validate-param rejected: {exc.error_code} — {exc.message}")
        return jsonify({
            "ok": False,
            "error": exc.message,
            "error_code": exc.error_code,
        }), 200
    except Exception as e:
        logger.error(f"Error in /validate-param: {str(e)}\n{traceback.format_exc()}")
        return jsonify({"error": _internal_error_message(e)}), 500


@bp.route('/start', methods=['POST'])
@require_auth
@require_subscription
@check_processing_limit
@log_request
def anomaly_start() -> Tuple[Response, int]:
    """
    Run preprocess → constants → zeros → range → SBAD, then route to next state.

    Body: {"uploadId": str, "params": {...}, "lang": "en"|"de"}
    Returns one of:
      - {status: "awaiting_stl_threshold", plots: {stl: [...]}, sessionId}
      - {status: "awaiting_lstm_threshold", plots: {lstmError}, sessionId}
      - {status: "complete", plots: {processed}, processedCsvUrl?, sessionId}
      - {error: localized}
    """
    cleanup_all_expired_data()

    try:
        data = request.get_json(silent=True) or {}
        upload_id, _uid_err = _coerce_upload_id(data)
        if _uid_err:
            return _uid_err
        raw_params = data.get("params") or {}
        lang = _normalize_lang(data.get("lang"))

        if not upload_id:
            return jsonify({"error": "uploadId is required"}), 400

        state = _get_anomaly(upload_id, g.user_id)
        if state is None or state.get("original_df") is None:
            return jsonify({"error": "Session not loaded — call /load first"}), 404

        # Build par dict with localized names so error messages match Python script.
        par = _build_par_dict(raw_params)

        # Preflight validation — same rules as Python L790-921.
        try:
            _validate_par_dict(par, dt_avg=state.get("dt_avg"), lang=lang)
        except ValueError as e:
            logger.info("adjustmentsOfData/start rejected (param validation) for upload %s: %s", upload_id, e)
            return jsonify({"error": str(e)}), 400

        # Atomic claim — refuses concurrent /start for the same upload.
        if not _try_acquire_pipeline(upload_id, g.user_id):
            return jsonify({"error": "Pipeline already running for this upload"}), 409

        state["params"] = par
        state["lang"] = lang
        pipeline_started_at = time.time()
        progress_cb = _make_progress_callback(
            upload_id, started_at=pipeline_started_at,
            phase_keys=_compute_phase_keys(par, "start"),
        )

        # Run the deterministic phases.
        try:
            _run_preprocess_and_sbad(state, par, lang, progress_cb)
        except ValueError as e:
            state["pipeline_status"] = _PipelineStatus.ERROR
            logger.info("adjustmentsOfData/start rejected (preprocess/SBAD) for upload %s: %s", upload_id, e)
            return jsonify({"error": str(e)}), 400

        plots = state.get("plots", {})

        # Branch: STL pause first if enabled. When STL.run AND LSTM.run are
        # both true, LSTM is run inside /stl-threshold AFTER the user submits
        # the STL threshold (see T5).
        if par["STL"]["run"]:
            try:
                period = int(par["STL"]["var"]["PERIOD"]["value"])
                stl_result, time_values = _prepare_stl(
                    state["processed_df"], period, lang=lang,
                    progress_callback=progress_cb,
                )
            except ValueError as e:
                state["pipeline_status"] = _PipelineStatus.ERROR
                logger.info("adjustmentsOfData/start rejected (STL) for upload %s: %s", upload_id, e)
                return jsonify({"error": str(e)}), 400

            state["intermediate"]["stl_result"] = stl_result
            state["pipeline_status"] = _PipelineStatus.AWAITING_STL_THRESHOLD

            stl_plots = _build_stl_decomposition(stl_result, time_values, lang=lang)
            plots["stlDecomposition"] = stl_plots
            state["plots"] = plots

            dlog("START_RESPONSE", status=_PipelineStatus.AWAITING_STL_THRESHOLD)
            return jsonify({
                "status": _PipelineStatus.AWAITING_STL_THRESHOLD,
                "plots": {"stlDecomposition": stl_plots},
                "sessionId": upload_id,
            }), 200

        # Else branch: LSTM pause if enabled (without STL).
        if par["LSTM"]["run"]:
            from domains.adjustments.services.anomaly_pipeline import prepare_lstm as _prep_lstm
            try:
                # validate_par_dict guarantees PERIOD is populated for LSTM.run=true.
                period = int(par["LSTM"]["var"]["PERIOD"]["value"])
                results_df, _model = _prep_lstm(
                    state["processed_df"],
                    period,
                    par["LSTM"]["var"]["NEURONS"]["value"],
                    par["LSTM"]["var"]["EPOCHS"]["value"],
                    par["LSTM"]["var"]["BATCH_SIZE"]["value"],
                    lang=lang,
                    progress_callback=progress_cb,
                )
            except ValueError as e:
                state["pipeline_status"] = _PipelineStatus.ERROR
                return jsonify({"error": str(e)}), 400

            state["intermediate"]["lstm_results_df"] = results_df
            state["pipeline_status"] = _PipelineStatus.AWAITING_LSTM_THRESHOLD

            lstm_error_plot = _build_lstm_error(results_df, lang=lang)
            plots["lstmError"] = lstm_error_plot
            state["plots"] = plots

            dlog("START_RESPONSE", status=_PipelineStatus.AWAITING_LSTM_THRESHOLD)
            return jsonify({
                "status": _PipelineStatus.AWAITING_LSTM_THRESHOLD,
                "plots": {"lstmError": lstm_error_plot},
                "sessionId": upload_id,
            }), 200

        # Else: no pauses → finalize directly.
        plots = _finalize_complete(state, par, lang, progress_cb)
        dlog("START_RESPONSE", status=_PipelineStatus.COMPLETE)
        return jsonify({
            "status": _PipelineStatus.COMPLETE,
            "plots": {"processed": plots["processed"]},
            "sessionId": upload_id,
        }), 200

    except AnomalyException as exc:
        # Reset pipeline_status so subsequent /start can retry without 409.
        try:
            err_state = _get_anomaly(upload_id, g.user_id) if upload_id else None
            if err_state is not None:
                err_state["pipeline_status"] = _PipelineStatus.ERROR
        except Exception:
            pass
        logger.warning(f"Anomaly validation rejected in /start: {exc.error_code} — {exc.message}")
        return jsonify({
            "ok": False,
            "error": exc.message,
            "error_code": exc.error_code,
            "details": exc.details,
            "suggestions": exc.suggestions,
        }), 400
    except Exception as e:
        # Reset pipeline_status so subsequent /start can retry without 409.
        try:
            err_state = _get_anomaly(upload_id, g.user_id) if upload_id else None
            if err_state is not None:
                err_state["pipeline_status"] = _PipelineStatus.ERROR
        except Exception:
            pass
        logger.error(f"Error in /start: {str(e)}\n{traceback.format_exc()}")
        return jsonify({"error": _internal_error_message(e)}), 500


# ---------------------------------------------------------------------------
# Pause-resume endpoints
# ---------------------------------------------------------------------------

def _save_processed_csv(state, lang) -> Optional[str]:
    """
    Persist processed_df next to the original upload as `<stem>_1.csv` (port of
    Python L1568 with the filename bug fixed: 'test2.csv_1.csv' → 'test2_1.csv').
    Returns the absolute path on disk, or None if no processed data.
    """
    df = state.get("processed_df")
    if df is None:
        return None
    file_path = state.get("file_path")
    if not file_path:
        return None
    src = _Path(file_path)
    out = src.with_name(f"{src.stem}_1{src.suffix}")
    df.to_csv(out, index=False, sep=";")
    return str(out)


def _ensure_stl_intermediate(state, lang) -> None:
    """Recompute state['intermediate']['stl_result'] if it has been wiped.

    The threshold endpoint used to return 409 "STL intermediate state
    missing — re-run /start" in this case, forcing the user to start over.
    STL decomposition is deterministic from `state['processed_df']` plus
    `state['params']['STL']['var']['PERIOD']`, both of which survive the
    same TTL/restart events that can lose the intermediate. Recovery is
    therefore local and free of side effects on other endpoints.
    """
    if state.get("intermediate", {}).get("stl_result") is not None:
        return
    processed_df = state.get("processed_df")
    par = state.get("params") or {}
    period_descriptor = (par.get("STL", {}).get("var", {}).get("PERIOD") or {})
    period_raw = period_descriptor.get("value")
    if processed_df is None or period_raw is None:
        # Nothing to recompute from — let the caller emit the original 409.
        return
    dlog("STL_INTERMEDIATE_RECOMPUTED", filename=state.get("filename"))
    period = int(period_raw)
    stl_result, _time_values = _prepare_stl(processed_df, period, lang=lang)
    state.setdefault("intermediate", {})["stl_result"] = stl_result


def _ensure_lstm_intermediate(state, lang) -> None:
    """Recompute state['intermediate']['lstm_results_df'] if it has been
    wiped — same idempotency contract as _ensure_stl_intermediate."""
    if state.get("intermediate", {}).get("lstm_results_df") is not None:
        return
    processed_df = state.get("processed_df")
    par = state.get("params") or {}
    lstm = par.get("LSTM", {}).get("var", {})
    period_raw = (lstm.get("PERIOD") or {}).get("value")
    neurons = (lstm.get("NEURONS") or {}).get("value")
    epochs = (lstm.get("EPOCHS") or {}).get("value")
    batch_size = (lstm.get("BATCH_SIZE") or {}).get("value")
    if processed_df is None or None in (period_raw, neurons, epochs, batch_size):
        return
    dlog("LSTM_INTERMEDIATE_RECOMPUTED", filename=state.get("filename"))
    period = int(period_raw)
    results_df, _model = _prepare_lstm(
        processed_df, period, neurons, epochs, batch_size, lang=lang,
    )
    state.setdefault("intermediate", {})["lstm_results_df"] = results_df


@bp.route('/stl-threshold', methods=['POST'])
@require_auth
@require_subscription
@log_request
def anomaly_stl_threshold() -> Tuple[Response, int]:
    """
    Apply user-supplied STL threshold; chain to LSTM pause or finalize.

    Body: {"uploadId": str, "threshold": float, "lang": "en"|"de"}
    """
    cleanup_all_expired_data()

    upload_id = None
    try:
        data = request.get_json(silent=True) or {}
        upload_id, _uid_err = _coerce_upload_id(data)
        if _uid_err:
            return _uid_err
        threshold = data.get("threshold")
        lang = _normalize_lang(data.get("lang"))

        if not upload_id:
            return jsonify({"error": "uploadId is required"}), 400

        state = _get_anomaly(upload_id, g.user_id)
        if state is None:
            return jsonify({"error": f"No upload found for ID '{upload_id}'"}), 404
        if state.get("pipeline_status") != _PipelineStatus.AWAITING_STL_THRESHOLD:
            return jsonify({"error": "Pipeline is not awaiting an STL threshold"}), 409
        try:
            _ensure_stl_intermediate(state, lang)
        except ValueError as e:
            state["pipeline_status"] = _PipelineStatus.ERROR
            return jsonify({"error": str(e)}), 400
        stl_result = state.get("intermediate", {}).get("stl_result")
        if stl_result is None:
            # Recovery wasn't possible (processed_df or params evicted) —
            # original 409 path stands.
            return jsonify({"error": "STL intermediate state missing — re-run /start"}), 409

        # Validate threshold using same rules as Python L1260-1271.
        # check_float failures (PARAM_NOT_FLOAT) propagate to the outer
        # except AnomalyException handler. A negative numeric threshold
        # emits THRESHOLD_OUT_OF_RANGE (route-specific error_code, not the
        # generic PARAM_OUT_OF_RANGE).
        t_descriptor = {"value": threshold, "unit": None,
                        "name": {"en": "Threshold for anomaly detection",
                                 "de": "Schwellwert für die Anomalieerkennung"}}
        from domains.adjustments.services.anomaly_validators import (
            check_float as _check_float,
        )
        t_descriptor["value"] = _check_float(t_descriptor, lang)
        if not math.isfinite(t_descriptor["value"]) or t_descriptor["value"] < 0:
            raise ThresholdOutOfRangeError(
                value=t_descriptor["value"],
                min_value=0,
                suggestions=[_tr(
                    "Threshold must be a non-negative number.",
                    "Der Schwellenwert muss eine nicht-negative Zahl sein.",
                    lang,
                )],
            )

        if not _try_acquire_pipeline(upload_id, g.user_id):
            return jsonify({"error": "Pipeline already running for this upload"}), 409
        state["pipeline_status"] = _PipelineStatus.APPLYING_STL

        par = state["params"]
        pipeline_started_at = time.time()
        progress_cb = _make_progress_callback(
            upload_id, started_at=pipeline_started_at,
            phase_keys=_compute_phase_keys(par, "stl_threshold"),
        )

        # Apply STL — NaN anomalies and interpolate.
        df, anomaly_mask = _apply_stl_threshold(
            state["processed_df"],
            stl_result,
            t_descriptor["value"],
            par["GAP_MAX"]["value"],
            par["DEC"]["value"],
            lang=lang,
        )
        state["processed_df"] = df

        # Build STL anomalies overlay (Python L1277-1305)
        # Use original_df as the canvas (NaN-ed values are gone from df).
        stl_overlay = _build_anomaly_overlay(
            state["original_df"],
            state["original_df"].columns[0],
            state["original_df"].columns[1],
            anomaly_mask,
            title=_tr("Anomalies detected by STL", "Entfernte Anomalien durch STL", lang),
            yaxis_label=str(state["original_df"].columns[1]),
            lang=lang,
            base_label=_tr("Original data", "Originaldaten", lang),
        )
        state["plots"]["stlAnomalies"] = stl_overlay
        # STL intermediate is consumed — free the memory.
        state["intermediate"]["stl_result"] = None

        # Chain into LSTM pause if also enabled.
        if par["LSTM"]["run"]:
            try:
                period = int(par["LSTM"]["var"]["PERIOD"]["value"])
                results_df, _model = _prepare_lstm(
                    df,
                    period,
                    par["LSTM"]["var"]["NEURONS"]["value"],
                    par["LSTM"]["var"]["EPOCHS"]["value"],
                    par["LSTM"]["var"]["BATCH_SIZE"]["value"],
                    lang=lang,
                    progress_callback=progress_cb,
                )
            except ValueError as e:
                # Recoverable validation failure (e.g. NaN-from-LSTM): keep pipeline
                # retryable so the user can submit a different STL threshold. Only the
                # outer except-Exception block (catastrophic errors) should set ERROR.
                state["pipeline_status"] = _PipelineStatus.AWAITING_STL_THRESHOLD
                return jsonify({"error": str(e)}), 400

            state["intermediate"]["lstm_results_df"] = results_df
            state["pipeline_status"] = _PipelineStatus.AWAITING_LSTM_THRESHOLD
            lstm_error_plot = _build_lstm_error(results_df, lang=lang)
            state["plots"]["lstmError"] = lstm_error_plot
            return jsonify({
                "status": _PipelineStatus.AWAITING_LSTM_THRESHOLD,
                "plots": {"stlAnomalies": stl_overlay, "lstmError": lstm_error_plot},
                "sessionId": upload_id,
            }), 200

        # Else: finalize.
        plots = _finalize_complete(state, par, lang, progress_cb)
        csv_path = _save_processed_csv(state, lang)
        return jsonify({
            "status": _PipelineStatus.COMPLETE,
            "plots": {"stlAnomalies": stl_overlay, "processed": plots["processed"]},
            "processedCsvFilename": _Path(csv_path).name if csv_path else None,
            "sessionId": upload_id,
        }), 200

    except AnomalyException as exc:
        # Recoverable validation failure — keep pipeline_status retryable so the
        # user can submit a different threshold. Only catastrophic exceptions
        # (generic except Exception below) should set ERROR.
        logger.warning(f"Anomaly validation rejected in /stl-threshold: {exc.error_code} — {exc.message}")
        return jsonify({
            "ok": False,
            "error": exc.message,
            "error_code": exc.error_code,
            "details": exc.details,
            "suggestions": exc.suggestions,
        }), 400
    except Exception as e:
        try:
            err_state = _get_anomaly(upload_id, g.user_id) if upload_id else None
            if err_state is not None:
                err_state["pipeline_status"] = _PipelineStatus.ERROR
        except Exception:
            pass
        logger.error(f"Error in /stl-threshold: {str(e)}\n{traceback.format_exc()}")
        return jsonify({"error": _internal_error_message(e)}), 500


@bp.route('/lstm-threshold', methods=['POST'])
@require_auth
@require_subscription
@log_request
def anomaly_lstm_threshold() -> Tuple[Response, int]:
    """Apply user-supplied LSTM threshold and finalize. Body: {uploadId, threshold, lang}."""
    cleanup_all_expired_data()

    upload_id = None
    try:
        data = request.get_json(silent=True) or {}
        upload_id, _uid_err = _coerce_upload_id(data)
        if _uid_err:
            return _uid_err
        threshold = data.get("threshold")
        lang = _normalize_lang(data.get("lang"))

        if not upload_id:
            return jsonify({"error": "uploadId is required"}), 400

        state = _get_anomaly(upload_id, g.user_id)
        if state is None:
            return jsonify({"error": f"No upload found for ID '{upload_id}'"}), 404
        if state.get("pipeline_status") != _PipelineStatus.AWAITING_LSTM_THRESHOLD:
            return jsonify({"error": "Pipeline is not awaiting an LSTM threshold"}), 409
        try:
            _ensure_lstm_intermediate(state, lang)
        except ValueError as e:
            state["pipeline_status"] = _PipelineStatus.ERROR
            return jsonify({"error": str(e)}), 400
        results_df = state.get("intermediate", {}).get("lstm_results_df")
        if results_df is None:
            return jsonify({"error": "LSTM intermediate state missing — re-run /start"}), 409

        # check_float failures (PARAM_NOT_FLOAT) propagate to the outer
        # except AnomalyException handler. A negative numeric threshold
        # emits THRESHOLD_OUT_OF_RANGE (route-specific error_code, not the
        # generic PARAM_OUT_OF_RANGE).
        t_descriptor = {"value": threshold, "unit": None,
                        "name": {"en": "Threshold for anomaly detection",
                                 "de": "Schwellwert für die Anomalieerkennung"}}
        from domains.adjustments.services.anomaly_validators import (
            check_float as _check_float,
        )
        t_descriptor["value"] = _check_float(t_descriptor, lang)
        if not math.isfinite(t_descriptor["value"]) or t_descriptor["value"] < 0:
            raise ThresholdOutOfRangeError(
                value=t_descriptor["value"],
                min_value=0,
                suggestions=[_tr(
                    "Threshold must be a non-negative number.",
                    "Der Schwellenwert muss eine nicht-negative Zahl sein.",
                    lang,
                )],
            )

        if not _try_acquire_pipeline(upload_id, g.user_id):
            return jsonify({"error": "Pipeline already running for this upload"}), 409
        state["pipeline_status"] = _PipelineStatus.APPLYING_LSTM

        par = state["params"]
        pipeline_started_at = time.time()
        progress_cb = _make_progress_callback(
            upload_id, started_at=pipeline_started_at,
            phase_keys=_compute_phase_keys(par, "lstm_threshold"),
        )

        df, anomalies_df = _apply_lstm_threshold(
            state["processed_df"],
            results_df,
            t_descriptor["value"],
            par["GAP_MAX"]["value"],
            par["DEC"]["value"],
            lang=lang,
        )
        state["processed_df"] = df

        # Build LSTM anomalies overlay (Python L1423-1453) — uses the
        # results_df rows; NOT every original_df row had a forecast.
        anomaly_mask_full = (
            state["original_df"].iloc[:, 0]
                .isin(anomalies_df["timestamp"])
                .values
        )
        lstm_overlay = _build_anomaly_overlay(
            state["original_df"],
            state["original_df"].columns[0],
            state["original_df"].columns[1],
            anomaly_mask_full,
            title=_tr("Detected anomalies by LSTM", "Erkannte Anomalien durch LSTM", lang),
            yaxis_label=str(state["original_df"].columns[1]),
            lang=lang,
            base_label=_tr("Original value", "Originalwert", lang),
        )
        state["plots"]["lstmAnomalies"] = lstm_overlay
        state["intermediate"]["lstm_results_df"] = None

        plots = _finalize_complete(state, par, lang, progress_cb)
        csv_path = _save_processed_csv(state, lang)
        return jsonify({
            "status": _PipelineStatus.COMPLETE,
            "plots": {"lstmAnomalies": lstm_overlay, "processed": plots["processed"]},
            "processedCsvFilename": _Path(csv_path).name if csv_path else None,
            "sessionId": upload_id,
        }), 200

    except AnomalyException as exc:
        # Recoverable validation failure — keep pipeline_status retryable so the
        # user can submit a different threshold. Only catastrophic exceptions
        # (generic except Exception below) should set ERROR.
        logger.warning(f"Anomaly validation rejected in /lstm-threshold: {exc.error_code} — {exc.message}")
        return jsonify({
            "ok": False,
            "error": exc.message,
            "error_code": exc.error_code,
            "details": exc.details,
            "suggestions": exc.suggestions,
        }), 400
    except Exception as e:
        try:
            err_state = _get_anomaly(upload_id, g.user_id) if upload_id else None
            if err_state is not None:
                err_state["pipeline_status"] = _PipelineStatus.ERROR
        except Exception:
            pass
        logger.error(f"Error in /lstm-threshold: {str(e)}\n{traceback.format_exc()}")
        return jsonify({"error": _internal_error_message(e)}), 500


@bp.route('/use-processed', methods=['POST'])
@require_auth
@require_subscription
@log_request
def anomaly_use_processed() -> Tuple[Response, int]:
    """
    Promote processed_df to original_df so the user can iterate the pipeline
    on the cleaned data without re-uploading. Returns refreshed {original, slope} plots.

    Markus's requirement: "Wenn man diesen Button betätigt, dann sollen die
    verarbeiteten Daten zu den Originaldaten werden."
    """
    cleanup_all_expired_data()

    try:
        data = request.get_json(silent=True) or {}
        upload_id, _uid_err = _coerce_upload_id(data)
        if _uid_err:
            return _uid_err
        lang = _normalize_lang(data.get("lang"))

        if not upload_id:
            return jsonify({"error": "uploadId is required"}), 400

        state = _get_anomaly(upload_id, g.user_id)
        if state is None:
            return jsonify({"error": f"No upload found for ID '{upload_id}'"}), 404
        if state.get("processed_df") is None:
            return jsonify({"error": "No processed data available — run pipeline first"}), 409
        # Block while pipeline is mid-run; only promote after a clean COMPLETE
        # to avoid clobbering /start state mid-flight (race with /start, etc.).
        if state.get("pipeline_status") != _PipelineStatus.COMPLETE:
            return jsonify({"error": "Pipeline must complete before promoting processed data"}), 409

        # Promote
        state["original_df"] = state["processed_df"].copy()
        state["processed_df"] = None
        state["pipeline_status"] = _PipelineStatus.LOADED
        state["plots"] = {}
        state["intermediate"] = {"stl_result": None, "lstm_results_df": None}
        state["lang"] = lang

        df = state["original_df"]
        original_plot = _build_original_plot(df, lang=lang)
        df_slope = _slope_calc(df)
        slope_plot = _build_slope_plot(df_slope, lang=lang)

        return jsonify({
            "status": _PipelineStatus.LOADED,
            "plots": {"original": original_plot, "slope": slope_plot},
            "rowCount": int(len(df)),
            "sessionId": upload_id,
        }), 200

    except Exception as e:
        logger.error(f"Error in /use-processed: {str(e)}\n{traceback.format_exc()}")
        return jsonify({"error": _internal_error_message(e)}), 500


@bp.route('/cancel-pipeline', methods=['POST'])
@require_auth
@require_subscription
@log_request
def cancel_pipeline():
    """Release the pipeline lock so the user can immediately retry /start.

    Idempotent: if pipeline already idle/loaded, still returns 200.
    Returns 404 if upload_id is unknown OR owned by a different user.
    """
    data = request.get_json(silent=True) or {}
    upload_id = data.get('uploadId')
    if not isinstance(upload_id, str) or not upload_id:
        return jsonify({'error': 'uploadId required'}), 400

    from domains.adjustments.services.state_manager import (
        get_anomaly_state as _cancel_get_state,
        reset_anomaly_intermediate as _cancel_reset,
    )
    state = _cancel_get_state(upload_id, g.user_id)
    if state is None:
        return jsonify({'error': 'Upload not found'}), 404

    _cancel_reset(upload_id, g.user_id)
    logger.info(f'Pipeline cancelled for upload_id={upload_id} by user={g.user_id[:8]}...')
    return jsonify({'status': 'cancelled'}), 200


@bp.route('/processed/<upload_id>/<filename>', methods=['GET'])
@require_auth
@require_subscription
@log_request
def anomaly_processed_download(upload_id: str, filename: str):
    """
    Stream a previously written processed CSV. Authenticated; the file must
    belong to a session owned by the caller (otherwise 404 — enumeration-safe).
    """
    cleanup_all_expired_data()

    state = _get_anomaly(upload_id, g.user_id)
    if state is None:
        return jsonify({"error": "Not found"}), 404

    try:
        clean_filename = sanitize_filename(filename)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    upload_dir = _Path(UPLOAD_FOLDER) / upload_id
    candidate = (upload_dir / clean_filename).resolve()
    try:
        candidate.relative_to(_Path(UPLOAD_FOLDER).resolve())
    except ValueError:
        return jsonify({"error": "Invalid path"}), 400

    if not candidate.exists() or not candidate.is_file():
        return jsonify({"error": "Not found"}), 404

    from flask import send_file
    return send_file(
        candidate,
        mimetype='text/csv',
        as_attachment=True,
        download_name=clean_filename,
    )
