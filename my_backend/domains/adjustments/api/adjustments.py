"""
Adjustments API Routes
Handles data adjustment operations including chunked uploads and data processing
"""
import os
import time
import math
import logging
import traceback
from typing import Tuple

from flask import Blueprint, request, jsonify, g, Response
import pandas as pd

from shared.auth.jwt import require_auth
from shared.auth.subscription import require_subscription, check_processing_limit
from shared.tracking.usage import increment_processing_count, update_storage_usage

from domains.adjustments.config import UPLOAD_FOLDER, VALID_METHODS
from domains.adjustments.services.state_manager import (
    adjustment_chunks,
    adjustment_chunks_timestamps,
    chunk_buffer,
    chunk_buffer_timestamps,
    cleanup_all_expired_data,
    get_file_info_from_cache,
    check_files_need_methods
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
                result = analyse_data(final_path, upload_id)

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


@bp.route('/adjustdata/complete', methods=['POST', 'OPTIONS'])
@require_auth
@require_subscription
@check_processing_limit
def complete_adjustment() -> Tuple[Response, int]:
    """
    Complete the adjustment processing.
    """
    if request.method == 'OPTIONS':
        response = jsonify({'success': True})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'POST,OPTIONS')
        return response, 200

    try:
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
        total_rows = sum(len(df) for df in dataframes.values())
        file_rows_dict = {filename: len(df) for filename, df in dataframes.items()}

        tracker = ProgressTracker(
            upload_id=upload_id,
            total_files=len(filenames),
            total_rows=total_rows
        )
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
