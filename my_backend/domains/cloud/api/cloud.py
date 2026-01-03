"""
Cloud API Routes
Handles cloud-based data analysis with regression and interpolation
"""
import os
import json
import shutil
import base64
import logging
import traceback
from io import StringIO

import pandas as pd
from flask import request, jsonify, Blueprint, Response

from domains.cloud.config import (
    VALID_FILE_TYPES,
    STREAMING_CHUNK_SIZE,
    FILE_BUFFER_SIZE
)
from domains.cloud.services import (
    CloudProgressTracker,
    chunk_uploads,
    sanitize_upload_id,
    validate_csv_size,
    validate_dataframe,
    get_chunk_dir,
    process_data_frames
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

bp = Blueprint('cloud', __name__)


@bp.route('/upload-chunk', methods=['POST'])
def upload_chunk():
    """
    Handle chunk upload for large files (5MB chunks).
    Frontend expects: { success: bool, data: { uploadId, progress, ... } }
    """
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'data': {'error': 'No file part in the request'}}), 400

        file_chunk = request.files['file']
        upload_id = request.form.get('uploadId')
        file_type = request.form.get('fileType')
        try:
            chunk_index = int(request.form.get('chunkIndex', 0))
            total_chunks = int(request.form.get('totalChunks', 1))
        except Exception:
            return jsonify({'success': False, 'data': {'error': 'Invalid chunk index or total chunks'}}), 400

        if not upload_id:
            return jsonify({'success': False, 'data': {'error': 'No upload ID provided'}}), 400
        if not file_type or file_type not in VALID_FILE_TYPES:
            return jsonify({'success': False, 'data': {'error': 'Invalid file type'}}), 400

        try:
            upload_id = sanitize_upload_id(upload_id)
        except ValueError as e:
            return jsonify({'success': False, 'data': {'error': str(e)}}), 400

        chunk_dir = get_chunk_dir(upload_id)

        if upload_id not in chunk_uploads:
            chunk_uploads[upload_id] = {
                'temp_file': {'total_chunks': 0, 'received_chunks': set(), 'filename': None},
                'load_file': {'total_chunks': 0, 'received_chunks': set(), 'filename': None},
                'interpolate_file': {'total_chunks': 0, 'received_chunks': set(), 'filename': None}
            }

        chunk_uploads[upload_id][file_type]['total_chunks'] = total_chunks
        chunk_uploads[upload_id][file_type]['received_chunks'].add(chunk_index)
        chunk_uploads[upload_id][file_type]['filename'] = file_chunk.filename

        chunk_path = os.path.join(chunk_dir, f"{file_type}_{chunk_index}")
        file_chunk.save(chunk_path)

        return jsonify({
            'success': True,
            'data': {
                'uploadId': upload_id,
                'progress': len(chunk_uploads[upload_id][file_type]['received_chunks']) / total_chunks,
                'chunkIndex': chunk_index,
                'totalChunks': total_chunks,
                'fileType': file_type
            }
        })
    except Exception as e:
        logger.error(f"Error in chunk upload: {str(e)}")
        traceback.print_exc()
        return jsonify({'success': False, 'data': {'error': str(e)}}), 500


@bp.route('/complete', methods=['POST', 'OPTIONS'])
def complete_redirect():
    """Handle chunked upload completion directly instead of redirecting."""
    try:
        if request.method == 'OPTIONS':
            response = jsonify({
                'success': True,
                'data': {'message': 'CORS preflight request successful'}
            })
            response.headers.add('Access-Control-Allow-Origin', '*')
            response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
            response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
            return response

        logger.info("=== HANDLING COMPLETE UPLOAD REQUEST ===")
        logger.info(f"Request method: {request.method}")
        logger.info(f"Request content type: {request.content_type}")

        if request.content_type and 'multipart/form-data' in request.content_type:
            logger.info("Processing FormData request")
            data = request.form.to_dict()
            logger.info(f"Form data: {data}")
        else:
            logger.info("Processing JSON request")
            try:
                data = request.get_json(force=True)
                logger.info(f"JSON data: {data}")
            except Exception as e:
                logger.error(f"Error parsing JSON: {str(e)}")
                return jsonify({
                    'success': False,
                    'data': {'error': 'Invalid request format. Expected JSON or FormData.'}
                }), 400

        upload_id = data.get('uploadId')
        logger.info(f"Completing upload for ID: {upload_id}")

        if not upload_id:
            logger.error("No upload ID provided")
            return jsonify({'success': False, 'data': {'error': 'No upload ID provided'}}), 400

        try:
            upload_id = sanitize_upload_id(upload_id)
        except ValueError as e:
            return jsonify({'success': False, 'data': {'error': str(e)}}), 400

        # Initialize progress tracker
        tracker = CloudProgressTracker(upload_id)
        tracker.emit('initializing', 5, 'cloud_initializing', force=True)

        if upload_id not in chunk_uploads:
            logger.error(f"Invalid upload ID: {upload_id}")
            return jsonify({'success': False, 'data': {'error': 'Invalid upload ID'}}), 400

        upload_info = chunk_uploads[upload_id]
        chunk_dir = get_chunk_dir(upload_id)

        temp_info = upload_info['temp_file']
        load_info = upload_info['load_file']

        logger.info(f"Temp file: {temp_info['received_chunks']}/{temp_info['total_chunks']} chunks")
        logger.info(f"Load file: {load_info['received_chunks']}/{load_info['total_chunks']} chunks")

        if temp_info['total_chunks'] == 0 or load_info['total_chunks'] == 0:
            logger.error(f"Missing file uploads. Temp chunks: {temp_info['total_chunks']}, Load chunks: {load_info['total_chunks']}")
            return jsonify({
                'success': False,
                'data': {
                    'error': 'Not all chunks received',
                    'temp_progress': len(temp_info['received_chunks']) / max(temp_info['total_chunks'], 1),
                    'load_progress': len(load_info['received_chunks']) / max(load_info['total_chunks'], 1)
                }
            }), 400

        if (len(temp_info['received_chunks']) != temp_info['total_chunks'] or
            len(load_info['received_chunks']) != load_info['total_chunks']):
            logger.error(f"Not all chunks received. Temp: {len(temp_info['received_chunks'])}/{temp_info['total_chunks']}, Load: {len(load_info['received_chunks'])}/{load_info['total_chunks']}")
            return jsonify({
                'success': False,
                'data': {
                    'error': 'Not all chunks received',
                    'temp_progress': len(temp_info['received_chunks']) / max(temp_info['total_chunks'], 1),
                    'load_progress': len(load_info['received_chunks']) / max(load_info['total_chunks'], 1)
                }
            }), 400

        logger.info("All chunks received, reassembling files")
        tracker.emit('assembling', 10, 'cloud_assembling_files', force=True)

        temp_file_path = os.path.join(chunk_dir, 'temp_out.csv')
        load_file_path = os.path.join(chunk_dir, 'load.csv')

        with open(temp_file_path, 'wb') as temp_file:
            for i in range(temp_info['total_chunks']):
                chunk_path = os.path.join(chunk_dir, f"temp_file_{i}")
                with open(chunk_path, 'rb') as chunk:
                    temp_file.write(chunk.read())

        tracker.emit('assembling', 15, 'cloud_assembling_temp_done')

        with open(load_file_path, 'wb') as load_file:
            for i in range(load_info['total_chunks']):
                chunk_path = os.path.join(chunk_dir, f"load_file_{i}")
                with open(chunk_path, 'rb') as chunk:
                    load_file.write(chunk.read())

        tracker.emit('assembling', 20, 'cloud_assembling_complete')
        logger.info("Files reassembled, processing data")

        try:
            tracker.emit('validating', 25, 'cloud_validating_size')
            validate_csv_size(temp_file_path)
            validate_csv_size(load_file_path)
            tracker.emit('validating', 30, 'cloud_size_validated')
        except ValueError as e:
            logger.error(f"File size validation failed: {str(e)}")
            tracker.emit('error', 0, 'cloud_validation_error', message_params={'error': str(e)}, force=True)
            return jsonify({'success': False, 'data': {'error': str(e)}}), 400

        try:
            tracker.emit('parsing', 35, 'cloud_parsing_csv')
            df1 = pd.read_csv(temp_file_path, sep=';')
            tracker.emit('parsing', 40, 'cloud_parsing_temp_done')
            df2 = pd.read_csv(load_file_path, sep=';')
            tracker.emit('parsing', 50, 'cloud_parsing_complete')

            tracker.emit('validating', 55, 'cloud_validating_dataframes')
            validate_dataframe(df1, "Temperature file")
            validate_dataframe(df2, "Load file")
            tracker.emit('validating', 60, 'cloud_dataframes_validated')

            processing_params = {
                'REG': data.get('REG', 'lin'),
                'TR': data.get('TR', 'cnt'),
                'TOL_CNT': data.get('TOL_CNT', '0'),
                'TOL_DEP': data.get('TOL_DEP', '0'),
                'TOL_DEP_EXTRA': data.get('TOL_DEP_EXTRA', '0'),
                'decimalPrecision': data.get('decimalPrecision', 'full')
            }

            logger.info(f"Processing data with parameters: {processing_params}")
            tracker.emit('processing', 65, 'cloud_processing_start', message_params={'reg_type': processing_params['REG']})

            result = process_data_frames(df1, df2, processing_params)
            tracker.emit('processing', 95, 'cloud_processing_complete')

            try:
                chunk_dir = get_chunk_dir(upload_id)
                if os.path.exists(chunk_dir):
                    shutil.rmtree(chunk_dir)
                if upload_id in chunk_uploads:
                    del chunk_uploads[upload_id]
                logger.info(f"Successfully cleaned up chunks for upload ID: {upload_id}")
            except Exception as e:
                logger.warning(f"Error cleaning up chunks: {str(e)}")

            tracker.emit('complete', 100, 'cloud_complete', force=True)
            return result
        except Exception as e:
            logger.error(f"Error processing uploaded files: {str(e)}")
            return jsonify({'success': False, 'data': {'error': f'Error processing uploaded files: {str(e)}'}}), 500
    except Exception as e:
        logger.error(f"Error in complete_redirect: {str(e)}")
        return jsonify({'success': False, 'data': {'error': str(e)}}), 500


@bp.route('/clouddata', methods=['POST'])
def clouddata():
    """Handle direct cloud data processing (non-chunked)."""
    if request.method == 'OPTIONS':
        response = jsonify({'success': True})
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
        return response

    try:
        logger.info("Received request to /clouddata")
        return _process_data()
    except Exception as e:
        logger.error(f"Error in clouddata endpoint: {str(e)}")
        traceback.print_exc()
        return jsonify({'success': False, 'data': {'error': str(e)}}), 500


def _process_data():
    """Process direct cloud data from base64 encoded files."""
    try:
        logger.info("Received request to /clouddata")
        data = request.json
        logger.info(f"Received data: {data}")

        if data is None:
            logger.error("No data received")
            return jsonify({'success': False, 'data': {'error': 'No data received'}}), 400

        temp_data = data['files'].get('temp_out.csv')
        load_data = data['files'].get('load.csv')

        if not temp_data or not load_data:
            logger.error("One or both files are empty")
            return jsonify({'success': False, 'data': {'error': 'One or both files are empty'}}), 400

        try:
            logger.info("Attempting to decode and read temperature data...")
            temp_decoded = base64.b64decode(temp_data).decode('utf-8')
            logger.debug(f"Temperature data preview: {temp_decoded[:200]}")
            df1 = pd.read_csv(StringIO(temp_decoded), sep=';')

            logger.info("Attempting to decode and read load data...")
            load_decoded = base64.b64decode(load_data).decode('utf-8')
            logger.debug(f"Load data preview: {load_decoded[:200]}")
            df2 = pd.read_csv(StringIO(load_decoded), sep=';')

            logger.info(f"Successfully read data. Shapes: {df1.shape}, {df2.shape}")
            logger.info(f"Columns in temperature file: {df1.columns.tolist()}")
            logger.info(f"Columns in load file: {df2.columns.tolist()}")
            logger.debug("First few rows of temperature file:")
            logger.debug(f"{df1.head()}")
            logger.debug(f"{df2.head()}")

            validate_dataframe(df1, "Temperature file")
            validate_dataframe(df2, "Load file")

            return process_data_frames(df1, df2, data)
        except ValueError as e:
            logger.error(f"Validation error: {str(e)}")
            return jsonify({'success': False, 'data': {'error': str(e)}}), 400
        except Exception as e:
            logger.error(f"Error reading CSV files: {str(e)}")
            return jsonify({'success': False, 'data': {'error': f'Error reading CSV files: {str(e)}'}}), 400

    except Exception as e:
        logger.error(f"Error processing data: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'success': False, 'data': {'error': str(e)}}), 500


@bp.route('/interpolate-chunked', methods=['POST'])
def interpolate_chunked():
    """
    Process a chunked file upload for interpolation.
    All responses are in format: {'success': bool, 'data': ...}
    """
    try:
        logger.info("Received request to /interpolate-chunked")

        data = request.json
        if not data or 'uploadId' not in data:
            logger.error("Missing uploadId in request")
            return jsonify({'success': False, 'data': {'error': 'Upload ID is required'}}), 400

        upload_id = data['uploadId']
        logger.info(f"Processing upload ID: {upload_id}")

        try:
            upload_id = sanitize_upload_id(upload_id)
        except ValueError as e:
            return jsonify({'success': False, 'data': {'error': str(e)}}), 400

        # Initialize progress tracker
        tracker = CloudProgressTracker(upload_id)
        tracker.emit('initializing', 5, 'cloud_interpolate_init', force=True)

        try:
            max_time_span = float(data.get('max_time_span', '60'))
            logger.info(f"Using max_time_span: {max_time_span}")
        except ValueError as e:
            logger.error(f"Invalid max_time_span value: {data.get('max_time_span')}")
            tracker.emit('error', 0, 'cloud_invalid_params', force=True)
            return jsonify({'success': False, 'data': {'error': 'Invalid max_time_span parameter'}}), 400

        if upload_id not in chunk_uploads:
            logger.error(f"Upload ID not found: {upload_id}")
            return jsonify({'success': False, 'data': {'error': 'Upload ID not found'}}), 404

        upload_info = chunk_uploads[upload_id]['interpolate_file']
        if len(upload_info['received_chunks']) < upload_info['total_chunks']:
            logger.error(f"Not all chunks received for upload {upload_id}")
            return jsonify({'success': False, 'data': {'error': f"Incomplete upload: Only {len(upload_info['received_chunks'])}/{upload_info['total_chunks']} chunks received"}}), 400

        chunk_dir = get_chunk_dir(upload_id)
        combined_file_path = os.path.join(chunk_dir, 'combined_interpolate_file.csv')

        tracker.emit('assembling', 10, 'cloud_assembling_chunks', force=True)

        with open(combined_file_path, 'wb') as outfile:
            for i in range(upload_info['total_chunks']):
                chunk_path = os.path.join(chunk_dir, f"interpolate_file_{i}")
                if os.path.exists(chunk_path):
                    with open(chunk_path, 'rb') as infile:
                        shutil.copyfileobj(infile, outfile, FILE_BUFFER_SIZE)

        logger.info(f"Combined file created at: {combined_file_path}")
        tracker.emit('assembling', 15, 'cloud_assembling_complete')

        try:
            tracker.emit('validating', 18, 'cloud_validating_size')
            validate_csv_size(combined_file_path)
            tracker.emit('validating', 20, 'cloud_size_validated')
        except ValueError as e:
            logger.error(f"File size validation failed: {str(e)}")
            tracker.emit('error', 0, 'cloud_validation_error', message_params={'error': str(e)}, force=True)
            return jsonify({'success': False, 'data': {'error': str(e)}}), 400

        try:
            tracker.emit('parsing', 25, 'cloud_detecting_separator')
            with open(combined_file_path, 'r', encoding='utf-8') as f:
                first_line = f.readline()

            if ';' in first_line:
                sep = ';'
            elif ',' in first_line:
                sep = ','
            else:
                sep = None

            logger.info(f"Using separator: {sep}")

            tracker.emit('parsing', 30, 'cloud_parsing_csv')
            df2 = pd.read_csv(combined_file_path,
                             sep=sep,
                             decimal=',',
                             engine='c')
            tracker.emit('parsing', 35, 'cloud_csv_parsed', message_params={'rows': len(df2)})

            tracker.emit('validating', 38, 'cloud_validating_dataframe')
            validate_dataframe(df2, "Interpolation file")
            tracker.emit('validating', 40, 'cloud_dataframe_validated')

        except ValueError as e:
            logger.error(f"Validation error: {str(e)}")
            tracker.emit('error', 0, 'cloud_validation_error', message_params={'error': str(e)}, force=True)
            return jsonify({'success': False, 'data': {'error': str(e)}}), 400
        except Exception as e:
            logger.error(f"Error reading CSV file: {str(e)}")
            tracker.emit('error', 0, 'cloud_parse_error', message_params={'error': str(e)}, force=True)
            return jsonify({'success': False, 'data': {'error': f'Error reading CSV file: {str(e)}'}}), 400

        if 'UTC' not in df2.columns:
            logger.error("UTC column not found in the file")
            tracker.emit('error', 0, 'cloud_utc_column_missing', force=True)
            return jsonify({'success': False, 'data': {'error': 'UTC column not found. The file must contain a UTC column with timestamps'}}), 400

        tracker.emit('processing', 45, 'cloud_detecting_columns')

        load_terms = set(['last', 'load', 'leistung', 'kw', 'w'])
        load_cols = [col for col in df2.columns if
                    any(term in str(col).lower() for term in load_terms)]

        if not load_cols:
            non_utc_cols = [col for col in df2.columns if col != 'UTC']
            if non_utc_cols:
                y_col = non_utc_cols[0]
                logger.info(f"No specific load column found, using first non-UTC column: {y_col}")
            else:
                logger.error("No suitable load column found")
                tracker.emit('error', 0, 'cloud_load_column_missing', force=True)
                return jsonify({
                    'success': False,
                    'error': 'Load column not found',
                    'message': 'The file must contain a column with load data'
                }), 400
        else:
            y_col = load_cols[0]
            logger.info(f"Found load column: {y_col}")

        tracker.emit('processing', 48, 'cloud_columns_detected', message_params={'column': y_col})

        df2 = df2[['UTC', y_col]].copy()

        if not pd.api.types.is_numeric_dtype(df2[y_col]):
            tracker.emit('processing', 50, 'cloud_converting_numeric')
            df2[y_col] = pd.to_numeric(df2[y_col].astype(str).str.replace(',', '.').str.replace(r'[^\d\-\.]', '', regex=True), errors='coerce')

        try:
            tracker.emit('datetime', 52, 'cloud_datetime_conversion')
            df2['UTC'] = pd.to_datetime(df2['UTC'], errors='coerce', cache=True)
            df2.dropna(subset=['UTC'], inplace=True)
            tracker.emit('datetime', 55, 'cloud_datetime_converted', force=True)

        except Exception as e:
            logger.error(f"Error converting UTC to datetime: {str(e)}")
            tracker.emit('error', 0, 'cloud_datetime_error', message_params={'error': str(e)}, force=True)
            return jsonify({
                'success': False,
                'error': f'Error processing timestamps: {str(e)}',
                'message': 'Please check the timestamp format in the file'
            }), 400

        df2.sort_values('UTC', inplace=True)

        df2.rename(columns={y_col: 'load'}, inplace=True)
        df_load = df2.set_index('UTC')

        if len(df_load) < 2:
            logger.error("Not enough valid data points for interpolation")
            tracker.emit('error', 0, 'cloud_insufficient_data', force=True)
            return jsonify({
                'success': False,
                'error': 'Not enough valid data points',
                'message': 'The file must contain at least 2 valid data points for interpolation'
            }), 400

        tracker.emit('interpolation', 60, 'cloud_analyzing_data', force=True)

        time_diffs = (df_load.index[1:] - df_load.index[:-1]).total_seconds() / 60
        max_gap = time_diffs.max() if len(time_diffs) > 0 else 0
        logger.info(f"Maximum time gap in data: {max_gap} minutes")

        total_minutes = (df_load.index[-1] - df_load.index[0]).total_seconds() / 60

        if total_minutes > 10000:
            resample_interval = '5min'
            logger.info(f"Large time span detected ({total_minutes} minutes), using 5-minute intervals")
        else:
            resample_interval = '1min'
            logger.info(f"Using standard 1-minute intervals")

        if not pd.api.types.is_numeric_dtype(df_load['load']):
            logger.info("Converting load column to numeric before interpolation")
            df_load['load'] = pd.to_numeric(df_load['load'], errors='coerce')

        limit = int(max_time_span)
        logger.info(f"Using interpolation limit of {limit} minutes")

        tracker.emit('interpolation', 70, 'cloud_interpolating', message_params={'limit': limit}, force=True)

        df2_resampled = df_load.copy()
        df2_resampled['load'] = df_load['load'].interpolate(method='linear', limit=limit)

        df2_resampled.reset_index(inplace=True)
        tracker.emit('interpolation', 80, 'cloud_interpolation_complete', force=True)

        original_points = len(df2)
        total_points = len(df2_resampled)
        added_points = total_points - original_points

        logger.info(f"Original points: {original_points}")
        logger.info(f"Interpolated points: {total_points}")
        logger.info(f"Added points: {added_points}")

        tracker.emit('streaming', 85, 'cloud_preparing_stream', message_params={
            'original': original_points,
            'interpolated': total_points,
            'added': added_points
        }, force=True)

        # Vectorized chart data creation (much faster than iterating)
        valid_mask = ~df2_resampled['UTC'].isna()
        valid_df = df2_resampled[valid_mask].copy()

        valid_df['load'] = valid_df['load'].apply(lambda x: 'NaN' if pd.isna(x) else x)
        valid_df['UTC'] = valid_df['UTC'].dt.strftime("%Y-%m-%d %H:%M:%S")

        chart_df = valid_df.rename(columns={'load': 'value'})[['UTC', 'value']]

        total_rows = len(chart_df)
        CHUNK_SIZE = STREAMING_CHUNK_SIZE
        total_chunks = (total_rows + CHUNK_SIZE - 1) // CHUNK_SIZE

        # Log sample for debugging
        sample_data = chart_df.head(5).to_dict('records') if len(chart_df) > 0 else []
        logger.info(f"Sample of chart data being sent: {sample_data}")
        logger.info(f"Total points in chart data: {total_rows}")

        if total_rows == 0:
            logger.error("No valid data points after processing")
            return jsonify({
                'success': False,
                'data': {
                    'error': 'No valid data points after processing',
                    'message': 'The file contains no valid data points for interpolation'
                }
            }), 400

        original_points_count = original_points

        logger.info(f"Total rows: {total_rows}, will be sent in {total_chunks} chunks")
        tracker.emit('streaming', 90, 'cloud_starting_stream', message_params={'chunks': total_chunks}, force=True)

        def generate_chunks():
            meta_data = {
                'type': 'meta',
                'total_rows': total_rows,
                'total_chunks': total_chunks,
                'added_points': added_points,
                'original_points': original_points_count,
                'success': True
            }
            yield json.dumps(meta_data, separators=(',', ':')) + '\n'

            for chunk_idx in range(total_chunks):
                start_idx = chunk_idx * CHUNK_SIZE
                end_idx = min(start_idx + CHUNK_SIZE, total_rows)

                chunk_data_list = chart_df.iloc[start_idx:end_idx].to_dict('records')

                chunk_data = {
                    'type': 'data',
                    'chunk_index': chunk_idx,
                    'data': chunk_data_list
                }

                yield json.dumps(chunk_data, separators=(',', ':')) + '\n'

            # Emit completion progress
            tracker.emit('complete', 100, 'cloud_interpolate_complete', force=True)

            yield json.dumps({
                'type': 'complete',
                'message': 'Data streaming completed',
                'success': True
            }, separators=(',', ':')) + '\n'

        def cleanup():
            """Guaranteed cleanup via call_on_close - runs even on client disconnect"""
            try:
                cleanup_dir = get_chunk_dir(upload_id)
                if os.path.exists(cleanup_dir):
                    shutil.rmtree(cleanup_dir)
                if upload_id in chunk_uploads:
                    del chunk_uploads[upload_id]
                logger.info(f"Cleanup completed for upload {upload_id}")
            except Exception as e:
                logger.warning(f"Cleanup error for {upload_id}: {e}")

        response = Response(generate_chunks(), mimetype='application/x-ndjson')
        response.call_on_close(cleanup)
        return response
    except Exception as e:
        logger.error(f"Error in interpolation-chunked endpoint: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'An error occurred during interpolation'
        }), 500
