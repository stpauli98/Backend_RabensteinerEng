"""
CSV Data Upload and Processing Routes

This module handles chunked CSV file uploads with the following features:
- Chunked upload support for large files
- Multiple datetime format support with auto-detection
- Timezone conversion to UTC
- Real-time progress tracking via WebSocket
- Usage tracking and quota enforcement

Main endpoints:
- POST /upload-chunk: Upload individual file chunks
- POST /finalize-upload: Complete upload and process file
- POST /cancel-upload: Cancel in-progress upload
- POST /prepare-save: Prepare processed data for download
- GET /download/<file_id>: Download processed CSV file
"""

import json
import csv
import traceback
from io import StringIO
from typing import Dict, Tuple, Optional, Any

from flask import Blueprint, request, jsonify, current_app, g, Response, redirect
import pandas as pd

from shared.auth.jwt import require_auth
from shared.auth.subscription import require_subscription, check_processing_limit
from shared.tracking.usage import increment_processing_count, update_storage_usage
from shared.storage.service import storage_service
from shared.exceptions import (
    MissingParameterError,
    DelimiterMismatchError,
    CSVParsingError,
    DateTimeParsingError,
    EncodingError,
    UploadNotFoundError,
    TimezoneConversionError,
    LoadDataException
)

from domains.upload.config import SUPPORTED_ENCODINGS
from domains.upload.services.progress import ProgressTracker
from domains.upload.services.datetime_parser import DateTimeParser
from domains.upload.services.state_manager import chunk_storage, temp_files
from domains.upload.services.csv_utils import (
    detect_delimiter,
    clean_time,
    clean_file_content
)


# Global parser instance
_datetime_parser = DateTimeParser()


def get_socketio():
    """Get the SocketIO instance from the Flask app extensions."""
    return current_app.extensions['socketio']


bp = Blueprint('load_row_data', __name__)


def _error_response(error_code: str, message: str, status_code: int = 400) -> Tuple[Response, int]:
    """
    Create standardized error response.
    """
    return jsonify({"error": error_code, "message": message}), status_code


@bp.route('/upload-chunk', methods=['POST'])
@require_auth
def upload_chunk() -> Tuple[Response, int]:
    """
    Handle chunked file upload.
    """
    try:
        if 'fileChunk' not in request.files:
            return jsonify({"error": "Chunk file not found"}), 400

        required_params = ['uploadId', 'chunkIndex', 'totalChunks', 'delimiter', 'selected_columns', 'timezone', 'dropdown_count', 'hasHeader']
        missing_params = [param for param in required_params if param not in request.form]
        if missing_params:
            return jsonify({"error": f"Missing required parameters: {', '.join(missing_params)}"}), 400

        upload_id = request.form['uploadId']
        chunk_index = int(request.form['chunkIndex'])
        total_chunks = int(request.form['totalChunks'])

        if upload_id not in chunk_storage:
            try:
                selected_columns_str = request.form.get('selected_columns')
                if not selected_columns_str:
                    return jsonify({"error": "selected_columns parameter is required"}), 400
                selected_columns = json.loads(selected_columns_str)
                if not isinstance(selected_columns, dict):
                    return jsonify({"error": "selected_columns must be a JSON object"}), 400

                chunk_storage[upload_id] = {
                    'chunks': {},
                    'total_chunks': total_chunks,
                    'received_chunks': 0,
                    'last_activity': 0,
                    'parameters': {
                        'delimiter': request.form.get('delimiter'),
                        'timezone': request.form.get('timezone', 'UTC'),
                        'has_header': request.form.get('hasHeader', 'nein'),
                        'selected_columns': selected_columns,
                        'custom_date_format': request.form.get('custom_date_format'),
                        'value_column_name': request.form.get('valueColumnName', '').strip(),
                        'dropdown_count': int(request.form.get('dropdown_count', '2'))
                    }
                }
            except json.JSONDecodeError:
                return jsonify({"error": "Invalid JSON format for selected_columns"}), 400
        else:
            chunk_storage[upload_id]['total_chunks'] = max(chunk_storage[upload_id]['total_chunks'], total_chunks)

        file_chunk = request.files['fileChunk']
        chunk_content = file_chunk.read()
        chunk_storage[upload_id]['chunks'][chunk_index] = chunk_content
        chunk_storage[upload_id]['received_chunks'] += 1

        return jsonify({
            "message": f"Chunk {chunk_index + 1}/{chunk_storage[upload_id]['total_chunks']} received",
            "uploadId": upload_id,
            "remainingChunks": chunk_storage[upload_id]['total_chunks'] - chunk_storage[upload_id]['received_chunks']
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@bp.route('/finalize-upload', methods=['POST'])
@require_auth
@require_subscription
@check_processing_limit
def finalize_upload() -> Tuple[Response, int]:
    """
    Finalize chunked upload and process the complete file.
    """
    try:
        data = request.get_json(force=True, silent=True)

        if not data or 'uploadId' not in data:
            return jsonify({"error": "uploadId is required"}), 400

        upload_id = data['uploadId']

        if upload_id not in chunk_storage:
            return jsonify({"error": "Upload not found or already processed"}), 404

        if chunk_storage[upload_id]['received_chunks'] != chunk_storage[upload_id]['total_chunks']:
            remaining = chunk_storage[upload_id]['total_chunks'] - chunk_storage[upload_id]['received_chunks']
            return jsonify({
                "error": f"Not all chunks received. Missing {remaining} chunks.",
                "received": chunk_storage[upload_id]['received_chunks'],
                "total": chunk_storage[upload_id]['total_chunks']
            }), 400

        return process_chunks(upload_id)
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@bp.route('/cancel-upload', methods=['POST'])
@require_auth
def cancel_upload() -> Tuple[Response, int]:
    """
    Cancel an in-progress upload.
    """
    try:
        data = request.get_json(force=True, silent=True)

        if not data or 'uploadId' not in data:
            return jsonify({"error": "uploadId is required"}), 400

        upload_id = data['uploadId']

        if upload_id in chunk_storage:
            del chunk_storage[upload_id]

            socketio = get_socketio()
            socketio.emit('upload_progress', {
                'uploadId': upload_id,
                'progress': 0,
                'status': 'error',
                'message': 'Upload canceled by user'
            }, room=upload_id)

        return jsonify({
            "success": True,
            "message": "Upload canceled successfully"
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400


def process_chunks(upload_id: str) -> Tuple[Response, int]:
    """
    Process and decode uploaded file chunks.
    """
    try:
        socketio = get_socketio()

        upload_data = chunk_storage[upload_id]
        chunks = [upload_data['chunks'][i] for i in range(upload_data['total_chunks'])]

        combined_bytes = b"".join(chunks)

        encodings = SUPPORTED_ENCODINGS
        full_content = None

        for encoding in encodings:
            try:
                decoded = combined_bytes.decode(encoding)

                first_line = decoded.split('\n')[0] if decoded else ''

                has_delimiter = any(d in first_line for d in [',', ';', '\t'])
                printable_ratio = sum(1 for c in first_line[:200] if ord(c) < 256 and (c.isprintable() or c in '\n\r\t')) / max(len(first_line[:200]), 1)

                if has_delimiter and printable_ratio > 0.9:
                    full_content = decoded
                    break

            except UnicodeDecodeError:
                continue

        if full_content is None:
            error = EncodingError(
                reason="Could not decode file content with any supported encoding",
                details={'tried_encodings': encodings}
            )
            socketio.emit('upload_progress', {
                'uploadId': upload_id,
                'progress': 0,
                'status': 'error',
                'message': error.message
            }, room=upload_id)
            return jsonify(error.to_dict()), 400

        params = upload_data['parameters']
        params['uploadId'] = upload_id
        del chunk_storage[upload_id]

        return upload_files(full_content, params)
    except LoadDataException as e:
        from flask import has_app_context
        if has_app_context():
            return jsonify(e.to_dict()), 400
        else:
            raise
    except KeyError:
        error = UploadNotFoundError(upload_id)
        from flask import has_app_context
        if has_app_context():
            return jsonify(error.to_dict()), 404
        else:
            raise error
    except Exception as e:
        from flask import has_app_context
        if has_app_context():
            return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500
        else:
            raise


def _validate_and_extract_params(
    params: Dict[str, Any],
    file_content: str
) -> Dict[str, Any]:
    """
    Validate and extract parameters from upload request.
    """
    delimiter = params.get('delimiter')
    if not delimiter:
        raise MissingParameterError('delimiter')

    # Validate delimiter against detected
    detected_delimiter = detect_delimiter(file_content)
    if delimiter != detected_delimiter:
        raise DelimiterMismatchError(
            provided=delimiter,
            detected=detected_delimiter
        )

    timezone = params.get('timezone', 'UTC')
    selected_columns = params.get('selected_columns', {})
    custom_date_format = params.get('custom_date_format')
    value_column_name = params.get('value_column_name', '').strip()
    dropdown_count = int(params.get('dropdown_count', '2'))
    has_separate_date_time = dropdown_count == 3
    has_header = params.get('has_header', False)
    upload_id = params.get('uploadId')

    date_column = selected_columns.get('column1')
    time_column = selected_columns.get('column2') if has_separate_date_time else None
    value_column = (
        selected_columns.get('column3') if has_separate_date_time
        else selected_columns.get('column2')
    )

    return {
        'upload_id': upload_id,
        'delimiter': delimiter,
        'timezone': timezone,
        'custom_date_format': custom_date_format,
        'value_column_name': value_column_name,
        'has_separate_date_time': has_separate_date_time,
        'has_header': has_header,
        'date_column': date_column,
        'time_column': time_column,
        'value_column': value_column,
    }


def _parse_csv_to_dataframe(
    file_content: str,
    validated_params: Dict[str, Any]
) -> pd.DataFrame:
    """
    Parse CSV content to pandas DataFrame.
    """
    delimiter = validated_params['delimiter']
    has_header = validated_params['has_header']
    value_column = validated_params['value_column']

    cleaned_content = clean_file_content(file_content, delimiter)

    try:
        df = pd.read_csv(
            StringIO(cleaned_content),
            delimiter=delimiter,
            header=0 if has_header == 'ja' else None
        )

        if has_header == 'nein':
            df.columns = [str(i) for i in range(len(df.columns))]
        else:
            df.columns = [col.strip() for col in df.columns]

        df = df.dropna(axis=1, how='all')
        df.columns = df.columns.astype(str)
        df.columns = [col.strip() for col in df.columns]

    except Exception as e:
        raise CSVParsingError(
            reason=str(e),
            original_exception=e
        )

    if df.empty:
        raise CSVParsingError(reason="No data loaded from file")

    # Convert value column to numeric
    if value_column and value_column in df.columns:
        df[value_column] = pd.to_numeric(df[value_column], errors='coerce')

    return df


def _process_datetime_columns(
    df: pd.DataFrame,
    validated_params: Dict[str, Any]
) -> pd.DataFrame:
    """
    Process and parse datetime columns in DataFrame.
    """
    has_separate_date_time = validated_params['has_separate_date_time']
    date_column = validated_params['date_column']
    time_column = validated_params['time_column']
    custom_date_format = validated_params['custom_date_format']

    try:
        datetime_col = date_column or df.columns[0]

        if has_separate_date_time and date_column and time_column:
            # Clean time columns
            df[time_column] = df[time_column].apply(clean_time)
            df[date_column] = df[date_column].apply(clean_time)

            # Extract date-only part if date_column contains datetime with dummy time
            sample_date = str(df[date_column].iloc[0])
            if ' ' in sample_date or 'T' in sample_date:
                df['date_only'] = df[date_column].astype(str).str.split(' ').str[0].str.split('T').str[0]
            else:
                df['date_only'] = df[date_column].astype(str)

            # Combine date + time
            df['datetime'] = (
                df['date_only'] + ' ' +
                df[time_column].astype(str)
            )

            # Try parsing
            success, parsed_dates, err = _datetime_parser.parse_series(df['datetime'])

            # Retry with custom format if needed
            if not success and custom_date_format:
                success, parsed_dates, err = _datetime_parser.parse_series(
                    df['datetime'], custom_format=custom_date_format
                )

            if not success:
                raise DateTimeParsingError(
                    column='datetime',
                    format_info="Format nicht unterstützt. Beispielwert: %d.%m.%Y %H:%M:%S"
                )
        else:
            # Single datetime column
            success, parsed_dates, err = _datetime_parser.parse_series(
                df[datetime_col], custom_format=custom_date_format
            )

            if not success:
                raise DateTimeParsingError(
                    column=datetime_col,
                    format_info=err or "Unsupported datetime format"
                )

        df['datetime'] = parsed_dates
        return df

    except DateTimeParsingError:
        raise
    except Exception as e:
        raise DateTimeParsingError(
            format_info=f"Error parsing date/time: {str(e)}",
            original_exception=e
        )


def convert_to_utc(df: pd.DataFrame, date_column: str, timezone: str = 'UTC') -> pd.DataFrame:
    """
    Convert datetime column to UTC timezone.
    """
    try:
        df = df.copy()
        df[date_column] = _datetime_parser.convert_to_utc(df[date_column], timezone)
        return df
    except Exception as e:
        raise TimezoneConversionError(
            from_tz=timezone,
            to_tz='UTC',
            reason=str(e),
            original_exception=e
        )


def upload_files(file_content: str, params: Dict[str, Any]) -> Tuple[Response, int]:
    """
    Process uploaded CSV file and convert to standardized format.
    """
    try:
        socketio = get_socketio()
        upload_id = params.get('uploadId')

        file_size_bytes = len(file_content.encode('utf-8'))

        # Initialize ProgressTracker
        tracker = ProgressTracker(upload_id, socketio, file_size_bytes=file_size_bytes)
        tracker.total_steps = 5

        # Step 1: Validate and extract parameters (5-15%)
        tracker.current_step = 1
        tracker.start_phase('validation')
        tracker.emit('validating', 5, 'validating_params', force=True)

        try:
            validated_params = _validate_and_extract_params(params, file_content)
        except LoadDataException as e:
            return jsonify(e.to_dict()), 400

        tracker.end_phase('validation')
        tracker.emit('validating', 15, 'params_validated', force=True)

        # Step 2: Parse CSV to DataFrame (15-40%)
        tracker.current_step = 2
        tracker.start_phase('parsing')
        tracker.emit('parsing', 15, 'parsing_csv', force=True)

        try:
            df = _parse_csv_to_dataframe(file_content, validated_params)
            total_rows = len(df)
            tracker.set_total_rows(total_rows)
            tracker.emit('parsing', 40, 'csv_parsed', force=True, message_params={'rowCount': total_rows})
        except LoadDataException as e:
            return jsonify(e.to_dict()), 400

        tracker.end_phase('parsing')

        # Step 3: Process datetime columns (40-60%)
        tracker.current_step = 3
        tracker.start_phase('datetime')
        tracker.emit('datetime', 40, 'processing_datetime', force=True)

        try:
            df = _process_datetime_columns(df, validated_params)
            tracker.emit('datetime', 60, 'datetime_processed', force=True)
        except LoadDataException as e:
            return jsonify(e.to_dict()), 400

        tracker.end_phase('datetime')

        # Step 4: Convert to UTC (60-75%)
        tracker.current_step = 4
        tracker.start_phase('utc')
        tracker.emit('utc', 60, 'converting_to_utc', force=True, message_params={'timezone': validated_params["timezone"]})

        try:
            df = convert_to_utc(df, 'datetime', validated_params['timezone'])
            tracker.emit('utc', 75, 'utc_conversion_complete', force=True)
        except LoadDataException as e:
            tracker.emit('error', 0, 'error_occurred', force=True, message_params={'error': e.message})
            return jsonify(e.to_dict()), 400

        tracker.end_phase('utc')

        # Step 5: Build result DataFrame and save (75-100%)
        tracker.current_step = 5
        tracker.start_phase('saving')
        tracker.emit('saving', 75, 'creating_result_dataframe', force=True)

        try:
            value_column = validated_params['value_column']
            value_column_name = validated_params['value_column_name']

            if not value_column or value_column not in df.columns:
                raise ValueError("Datum, Wert 1 oder Wert 2 nicht ausgewählt")

            result_df = pd.DataFrame()
            result_df['UTC'] = df['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')

            final_value_column = value_column_name if value_column_name else value_column
            result_df[final_value_column] = df[value_column].apply(
                lambda x: str(x) if pd.notnull(x) else ""
            )

            result_df.dropna(subset=['UTC'], inplace=True)
            result_df.sort_values('UTC', inplace=True)
            result_df.reset_index(drop=True, inplace=True)

            total_rows = len(result_df)
            tracker.emit('saving', 85, 'result_created', force=True, message_params={'rowCount': total_rows})

            # Save to Supabase Storage
            tracker.emit('saving', 90, 'saving_to_cloud_storage', force=True)

            csv_content = result_df.to_csv(sep=';', index=False)

            user_id = g.user_id
            file_id = storage_service.upload_csv(
                user_id=user_id,
                csv_content=csv_content,
                original_filename=f"processed_{upload_id}.csv",
                metadata={
                    'totalRows': total_rows,
                    'headers': result_df.columns.tolist(),
                    'uploadId': upload_id
                }
            )

            if not file_id:
                raise ValueError("Failed to upload file to storage")

            tracker.emit('saving', 95, 'cloud_storage_saved', force=True)

            # Generate preview (first 100 rows)
            preview_rows = min(100, total_rows)
            preview_data = []

            preview_data.append(result_df.columns.tolist())

            for _, row in result_df.head(preview_rows).iterrows():
                preview_data.append([row['UTC'], row[final_value_column]])

            tracker.emit('complete', 100, 'processing_complete', force=True, message_params={'rowCount': total_rows})

        except Exception as e:
            tracker.emit('error', 0, 'error_occurred', force=True, message_params={'error': str(e)})
            return jsonify({"error": str(e)}), 400

        tracker.end_phase('saving')

        # Track usage metrics
        try:
            increment_processing_count(g.user_id)
            file_size_mb = file_size_bytes / (1024 * 1024)
            update_storage_usage(g.user_id, file_size_mb)
        except Exception:
            pass

        return jsonify({
            "success": True,
            "message": "File processed successfully",
            "fileId": file_id,
            "totalRows": total_rows,
            "headers": result_df.columns.tolist(),
            "previewRowCount": preview_rows,
            "preview": preview_data
        }), 200

    except LoadDataException as e:
        socketio = get_socketio()
        socketio.emit('upload_progress', {
            'uploadId': params.get('uploadId'),
            'progress': 0,
            'status': 'error',
            'message': f'Error: {e.message}'
        }, room=params.get('uploadId'))

        return jsonify(e.to_dict()), 400
    except Exception as e:
        socketio = get_socketio()
        socketio.emit('upload_progress', {
            'uploadId': params.get('uploadId'),
            'progress': 0,
            'status': 'error',
            'message': f'Unexpected error: {str(e)}'
        }, room=params.get('uploadId'))

        return jsonify({"error": str(e)}), 500


@bp.route('/prepare-save', methods=['POST'])
@require_auth
def prepare_save() -> Tuple[Response, int]:
    """
    Prepare merged/processed data for download.
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

        output = StringIO()
        writer = csv.writer(output, delimiter=';')
        for row in save_data:
            writer.writerow(row)
        csv_content = output.getvalue()

        user_id = g.user_id

        file_id = storage_service.upload_csv(
            user_id=user_id,
            csv_content=csv_content,
            original_filename=file_name or "merged_data.csv",
            metadata={
                'totalRows': len(save_data) - 1,
                'source': 'prepare-save',
                'merged': True
            }
        )

        if not file_id:
            return jsonify({"error": "Failed to save file to storage"}), 500

        return jsonify({
            "message": "File prepared for download",
            "fileId": file_id,
            "totalRows": len(save_data) - 1
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.route('/merge-and-prepare', methods=['POST'])
@require_auth
def merge_and_prepare() -> Tuple[Response, int]:
    """
    Merge multiple processed files from Supabase Storage into one file.
    """
    try:
        data = request.json

        if not data:
            return jsonify({"error": "No data received"}), 400

        file_ids = data.get('fileIds', [])
        file_name = data.get('fileName', 'merged_data.csv')

        if not file_ids:
            return jsonify({"error": "No file IDs provided"}), 400

        if len(file_ids) == 1:
            return jsonify({
                "message": "Single file, no merge needed",
                "fileId": file_ids[0],
                "downloadFileId": file_ids[0]
            }), 200

        all_dataframes = []
        headers = None

        for file_id in file_ids:
            csv_content = storage_service.download_csv(file_id)

            if not csv_content:
                return jsonify({"error": f"Failed to download file: {file_id}"}), 404

            df = pd.read_csv(StringIO(csv_content), sep=';')

            if headers is None:
                headers = list(df.columns)
            all_dataframes.append(df)

        merged_df = pd.concat(all_dataframes, ignore_index=True)

        if 'UTC' in merged_df.columns:
            merged_df['UTC'] = pd.to_datetime(merged_df['UTC'], errors='coerce')
            merged_df = merged_df.sort_values('UTC')
            merged_df['UTC'] = merged_df['UTC'].dt.strftime('%Y-%m-%d %H:%M:%S')

        merged_df = merged_df.drop_duplicates()

        csv_content = merged_df.to_csv(sep=';', index=False)

        user_id = g.user_id

        merged_file_id = storage_service.upload_csv(
            user_id=user_id,
            csv_content=csv_content,
            original_filename=file_name,
            metadata={
                'totalRows': len(merged_df),
                'source': 'merge-and-prepare',
                'merged': True,
                'sourceFiles': len(file_ids)
            }
        )

        if not merged_file_id:
            return jsonify({"error": "Failed to save merged file"}), 500

        # Clean up source files
        deleted_count = 0
        for file_id in file_ids:
            try:
                if storage_service.delete_file(file_id):
                    deleted_count += 1
            except Exception:
                pass

        return jsonify({
            "message": "Files merged successfully",
            "fileId": merged_file_id,
            "downloadFileId": merged_file_id,
            "totalRows": len(merged_df),
            "sourceFilesCount": len(file_ids),
            "deletedSourceFiles": deleted_count
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.route('/download/<path:file_id>', methods=['GET'])
@require_auth
def download_file(file_id: str) -> Response:
    """
    Download prepared CSV file from Supabase Storage.
    """
    try:
        signed_url = storage_service.get_download_url(file_id, expires_in=3600)

        if signed_url:
            return redirect(signed_url)

        csv_content = storage_service.download_csv(file_id)

        if csv_content:
            response = Response(
                csv_content,
                mimetype='text/csv',
                headers={
                    'Content-Disposition': f'attachment; filename="processed_{file_id.split("/")[-1]}.csv"'
                }
            )
            return response

        return jsonify({"error": "File not found"}), 404

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.route('/cleanup-files', methods=['POST'])
@require_auth
def cleanup_files() -> Tuple[Response, int]:
    """
    Delete files from Supabase Storage after successful download.
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
                else:
                    failed_ids.append(file_id)
            except Exception:
                failed_ids.append(file_id)

        return jsonify({
            "message": "Cleanup complete",
            "deletedCount": deleted_count,
            "totalRequested": len(file_ids),
            "failedIds": failed_ids
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
