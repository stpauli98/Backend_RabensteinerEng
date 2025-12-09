"""
CSV processing service for first_processing domain.
Handles CSV parsing, processing methods (mean, interpolation, nearest).
"""
import datetime
import json
import logging
import math
import statistics
import time
from io import StringIO

import numpy as np
import pandas as pd
from flask import Response, jsonify

from domains.processing.services.progress import ProgressTracker
from domains.processing.config import STREAMING_CHUNK_SIZE, BACKPRESSURE_DELAY

logger = logging.getLogger(__name__)


def clean_for_json(obj):
    """Convert numpy and pandas types to Python native types. Handle NaN/Inf."""
    # First check NaN/None
    if obj is None:
        return None
    if pd.isna(obj):
        return None

    # Convert numpy types
    if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32,
                         np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, (np.float16, np.float32, np.float64)):
        val = float(obj)
        # Check NaN/Inf after conversion
        if math.isnan(val) or math.isinf(val):
            return None
        return val
    elif isinstance(obj, float):
        # Native Python float - check NaN/Inf
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    elif isinstance(obj, (pd.Timestamp)):
        return obj.isoformat()
    return obj


def is_numeric(value):
    """Check if value is numeric"""
    try:
        float(value)
        return True
    except (ValueError, TypeError):
        return False


def process_csv(file_content, tss, offset, mode_input, intrpl_max, upload_id=None, tracker=None, decimal_precision='full'):
    """
    Process CSV content and return result as gzip-compressed JSON response.

    Args:
        file_content: CSV file content as string
        tss: Time step size in minutes
        offset: Offset in minutes
        mode_input: Processing mode ('mean', 'intrpl', 'nearest', 'nearest (mean)')
        intrpl_max: Maximum time for interpolation in minutes
        upload_id: Optional upload ID for Socket.IO progress tracking
        tracker: Optional existing ProgressTracker instance (from chunk assembly)
        decimal_precision: Decimal precision for rounding ('full' or integer)
    """
    # Use existing tracker or create new one
    if not tracker and upload_id:
        tracker = ProgressTracker(upload_id)

    def apply_precision(value):
        """Round value to specified decimal places."""
        if decimal_precision == 'full' or value == "nan" or not is_numeric(value):
            return value
        try:
            return round(float(value), int(decimal_precision))
        except (ValueError, TypeError):
            return value

    try:
        # === PHASE 1: PARSING (10-25%) ===
        if tracker:
            tracker.start_phase('parsing')
            tracker.emit('parsing', 10, 'fp_parsing_start', force=True)

        try:
            if tracker:
                tracker.emit('parsing', 12, 'fp_loading_csv')

            lines = file_content.strip().split('\n')
            logger.info(f"Total lines in CSV: {len(lines)}")

            if tracker:
                tracker.emit('parsing', 15, 'fp_lines_loaded', message_params={'lineCount': len(lines)})

            if len(lines) > 0:
                header = lines[0]
                logger.info(f"Header: '{header}'")
                logger.info(f"Header fields: {header.split(';')}")

            if tracker:
                tracker.emit('parsing', 17, 'fp_pandas_parsing')

            try:
                df = pd.read_csv(StringIO(file_content), delimiter=';', skipinitialspace=True, on_bad_lines='skip')
                logger.info(f"Successfully parsed CSV with {len(df)} rows after skipping bad lines")

                if tracker:
                    tracker.emit('parsing', 28, 'fp_parsing_complete', message_params={'rowCount': len(df)})
                    tracker.end_phase('parsing')
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

            # === PHASE 2: PREPROCESSING (20-30%) ===
            if tracker:
                tracker.start_phase('preprocessing')
                tracker.emit('preprocessing', 30, 'fp_type_conversion')

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

        # RAW DATA PREPARATION
        if tracker:
            tracker.emit('preprocessing', 32, 'fp_removing_duplicates')

        # Remove duplicates in raw data
        df = df.drop_duplicates(subset=[utc_col_name]).reset_index(drop=True)

        if tracker:
            tracker.emit('preprocessing', 34, 'fp_sorting_data')

        # Sort raw data by UTC
        df = df.sort_values(by=[utc_col_name])

        # Reset index in raw data
        df = df.reset_index(drop=True)

        if df.empty:
            return jsonify({"error": "Keine Daten gefunden"}), 400

        if tracker:
            tracker.emit('preprocessing', 35, 'fp_datetime_conversion')

        # TIME BOUNDARIES
        # Convert UTC to datetime objects
        df[utc_col_name] = pd.to_datetime(df[utc_col_name], format='%Y-%m-%d %H:%M:%S')

        time_min_raw = df[utc_col_name].iloc[0].to_pydatetime()
        time_max_raw = df[utc_col_name].iloc[-1].to_pydatetime()

        logger.info(f"Time range: {time_min_raw} to {time_max_raw}")

        if tracker:
            tracker.emit('preprocessing', 37, 'fp_preprocessing_complete')
            tracker.end_phase('preprocessing')

        # CONTINUOUS TIMESTAMP
        # Offset of lower time boundary in raw data
        offset_strt = datetime.timedelta(
            minutes=time_min_raw.minute,
            seconds=time_min_raw.second,
            microseconds=time_min_raw.microsecond
        )

        # Real offset in processed data [min]
        normalized_offset = abs(offset) % tss if offset >= 0 else 0

        # Lower time boundary in processed data
        time_min = time_min_raw - offset_strt
        if normalized_offset > 0:
            time_min += datetime.timedelta(minutes=normalized_offset)

        logger.info(f"Applying offset of {normalized_offset} minutes to {time_min_raw}")
        logger.info(f"Resulting start time: {time_min}")

        # Generate continuous timestamp
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

        # === PHASE 3: PROCESSING (37-90%) ===
        if tracker:
            tracker.start_phase('processing')
            tracker.total_steps = 1
            tracker.current_step = 1
            tracker.start_step(len(time_list))
            tracker.emit('processing', 37, 'fp_processing_start', force=True, message_params={'mode': mode_input})

        # Counter for raw data iteration
        i_raw = 0

        # Initialize list for processed values
        value_list = []

        # Emit frequency - every ~2% of steps or min 500 rows
        emit_frequency = max(500, len(time_list) // 50)

        # METHOD: MEAN
        if mode_input == "mean":
            if tracker:
                tracker.emit('processing', 37, 'fp_processing_start', force=True, message_params={'mode': 'mean'})

            for i in range(0, len(time_list)):
                if tracker and i % emit_frequency == 0 and i > 0:
                    tracker.update_step_progress(i)
                    progress = 37 + (i / len(time_list)) * 53
                    tracker.emit('processing', progress, 'fp_processing_progress', message_params={'current': i, 'total': len(time_list)})

                time_int_min = time_list[i] - datetime.timedelta(minutes=tss/2)
                time_int_max = time_list[i] + datetime.timedelta(minutes=tss/2)

                if i > 0:
                    i_raw -= 1
                if i > 0 and df[utc_col_name].iloc[i_raw].to_pydatetime() < time_int_min:
                    i_raw += 1

                value_int_list = []

                while (i_raw < len(df) and
                       df[utc_col_name].iloc[i_raw].to_pydatetime() <= time_int_max and
                       df[utc_col_name].iloc[i_raw].to_pydatetime() >= time_int_min):
                    if is_numeric(df[value_col_name].iloc[i_raw]):
                        value_int_list.append(float(df[value_col_name].iloc[i_raw]))
                    i_raw += 1

                if len(value_int_list) > 0:
                    value_list.append(apply_precision(statistics.mean(value_int_list)))
                else:
                    value_list.append("nan")

        # METHOD: LINEAR INTERPOLATION
        elif mode_input == "intrpl":
            if tracker:
                tracker.emit('processing', 37, 'fp_processing_start', force=True, message_params={'mode': 'intrpl'})

            i_raw = 0
            direct = 1
            time_next = None
            value_next = None
            time_prior = None
            value_prior = None

            for i in range(0, len(time_list)):
                if tracker and i % emit_frequency == 0 and i > 0:
                    tracker.update_step_progress(i)
                    progress = 37 + (i / len(time_list)) * 53
                    tracker.emit('processing', progress, 'fp_interpolation_progress', message_params={'current': i, 'total': len(time_list)})

                if direct == 1:
                    loop = True
                    while i_raw < len(df) and loop == True:
                        if df[utc_col_name].iloc[i_raw].to_pydatetime() >= time_list[i]:
                            if is_numeric(df[value_col_name].iloc[i_raw]):
                                time_next = df[utc_col_name].iloc[i_raw].to_pydatetime()
                                value_next = float(df[value_col_name].iloc[i_raw])
                                loop = False
                            else:
                                i_raw += 1
                        else:
                            i_raw += 1

                    if i_raw + 1 > len(df):
                        value_list.append("nan")
                        i_raw = 0
                        direct = 1
                    else:
                        direct = -1

                if direct == -1:
                    loop = True
                    while i_raw >= 0 and loop == True:
                        if df[utc_col_name].iloc[i_raw].to_pydatetime() <= time_list[i]:
                            if is_numeric(df[value_col_name].iloc[i_raw]):
                                time_prior = df[utc_col_name].iloc[i_raw].to_pydatetime()
                                value_prior = float(df[value_col_name].iloc[i_raw])
                                loop = False
                            else:
                                i_raw -= 1
                        else:
                            i_raw -= 1

                    if i_raw < 0:
                        value_list.append("nan")
                        i_raw = 0
                        direct = 1
                    else:
                        delta_time = time_next - time_prior
                        delta_time_sec = delta_time.total_seconds()
                        delta_value = value_prior - value_next

                        if delta_time_sec == 0 or (delta_value == 0 and delta_time_sec <= intrpl_max*60):
                            value_list.append(apply_precision(value_prior))
                        elif delta_time_sec > intrpl_max*60:
                            value_list.append("nan")
                        else:
                            delta_time_prior_sec = (time_list[i] - time_prior).total_seconds()
                            value_list.append(apply_precision(value_prior - delta_value/delta_time_sec*delta_time_prior_sec))

                        direct = 1

        # METHOD: NEAREST VALUE
        elif mode_input == "nearest" or mode_input == "nearest (mean)":
            if tracker:
                tracker.emit('processing', 37, 'fp_processing_start', force=True, message_params={'mode': mode_input})

            i_raw = 0

            for i in range(0, len(time_list)):
                if tracker and i % emit_frequency == 0 and i > 0:
                    tracker.update_step_progress(i)
                    progress = 37 + (i / len(time_list)) * 53
                    tracker.emit('processing', progress, 'fp_mode_progress', message_params={'mode': mode_input, 'current': i, 'total': len(time_list)})

                try:
                    time_int_min = time_list[i] - datetime.timedelta(minutes=tss/2)
                    time_int_max = time_list[i] + datetime.timedelta(minutes=tss/2)

                    value_int_list = []
                    delta_time_int_list = []

                    while i_raw < len(df):
                        current_time = df[utc_col_name].iloc[i_raw].to_pydatetime()

                        if current_time > time_int_max:
                            break

                        if current_time >= time_int_min:
                            if is_numeric(df[value_col_name].iloc[i_raw]):
                                value_int_list.append(float(df[value_col_name].iloc[i_raw]))
                                delta_time_int_list.append(abs((time_list[i] - current_time).total_seconds()))

                        i_raw += 1

                    if i_raw > 0:
                        i_raw -= 1

                    if value_int_list:
                        if mode_input == "nearest":
                            min_time = min(delta_time_int_list)
                            min_idx = delta_time_int_list.index(min_time)
                            value_list.append(apply_precision(value_int_list[min_idx]))
                        else:  # nearest (mean)
                            min_time = min(delta_time_int_list)
                            nearest_values = [
                                value_int_list[idx]
                                for idx, delta in enumerate(delta_time_int_list)
                                if abs(delta - min_time) < 0.001
                            ]
                            value_list.append(apply_precision(statistics.mean(nearest_values)))
                    else:
                        value_list.append("nan")

                except Exception as e:
                    logger.error(f"Error processing time step {i}: {str(e)}")
                    value_list.append("nan")

        # DATA FRAME WITH PROCESSED DATA
        logger.info(f"Length of time_list: {len(time_list)}")
        logger.info(f"Length of value_list: {len(value_list)}")

        if tracker:
            tracker.emit('processing', 90, 'fp_processing_done', force=True)
            tracker.end_phase('processing')
            tracker.current_step = 0
            tracker.total_steps = 0

        # Create result dataframe
        result_df = pd.DataFrame({"UTC": time_list, value_col_name: value_list})

        # Format UTC column to desired format
        result_df['UTC'] = result_df['UTC'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))

        # Create closure for tracker so generator can use it
        tracker_ref = tracker
        total_rows = len(result_df)

        def generate():
            """Generator for streaming NDJSON with progress tracking - OPTIMIZED"""
            try:
                # === PHASE 4: STREAMING (90-100%) ===
                if tracker_ref:
                    tracker_ref.start_phase('streaming')
                    tracker_ref.emit('streaming', 90, 'fp_streaming_start', force=True, message_params={'totalRows': total_rows})

                # Send total row count as first chunk
                yield json.dumps({"total_rows": total_rows}) + "\n"

                chunk_size = STREAMING_CHUNK_SIZE
                total_chunks_to_stream = (total_rows // chunk_size) + 1
                streaming_start_time = time.time()

                for i in range(0, total_rows, chunk_size):
                    try:
                        chunk_progress = 90 + ((i / total_rows) * 9)
                        current_chunk = (i // chunk_size) + 1

                        streaming_eta = None
                        if current_chunk > 1:
                            elapsed = time.time() - streaming_start_time
                            chunks_done = current_chunk - 1
                            chunks_remaining = total_chunks_to_stream - current_chunk + 1
                            time_per_chunk = elapsed / chunks_done
                            streaming_eta = int(chunks_remaining * time_per_chunk)

                        if tracker_ref:
                            tracker_ref.emit('streaming', chunk_progress,
                                            'fp_streaming_chunk',
                                            eta_seconds=streaming_eta,
                                            message_params={'current': current_chunk, 'total': total_chunks_to_stream})

                        chunk = result_df.iloc[i:i + chunk_size]

                        chunk_subset = chunk[['UTC', value_col_name]].copy()
                        chunk_subset[value_col_name] = chunk_subset[value_col_name].astype(object)
                        chunk_subset[value_col_name] = chunk_subset[value_col_name].where(
                            pd.notna(chunk_subset[value_col_name]), None
                        )

                        for record in chunk_subset.to_dict('records'):
                            yield json.dumps(record) + "\n"

                        time.sleep(BACKPRESSURE_DELAY)

                    except Exception as chunk_error:
                        logger.error(f"Streaming error at chunk {i}: {chunk_error}")
                        yield json.dumps({"error": f"Chunk {i} failed: {str(chunk_error)}", "partial": True}) + "\n"
                        break

                if tracker_ref:
                    tracker_ref.end_phase('streaming')
                    tracker_ref.emit('complete', 100,
                                   'fp_complete', force=True, message_params={'totalRows': total_rows})

                yield json.dumps({"status": "complete"}) + "\n"

            except GeneratorExit:
                logger.info("Client disconnected during streaming")
            except BrokenPipeError:
                logger.warning("Broken pipe - client forcefully disconnected")
            except Exception as e:
                logger.error(f"Generator error: {e}")
                try:
                    yield json.dumps({"error": str(e)}) + "\n"
                except:
                    pass

        return Response(generate(), mimetype="application/x-ndjson")
    except Exception as e:
        error_msg = f"Error processing CSV: {str(e)}"
        logger.error(error_msg)
        return jsonify({"error": str(e)}), 400
