"""
Utility Functions for Adjustments Domain
File handling, validation, and data analysis utilities
"""
import os
import time
import logging
import traceback
from io import StringIO
from typing import Dict, Any, Optional, List, Tuple

import pandas as pd
import numpy as np

from domains.adjustments.config import UTC_FORMAT
from domains.adjustments.services.state_manager import (
    stored_data,
    stored_data_timestamps,
    adjustment_chunks,
    adjustment_chunks_timestamps,
    info_df_cache,
    info_df_cache_timestamps
)

logger = logging.getLogger(__name__)

# Global info DataFrame
info_df = pd.DataFrame(columns=[
    'Name der Datei', 'Name der Messreihe', 'Startzeit (UTC)', 'Endzeit (UTC)',
    'Zeitschrittweite [min]', 'Offset [min]', 'Anzahl der Datenpunkte',
    'Anzahl der numerischen Datenpunkte', 'Anteil an numerischen Datenpunkten'
])


def detect_delimiter(file_content: str) -> str:
    """
    Detect the delimiter used in a CSV file content
    """
    delimiters = [';', ',', '\t']
    first_line = file_content.split('\n')[0]

    counts = {d: first_line.count(d) for d in delimiters}

    max_count = max(counts.values())
    if max_count > 0:
        return max(counts.items(), key=lambda x: x[1])[0]
    return ';'


def get_time_column(df: pd.DataFrame) -> Optional[str]:
    """
    Check if DataFrame has exactly 'UTC' column
    """
    if 'UTC' in df.columns:
        return 'UTC'
    return None


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename to prevent path traversal attacks

    Args:
        filename: User-provided filename

    Returns:
        Sanitized filename safe for filesystem operations

    Raises:
        ValueError: If filename is invalid or contains path traversal attempts
    """
    if not filename:
        raise ValueError("Filename cannot be empty")

    safe_filename = os.path.basename(filename)

    if '..' in safe_filename:
        raise ValueError("Invalid filename: path traversal detected")

    return safe_filename


def analyse_data(file_path: str, upload_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Analyze CSV file and extract relevant information

    Args:
        file_path: Path to the CSV file to analyze
        upload_id: ID of the upload if this is part of a chunked upload
    """
    try:
        global stored_data, info_df

        all_file_info = []

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
                'Startzeit (UTC)': df['UTC'].iloc[0].strftime(UTC_FORMAT) if 'UTC' in df.columns else None,
                'Endzeit (UTC)': df['UTC'].iloc[-1].strftime(UTC_FORMAT) if 'UTC' in df.columns else None,
                'Zeitschrittweite [min]': float(time_step) if time_step is not None else None,
                'Offset [min]': float(offset),
                'Anzahl der Datenpunkte': int(len(df)),
                'Anzahl der numerischen Datenpunkte': int(df[measurement_col].count()),
                'Anteil an numerischen Datenpunkten': float(df[measurement_col].count() / len(df) * 100)
            }
            all_file_info.append(file_info)

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
