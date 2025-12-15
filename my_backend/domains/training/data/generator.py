"""
Dataset Generation Service
Business logic for generating datasets and violin plots for training visualization

This service handles:
- Loading session CSV data from Supabase Storage
- Processing input/output data for visualization
- Generating violin plots without model training
- Phase 1 of training workflow (visualization only)

Created: 2025-10-24
Phase 6a of training.py refactoring
"""

import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, Optional

logger = logging.getLogger(__name__)


def generate_violin_plots_for_session(
    session_id: str,
    model_parameters: Optional[Dict] = None,
    training_split: Optional[Dict] = None,
    progress_tracker=None
) -> Dict:
    """
    Generate datasets and violin plots WITHOUT training models.

    This is phase 1 of the training workflow - data visualization only.
    Loads CSV files, processes numeric data, and generates violin plots.

    Args:
        session_id: Training session ID
        model_parameters: Model configuration (not used for visualization)
        training_split: Training split configuration (not used for visualization)
        progress_tracker: Optional ViolinProgressTracker for emitting progress updates

    Returns:
        dict: {
            'success': bool,
            'violin_plots': dict,
            'message': str,
            'data_info': dict (optional)
        }

    Raises:
        ValueError: If no data available or no numeric data found
        Exception: If CSV loading or plot generation fails
    """
    from domains.training.data.loader import DataLoader
    from domains.training.services.violin import generate_violin_plots_from_data

    logger.info(f"Generating violin plots for session {session_id} WITHOUT training")

    # Emit start progress
    if progress_tracker:
        progress_tracker.start()

    data_loader = DataLoader()

    session_data = data_loader.load_session_data(session_id)
    files_info = session_data.get('files', [])

    if not files_info:
        if progress_tracker:
            progress_tracker.error('No data available for visualization')
        raise ValueError('No data available for visualization. Please upload CSV files first')

    # Download files with progress tracking
    # Now returns dict with bezeichnung as key: {bezeichnung: {path, type, file_name}}
    downloaded_files = data_loader.download_session_files(session_id, progress_tracker=progress_tracker)

    # Emit parsing phase
    if progress_tracker:
        progress_tracker.parsing_files()

    # Parse CSV files - now keyed by bezeichnung
    csv_data = {}
    for bezeichnung, file_info in downloaded_files.items():
        file_path = file_info['path']
        file_type = file_info['type']
        file_name = file_info['file_name']

        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path, sep=';')
                if df.shape[1] == 1:
                    df = pd.read_csv(file_path, sep=',')
            except:
                df = pd.read_csv(file_path)

            csv_data[bezeichnung] = {
                'df': df,
                'type': file_type,
                'file_name': file_name
            }
            logger.info(f"Loaded '{bezeichnung}' ({file_type}) with {len(df)} rows and {len(df.columns)} columns")

    if not csv_data:
        if progress_tracker:
            progress_tracker.error('Could not load CSV data')
        raise Exception('Could not load CSV data. CSV files could not be read')

    # Emit parsing complete
    if progress_tracker:
        progress_tracker.parsing_complete()

    # Separate files by type and prepare data lists
    input_files_data = []
    output_files_data = []

    for bezeichnung, data in csv_data.items():
        df = data['df']
        file_type = data['type']
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if numeric_cols:
            file_data = {
                'bezeichnung': bezeichnung,
                'data': df[numeric_cols].values,
                'features': numeric_cols,
                'type': file_type
            }
            if file_type == 'input':
                input_files_data.append(file_data)
            else:
                output_files_data.append(file_data)

    if not input_files_data and not output_files_data:
        if progress_tracker:
            progress_tracker.error('No numeric data found in CSV files')
        raise ValueError('No numeric data found in CSV files. CSV files must contain numeric columns for visualization')

    # Combine all files data for violin plot generation
    all_files_data = input_files_data + output_files_data

    # Generate plots with progress tracking - one plot per bezeichnung
    plot_result = generate_violin_plots_from_data(
        session_id,
        files_data=all_files_data,
        progress_tracker=progress_tracker
    )

    # Collect features info
    all_input_features = []
    all_output_features = []
    for fd in input_files_data:
        all_input_features.extend(fd['features'])
    for fd in output_files_data:
        all_output_features.extend(fd['features'])

    result = {
        'success': plot_result['success'],
        'violin_plots': plot_result.get('plots', {}),
        'message': 'Violin plots generated successfully. Ready for model training.',
        'data_info': {
            'input_features': list(set(all_input_features)),
            'output_features': list(set(all_output_features)),
            'input_files_count': len(input_files_data),
            'output_files_count': len(output_files_data),
            'total_files': len(all_files_data)
        }
    }

    logger.info(f"✅ Violin plots generated for session {session_id}: {len(all_files_data)} files processed")

    return result
