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
    downloaded_files = data_loader.download_session_files(session_id, progress_tracker=progress_tracker)

    # Emit parsing phase
    if progress_tracker:
        progress_tracker.parsing_files()

    csv_data = {}
    for file_type, file_path in downloaded_files.items():
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path, sep=';')
                if df.shape[1] == 1:
                    df = pd.read_csv(file_path, sep=',')
            except:
                df = pd.read_csv(file_path)

            csv_data[file_type] = df
            logger.info(f"Loaded {file_type} file with {len(df)} rows and {len(df.columns)} columns: {list(df.columns)}")

    if not csv_data:
        if progress_tracker:
            progress_tracker.error('Could not load CSV data')
        raise Exception('Could not load CSV data. CSV files could not be read')

    # Emit parsing complete
    if progress_tracker:
        progress_tracker.parsing_complete()

    input_data = None
    output_data = None
    input_features = []
    output_features = []

    if 'input' in csv_data:
        input_df = csv_data['input']
        numeric_cols = input_df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            input_data = input_df[numeric_cols].values
            input_features = numeric_cols

    if 'output' in csv_data:
        output_df = csv_data['output']
        numeric_cols = output_df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            output_data = output_df[numeric_cols].values
            output_features = numeric_cols

    if input_data is None and output_data is None:
        if progress_tracker:
            progress_tracker.error('No numeric data found in CSV files')
        raise ValueError('No numeric data found in CSV files. CSV files must contain numeric columns for visualization')

    data_info = {
        'success': True,
        'input_data': input_data,
        'output_data': output_data,
        'input_features': input_features,
        'output_features': output_features
    }

    # Generate plots with progress tracking
    plot_result = generate_violin_plots_from_data(
        session_id,
        input_data=data_info.get('input_data'),
        output_data=data_info.get('output_data'),
        input_features=data_info.get('input_features'),
        output_features=data_info.get('output_features'),
        progress_tracker=progress_tracker
    )

    result = {
        'success': plot_result['success'],
        'violin_plots': plot_result.get('plots', {}),
        'message': 'Violin plots generated successfully. Ready for model training.',
        'data_info': {
            'input_features': input_features,
            'output_features': output_features,
            'input_shape': input_data.shape if input_data is not None else None,
            'output_shape': output_data.shape if output_data is not None else None
        }
    }

    logger.info(f"✅ Violin plots generated for session {session_id}")

    return result
