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

Updated: 2026-01-14
Refactored to generate 3 separate violin plots:
- ONE plot for INPUT features only
- ONE plot for TIME components only
- ONE plot for OUTPUT features only
"""

import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, Optional, List, Tuple

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

    Creates 3 plots total:
    - ONE plot for INPUT features only (Eingabedaten)
    - ONE plot for TIME components only (Zeitkomponenten)
    - ONE plot for OUTPUT features only (Ausgabedaten)

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
    # Returns dict with bezeichnung as key: {bezeichnung: {path, type, file_name}}
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

    # ═══════════════════════════════════════════════════════════════════════════
    # STRUCTURE: Separate input, time, and output features
    # Each feature is a tuple: (feature_name, values_array)
    # ═══════════════════════════════════════════════════════════════════════════
    input_features: List[Tuple[str, np.ndarray]] = []
    time_features: List[Tuple[str, np.ndarray]] = []
    output_features: List[Tuple[str, np.ndarray]] = []

    input_feature_names = []
    time_feature_names = []
    output_feature_names = []

    for bezeichnung, data in csv_data.items():
        df = data['df']
        file_type = data['type']
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if numeric_cols:
            for col in numeric_cols:
                if file_type == 'input':
                    input_features.append((col, df[col].values))
                    input_feature_names.append(col)
                else:
                    output_features.append((col, df[col].values))
                    output_feature_names.append(col)

            logger.info(f"Added {len(numeric_cols)} features from '{bezeichnung}' ({file_type})")

    if not input_features and not output_features:
        if progress_tracker:
            progress_tracker.error('No numeric data found in CSV files')
        raise ValueError('No numeric data found in CSV files. CSV files must contain numeric columns for visualization')

    # ═══════════════════════════════════════════════════════════════════════════
    # Generate TIME components as SEPARATE time_features list
    # ═══════════════════════════════════════════════════════════════════════════
    time_info = session_data.get('time_info', {})
    if time_info and any([time_info.get('jahr'), time_info.get('monat'),
                          time_info.get('woche'), time_info.get('tag'),
                          time_info.get('feiertag')]):
        try:
            from domains.training.data.processor import TimeFeatures

            # Get first input file's DataFrame to extract timestamps
            first_input_df = None
            for bezeichnung, data in csv_data.items():
                if data['type'] == 'input':
                    first_input_df = data['df']
                    break

            if first_input_df is None and csv_data:
                # Use any file if no input file
                first_input_df = list(csv_data.values())[0]['df']

            if first_input_df is not None and 'UTC' in first_input_df.columns:
                # Ensure UTC column is datetime
                df_copy = first_input_df.copy()
                df_copy['UTC'] = pd.to_datetime(df_copy['UTC'])

                timezone = time_info.get('zeitzone', 'UTC')
                processor = TimeFeatures(timezone)

                # Generate time features
                time_features_df = processor.add_time_features(
                    df_copy, 'UTC', time_info
                )

                # Extract TIME columns (y_sin, y_cos, m_sin, m_cos, w_sin, w_cos, d_sin, d_cos)
                time_cols = [c for c in time_features_df.columns
                             if c.endswith('_sin') or c.endswith('_cos')]

                if time_cols:
                    # Add TIME features to SEPARATE time_features list
                    for col in time_cols:
                        # Create display names: y_sin -> Y_sin, w_cos -> W_cos
                        display_name = col[0].upper() + col[1:]
                        time_features.append((display_name, time_features_df[col].values))
                        time_feature_names.append(display_name)

                    logger.info(f"Added {len(time_cols)} TIME features: {[c[0].upper() + c[1:] for c in time_cols]}")
                else:
                    logger.warning("No TIME feature columns generated")
            else:
                logger.warning("No UTC column found in input files, skipping TIME components")

        except Exception as e:
            logger.error(f"Error generating TIME components for violin plots: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())

    # ═══════════════════════════════════════════════════════════════════════════
    # Generate plots with progress tracking
    # Pass input_features, time_features, and output_features separately
    # ═══════════════════════════════════════════════════════════════════════════
    plot_result = generate_violin_plots_from_data(
        session_id,
        input_features=input_features,
        time_features=time_features,
        output_features=output_features,
        progress_tracker=progress_tracker
    )

    result = {
        'success': plot_result['success'],
        'violin_plots': plot_result.get('plots', {}),
        'message': 'Violin plots generated successfully. Ready for model training.',
        'data_info': {
            'input_features': list(set(input_feature_names)),
            'time_features': list(set(time_feature_names)),
            'output_features': list(set(output_feature_names)),
            'input_features_count': len(input_features),
            'time_features_count': len(time_features),
            'output_features_count': len(output_features),
            'total_features': len(input_features) + len(time_features) + len(output_features)
        }
    }

    logger.info(f"Violin plots generated for session {session_id}: "
                f"{len(input_features)} input, {len(time_features)} time, {len(output_features)} output features")

    return result
