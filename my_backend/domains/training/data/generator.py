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
from typing import Dict, Optional, List, Tuple, Any

logger = logging.getLogger(__name__)


def _safe_float_to_int(value: Any, default: int) -> int:
    """Safely convert a value to int, handling empty strings and None."""
    if value is None or value == '' or value == 'None':
        return default
    try:
        return int(float(value))
    except (ValueError, TypeError):
        return default



def _calculate_n_dat(session_data: Dict, csv_data: Dict) -> int:
    """
    Calculate the exact n_dat using create_training_arrays.
    Replicates the data preparation logic from middleman.py.

    Args:
        session_data: Session data loaded from database
        csv_data: Parsed CSV data dict with bezeichnung as key

    Returns:
        n_dat: Number of valid training samples (i_array_3D.shape[0])
    """
    from domains.training.data.loader import load, transf
    from domains.training.data.transformer import create_training_arrays
    from domains.training.config import MTS, T

    try:
        # 1. Configure MTS from zeitschritte
        zeitschritte = session_data.get('zeitschritte', {})
        mts_config = MTS()
        mts_config.I_N = int(zeitschritte.get('eingabe', mts_config.I_N))
        mts_config.O_N = int(zeitschritte.get('ausgabe', mts_config.O_N))
        mts_config.DELT = float(zeitschritte.get('zeitschrittweite', mts_config.DELT))
        mts_config.OFST = float(zeitschritte.get('offset', mts_config.OFST))


        # 2. Configure TIME components from time_info
        time_info = session_data.get('time_info', {})
        T.Y.IMP = time_info.get('jahr', False)
        T.M.IMP = time_info.get('monat', False)
        T.W.IMP = time_info.get('woche', False)
        T.D.IMP = time_info.get('tag', False)
        T.H.IMP = time_info.get('feiertag', False)
        T.TZ = time_info.get('zeitzone', 'UTC')

        # Configure detailed TIME settings from category_data
        category_data = time_info.get('category_data', {})
        if category_data:
            if 'jahr' in category_data:
                jahr_cfg = category_data['jahr']
                T.Y.SPEC = jahr_cfg.get('datenform', 'Zeithorizont')
                T.Y.TH_STRT = _safe_float_to_int(jahr_cfg.get('zeithorizontStart'), -24)
                T.Y.TH_END = _safe_float_to_int(jahr_cfg.get('zeithorizontEnd'), 0)
            if 'woche' in category_data:
                woche_cfg = category_data['woche']
                T.W.SPEC = woche_cfg.get('datenform', 'Zeithorizont')
                T.W.TH_STRT = _safe_float_to_int(woche_cfg.get('zeithorizontStart'), -24)
                T.W.TH_END = _safe_float_to_int(woche_cfg.get('zeithorizontEnd'), 0)
            if 'tag' in category_data:
                tag_cfg = category_data['tag']
                T.D.SPEC = tag_cfg.get('datenform', 'Zeithorizont')
                T.D.TH_STRT = _safe_float_to_int(tag_cfg.get('zeithorizontStart'), -24)
                T.D.TH_END = _safe_float_to_int(tag_cfg.get('zeithorizontEnd'), 0)

        # 3. Prepare i_dat, o_dat from csv_data
        i_dat = {}
        o_dat = {}
        for bezeichnung, data in csv_data.items():
            df_copy = data['df'].copy()
            # Ensure columns are named correctly for load() function
            if len(df_copy.columns) >= 2:
                df_copy.columns = ['UTC', 'data_value'] + list(df_copy.columns[2:])
            if data['type'] == 'input':
                i_dat[bezeichnung] = df_copy
            else:
                o_dat[bezeichnung] = df_copy

        if not i_dat or not o_dat:
            logger.warning("n_dat calculation: Missing input or output data")
            return 0

        # 4. Create files metadata mapping
        files_metadata = {}
        for file_info in session_data.get('files', []):
            bezeichnung = file_info.get('bezeichnung', file_info['file_name'].replace('.csv', ''))
            files_metadata[bezeichnung] = file_info

        # 5. Initialize i_dat_inf and o_dat_inf DataFrames
        inf_columns = [
            "utc_min", "utc_max", "delt", "ofst", "n_all", "n_num",
            "rate_num", "val_min", "val_max", "spec", "th_strt",
            "th_end", "meth", "avg", "delt_transf", "ofst_transf",
            "scal", "scal_max", "scal_min"
        ]
        i_dat_inf = pd.DataFrame(columns=inf_columns)
        o_dat_inf = pd.DataFrame(columns=inf_columns)

        # 6. Process with load()
        i_dat, i_dat_inf = load(i_dat, i_dat_inf)
        o_dat, o_dat_inf = load(o_dat, o_dat_inf)

        # 7. Set zeithorizont from metadata (AFTER load())
        for key in i_dat_inf.index:
            metadata = files_metadata.get(key, {})
            i_dat_inf.loc[key, "spec"] = "Historische Daten"
            i_dat_inf.loc[key, "th_strt"] = _safe_float_to_int(metadata.get('zeithorizont_start'), -1)
            i_dat_inf.loc[key, "th_end"] = _safe_float_to_int(metadata.get('zeithorizont_end'), 0)
            i_dat_inf.loc[key, "meth"] = "Lineare Interpolation"
            i_dat_inf.loc[key, "avg"] = False
            i_dat_inf.loc[key, "scal"] = True
            i_dat_inf.loc[key, "scal_max"] = 1
            i_dat_inf.loc[key, "scal_min"] = 0

        for key in o_dat_inf.index:
            metadata = files_metadata.get(key, {})
            o_dat_inf.loc[key, "spec"] = "Historische Daten"
            o_dat_inf.loc[key, "th_strt"] = _safe_float_to_int(metadata.get('zeithorizont_start'), 0)
            o_dat_inf.loc[key, "th_end"] = _safe_float_to_int(metadata.get('zeithorizont_end'), 1)
            o_dat_inf.loc[key, "meth"] = "Lineare Interpolation"
            o_dat_inf.loc[key, "avg"] = False
            o_dat_inf.loc[key, "scal"] = True
            o_dat_inf.loc[key, "scal_max"] = 1
            o_dat_inf.loc[key, "scal_min"] = 0

        # 8. Apply transf()
        i_dat_inf = transf(i_dat_inf, mts_config.I_N, mts_config.OFST)
        o_dat_inf = transf(o_dat_inf, mts_config.O_N, mts_config.OFST)

        # 9. Determine time range (CRITICAL: .min() for utc_end!)
        utc_strt = i_dat_inf["utc_min"].min()
        utc_end = i_dat_inf["utc_max"].min()  # CRITICAL: Use .min() not .max()!


        # 10. Call create_training_arrays
        i_array_3D, o_array_3D, _, _, _ = create_training_arrays(
            i_dat=i_dat,
            o_dat=o_dat,
            i_dat_inf=i_dat_inf,
            o_dat_inf=o_dat_inf,
            utc_strt=utc_strt,
            utc_end=utc_end,
            mts_config=mts_config
        )

        n_dat = i_array_3D.shape[0] if len(i_array_3D) > 0 else 0
        return n_dat

    except Exception as e:
        logger.error(f"Error calculating n_dat: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 0


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

    if not csv_data:
        if progress_tracker:
            progress_tracker.error('Could not load CSV data')
        raise Exception('Could not load CSV data. CSV files could not be read')

    # Emit parsing complete
    if progress_tracker:
        progress_tracker.parsing_complete()

    # ═══════════════════════════════════════════════════════════════════════════
    # STRUCTURE: Separate input, time, and output features
    # Input/Output features: (bezeichnung, column_name, values_array)
    # Time features: (display_name, values_array) - no column_name for generated features
    # ═══════════════════════════════════════════════════════════════════════════
    input_features: List[Tuple[str, str, np.ndarray]] = []  # (bezeichnung, column_name, values)
    time_features: List[Tuple[str, np.ndarray]] = []
    output_features: List[Tuple[str, str, np.ndarray]] = []  # (bezeichnung, column_name, values)

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
                    input_features.append((bezeichnung, col, df[col].values))
                    input_feature_names.append(bezeichnung)
                else:
                    output_features.append((bezeichnung, col, df[col].values))
                    output_feature_names.append(bezeichnung)

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

    # ═══════════════════════════════════════════════════════════════════════════
    # Calculate n_dat (number of valid training samples after 3D transformation)
    # This is the actual dataset count that will be used for training
    # ═══════════════════════════════════════════════════════════════════════════
    if progress_tracker:
        progress_tracker.calculating_dataset_count()

    n_dat = _calculate_n_dat(session_data, csv_data)

    if progress_tracker:
        progress_tracker.dataset_count_complete(n_dat)
        progress_tracker.complete()

    result = {
        'success': plot_result['success'],
        'violin_plots': plot_result.get('plots', {}),
        'message': 'Violin plots generated successfully. Ready for model training.',
        'n_dat': n_dat,
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

    return result
