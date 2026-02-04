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



def _save_training_arrays_to_storage(uuid_session_id: str, i_array_3D: np.ndarray, o_array_3D: np.ndarray) -> None:
    """Save i_array_3D and o_array_3D to Supabase Storage as a compressed pickle for download."""
    import io
    import gzip
    import pickle
    from shared.database.client import get_supabase_admin_client

    data = {'i_array_3D': i_array_3D, 'o_array_3D': o_array_3D}
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode='wb', compresslevel=6) as gz:
        pickle.dump(data, gz, protocol=pickle.HIGHEST_PROTOCOL)
    upload_bytes = buffer.getvalue()

    file_path = f"{uuid_session_id}/training_arrays.pkl.gz"
    supabase = get_supabase_admin_client()

    # Remove old file if exists (upsert)
    try:
        supabase.storage.from_('training-results').remove([file_path])
    except Exception:
        pass

    supabase.storage.from_('training-results').upload(
        path=file_path,
        file=upload_bytes,
        file_options={'content-type': 'application/gzip', 'cache-control': '3600'}
    )
    size_mb = len(upload_bytes) / 1024 / 1024
    logger.info(f"VIOLIN: Saved training arrays to storage: {file_path} ({size_mb:.1f}MB)")


def _prepare_processed_data(session_data: Dict, csv_data: Dict, progress_tracker=None, uuid_session_id: Optional[str] = None) -> Dict:
    """
    Prepare processed data using create_training_arrays.
    Returns n_dat, combined arrays, and feature name lists for violin plots.

    Args:
        session_data: Session data loaded from database
        csv_data: Parsed CSV data dict with bezeichnung as key

    Returns:
        Dict with keys: n_dat, i_combined_array, o_combined_array,
        i_list, o_list, time_list, n_input_files
    """
    from domains.training.data.loader import load, transf
    from domains.training.data.transformer import create_training_arrays
    from domains.training.config import MTS, T

    try:
        logger.debug("=" * 60)
        logger.debug("VIOLIN: _prepare_processed_data() START")
        logger.debug("=" * 60)

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

        logger.debug(f"VIOLIN: i_dat keys (input files): {list(i_dat.keys())}")
        logger.debug(f"VIOLIN: o_dat keys (output files): {list(o_dat.keys())}")
        for key, df in i_dat.items():
            logger.debug(f"VIOLIN:   i_dat['{key}'] shape={df.shape}, "
                         f"UTC range=[{df['UTC'].iloc[0]} -> {df['UTC'].iloc[-1]}], "
                         f"value range=[{df.iloc[:,1].min():.2f} -> {df.iloc[:,1].max():.2f}]")

        if not i_dat or not o_dat:
            logger.warning("_prepare_processed_data: Missing input or output data")
            return None

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
        if progress_tracker:
            progress_tracker.ndat_loading_data()
        i_dat, i_dat_inf = load(i_dat, i_dat_inf)
        o_dat, o_dat_inf = load(o_dat, o_dat_inf)

        logger.debug(f"VIOLIN: i_dat_inf index (order): {list(i_dat_inf.index)}")
        logger.debug(f"VIOLIN: o_dat_inf index (order): {list(o_dat_inf.index)}")

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
        if progress_tracker:
            progress_tracker.ndat_transforming()
        i_dat_inf = transf(i_dat_inf, mts_config.I_N, mts_config.OFST)
        o_dat_inf = transf(o_dat_inf, mts_config.O_N, mts_config.OFST)

        logger.debug(f"VIOLIN: MTS config: I_N={mts_config.I_N}, O_N={mts_config.O_N}, "
                     f"DELT={mts_config.DELT}, OFST={mts_config.OFST}")

        # 9. Determine time range (CRITICAL: .min() for utc_end!)
        utc_strt = i_dat_inf["utc_min"].min()
        utc_end = i_dat_inf["utc_max"].min()  # CRITICAL: Use .min() not .max()!

        logger.debug(f"VIOLIN: UTC range: {utc_strt} -> {utc_end}")

        # 10. Build feature name lists matching combined array column order
        i_list = list(i_dat_inf.index)  # File bezeichnungen in deterministic order

        # TIME component names (same order as create_training_arrays adds them)
        time_list = []
        if time_info.get('jahr'):
            time_list.extend(['Y_sin', 'Y_cos'])
        if time_info.get('monat'):
            time_list.extend(['M_sin', 'M_cos'])
        if time_info.get('woche'):
            time_list.extend(['W_sin', 'W_cos'])
        if time_info.get('tag'):
            time_list.extend(['D_sin', 'D_cos'])
        if time_info.get('feiertag'):
            time_list.append('H')

        o_list = list(o_dat_inf.index)  # Output bezeichnungen

        # 11. Call create_training_arrays (uses optimized version when USE_OPTIMIZED_TRANSFORMER=true)
        if progress_tracker:
            progress_tracker.ndat_creating_arrays()
        i_array_3D, o_array_3D, i_combined_array, o_combined_array, _ = create_training_arrays(
            i_dat=i_dat,
            o_dat=o_dat,
            i_dat_inf=i_dat_inf,
            o_dat_inf=o_dat_inf,
            utc_strt=utc_strt,
            utc_end=utc_end,
            mts_config=mts_config
        )

        n_dat = i_array_3D.shape[0] if len(i_array_3D) > 0 else 0

        logger.debug(f"VIOLIN: i_array_3D shape: {i_array_3D.shape}")
        logger.debug(f"VIOLIN: o_array_3D shape: {o_array_3D.shape}")
        logger.debug(f"VIOLIN: i_combined_array shape: {i_combined_array.shape}")
        logger.debug(f"VIOLIN: o_combined_array shape: {o_combined_array.shape}")
        logger.debug(f"VIOLIN: n_dat (training samples): {n_dat}")
        logger.debug(f"VIOLIN: Feature lists -> i_list={i_list}, time_list={time_list}, o_list={o_list}")

        # Save 3D arrays to Supabase Storage for debugging/comparison downloads
        if uuid_session_id:
            try:
                _save_training_arrays_to_storage(uuid_session_id, i_array_3D, o_array_3D)
            except Exception as e:
                logger.warning(f"VIOLIN: Failed to save training arrays to storage: {e}")

        # Free 3D arrays early - only n_dat was needed from them (~820MB freed)
        del i_array_3D, o_array_3D
        import gc
        gc.collect()
        logger.debug("VIOLIN: Freed 3D arrays, keeping combined arrays only")
        logger.debug("VIOLIN: _prepare_processed_data() DONE")

        if progress_tracker:
            progress_tracker.ndat_arrays_complete()

        return {
            'n_dat': n_dat,
            'i_combined_array': i_combined_array,
            'o_combined_array': o_combined_array,
            'i_list': i_list,
            'o_list': o_list,
            'time_list': time_list,
            'n_input_files': len(i_dat),
        }

    except Exception as e:
        logger.error(f"Error in _prepare_processed_data: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


def generate_violin_plots_for_session(
    session_id: str,
    model_parameters: Optional[Dict] = None,
    training_split: Optional[Dict] = None,
    progress_tracker=None,
    uuid_session_id: Optional[str] = None
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

            # Capture original column name before renaming (e.g., "tl [°C]")
            column_name = df.columns[1] if len(df.columns) >= 2 else bezeichnung

            csv_data[bezeichnung] = {
                'df': df,
                'type': file_type,
                'file_name': file_name,
                'column_name': column_name
            }

    if not csv_data:
        if progress_tracker:
            progress_tracker.error('Could not load CSV data')
        raise Exception('Could not load CSV data. CSV files could not be read')

    # Emit parsing complete
    if progress_tracker:
        progress_tracker.parsing_complete()

    # ═══════════════════════════════════════════════════════════════════════════
    # Run full data processing pipeline to get processed (interpolated/scaled) data
    # This replaces both raw CSV reading AND the separate _calculate_n_dat() call
    # ═══════════════════════════════════════════════════════════════════════════
    if progress_tracker:
        progress_tracker.calculating_dataset_count()

    processed = _prepare_processed_data(session_data, csv_data, progress_tracker, uuid_session_id=uuid_session_id)

    if processed is None:
        if progress_tracker:
            progress_tracker.error('Data processing failed')
        raise Exception('Data processing pipeline failed. Check logs for details.')

    n_dat = processed['n_dat']
    i_combined = processed['i_combined_array']
    o_combined = processed['o_combined_array']
    i_list = processed['i_list']
    o_list = processed['o_list']
    time_list = processed['time_list']
    n_input_files = processed['n_input_files']

    if progress_tracker:
        progress_tracker.dataset_count_complete(n_dat)

    # ═══════════════════════════════════════════════════════════════════════════
    # Build feature tuples from processed arrays
    # i_combined columns: [file1, file2, ..., Y_sin, Y_cos, M_sin, ...]
    # Input/Output: 3-tuple (bezeichnung, column_name, values) - column_name as y-label
    # Time: 2-tuple (display_name, values) - no y-label for generated features
    # .copy() creates independent arrays so we can free the large combined arrays
    # ═══════════════════════════════════════════════════════════════════════════
    input_features: List[Tuple] = []
    time_features: List[Tuple[str, np.ndarray]] = []
    output_features: List[Tuple] = []

    for i in range(n_input_files):
        bezeichnung = i_list[i]
        column_name = csv_data.get(bezeichnung, {}).get('column_name', bezeichnung)
        input_features.append((bezeichnung, column_name, i_combined[:, i].copy()))

    for i, name in enumerate(time_list):
        col_idx = n_input_files + i
        time_features.append((name, i_combined[:, col_idx].copy()))

    for i in range(len(o_list)):
        bezeichnung = o_list[i]
        column_name = csv_data.get(bezeichnung, {}).get('column_name', bezeichnung)
        output_features.append((bezeichnung, column_name, o_combined[:, i].copy()))

    # Free the large combined arrays (~1.5GB) before rendering
    del i_combined, o_combined, processed
    import gc
    gc.collect()
    logger.debug("VIOLIN: Freed combined arrays, column slices retained")

    logger.debug("=" * 60)
    logger.debug("VIOLIN: Feature tuples built from processed arrays")
    logger.debug("=" * 60)
    logger.debug(f"VIOLIN: input_features ({len(input_features)}):")
    for name, vals in input_features:
        logger.debug(f"VIOLIN:   '{name}': shape={vals.shape}, "
                     f"range=[{np.nanmin(vals):.4f} -> {np.nanmax(vals):.4f}], "
                     f"NaN count={np.isnan(vals).sum()}")
    logger.debug(f"VIOLIN: time_features ({len(time_features)}):")
    for name, vals in time_features:
        logger.debug(f"VIOLIN:   '{name}': shape={vals.shape}, "
                     f"range=[{np.nanmin(vals):.4f} -> {np.nanmax(vals):.4f}]")
    logger.debug(f"VIOLIN: output_features ({len(output_features)}):")
    for name, vals in output_features:
        logger.debug(f"VIOLIN:   '{name}': shape={vals.shape}, "
                     f"range=[{np.nanmin(vals):.4f} -> {np.nanmax(vals):.4f}], "
                     f"NaN count={np.isnan(vals).sum()}")

    if not input_features and not output_features:
        if progress_tracker:
            progress_tracker.error('No features found in processed data')
        raise ValueError('No features found in processed data')

    # ═══════════════════════════════════════════════════════════════════════════
    # Generate plots with progress tracking
    # ═══════════════════════════════════════════════════════════════════════════
    plot_result = generate_violin_plots_from_data(
        session_id,
        input_features=input_features,
        time_features=time_features,
        output_features=output_features,
        progress_tracker=progress_tracker
    )

    input_feature_names = [name for name, _ in input_features]
    time_feature_names = [name for name, _ in time_features]
    output_feature_names = [name for name, _ in output_features]

    result = {
        'success': plot_result['success'],
        'violin_plots': plot_result.get('plots', {}),
        'message': 'Violin plots generated successfully. Ready for model training.',
        'n_dat': n_dat,
        'data_info': {
            'input_features': input_feature_names,
            'time_features': time_feature_names,
            'output_features': output_feature_names,
            'input_features_count': len(input_features),
            'time_features_count': len(time_features),
            'output_features_count': len(output_features),
            'total_features': len(input_features) + len(time_features) + len(output_features)
        }
    }

    return result
