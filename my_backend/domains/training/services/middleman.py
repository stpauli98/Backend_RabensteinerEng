"""
ModernMiddlemanRunner - Connects API with verified training pipeline
Uses pipeline_exact.py which has been tested to produce identical results to original
"""

import sys
import os
import logging
from typing import Dict, Optional, Any
import traceback
import pandas as pd
from datetime import datetime

from shared.database.operations import create_or_get_session_uuid
from shared.database.client import get_supabase_admin_client

from domains.training.ml.exact import run_exact_training_pipeline
from domains.training.data.loader import DataLoader
from domains.training.config import MDL, T

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BACKEND_URL = "http://127.0.0.1:8080"

def safe_float_to_int(value: Any, default: int) -> int:
    """Safely convert a value to int, handling empty strings and None."""
    if value is None or value == '' or value == 'None':
        return default
    try:
        return int(float(value))
    except (ValueError, TypeError):
        return default

class ModernMiddlemanRunner:
    """
    Modern middleman runner that uses run_exact_training_pipeline
    Maintains the same API but uses extracted modules internally
    """
    
    def __init__(self):
        self.supabase = get_supabase_admin_client()
        self.socketio = None
        
    def set_socketio(self, socketio_instance):
        """Set SocketIO instance for real-time progress updates"""
        self.socketio = socketio_instance
        
    def run_training_script(self, session_id: str, model_params: Optional[Dict] = None) -> Dict:
        """
        Main training execution using verified pipeline_exact.py

        Args:
            session_id: Session identifier
            model_params: Optional model configuration from frontend

        Returns:
            Dict containing training results and status
        """
        try:
            logger.info(f"📍 Step 1: Starting run_training_script for session {session_id}")

            import glob
            temp_dir = "temp_training_data"
            pattern = f"{temp_dir}/session_{session_id}_*"
            found_files = glob.glob(pattern)
            
            if not found_files:
                if not self._validate_session(session_id):
                    raise ValueError(f"Session {session_id} not found or invalid")
            
            data_loader = DataLoader(self.supabase)
            
            session_data = data_loader.load_session_data(session_id)

            # ===================================================================
            # CONFIGURE TIME COMPONENTS FROM DATABASE (CRITICAL!)
            # Without this, T.Y.IMP etc. stay False and TIME features aren't added
            # ===================================================================
            time_info = session_data.get('time_info', {})
            category_data = time_info.get('category_data', {})

            # Set main import flags
            T.Y.IMP = time_info.get('jahr', False)
            T.M.IMP = time_info.get('monat', False)
            T.W.IMP = time_info.get('woche', False)
            T.D.IMP = time_info.get('tag', False)
            T.H.IMP = time_info.get('feiertag', False)

            # Set timezone
            T.TZ = time_info.get('zeitzone', 'UTC')

            # Configure detailed settings from category_data
            if category_data:
                # Year (jahr) settings
                if 'jahr' in category_data:
                    jahr_cfg = category_data['jahr']
                    T.Y.SPEC = jahr_cfg.get('datenform', 'Zeithorizont')
                    T.Y.TH_STRT = safe_float_to_int(jahr_cfg.get('zeithorizontStart'), -24)
                    T.Y.TH_END = safe_float_to_int(jahr_cfg.get('zeithorizontEnd'), 0)
                    T.Y.SCAL = jahr_cfg.get('skalierung') == 'ja'
                    T.Y.SCAL_MIN = float(jahr_cfg.get('skalierungMin', 0))
                    T.Y.SCAL_MAX = float(jahr_cfg.get('skalierungMax', 1))
                    T.Y.LT = jahr_cfg.get('detaillierteBerechnung', False)

                # Month (monat) settings
                if 'monat' in category_data:
                    monat_cfg = category_data['monat']
                    T.M.SPEC = monat_cfg.get('datenform', 'Zeithorizont')
                    T.M.TH_STRT = safe_float_to_int(monat_cfg.get('zeithorizontStart'), -1)
                    T.M.TH_END = safe_float_to_int(monat_cfg.get('zeithorizontEnd'), 0)
                    T.M.SCAL = monat_cfg.get('skalierung') == 'ja'
                    T.M.SCAL_MIN = float(monat_cfg.get('skalierungMin', 0))
                    T.M.SCAL_MAX = float(monat_cfg.get('skalierungMax', 1))
                    T.M.LT = monat_cfg.get('detaillierteBerechnung', False)

                # Week (woche) settings
                if 'woche' in category_data:
                    woche_cfg = category_data['woche']
                    T.W.SPEC = woche_cfg.get('datenform', 'Zeithorizont')
                    T.W.TH_STRT = safe_float_to_int(woche_cfg.get('zeithorizontStart'), -24)
                    T.W.TH_END = safe_float_to_int(woche_cfg.get('zeithorizontEnd'), 0)
                    T.W.SCAL = woche_cfg.get('skalierung') == 'ja'
                    T.W.SCAL_MIN = float(woche_cfg.get('skalierungMin', 0))
                    T.W.SCAL_MAX = float(woche_cfg.get('skalierungMax', 1))
                    T.W.LT = woche_cfg.get('detaillierteBerechnung', False)

                # Day (tag) settings
                if 'tag' in category_data:
                    tag_cfg = category_data['tag']
                    T.D.SPEC = tag_cfg.get('datenform', 'Zeithorizont')
                    T.D.TH_STRT = safe_float_to_int(tag_cfg.get('zeithorizontStart'), -24)
                    T.D.TH_END = safe_float_to_int(tag_cfg.get('zeithorizontEnd'), 0)
                    T.D.SCAL = tag_cfg.get('skalierung') == 'ja'
                    T.D.SCAL_MIN = float(tag_cfg.get('skalierungMin', 0))
                    T.D.SCAL_MAX = float(tag_cfg.get('skalierungMax', 1))
                    T.D.LT = tag_cfg.get('detaillierteBerechnung', False)

                # Holiday (feiertag) settings
                if 'feiertag' in category_data:
                    feiertag_cfg = category_data['feiertag']
                    T.H.SPEC = feiertag_cfg.get('datenform', 'Zeithorizont')
                    T.H.TH_STRT = safe_float_to_int(feiertag_cfg.get('zeithorizontStart'), -24)
                    T.H.TH_END = safe_float_to_int(feiertag_cfg.get('zeithorizontEnd'), 0)
                    T.H.SCAL = feiertag_cfg.get('skalierung') == 'ja'
                    T.H.SCAL_MIN = float(feiertag_cfg.get('skalierungMin', 0))
                    T.H.SCAL_MAX = float(feiertag_cfg.get('skalierungMax', 1))
                    T.H.LAND = feiertag_cfg.get('land', 'AT')

            logger.info(f"📍 TIME components configured: Y={T.Y.IMP}, M={T.M.IMP}, W={T.W.IMP}, D={T.D.IMP}, H={T.H.IMP}")
            logger.info(f"📍 T.Y.SPEC={T.Y.SPEC}, T.W.SPEC={T.W.SPEC}, T.D.SPEC={T.D.SPEC}")

            if found_files:
                input_files = [f for f in found_files if "Leistung" in f]
                output_files = [f for f in found_files if "Temp" in f]
            else:
                input_files, output_files = data_loader.prepare_file_paths(session_id)
            
            logger.info(f"📍 Step 2: Loading CSV data from {len(input_files)} input files and {len(output_files)} output files")

            i_dat = {}
            o_dat = {}

            # Create mapping of bezeichnung to metadata AND file_name to bezeichnung
            files_metadata = {}
            filename_to_bezeichnung = {}
            for file_info in session_data.get('files', []):
                base_name = file_info['file_name'].replace('.csv', '')
                bezeichnung = file_info.get('bezeichnung', base_name)
                files_metadata[bezeichnung] = file_info
                filename_to_bezeichnung[base_name] = bezeichnung
                logger.info(f"   Loaded metadata for '{bezeichnung}' (file: {base_name}): type={file_info['type']}, zeithorizont={file_info.get('zeithorizont_start')}-{file_info.get('zeithorizont_end')}")

            for file_path in input_files:
                df = data_loader.load_csv_data(file_path, delimiter=';')
                # Local files are now named: session_{uuid}_{bezeichnung}.csv
                local_name = file_path.split('/')[-1].replace('.csv', '')
                if '_' in local_name:
                    parts = local_name.split('_')
                    if len(parts) > 2 and parts[0] == 'session':
                        # bezeichnung is everything after session_{uuid}_
                        bezeichnung = '_'.join(parts[2:])
                    else:
                        bezeichnung = local_name
                else:
                    bezeichnung = local_name
                i_dat[bezeichnung] = df
                logger.info(f"   📊 Loaded input file '{bezeichnung}': shape={df.shape}, columns={list(df.columns[:3])}")

            for file_path in output_files:
                df = data_loader.load_csv_data(file_path, delimiter=';')
                # Local files are now named: session_{uuid}_{bezeichnung}.csv
                local_name = file_path.split('/')[-1].replace('.csv', '')
                if '_' in local_name:
                    parts = local_name.split('_')
                    if len(parts) > 2 and parts[0] == 'session':
                        # bezeichnung is everything after session_{uuid}_
                        bezeichnung = '_'.join(parts[2:])
                    else:
                        bezeichnung = local_name
                else:
                    bezeichnung = local_name
                o_dat[bezeichnung] = df
                logger.info(f"   📊 Loaded output file '{bezeichnung}': shape={df.shape}, columns={list(df.columns[:3])}")
            
            from domains.training.data.loader import load, transf
            from domains.training.config import MTS

            # ===================================================================
            # CONFIGURE MTS FROM DATABASE (CRITICAL!)
            # zeitschritte contains eingabe, ausgabe, zeitschrittweite, offset
            # ===================================================================
            zeitschritte = session_data.get('zeitschritte', {})

            mts_config = MTS()
            mts_config.I_N = int(zeitschritte.get('eingabe', mts_config.I_N))
            mts_config.O_N = int(zeitschritte.get('ausgabe', mts_config.O_N))
            mts_config.DELT = float(zeitschritte.get('zeitschrittweite', mts_config.DELT))
            mts_config.OFST = float(zeitschritte.get('offset', mts_config.OFST))

            logger.info(f"📍 MTS configured from database: I_N={mts_config.I_N}, O_N={mts_config.O_N}, DELT={mts_config.DELT}, OFST={mts_config.OFST}")
            
            i_dat_inf = pd.DataFrame(columns=[
                "utc_min", "utc_max", "delt", "ofst", "n_all", "n_num", 
                "rate_num", "val_min", "val_max", "spec", "th_strt", 
                "th_end", "meth", "avg", "delt_transf", "ofst_transf", 
                "scal", "scal_max", "scal_min"
            ])
            
            o_dat_inf = pd.DataFrame(columns=[
                "utc_min", "utc_max", "delt", "ofst", "n_all", "n_num",
                "rate_num", "val_min", "val_max", "spec", "th_strt",
                "th_end", "meth", "avg", "delt_transf", "ofst_transf",
                "scal", "scal_max", "scal_min"
            ])
            
            logger.info(f"📍 Step 3: Processing input files through load() function")

            for file_name, df in i_dat.items():
                logger.info(f"   Input file '{file_name}': {len(df)} rows, {len(df.columns)} columns")

            try:
                import time
                start_time = time.time()
                i_dat, i_dat_inf = load(i_dat, i_dat_inf)
                elapsed = time.time() - start_time
                logger.info(f"📍 Step 3 complete: Input files processed successfully in {elapsed:.2f}s")
            except Exception as e:
                error_msg = f"Failed to process input files: {str(e)}"
                logger.error(f"❌ Step 3 FAILED: {error_msg}")
                logger.error(f"Exception details: {traceback.format_exc()}")
                return {
                    'success': False,
                    'error': error_msg,
                    'details': str(e),
                    'stage': 'load_input_files'
                }

            logger.info(f"📍 Step 4: Processing output files through load() function")

            for file_name, df in o_dat.items():
                logger.info(f"   Output file '{file_name}': {len(df)} rows, {len(df.columns)} columns")

            try:
                import time
                start_time = time.time()
                o_dat, o_dat_inf = load(o_dat, o_dat_inf)
                elapsed = time.time() - start_time
                logger.info(f"📍 Step 4 complete: Output files processed successfully in {elapsed:.2f}s")
            except Exception as e:
                error_msg = f"Failed to process output files: {str(e)}"
                logger.error(f"❌ Step 4 FAILED: {error_msg}")
                logger.error(f"Exception details: {traceback.format_exc()}")
                return {
                    'success': False,
                    'error': error_msg,
                    'details': str(e),
                    'stage': 'load_output_files'
                }
            
            for key in i_dat_inf.index:
                # Get time horizon from database metadata
                metadata = files_metadata.get(key, {})
                th_start = safe_float_to_int(metadata.get('zeithorizont_start'), -1)
                th_end = safe_float_to_int(metadata.get('zeithorizont_end'), 0)
                logger.info(f"   Input file '{key}': th_strt={th_start}, th_end={th_end}")
                
                i_dat_inf.loc[key, "spec"] = "Historische Daten"
                i_dat_inf.loc[key, "th_strt"] = th_start
                i_dat_inf.loc[key, "th_end"] = th_end
                i_dat_inf.loc[key, "meth"] = "Lineare Interpolation"
                i_dat_inf.loc[key, "avg"] = False
                i_dat_inf.loc[key, "scal"] = True
                i_dat_inf.loc[key, "scal_max"] = 1
                i_dat_inf.loc[key, "scal_min"] = 0
            
            for key in o_dat_inf.index:
                # Get time horizon from database metadata
                metadata = files_metadata.get(key, {})
                th_start = safe_float_to_int(metadata.get('zeithorizont_start'), 0)
                th_end = safe_float_to_int(metadata.get('zeithorizont_end'), 1)
                logger.info(f"   Output file '{key}': th_strt={th_start}, th_end={th_end}")
                
                o_dat_inf.loc[key, "spec"] = "Historische Daten"
                o_dat_inf.loc[key, "th_strt"] = th_start
                o_dat_inf.loc[key, "th_end"] = th_end
                o_dat_inf.loc[key, "meth"] = "Lineare Interpolation"
                o_dat_inf.loc[key, "avg"] = False
                o_dat_inf.loc[key, "scal"] = True
                o_dat_inf.loc[key, "scal_max"] = 1
                o_dat_inf.loc[key, "scal_min"] = 0
            
            logger.info(f"📍 Step 5: Applying transformations")
            i_dat_inf = transf(i_dat_inf, mts_config.I_N, mts_config.OFST)
            o_dat_inf = transf(o_dat_inf, mts_config.O_N, mts_config.OFST)
            logger.info(f"📍 Step 5 complete: Transformations applied")

            # Validate zeithorizont values before proceeding
            logger.info(f"📍 Step 6: Validating zeithorizont configuration")
            max_zeithorizont_span = 0
            zeithorizont_details = []

            for key in i_dat_inf.index:
                th_start = i_dat_inf.loc[key, "th_strt"]
                th_end = i_dat_inf.loc[key, "th_end"]
                span = abs(th_end - th_start)
                max_zeithorizont_span = max(max_zeithorizont_span, span)
                zeithorizont_details.append(f"Input '{key}': {th_start}h to {th_end}h (span: {span}h)")

            for key in o_dat_inf.index:
                th_start = o_dat_inf.loc[key, "th_strt"]
                th_end = o_dat_inf.loc[key, "th_end"]
                span = abs(th_end - th_start)
                max_zeithorizont_span = max(max_zeithorizont_span, span)
                zeithorizont_details.append(f"Output '{key}': {th_start}h to {th_end}h (span: {span}h)")

            logger.info(f"   Zeithorizont configuration:")
            for detail in zeithorizont_details:
                logger.info(f"      {detail}")
            logger.info(f"   Maximum zeithorizont span: {max_zeithorizont_span}h")

            logger.info(f"📍 Step 6 complete: Zeithorizont configuration logged (span: {max_zeithorizont_span}h)")

            # EXACT MATCH to original training.py lines 1097-1100:
            # utc_strt = i_dat_inf["utc_min"].min()  → minimum of all min timestamps
            # utc_end = i_dat_inf["utc_max"].min()   → MINIMUM of all max timestamps (NOT max!)
            # This ensures data is available for ALL input files at every timestep
            utc_strt = i_dat_inf["utc_min"].min()
            utc_end = i_dat_inf["utc_max"].min()  # CRITICAL: .min() not .max()!

            if not o_dat_inf.empty:
                utc_strt = max(utc_strt, o_dat_inf["utc_min"].min())
                utc_end = min(utc_end, o_dat_inf["utc_max"].min())  # CRITICAL: .min() not .max()!

            # NO OFFSET - exactly matching original training.py
            # Original lines 1097-1100 do NOT apply any offsets.
            # Iterations that fail interpolation are simply skipped.
            data_span_hours = (utc_end - utc_strt).total_seconds() / 3600
            logger.info(f"   📊 Data span (exact match to original, NO offsets): {data_span_hours:.1f} hours")
            logger.info(f"   utc_strt = {utc_strt}, utc_end = {utc_end}")

            mdl_config = None
            if model_params:
                mode_mapping = {
                    'Linear': 'LIN',
                    'Dense': 'Dense',
                    'CNN': 'CNN',
                    'LSTM': 'LSTM',
                    'AR LSTM': 'AR LSTM',
                    'SVR_dir': 'SVR_dir',
                    'SVR_MIMO': 'SVR_MIMO'
                }
                
                frontend_mode = model_params.get('MODE', 'Linear')
                backend_mode = mode_mapping.get(frontend_mode, 'LIN')
                
                mdl_config = MDL(mode=backend_mode)
                
                if backend_mode in ['Dense', 'CNN', 'LSTM', 'AR LSTM']:
                    if 'LAY' in model_params:
                        mdl_config.LAY = model_params['LAY']
                    if 'N' in model_params:
                        mdl_config.N = model_params['N']
                    if 'EP' in model_params:
                        mdl_config.EP = model_params['EP']
                    if 'ACTF' in model_params:
                        mdl_config.ACTF = model_params['ACTF']
                    if backend_mode == 'CNN' and 'K' in model_params:
                        mdl_config.K = model_params['K']
                elif backend_mode in ['SVR_dir', 'SVR_MIMO']:
                    if 'KERNEL' in model_params:
                        mdl_config.KERNEL = model_params['KERNEL']
                    if 'C' in model_params:
                        mdl_config.C = model_params['C']
                    if 'EPSILON' in model_params:
                        mdl_config.EPSILON = model_params['EPSILON']
            
            logger.info(f"📍 Step 6: Running training pipeline with utc_strt={utc_strt}, utc_end={utc_end}")
            logger.info(f"   Model config: MODE={getattr(mdl_config, 'MODE', 'default')}, LAY={getattr(mdl_config, 'LAY', None)}, N={getattr(mdl_config, 'N', None)}")

            try:
                results = run_exact_training_pipeline(
                    i_dat=i_dat,
                    o_dat=o_dat,
                    i_dat_inf=i_dat_inf,
                    o_dat_inf=o_dat_inf,
                    utc_strt=utc_strt,
                    utc_end=utc_end,
                    random_dat=model_params.get('random_dat', False) if model_params else False,
                    mdl_config=mdl_config,
                    mts_config=mts_config,
                    socketio=self.socketio,
                    session_id=session_id
                )
                logger.info(f"📍 Step 6 complete: Training pipeline finished successfully")
            except Exception as e:
                error_msg = f"Training pipeline failed: {str(e)}"
                logger.error(f"❌ Step 6 FAILED: {error_msg}")
                logger.error(f"Exception details: {traceback.format_exc()}")
                return {
                    'success': False,
                    'error': error_msg,
                    'details': str(e),
                    'stage': 'training_pipeline'
                }
            
            
            violin_plots = {}
            try:
                from domains.training.services.violin import create_violin_plots_from_viz_data
                import numpy as np

                viz_data = {
                    'i_combined_array': results.get('scalers', {}).get('i_combined_array'),
                    'o_combined_array': results.get('scalers', {}).get('o_combined_array')
                }

                if viz_data['i_combined_array'] is None:
                    train_x = results.get('train_data', {}).get('X_orig')
                    val_x = results.get('val_data', {}).get('X_orig')
                    test_x = results.get('test_data', {}).get('X_orig')

                    if train_x is not None and val_x is not None and test_x is not None:
                        all_x = np.vstack([train_x, val_x, test_x])
                        viz_data['i_combined_array'] = all_x.reshape(-1, all_x.shape[-1])

                    train_y = results.get('train_data', {}).get('y_orig')
                    val_y = results.get('val_data', {}).get('y_orig')
                    test_y = results.get('test_data', {}).get('y_orig')

                    if train_y is not None and val_y is not None and test_y is not None:
                        all_y = np.vstack([train_y, val_y, test_y])
                        viz_data['o_combined_array'] = all_y.reshape(-1, all_y.shape[-1])

                # Get feature names from results metadata (matches original training.py)
                # These come from i_dat_inf.index and o_dat_inf.index
                metadata = results.get('metadata', {})
                input_feature_names = metadata.get('input_features', [])
                output_feature_names = metadata.get('output_features', [])

                logger.info(f"Violin plot feature names from metadata: input={input_feature_names}, output={output_feature_names}")

                if viz_data['i_combined_array'] is not None or viz_data['o_combined_array'] is not None:
                    violin_plots = create_violin_plots_from_viz_data(
                        session_id,
                        viz_data,
                        input_feature_names=input_feature_names,
                        output_feature_names=output_feature_names
                    )
                else:
                    logger.warning("No data available for creating violin plots")
                    
            except Exception as e:
                logger.error(f"Error generating visualizations: {str(e)}")
                import traceback as tb
                logger.error(tb.format_exc())
            
            evaluation_metrics = results.get('evaluation_metrics', {})
            metrics = results.get('metrics', evaluation_metrics)
            
            return {
                'success': True,
                'session_id': session_id,
                'results': results,
                'violin_plots': violin_plots,
                'dataset_count': results.get('metadata', {}).get('n_dat', 0),
                'evaluation_metrics': evaluation_metrics,
                'metrics': metrics,
                'message': 'Training completed successfully using extracted modules'
            }
            
        except Exception as e:
            error_msg = f"Training failed for session {session_id}: {str(e)}"
            logger.error(error_msg)
            import traceback as tb
            logger.error(tb.format_exc())
            
            self._save_error_to_database(session_id, str(e), tb.format_exc())
            
            return {
                'success': False,
                'session_id': session_id,
                'error': str(e),
                'message': 'Training failed - see logs for details'
            }
    
    def _validate_session(self, session_id: str) -> bool:
        """
        Validate that session exists and has necessary data
        
        Args:
            session_id: Session identifier (can be string or UUID)
            
        Returns:
            True if valid, False otherwise
        """
        try:
            uuid_session_id = create_or_get_session_uuid(session_id, user_id=None)
            
            response = self.supabase.table('sessions').select('id').eq('id', uuid_session_id).execute()
            
            if not response.data:
                logger.error(f"Session {session_id} not found in database")
                return False
            
            session_uuid = response.data[0]['id']
            
            files_response = self.supabase.table('files').select('*').eq('session_id', session_uuid).execute()
            
            if not files_response.data:
                logger.error(f"No files found for session {session_id}")
                return False
            
            input_files = [f for f in files_response.data if f.get('type') == 'input']
            output_files = [f for f in files_response.data if f.get('type') == 'output']
            
            if not input_files:
                logger.error(f"No input files found for session {session_id}")
                return False
                
            if not output_files:
                logger.error(f"No output files found for session {session_id}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating session {session_id}: {str(e)}")
            return False
    
    def _save_error_to_database(self, session_id: str, error_message: str, error_traceback: str):
        """Save error information to database"""
        try:
            uuid_session_id = create_or_get_session_uuid(session_id, user_id=None)
            
            error_data = {
                'session_id': uuid_session_id,
                'error_message': error_message,
                'error_traceback': error_traceback,
                'status': 'failed'
            }
            
            self.supabase.table('training_results').insert(error_data).execute()
            
        except Exception as e:
            logger.error(f"Failed to save error to database: {str(e)}")

def run_training_script(session_id: str) -> Dict:
    """
    Legacy function that maintains the same API as the old middleman_runner
    Now uses the modern run_exact_training_pipeline approach
    
    Args:
        session_id: Session identifier
        
    Returns:
        Dict containing training results
    """
    runner = ModernMiddlemanRunner()
    return runner.run_training_script(session_id)

def main():
    """Main entry point for command line usage"""
    if len(sys.argv) < 2:
        print("Usage: python middleman_runner_new.py <session_id>")
        print("Please provide a session ID as a command-line argument.")
        sys.exit(1)
    
    session_id = sys.argv[1]
    
    print(f"🚀 Starting modern training pipeline for session: {session_id}")
    print("=" * 60)
    
    runner = ModernMiddlemanRunner()
    result = runner.run_training_script(session_id)
    
    if result['success']:
        print("\n✅ Training completed successfully!")
        print(f"Results: {result.get('message', 'Training completed')}")
        
        if 'results' in result and 'summary' in result['results']:
            summary = result['results']['summary']
            print(f"📊 Summary:")
            print(f"  - Total datasets: {summary.get('total_datasets', 'N/A')}")
            print(f"  - Total models: {summary.get('total_models', 'N/A')}")
            if 'best_model' in summary:
                best = summary['best_model']
                print(f"  - Best model: {best.get('name', 'N/A')} (MAE: {best.get('mae', 'N/A')})")
                
        sys.exit(0)
    else:
        print("\n❌ Training failed!")
        print(f"Error: {result.get('error', 'Unknown error')}")
        print(f"Message: {result.get('message', 'No additional details')}")
        sys.exit(1)

if __name__ == "__main__":
    main()
