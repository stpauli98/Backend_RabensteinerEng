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

# Import existing supabase client and UUID conversion
from utils.database import get_supabase_client, create_or_get_session_uuid

# Import verified pipeline that produces identical results to original
from services.training.pipeline_exact import run_exact_training_pipeline
from services.training.data_loader import DataLoader
from services.training.config import MDL

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
BACKEND_URL = "http://127.0.0.1:8080"

class ModernMiddlemanRunner:
    """
    Modern middleman runner that uses TrainingPipeline instead of subprocess
    Maintains the same API but uses extracted modules internally
    """
    
    def __init__(self):
        self.supabase = get_supabase_client()
        self.socketio = None  # Will be set if SocketIO is available
        
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
            logger.info(f"Starting verified training pipeline for session {session_id}")
            
            # Validate session exists
            if not self._validate_session(session_id):
                raise ValueError(f"Session {session_id} not found or invalid")
            
            # Load session data
            data_loader = DataLoader(self.supabase)
            session_data = data_loader.load_session_data(session_id)
            
            # Get file paths
            input_files, output_files = data_loader.prepare_file_paths(session_id)
            
            # Load CSV data
            i_dat = {}
            o_dat = {}
            
            # Load input files
            for file_path in input_files:
                df = data_loader.load_csv_data(file_path, delimiter=';')
                file_name = file_path.split('/')[-1].replace('.csv', '')
                i_dat[file_name] = df
                
            # Load output files  
            for file_path in output_files:
                df = data_loader.load_csv_data(file_path, delimiter=';')
                file_name = file_path.split('/')[-1].replace('.csv', '')
                o_dat[file_name] = df
            
            # Import load and transf functions from data_loader
            from services.training.data_loader import load, transf
            from services.training.config import MTS
            
            # Create MTS instance
            mts_config = MTS()
            
            # Initialize info DataFrames with all required columns as in training_original.py
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
            
            # Process each input file through load() to populate info
            # Pass the entire dictionary to load() so it can update the DataFrames
            for key in list(i_dat.keys()):
                temp_dict = {key: i_dat[key]}
                temp_dict, i_dat_inf = load(temp_dict, i_dat_inf)
                i_dat[key] = temp_dict[key]  # Update with processed DataFrame
            
            # Process each output file through load() to populate info
            for key in list(o_dat.keys()):
                temp_dict = {key: o_dat[key]}
                temp_dict, o_dat_inf = load(temp_dict, o_dat_inf)
                o_dat[key] = temp_dict[key]  # Update with processed DataFrame
            
            # Set required column values for all input files (as in training_original.py)
            for key in i_dat_inf.index:
                i_dat_inf.loc[key, "spec"] = "Historische Daten"
                i_dat_inf.loc[key, "th_strt"] = -1  # Time horizon start (hours before reference)
                i_dat_inf.loc[key, "th_end"] = 0    # Time horizon end (at reference time)
                i_dat_inf.loc[key, "meth"] = "Lineare Interpolation"
                i_dat_inf.loc[key, "avg"] = False   # No averaging by default
                i_dat_inf.loc[key, "scal"] = True   # Enable scaling
                i_dat_inf.loc[key, "scal_max"] = 1  # Max scaling value
                i_dat_inf.loc[key, "scal_min"] = 0  # Min scaling value
            
            # Set required column values for all output files
            for key in o_dat_inf.index:
                o_dat_inf.loc[key, "spec"] = "Historische Daten"
                o_dat_inf.loc[key, "th_strt"] = 0   # Time horizon start (at reference time)
                o_dat_inf.loc[key, "th_end"] = 1    # Time horizon end (1 hour after reference)
                o_dat_inf.loc[key, "meth"] = "Lineare Interpolation"
                o_dat_inf.loc[key, "avg"] = False   # No averaging by default
                o_dat_inf.loc[key, "scal"] = True   # Enable scaling
                o_dat_inf.loc[key, "scal_max"] = 1  # Max scaling value
                o_dat_inf.loc[key, "scal_min"] = 0  # Min scaling value
            
            # Apply transformations
            i_dat_inf = transf(i_dat_inf, mts_config.I_N, mts_config.OFST)
            o_dat_inf = transf(o_dat_inf, mts_config.O_N, mts_config.OFST)
            
            # Get time info from session
            time_info = session_data.get('timeInfo', {})
            utc_strt = datetime.fromisoformat(time_info.get('startzeitpunkt', '2023-01-01T00:00:00'))
            utc_end = datetime.fromisoformat(time_info.get('endzeitpunkt', '2023-12-31T23:59:59'))
            
            # Create MDL configuration from model_params or use defaults
            mdl_config = None
            if model_params:
                mdl_config = MDL()
                mdl_config.MODE = model_params.get('MODE', 'Linear')
                
                # Set parameters based on model type
                if mdl_config.MODE in ['Dense', 'CNN', 'LSTM', 'AR LSTM']:
                    mdl_config.LAY = model_params.get('LAY', 2)
                    mdl_config.N = model_params.get('N', 64)
                    mdl_config.EP = model_params.get('EP', 10)
                    mdl_config.ACTF = model_params.get('ACTF', 'relu')
                    if mdl_config.MODE == 'CNN':
                        mdl_config.K = model_params.get('K', 3)
                elif mdl_config.MODE in ['SVR_dir', 'SVR_MIMO']:
                    mdl_config.KERNEL = model_params.get('KERNEL', 'rbf')
                    mdl_config.C = model_params.get('C', 1.0)
                    mdl_config.EPSILON = model_params.get('EPSILON', 0.1)
            
            # Run the verified pipeline
            logger.info("Executing verified pipeline_exact with model configuration")
            results = run_exact_training_pipeline(
                i_dat=i_dat,
                o_dat=o_dat,
                i_dat_inf=i_dat_inf,
                o_dat_inf=o_dat_inf,
                utc_strt=utc_strt,
                utc_end=utc_end,
                random_dat=model_params.get('random_dat', False) if model_params else False,
                mdl_config=mdl_config
            )
            
            logger.info(f"Training pipeline completed successfully for session {session_id}")
            
            # Return structured response
            return {
                'success': True,
                'session_id': session_id,
                'results': results,
                'message': 'Training completed successfully using extracted modules'
            }
            
        except Exception as e:
            error_msg = f"Training failed for session {session_id}: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            
            # Save error to database
            self._save_error_to_database(session_id, str(e), traceback.format_exc())
            
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
            # Convert string session ID to UUID if needed
            uuid_session_id = create_or_get_session_uuid(session_id)
            
            # Check if session exists - sessions table uses UUID
            response = self.supabase.table('sessions').select('id').eq('id', uuid_session_id).execute()
            
            if not response.data:
                logger.error(f"Session {session_id} not found in database")
                return False
            
            session_uuid = response.data[0]['id']
            
            # Check if session has files
            files_response = self.supabase.table('files').select('*').eq('session_id', session_uuid).execute()
            
            if not files_response.data:
                logger.error(f"No files found for session {session_id}")
                return False
            
            # Check for input and output files
            input_files = [f for f in files_response.data if f.get('type') == 'input']
            output_files = [f for f in files_response.data if f.get('type') == 'output']
            
            if not input_files:
                logger.error(f"No input files found for session {session_id}")
                return False
                
            if not output_files:
                logger.error(f"No output files found for session {session_id}")
                return False
            
            logger.info(f"Session {session_id} validated: {len(input_files)} input files, {len(output_files)} output files")
            return True
            
        except Exception as e:
            logger.error(f"Error validating session {session_id}: {str(e)}")
            return False
    
    def _save_error_to_database(self, session_id: str, error_message: str, error_traceback: str):
        """Save error information to database"""
        try:
            # Convert string session ID to UUID if needed
            uuid_session_id = create_or_get_session_uuid(session_id)
            
            error_data = {
                'session_id': uuid_session_id,
                'error_message': error_message,
                'error_traceback': error_traceback,
                'status': 'failed'
            }
            
            self.supabase.table('training_results').insert(error_data).execute()
            logger.info(f"Error saved to database for session {session_id} (UUID: {uuid_session_id})")
            
        except Exception as e:
            logger.error(f"Failed to save error to database: {str(e)}")

def run_training_script(session_id: str) -> Dict:
    """
    Legacy function that maintains the same API as the old middleman_runner
    Now uses the modern TrainingPipeline approach
    
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
        
        # Print some key results if available
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