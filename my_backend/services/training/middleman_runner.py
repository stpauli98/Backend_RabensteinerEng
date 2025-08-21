"""
Refactored middleman runner that uses TrainingPipeline instead of subprocess
This replaces the old subprocess approach with direct module integration
"""

import sys
import os
import logging
from typing import Dict, Optional
import traceback

# Import existing supabase client and SocketIO
from utils.database import get_supabase_client
from services.training.training_pipeline import run_training_for_session

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
        
    def run_training_script(self, session_id: str) -> Dict:
        """
        Main training execution using TrainingPipeline
        
        Args:
            session_id: Session identifier
            
        Returns:
            Dict containing training results and status
        """
        try:
            logger.info(f"Starting modern training pipeline for session {session_id}")
            
            # Validate session exists
            if not self._validate_session(session_id):
                raise ValueError(f"Session {session_id} not found or invalid")
            
            # Run the training pipeline using extracted modules
            logger.info("Executing TrainingPipeline with real extracted functions")
            results = run_training_for_session(
                session_id=session_id,
                supabase_client=self.supabase,
                socketio_instance=self.socketio
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
            session_id: Session identifier
            
        Returns:
            True if valid, False otherwise
        """
        try:
            # Check if session exists - sessions table has 'id' column not 'uuid'
            response = self.supabase.table('sessions').select('id').eq('id', session_id).execute()
            
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
            error_data = {
                'session_id': session_id,
                'error_message': error_message,
                'error_traceback': error_traceback,
                'status': 'failed'
            }
            
            self.supabase.table('training_results').insert(error_data).execute()
            logger.info(f"Error saved to database for session {session_id}")
            
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