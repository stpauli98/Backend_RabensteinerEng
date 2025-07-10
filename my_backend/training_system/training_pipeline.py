"""
Training pipeline module for training system
Main orchestration module that coordinates all training components
"""

import logging
from typing import Dict, Optional, Any
from datetime import datetime
import traceback

from .config import MTS, MDL
from .data_loader import create_data_loader
from .data_processor import create_data_processor
from .model_trainer import create_model_trainer
from .results_generator import create_results_generator
from .visualization import create_visualizer

logger = logging.getLogger(__name__)


class TrainingPipeline:
    """
    Main training pipeline orchestrator
    Coordinates all training system components
    """
    
    def __init__(self, supabase_client, socketio_instance=None):
        self.supabase = supabase_client
        self.socketio = socketio_instance
        self.current_session_id = None
        self.progress = {
            'overall': 0,
            'current_step': 'Initializing',
            'total_steps': 7,
            'completed_steps': 0,
            'steps': [
                'Data Loading',
                'Data Processing', 
                'Model Training',
                'Evaluation',
                'Visualization',
                'Results Generation',
                'Saving Results'
            ]
        }
    
    def run_training_pipeline(self, session_id: str) -> Dict:
        """
        Main training pipeline execution
        
        Args:
            session_id: Session identifier
            
        Returns:
            Dict containing training results
        """
        try:
            self.current_session_id = session_id
            logger.info(f"Starting training pipeline for session {session_id}")
            
            # Initialize progress tracking
            self._update_progress(0, 'Initializing training pipeline')
            
            # Step 1: Load session data from database
            self._update_progress(1, 'Loading session data')
            session_data, input_files, output_files = self._load_session_data(session_id)
            
            # Step 2: Process data
            self._update_progress(2, 'Processing data')
            processed_data = self._process_data(session_data, input_files, output_files)
            
            # Step 3: Train models
            self._update_progress(3, 'Training models')
            training_results = self._train_models(processed_data, session_data)
            
            # Step 4: Evaluate models
            self._update_progress(4, 'Evaluating models')
            evaluation_results = self._evaluate_models(training_results, session_data)
            
            # Step 5: Create visualizations
            self._update_progress(5, 'Creating visualizations')
            visualizations = self._create_visualizations(training_results, evaluation_results)
            
            # Step 6: Generate final results
            self._update_progress(6, 'Generating results')
            final_results = self._generate_final_results(training_results, evaluation_results, visualizations)
            
            # Step 7: Save results to database
            self._update_progress(7, 'Saving results')
            self._save_results_to_database(session_id, final_results)
            
            # Mark as completed
            self._update_progress(7, 'Training completed successfully', completed=True)
            
            logger.info(f"Training pipeline completed successfully for session {session_id}")
            return final_results
            
        except Exception as e:
            error_msg = f"Training pipeline failed for session {session_id}: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            
            # Update progress with error
            self._update_progress(self.progress['completed_steps'], f'Training failed: {str(e)}', error=True)
            
            # Save error to database
            self._save_error_to_database(session_id, str(e), traceback.format_exc())
            
            raise
    
    def _load_session_data(self, session_id: str) -> tuple:
        """
        Load session data from database
        
        Args:
            session_id: Session identifier
            
        Returns:
            Tuple of (session_data, input_files, output_files)
        """
        try:
            # Create data loader
            data_loader = create_data_loader(self.supabase)
            
            # Load session data
            session_data = data_loader.load_session_data(session_id)
            
            # Download files
            input_files, output_files = data_loader.prepare_file_paths(session_id)
            
            logger.info(f"Loaded session data for {session_id}: {len(input_files)} input files, {len(output_files)} output files")
            
            return session_data, input_files, output_files
            
        except Exception as e:
            logger.error(f"Error loading session data: {str(e)}")
            raise
    
    def _process_data(self, session_data: Dict, input_files: list, output_files: list) -> Dict:
        """
        Process the loaded data
        
        Args:
            session_data: Session configuration
            input_files: List of input file paths
            output_files: List of output file paths
            
        Returns:
            Processed data
        """
        try:
            # Create MTS config from session data
            config = self._create_mts_config(session_data)
            
            # Create data processor
            data_processor = create_data_processor(config)
            
            # Process data
            processed_data = data_processor.process_session_data(session_data, input_files, output_files)
            
            logger.info(f"Data processing completed. Datasets: {len(processed_data.get('train_datasets', {}))}")
            
            return processed_data
            
        except Exception as e:
            logger.error(f"Error processing data: {str(e)}")
            raise
    
    def _train_models(self, processed_data: Dict, session_data: Dict) -> Dict:
        """
        Train all models
        
        Args:
            processed_data: Processed data
            session_data: Session configuration
            
        Returns:
            Training results
        """
        try:
            # Create MDL config from session data
            config = self._create_mdl_config(session_data)
            
            # Create model trainer
            model_trainer = create_model_trainer(config)
            
            # Train models
            training_results = model_trainer.train_all_models(
                processed_data.get('train_datasets', {}), 
                session_data
            )
            
            logger.info(f"Model training completed. Results: {len(training_results)} datasets")
            
            return training_results
            
        except Exception as e:
            logger.error(f"Error training models: {str(e)}")
            raise
    
    def _evaluate_models(self, training_results: Dict, session_data: Dict) -> Dict:
        """
        Evaluate trained models
        
        Args:
            training_results: Training results
            session_data: Session configuration
            
        Returns:
            Evaluation results
        """
        try:
            # Create results generator
            results_generator = create_results_generator()
            
            # Generate evaluation results
            evaluation_results = results_generator.generate_results(training_results, session_data)
            
            logger.info(f"Model evaluation completed")
            
            return evaluation_results
            
        except Exception as e:
            logger.error(f"Error evaluating models: {str(e)}")
            raise
    
    def _create_visualizations(self, training_results: Dict, evaluation_results: Dict) -> Dict:
        """
        Create visualizations
        
        Args:
            training_results: Training results
            evaluation_results: Evaluation results
            
        Returns:
            Visualizations
        """
        try:
            # Create visualizer
            visualizer = create_visualizer()
            
            # Create all visualizations
            visualizations = visualizer.create_all_visualizations(training_results, evaluation_results)
            
            logger.info(f"Visualizations created: {len(visualizations)} plots")
            
            return visualizations
            
        except Exception as e:
            logger.error(f"Error creating visualizations: {str(e)}")
            raise
    
    def _generate_final_results(self, training_results: Dict, evaluation_results: Dict, visualizations: Dict) -> Dict:
        """
        Generate final results structure
        
        Args:
            training_results: Training results
            evaluation_results: Evaluation results
            visualizations: Visualizations
            
        Returns:
            Final results structure
        """
        try:
            final_results = {
                'session_id': self.current_session_id,
                'timestamp': datetime.now().isoformat(),
                'status': 'completed',
                'training_results': training_results,
                'evaluation_results': evaluation_results,
                'visualizations': visualizations,
                'summary': {
                    'total_datasets': len(training_results),
                    'total_models': sum(len(models) for models in training_results.values()),
                    'total_plots': len(visualizations),
                    'best_model': self._find_best_model(evaluation_results)
                }
            }
            
            return final_results
            
        except Exception as e:
            logger.error(f"Error generating final results: {str(e)}")
            raise
    
    def _save_results_to_database(self, session_id: str, results: Dict) -> bool:
        """
        Save results to database
        
        Args:
            session_id: Session identifier
            results: Results to save
            
        Returns:
            True if successful
        """
        try:
            # Save main results
            result_data = {
                'session_id': session_id,
                'results': results,
                'status': 'completed',
                'completed_at': datetime.now().isoformat()
            }
            
            response = self.supabase.table('training_results').insert(result_data).execute()
            
            if response.data:
                logger.info(f"Results saved to database for session {session_id}")
                return True
            else:
                logger.error(f"Failed to save results to database for session {session_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error saving results to database: {str(e)}")
            return False
    
    def _save_error_to_database(self, session_id: str, error_message: str, error_traceback: str):
        """
        Save error information to database
        
        Args:
            session_id: Session identifier
            error_message: Error message
            error_traceback: Error traceback
        """
        try:
            error_data = {
                'session_id': session_id,
                'error_message': error_message,
                'error_traceback': error_traceback,
                'status': 'failed',
                'failed_at': datetime.now().isoformat()
            }
            
            self.supabase.table('training_results').insert(error_data).execute()
            
        except Exception as e:
            logger.error(f"Error saving error to database: {str(e)}")
    
    def _update_progress(self, step: int, message: str, completed: bool = False, error: bool = False):
        """
        Update training progress
        
        Args:
            step: Current step number
            message: Progress message
            completed: Whether training is completed
            error: Whether there was an error
        """
        try:
            self.progress['completed_steps'] = step
            self.progress['current_step'] = message
            self.progress['overall'] = int((step / self.progress['total_steps']) * 100)
            
            if completed:
                self.progress['overall'] = 100
                self.progress['current_step'] = 'Completed'
            elif error:
                self.progress['current_step'] = f'Error: {message}'
            
            # Emit progress via SocketIO if available
            if self.socketio and self.current_session_id:
                self.socketio.emit('training_progress', {
                    'session_id': self.current_session_id,
                    'progress': self.progress
                }, room=self.current_session_id)
            
            # Save progress to database
            self._save_progress_to_database()
            
            logger.info(f"Progress: {self.progress['overall']}% - {message}")
            
        except Exception as e:
            logger.error(f"Error updating progress: {str(e)}")
    
    def _save_progress_to_database(self):
        """Save progress to database"""
        try:
            if self.current_session_id:
                progress_data = {
                    'session_id': self.current_session_id,
                    'progress': self.progress,
                    'updated_at': datetime.now().isoformat()
                }
                
                # Upsert progress
                self.supabase.table('training_progress').upsert(progress_data).execute()
                
        except Exception as e:
            logger.error(f"Error saving progress to database: {str(e)}")
    
    def _create_mts_config(self, session_data: Dict) -> MTS:
        """
        Create MTS configuration from session data
        
        Args:
            session_data: Session data
            
        Returns:
            MTS configuration object
        """
        try:
            config = MTS()
            
            # Configure from session data
            time_info = session_data.get('time_info', {})
            zeitschritte = session_data.get('zeitschritte', {})
            
            # Set time features
            config.jahr = time_info.get('jahr', True)
            config.monat = time_info.get('monat', True)
            config.woche = time_info.get('woche', True)
            config.feiertag = time_info.get('feiertag', True)
            config.timezone = time_info.get('zeitzone', 'UTC')
            
            # Set time steps
            config.time_steps_in = int(zeitschritte.get('eingabe', 24))
            config.time_steps_out = int(zeitschritte.get('ausgabe', 1))
            config.time_step_size = int(zeitschritte.get('zeitschrittweite', 1))
            config.offset = int(zeitschritte.get('offset', 0))
            
            return config
            
        except Exception as e:
            logger.error(f"Error creating MTS config: {str(e)}")
            raise
    
    def _create_mdl_config(self, session_data: Dict) -> MDL:
        """
        Create MDL configuration from session data
        
        Args:
            session_data: Session data
            
        Returns:
            MDL configuration object
        """
        try:
            config = MDL()
            
            # TODO: Configure models based on session data
            # For now, use defaults
            
            return config
            
        except Exception as e:
            logger.error(f"Error creating MDL config: {str(e)}")
            raise
    
    def _find_best_model(self, evaluation_results: Dict) -> Dict:
        """
        Find the best performing model
        
        Args:
            evaluation_results: Evaluation results
            
        Returns:
            Best model information
        """
        try:
            best_model = {'name': 'unknown', 'dataset': 'unknown', 'mae': float('inf')}
            
            model_comparison = evaluation_results.get('model_comparison', {})
            
            for dataset_name, comparison in model_comparison.items():
                best_models = comparison.get('best_models', {})
                
                if 'mae' in best_models:
                    mae_info = best_models['mae']
                    if mae_info['value'] < best_model['mae']:
                        best_model = {
                            'name': mae_info['model'],
                            'dataset': dataset_name,
                            'mae': mae_info['value']
                        }
            
            return best_model
            
        except Exception as e:
            logger.error(f"Error finding best model: {str(e)}")
            return {'name': 'unknown', 'dataset': 'unknown', 'mae': float('inf')}


# Factory function to create training pipeline
def create_training_pipeline(supabase_client, socketio_instance=None) -> TrainingPipeline:
    """
    Create and return a TrainingPipeline instance
    
    Args:
        supabase_client: Supabase client instance
        socketio_instance: SocketIO instance for progress updates
        
    Returns:
        TrainingPipeline instance
    """
    return TrainingPipeline(supabase_client, socketio_instance)


# Main execution function for use in middleman_runner.py
def run_training_for_session(session_id: str, supabase_client, socketio_instance=None) -> Dict:
    """
    Main function to run training for a session
    This function can be called from middleman_runner.py
    
    Args:
        session_id: Session identifier
        supabase_client: Supabase client instance
        socketio_instance: SocketIO instance for progress updates
        
    Returns:
        Training results
    """
    try:
        # Create training pipeline
        pipeline = create_training_pipeline(supabase_client, socketio_instance)
        
        # Run training
        results = pipeline.run_training_pipeline(session_id)
        
        return results
        
    except Exception as e:
        logger.error(f"Error running training for session {session_id}: {str(e)}")
        raise