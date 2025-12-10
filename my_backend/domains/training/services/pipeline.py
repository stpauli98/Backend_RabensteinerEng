"""
Training pipeline module for training system
Main orchestration module that coordinates all training components
"""

import logging
from typing import Dict, Optional, Any
from datetime import datetime
import traceback

from domains.training.config import MTS, MDL
from domains.training.data.loader import create_data_loader
from domains.training.data.processor import create_data_processor
from domains.training.ml.trainer import create_model_trainer
from domains.training.services.results import create_results_generator
from domains.training.services.visualization import create_visualizer
from domains.training.ml.integration import (
    run_real_training_pipeline, 
    run_dataset_generation_pipeline, 
    run_model_training_pipeline
)

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
        Main training pipeline execution using real extracted functions
        
        Args:
            session_id: Session identifier
            
        Returns:
            Dict containing training results
        """
        try:
            self.current_session_id = session_id
            
            self._update_progress(0, 'Initializing real training pipeline')
            
            self._update_progress(1, 'Running real training pipeline')
            final_results = run_real_training_pipeline(session_id, self.supabase, self.socketio)
            
            self._update_progress(6, 'Saving results to database')
            self._save_results_to_database(session_id, final_results)
            
            self._update_progress(7, 'Training completed successfully', completed=True)
            
            return final_results
            
        except Exception as e:
            error_msg = f"Real training pipeline failed for session {session_id}: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            
            self._update_progress(self.progress['completed_steps'], f'Training failed: {str(e)}', error=True)
            
            self._save_error_to_database(session_id, str(e), traceback.format_exc())
            
            raise
    
    def run_dataset_generation(self, session_id: str) -> Dict:
        """
        Run dataset generation pipeline (first step of restructured workflow)
        
        Args:
            session_id: Session identifier
            
        Returns:
            Dict containing dataset info and violin plots
        """
        try:
            self.current_session_id = session_id
            
            self._update_progress(0, 'Generating datasets and violin plots')
            
            results = run_dataset_generation_pipeline(session_id, self.supabase, self.socketio)
            
            self._update_progress(1, 'Saving dataset generation results')
            self._save_dataset_generation_to_database(session_id, results)
            
            self._update_progress(2, 'Dataset generation completed successfully', completed=True)
            
            return results
            
        except Exception as e:
            error_msg = f"Dataset generation failed for session {session_id}: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            
            self._update_progress(self.progress['completed_steps'], f'Dataset generation failed: {str(e)}', error=True)
            
            self._save_error_to_database(session_id, str(e), traceback.format_exc())
            
            raise
    
    def run_model_training(self, session_id: str, model_params: Dict) -> Dict:
        """
        Run model training pipeline with user parameters (second step of restructured workflow)
        
        Args:
            session_id: Session identifier
            model_params: User-specified model parameters
            
        Returns:
            Dict containing training results
        """
        try:
            self.current_session_id = session_id
            
            self._update_progress(0, 'Training models with user parameters')
            
            results = run_model_training_pipeline(session_id, model_params, self.supabase, self.socketio)
            
            self._update_progress(6, 'Saving training results to database')
            self._save_results_to_database(session_id, results)
            
            self._update_progress(7, 'Model training completed successfully', completed=True)
            
            return results
            
        except Exception as e:
            error_msg = f"Model training failed for session {session_id}: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            
            self._update_progress(self.progress['completed_steps'], f'Model training failed: {str(e)}', error=True)
            
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
            data_loader = create_data_loader(self.supabase)
            
            session_data = data_loader.load_session_data(session_id)
            
            input_files, output_files = data_loader.prepare_file_paths(session_id)
            
            
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
            config = self._create_mts_config(session_data)
            
            data_processor = create_data_processor(config)
            
            processed_data = data_processor.process_session_data(session_data, input_files, output_files)
            
            
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
            config = self._create_mdl_config(session_data)
            
            model_trainer = create_model_trainer(config)
            
            training_results = model_trainer.train_all_models(
                processed_data.get('train_datasets', {}), 
                session_data
            )
            
            
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
            results_generator = create_results_generator()
            
            evaluation_results = results_generator.generate_results(training_results, session_data)
            
            
            return evaluation_results
            
        except Exception as e:
            logger.error(f"Error evaluating models: {str(e)}")
            raise
    
    def _create_visualizations(self, training_results: Dict, evaluation_results: Dict) -> Dict:
        """
        Create visualizations using consolidated approach
        
        Args:
            training_results: Training results
            evaluation_results: Evaluation results
            
        Returns:
            Visualizations
        """
        try:
            visualizations = {}
            
            visualizer = create_visualizer()
            
            forecast_plots = visualizer.create_forecast_plots(training_results, evaluation_results)
            visualizations.update(forecast_plots)
            
            comparison_plots = visualizer.create_metrics_comparison_plots(evaluation_results)
            visualizations.update(comparison_plots)
            
            history_plots = visualizer.create_training_history_plots(training_results)
            visualizations.update(history_plots)
            
            
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
    
    def _clean_json_data(self, data):
        """
        Clean data by replacing inf/nan values to make it JSON serializable
        
        Args:
            data: Data structure to clean
            
        Returns:
            Cleaned data structure
        """
        import math
        import numpy as np
        
        if isinstance(data, dict):
            return {key: self._clean_json_data(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self._clean_json_data(item) for item in data]
        elif isinstance(data, (int, str, bool)) or data is None:
            return data
        elif isinstance(data, float):
            if math.isnan(data) or math.isinf(data):
                return None
            return data
        elif isinstance(data, np.ndarray):
            return self._clean_json_data(data.tolist())
        elif hasattr(data, 'item'):
            return self._clean_json_data(data.item())
        else:
            return str(data)
    
    def _save_results_to_database(self, session_id: str, results: Dict) -> bool:
        """
        Save results to database using the new schema
        
        Args:
            session_id: Session identifier
            results: Results from real pipeline
            
        Returns:
            True if successful
        """
        try:
            evaluation_results = results.get('evaluation_results', {})
            training_results = results.get('training_results', {})
            visualizations = results.get('visualizations', {})
            summary = results.get('summary', {})
            
            evaluation_results = self._clean_json_data(evaluation_results)
            training_results = self._clean_json_data(training_results)
            summary = self._clean_json_data(summary)
            
            result_data = {
                'session_id': session_id,
                'evaluation_metrics': evaluation_results.get('evaluation_metrics', {}),
                'model_performance': training_results,
                'best_model': summary.get('best_model', {}),
                'summary': summary,
                'status': 'completed',
                'completed_at': datetime.now().isoformat()
            }
            
            response = self.supabase.table('training_results').insert(result_data).execute()
            
            if response.data:
                result_id = response.data[0]['id']
                
                self._save_visualizations_to_database(session_id, visualizations)
                
                return True
            else:
                logger.error(f"Failed to save results to database for session {session_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error saving results to database: {str(e)}")
            return False
    
    def _save_visualizations_to_database(self, session_id: str, visualizations: Dict):
        """Save visualizations to training_visualizations table"""
        try:
            for plot_name, plot_data_base64 in visualizations.items():
                viz_data = {
                    'session_id': session_id,
                    'plot_name': plot_name,
                    'plot_type': 'violin' if 'violin' in plot_name.lower() else 'unknown',
                    'plot_data_base64': plot_data_base64,
                    'metadata': {'generated_by': 'real_pipeline'}
                }
                
                self.supabase.table('training_visualizations').insert(viz_data).execute()
                
            
        except Exception as e:
            logger.error(f"Error saving visualizations: {str(e)}")
    
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
                'completed_at': datetime.now().isoformat()
            }
            
            self.supabase.table('training_results').insert(error_data).execute()
            
        except Exception as e:
            logger.error(f"Error saving error to database: {str(e)}")
    
    def _save_dataset_generation_to_database(self, session_id: str, results: Dict) -> bool:
        """
        Save dataset generation results to database
        
        Args:
            session_id: Session identifier
            results: Dataset generation results
            
        Returns:
            True if successful
        """
        try:
            dataset_data = {
                'session_id': session_id,
                'status': 'datasets_generated',
                'dataset_count': results.get('dataset_count', 0),
                'datasets_info': results.get('datasets_info', {}),
                'completed_at': datetime.now().isoformat()
            }
            
            response = self.supabase.table('training_results').insert(dataset_data).execute()
            
            if response.data:
                visualizations = results.get('visualizations', {})
                if visualizations:
                    self._save_visualizations_to_database(session_id, visualizations)
                
                return True
            else:
                logger.error(f"Failed to save dataset generation results for session {session_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error saving dataset generation to database: {str(e)}")
            return False
    
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
            
            if self.socketio and self.current_session_id:
                self.socketio.emit('training_progress', {
                    'session_id': self.current_session_id,
                    'progress': self.progress
                }, room=f"training_{self.current_session_id}")
            
            self._save_progress_to_database()
            
            
        except Exception as e:
            logger.error(f"Error updating progress: {str(e)}")
    
    def _save_progress_to_database(self):
        """Save progress to database"""
        try:
            if self.current_session_id:
                clean_progress = self._clean_json_data(self.progress)
                
                progress_data = {
                    'session_id': self.current_session_id,
                    'message': f'Training progress: {clean_progress.get("overall", 0)}%',
                    'level': 'INFO',
                    'step_number': int(clean_progress.get("completed_steps", 0)),
                    'step_name': clean_progress.get("current_step", "Training in progress"),
                    'progress_percentage': clean_progress.get("overall", 0)
                }
                
                self.supabase.table('training_logs').insert(progress_data).execute()
                
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
            
            time_info = session_data.get('time_info', {})
            zeitschritte = session_data.get('zeitschritte', {})
            
            config.jahr = time_info.get('jahr', True)
            config.monat = time_info.get('monat', True)
            config.woche = time_info.get('woche', True)
            config.feiertag = time_info.get('feiertag', True)
            config.timezone = time_info.get('zeitzone', 'UTC')
            
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
        pipeline = create_training_pipeline(supabase_client, socketio_instance)
        
        results = pipeline.run_training_pipeline(session_id)
        
        return results
        
    except Exception as e:
        logger.error(f"Error running training for session {session_id}: {str(e)}")
        raise
