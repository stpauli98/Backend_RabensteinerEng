"""
Training API module for training system
Provides API endpoints for training results and status
"""

from flask import Blueprint, jsonify, request
from typing import Dict, Optional
import logging
import sys
import os
from datetime import datetime

# Import existing supabase client
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from supabase_client import get_supabase_client, create_or_get_session_uuid

# Import the new pipeline function
from .pipeline_integration import run_complete_original_pipeline

logger = logging.getLogger(__name__)

# Create blueprint for training API
training_api_bp = Blueprint('training_api', __name__, url_prefix='/api/training')


@training_api_bp.route('/results/<session_id>', methods=['GET'])
def get_training_results(session_id: str):
    """
    Get training results for a session
    
    Args:
        session_id: Session identifier
        
    Returns:
        JSON response with training results
    """
    try:
        supabase = get_supabase_client()
        results = _get_results_from_database(session_id, supabase)
        
        if not results:
            return jsonify({
                'session_id': session_id,
                'status': 'not_found',
                'message': 'No training results found for this session'
            }), 404
        
        # Also get visualizations from training_visualizations table
        visualizations = _get_visualizations_from_database(session_id, supabase)
        logger.info(f"Retrieved visualizations for {session_id}: {visualizations}")
        
        response_data = {
            'session_id': session_id,
            'status': results.get('status', 'unknown'),
            'evaluation_metrics': results.get('evaluation_metrics', {}),
            'model_performance': results.get('model_performance', {}),
            'best_model': results.get('best_model', {}),
            'summary': results.get('summary', {}),
            'visualizations': visualizations.get('plots', {}),  # Include violin plots
            'created_at': results.get('created_at'),
            'completed_at': results.get('completed_at'),
            'message': 'Training results retrieved successfully'
        }
        logger.info(f"API response includes visualizations: {'visualizations' in response_data}")
        
        return jsonify(response_data), 200
        
    except Exception as e:
        logger.error(f"Error retrieving training results for {session_id}: {str(e)}")
        return jsonify({
            'error': 'Failed to retrieve training results',
            'message': str(e)
        }), 500


@training_api_bp.route('/status/<session_id>', methods=['GET'])
@training_api_bp.route('/session-status/<session_id>', methods=['GET'])  # Alias for frontend compatibility
def get_training_status(session_id: str):
    """
    Get training status for a session
    
    Args:
        session_id: Session identifier
        
    Returns:
        JSON response with training status
    """
    try:
        supabase = get_supabase_client()
        
        # Check training_results table first
        results_status = _get_results_from_database(session_id, supabase)
        
        # Check training_logs table for detailed progress
        progress_status = _get_status_from_database(session_id, supabase)
        
        if results_status:
            # Training is completed
            status = {
                'session_id': session_id,
                'status': results_status.get('status', 'completed'),
                'progress': 100,
                'current_step': 'Training completed',
                'total_steps': 7,
                'completed_steps': 7,
                'started_at': results_status.get('created_at'),
                'completed_at': results_status.get('completed_at'),
                'message': 'Training completed successfully'
            }
        elif progress_status and progress_status.get('status') != 'not_found':
            # Training is in progress
            progress_data = progress_status.get('progress', {})
            status = {
                'session_id': session_id,
                'status': 'in_progress',
                'progress': progress_data.get('overall', 0),
                'current_step': progress_data.get('current_step', 'Processing'),
                'total_steps': progress_data.get('total_steps', 7),
                'completed_steps': progress_data.get('completed_steps', 0),
                'started_at': progress_status.get('created_at'),
                'completed_at': None,
                'message': 'Training in progress'
            }
        else:
            # No training found
            status = {
                'session_id': session_id,
                'status': 'not_found',
                'progress': 0,
                'current_step': 'Not started',
                'total_steps': 7,
                'completed_steps': 0,
                'started_at': None,
                'completed_at': None,
                'message': 'No training found for this session'
            }
        
        return jsonify(status), 200
        
    except Exception as e:
        logger.error(f"Error retrieving training status for {session_id}: {str(e)}")
        return jsonify({
            'error': 'Failed to retrieve training status',
            'message': str(e)
        }), 500


@training_api_bp.route('/progress/<session_id>', methods=['GET'])
def get_training_progress(session_id: str):
    """
    Get detailed training progress for a session
    
    Args:
        session_id: Session identifier
        
    Returns:
        JSON response with training progress details
    """
    try:
        # TODO: Import supabase client
        # progress = _get_progress_from_database(session_id, supabase)
        
        # Placeholder implementation
        progress = {
            'session_id': session_id,
            'overall_progress': 100,
            'steps': [
                {'step': 'Data Loading', 'status': 'completed', 'progress': 100},
                {'step': 'Data Processing', 'status': 'completed', 'progress': 100},
                {'step': 'Model Training', 'status': 'completed', 'progress': 100},
                {'step': 'Evaluation', 'status': 'completed', 'progress': 100},
                {'step': 'Visualization', 'status': 'completed', 'progress': 100},
                {'step': 'Results Generation', 'status': 'completed', 'progress': 100},
                {'step': 'Saving Results', 'status': 'completed', 'progress': 100}
            ],
            'models': {
                'dense': {'status': 'completed', 'metrics': {'mae': 0.123, 'mse': 0.456}},
                'cnn': {'status': 'completed', 'metrics': {'mae': 0.234, 'mse': 0.567}},
                'lstm': {'status': 'completed', 'metrics': {'mae': 0.345, 'mse': 0.678}},
                'svr': {'status': 'completed', 'metrics': {'mae': 0.456, 'mse': 0.789}}
            },
            'current_model': None,
            'estimated_time_remaining': 0
        }
        
        return jsonify(progress), 200
        
    except Exception as e:
        logger.error(f"Error retrieving training progress for {session_id}: {str(e)}")
        return jsonify({
            'error': 'Failed to retrieve training progress',
            'message': str(e)
        }), 500


@training_api_bp.route('/visualizations/<session_id>', methods=['GET'])
def get_training_visualizations(session_id: str):
    """
    Get training visualizations for a session
    
    Args:
        session_id: Session identifier
        
    Returns:
        JSON response with training visualizations
    """
    try:
        supabase = get_supabase_client()
        visualizations = _get_visualizations_from_database(session_id, supabase)
        
        if not visualizations or not visualizations.get('plots'):
            return jsonify({
                'session_id': session_id,
                'plots': {},
                'message': 'No visualizations found for this session'
            }), 404
        
        response_data = {
            'session_id': session_id,
            'plots': visualizations.get('plots', {}),
            'metadata': visualizations.get('metadata', {}),
            'created_at': visualizations.get('created_at'),
            'message': 'Visualizations retrieved successfully'
        }
        
        return jsonify(response_data), 200
        
    except Exception as e:
        logger.error(f"Error retrieving training visualizations for {session_id}: {str(e)}")
        return jsonify({
            'error': 'Failed to retrieve training visualizations',
            'message': str(e)
        }), 500


@training_api_bp.route('/metrics/<session_id>', methods=['GET'])
def get_training_metrics(session_id: str):
    """
    Get training metrics summary for a session
    
    Args:
        session_id: Session identifier
        
    Returns:
        JSON response with training metrics
    """
    try:
        supabase = get_supabase_client()
        results = _get_results_from_database(session_id, supabase)
        
        if not results:
            return jsonify({
                'session_id': session_id,
                'message': 'No metrics found for this session'
            }), 404
        
        evaluation_metrics = results.get('evaluation_metrics', {})
        best_model = results.get('best_model', {})
        summary = results.get('summary', {})
        
        response_data = {
            'session_id': session_id,
            'best_model': best_model,
            'evaluation_metrics': evaluation_metrics,
            'model_comparison': evaluation_metrics,  # Same structure for now
            'summary': summary,
            'created_at': results.get('created_at'),
            'message': 'Metrics retrieved successfully'
        }
        
        return jsonify(response_data), 200
        
    except Exception as e:
        logger.error(f"Error retrieving training metrics for {session_id}: {str(e)}")
        return jsonify({
            'error': 'Failed to retrieve training metrics',
            'message': str(e)
        }), 500


@training_api_bp.route('/logs/<session_id>', methods=['GET'])
def get_training_logs(session_id: str):
    """
    Get training logs for a session
    
    Args:
        session_id: Session identifier
        
    Returns:
        JSON response with training logs
    """
    try:
        # Get optional parameters
        limit = request.args.get('limit', 100, type=int)
        level = request.args.get('level', 'INFO')
        
        supabase = get_supabase_client()
        logs_data = _get_logs_from_database(session_id, supabase, limit, level)
        
        response_data = {
            'session_id': session_id,
            'logs': logs_data.get('logs', []),
            'total_logs': logs_data.get('total_logs', 0),
            'limit': limit,
            'level_filter': level,
            'message': 'Logs retrieved successfully'
        }
        
        return jsonify(response_data), 200
        
    except Exception as e:
        logger.error(f"Error retrieving training logs for {session_id}: {str(e)}")
        return jsonify({
            'error': 'Failed to retrieve training logs',
            'message': str(e)
        }), 500


@training_api_bp.route('/cancel/<session_id>', methods=['POST'])
def cancel_training(session_id: str):
    """
    Cancel training for a session
    
    Args:
        session_id: Session identifier
        
    Returns:
        JSON response with cancellation status
    """
    try:
        # TODO: Implement training cancellation logic
        # This would need to communicate with the training process
        
        # Placeholder implementation
        result = {
            'session_id': session_id,
            'status': 'cancelled',
            'message': 'Training cancellation requested'
        }
        
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Error cancelling training for {session_id}: {str(e)}")
        return jsonify({
            'error': 'Failed to cancel training',
            'message': str(e)
        }), 500


@training_api_bp.route('/list-sessions', methods=['GET'])
def list_sessions():
    """
    Get list of all available sessions
    
    Returns:
        JSON response with list of sessions
    """
    try:
        supabase = get_supabase_client()
        
        # Get all sessions with basic info
        sessions_response = supabase.table('sessions').select('id, created_at, finalized, file_count').order('created_at', desc=True).execute()
        
        sessions_list = []
        
        if sessions_response.data:
            for session in sessions_response.data:
                session_id = session['id']
                
                # Get additional info for each session
                # Check if there are files
                files_response = supabase.table('csv_file_refs').select('id, name, type').eq('session_id', session_id).execute()
                files_count = len(files_response.data) if files_response.data else 0
                
                # Check if there's time info
                time_info_response = supabase.table('time_info').select('jahr, woche, monat, feiertag, tag').eq('session_id', session_id).execute()
                has_time_info = len(time_info_response.data) > 0 if time_info_response.data else False
                
                # Check if there's zeitschritte info
                zeitschritte_response = supabase.table('zeitschritte').select('eingabe, ausgabe').eq('session_id', session_id).execute()
                has_zeitschritte = len(zeitschritte_response.data) > 0 if zeitschritte_response.data else False
                
                # Check training status
                training_results = supabase.table('training_results').select('status, created_at').eq('session_id', session_id).execute()
                training_status = 'not_started'
                training_completed_at = None
                
                if training_results.data and len(training_results.data) > 0:
                    result = training_results.data[0]
                    training_status = result.get('status', 'unknown')
                    training_completed_at = result.get('created_at')
                
                session_info = {
                    'id': session_id,
                    'created_at': session['created_at'],
                    'finalized': session.get('finalized', False),
                    'file_count': files_count,
                    'has_time_info': has_time_info,
                    'has_zeitschritte': has_zeitschritte,
                    'training_status': training_status,
                    'training_completed_at': training_completed_at,
                    'url': f'/training?session={session_id}'
                }
                
                sessions_list.append(session_info)
        
        return jsonify({
            'success': True,
            'sessions': sessions_list,
            'total_count': len(sessions_list)
        }), 200
        
    except Exception as e:
        logger.error(f"Error listing sessions: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Failed to list sessions',
            'message': str(e),
            'sessions': []
        }), 500


@training_api_bp.route('/get-session-uuid/<session_id>', methods=['GET'])
def get_session_uuid(session_id: str):
    """
    Get UUID for a session ID (handles string to UUID conversion)
    
    Args:
        session_id: Session identifier (string or UUID)
        
    Returns:
        JSON response with session UUID
    """
    try:
        import re
        supabase = get_supabase_client()
        
        # Check if it's already a UUID format
        uuid_regex = r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
        if re.match(uuid_regex, session_id, re.IGNORECASE):
            # Already a UUID, just return it
            return jsonify({
                'session_id': session_id,
                'uuid': session_id,
                'success': True
            }), 200
        
        # Look up in session_mappings table
        response = supabase.table('session_mappings').select('uuid_session_id').eq('string_session_id', session_id).execute()
        
        if response.data and len(response.data) > 0:
            uuid_session_id = response.data[0]['uuid_session_id']
            return jsonify({
                'session_id': session_id,
                'uuid': uuid_session_id,
                'success': True
            }), 200
        else:
            return jsonify({
                'session_id': session_id,
                'error': 'Session not found',
                'success': False
            }), 404
            
    except Exception as e:
        logger.error(f"Error getting session UUID for {session_id}: {str(e)}")
        return jsonify({
            'error': 'Failed to get session UUID',
            'message': str(e),
            'success': False
        }), 500


@training_api_bp.route('/create-database-session', methods=['POST'])
def create_database_session():
    """
    Create a new database session with UUID
    
    Returns:
        JSON response with created session UUID
    """
    try:
        import uuid
        supabase = get_supabase_client()
        
        # Create new session in sessions table
        new_session_id = str(uuid.uuid4())
        session_response = supabase.table('sessions').insert({
            'id': new_session_id,
            'finalized': False,
            'file_count': 0
        }).execute()
        
        if session_response.data:
            return jsonify({
                'success': True,
                'session_id': new_session_id,
                'uuid': new_session_id
            }), 200
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to create session'
            }), 500
            
    except Exception as e:
        logger.error(f"Error creating database session: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Failed to create database session',
            'message': str(e)
        }), 500


@training_api_bp.route('/save-time-info', methods=['POST'])
def save_time_info():
    """
    Save time information for a session
    
    Returns:
        JSON response with save status
    """
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        time_info = data.get('time_info', {})
        
        if not session_id:
            return jsonify({
                'success': False,
                'error': 'Missing session_id'
            }), 400
            
        supabase = get_supabase_client()
        
        # Save to time_info table
        response = supabase.table('time_info').upsert({
            'session_id': session_id,
            'jahr': time_info.get('jahr', False),
            'woche': time_info.get('woche', False),
            'monat': time_info.get('monat', False),
            'feiertag': time_info.get('feiertag', False),
            'tag': time_info.get('tag', False),
            'zeitzone': time_info.get('zeitzone', ''),
            'category_data': time_info.get('category_data', {})
        }).execute()
        
        return jsonify({
            'success': True,
            'message': 'Time info saved successfully'
        }), 200
        
    except Exception as e:
        logger.error(f"Error saving time info: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Failed to save time info',
            'message': str(e)
        }), 500


@training_api_bp.route('/save-zeitschritte', methods=['POST'])
def save_zeitschritte():
    """
    Save zeitschritte information for a session
    
    Returns:
        JSON response with save status
    """
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        zeitschritte = data.get('zeitschritte', {})
        
        if not session_id:
            return jsonify({
                'success': False,
                'error': 'Missing session_id'
            }), 400
            
        supabase = get_supabase_client()
        
        # Save to zeitschritte table
        response = supabase.table('zeitschritte').upsert({
            'session_id': session_id,
            'eingabe': zeitschritte.get('eingabe', ''),
            'ausgabe': zeitschritte.get('ausgabe', ''),
            'zeitschrittweite': zeitschritte.get('zeitschrittweite', ''),
            'offset': zeitschritte.get('offset', '')
        }).execute()
        
        return jsonify({
            'success': True,
            'message': 'Zeitschritte saved successfully'
        }), 200
        
    except Exception as e:
        logger.error(f"Error saving zeitschritte: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Failed to save zeitschritte',
            'message': str(e)
        }), 500


@training_api_bp.route('/generate-datasets/<session_id>', methods=['POST'])
def generate_datasets(session_id: str):
    """
    Generate datasets for training using the 7-phase pipeline
    
    Returns:
        JSON response with generation status
    """
    import threading
    
    try:
        logger.info(f"Starting dataset generation for session {session_id}")
        
        # Get request data for model parameters and training split
        request_data = request.get_json() or {}
        model_parameters = request_data.get('model_parameters', {})
        training_split = request_data.get('training_split', {})
        
        logger.info(f"Received model parameters: {model_parameters}")
        logger.info(f"Received training split: {training_split}")
        
        # Start the complete 7-phase pipeline in background thread
        def run_pipeline_async():
            try:
                logger.info(f"Running complete original pipeline for session {session_id}")
                
                # Import and run the complete pipeline with user parameters
                result = run_complete_original_pipeline(
                    session_id=session_id,
                    model_parameters=model_parameters,
                    training_split=training_split,
                    progress_callback=lambda session_id, phase, msg, progress: logger.info(f"Phase {phase}: {msg} ({progress}%)")
                )
                
                if result.get('success'):
                    logger.info(f"Pipeline completed successfully for session {session_id}")
                    
                    # Save results to database
                    try:
                        from .results_generator import ResultsGenerator
                        results_gen = ResultsGenerator()
                        
                        # Extract and save evaluation results if they exist
                        if 'final_results' in result and 'evaluation_results' in result['final_results']:
                            results_gen.results = result['final_results']['evaluation_results']
                            success = results_gen.save_results_to_database(session_id, get_supabase_client())
                            if success:
                                logger.info(f"Results saved to database for session {session_id}")
                            else:
                                logger.warning(f"Failed to save results to database for session {session_id}")
                        
                        # Also save violin plots if they exist
                        if 'final_results' in result and 'visualizations' in result['final_results']:
                            visualizations = result['final_results']['visualizations']
                            logger.info(f"Found {len(visualizations)} visualizations to save")
                            
                            # Save each visualization to database
                            supabase = get_supabase_client()
                            for plot_name, plot_data in visualizations.items():
                                try:
                                    # Convert session_id to UUID
                                    uuid_session_id = create_or_get_session_uuid(session_id)
                                    if uuid_session_id:
                                        viz_record = {
                                            'session_id': uuid_session_id,
                                            'plot_name': plot_name,
                                            'image_data': plot_data,  # Use 'image_data' instead of 'plot_data'
                                            'plot_type': 'violin_plot' if 'distribution' in plot_name else 'other'
                                        }
                                        response = supabase.table('training_visualizations').insert(viz_record).execute()
                                        logger.info(f"Saved visualization {plot_name} for session {session_id}")
                                except Exception as viz_error:
                                    logger.error(f"Error saving visualization {plot_name}: {str(viz_error)}")
                            
                    except Exception as save_error:
                        logger.error(f"Error saving results to database: {str(save_error)}")
                else:
                    logger.error(f"Pipeline failed for session {session_id}: {result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                logger.error(f"Error in async pipeline execution: {str(e)}")
        
        # Start pipeline in background thread
        pipeline_thread = threading.Thread(target=run_pipeline_async)
        pipeline_thread.daemon = True
        pipeline_thread.start()
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'message': 'Dataset generation and training pipeline started',
            'status': 'generating'
        }), 200
        
    except Exception as e:
        logger.error(f"Error starting pipeline for {session_id}: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Failed to start dataset generation',
            'message': str(e)
        }), 500


@training_api_bp.route('/train-models/<session_id>', methods=['POST'])
def train_models(session_id: str):
    """
    Start model training for a session
    
    Returns:
        JSON response with training status
    """
    try:
        # TODO: Implement actual model training logic
        # This is a placeholder implementation
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'message': 'Model training started',
            'status': 'training'
        }), 200
        
    except Exception as e:
        logger.error(f"Error starting model training for {session_id}: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Failed to start model training',
            'message': str(e)
        }), 500


@training_api_bp.route('/get-training-status/<session_id>', methods=['GET'])
def get_training_status_details(session_id: str):
    """
    Get detailed training status for a session
    
    Returns:
        JSON response with detailed training status
    """
    try:
        # Use existing status function as base
        return get_training_status(session_id)
        
    except Exception as e:
        logger.error(f"Error getting training status details for {session_id}: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Failed to get training status',
            'message': str(e)
        }), 500


@training_api_bp.route('/get-training-results/<session_id>', methods=['GET'])
def get_training_results_details(session_id: str):
    """
    Get detailed training results for a session
    
    Returns:
        JSON response with detailed training results
    """
    try:
        # Use existing results function as base
        return get_training_results(session_id)
        
    except Exception as e:
        logger.error(f"Error getting training results details for {session_id}: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Failed to get training results',
            'message': str(e)
        }), 500


# Helper functions (to be implemented with actual database queries)

def _get_results_from_database(session_id: str, supabase_client) -> Dict:
    """
    Get training results from database
    
    Args:
        session_id: Session identifier (string or UUID)
        supabase_client: Supabase client instance
        
    Returns:
        Dict containing training results
    """
    try:
        # Convert string session_id to UUID if needed
        try:
            import uuid
            uuid.UUID(session_id)
            uuid_session_id = session_id
        except (ValueError, TypeError):
            uuid_session_id = create_or_get_session_uuid(session_id)
            if not uuid_session_id:
                logger.warning(f"Could not get UUID for session {session_id}")
                return {}
        
        response = supabase_client.table('training_results').select('*').eq('session_id', uuid_session_id).order('created_at', desc=True).execute()
        
        if response.data and len(response.data) > 0:
            return response.data[0]
        else:
            return {}
            
    except Exception as e:
        logger.error(f"Error getting results from database: {str(e)}")
        raise


def _get_status_from_database(session_id: str, supabase_client) -> Dict:
    """
    Get training status from database
    
    Args:
        session_id: Session identifier (string or UUID)
        supabase_client: Supabase client instance
        
    Returns:
        Dict containing training status
    """
    try:
        # Convert string session_id to UUID if needed
        try:
            import uuid
            uuid.UUID(session_id)
            uuid_session_id = session_id
        except (ValueError, TypeError):
            uuid_session_id = create_or_get_session_uuid(session_id)
            if not uuid_session_id:
                logger.warning(f"Could not get UUID for session {session_id}")
                return {'status': 'not_found'}
        
        response = supabase_client.table('training_logs').select('*').eq('session_id', uuid_session_id).order('created_at', desc=True).limit(1).execute()
        
        if response.data:
            return response.data[0]
        else:
            return {'status': 'not_found'}
            
    except Exception as e:
        logger.error(f"Error getting status from database: {str(e)}")
        raise


def _get_progress_from_database(session_id: str, supabase_client) -> Dict:
    """
    Get training progress from database
    
    Args:
        session_id: Session identifier (string or UUID)
        supabase_client: Supabase client instance
        
    Returns:
        Dict containing training progress
    """
    try:
        # Convert string session_id to UUID if needed
        try:
            import uuid
            uuid.UUID(session_id)
            uuid_session_id = session_id
        except (ValueError, TypeError):
            uuid_session_id = create_or_get_session_uuid(session_id)
            if not uuid_session_id:
                logger.warning(f"Could not get UUID for session {session_id}")
                return {'progress': 0}
        
        response = supabase_client.table('training_logs').select('*').eq('session_id', uuid_session_id).order('created_at', desc=True).limit(1).execute()
        
        if response.data:
            return response.data[0]
        else:
            return {'progress': 0}
            
    except Exception as e:
        logger.error(f"Error getting progress from database: {str(e)}")
        raise


def _get_visualizations_from_database(session_id: str, supabase_client) -> Dict:
    """
    Get training visualizations from database
    
    Args:
        session_id: Session identifier (string or UUID)
        supabase_client: Supabase client instance
        
    Returns:
        Dict containing training visualizations
    """
    try:
        # Convert string session_id to UUID if needed
        try:
            import uuid
            uuid.UUID(session_id)
            uuid_session_id = session_id
        except (ValueError, TypeError):
            uuid_session_id = create_or_get_session_uuid(session_id)
            if not uuid_session_id:
                logger.warning(f"Could not get UUID for session {session_id}")
                return {'plots': {}}
        
        response = supabase_client.table('training_visualizations').select('*').eq('session_id', uuid_session_id).execute()
        
        if response.data and len(response.data) > 0:
            # Organize plots by plot_name
            plots = {}
            metadata = {}
            created_at = None
            
            for viz in response.data:
                plot_name = viz.get('plot_name', 'unknown')
                plots[plot_name] = viz.get('image_data', '')  # Use 'image_data' instead of 'plot_data_base64'
                
                if not metadata:
                    metadata = viz.get('plot_metadata', {})  # Use 'plot_metadata' from schema
                    created_at = viz.get('created_at')
            
            return {
                'plots': plots,
                'metadata': metadata,
                'created_at': created_at
            }
        else:
            return {'plots': {}}
            
    except Exception as e:
        logger.error(f"Error getting visualizations from database: {str(e)}")
        raise


def _get_metrics_from_database(session_id: str, supabase_client) -> Dict:
    """
    Get training metrics from database
    
    Args:
        session_id: Session identifier (string or UUID)
        supabase_client: Supabase client instance
        
    Returns:
        Dict containing training metrics
    """
    try:
        # Convert string session_id to UUID if needed
        try:
            import uuid
            uuid.UUID(session_id)
            uuid_session_id = session_id
        except (ValueError, TypeError):
            uuid_session_id = create_or_get_session_uuid(session_id)
            if not uuid_session_id:
                logger.warning(f"Could not get UUID for session {session_id}")
                return {}
        
        response = supabase_client.table('training_results').select('*').eq('session_id', uuid_session_id).execute()
        
        if response.data:
            return response.data[0].get('evaluation_metrics', {})
        else:
            return {}
            
    except Exception as e:
        logger.error(f"Error getting metrics from database: {str(e)}")
        raise


def _get_logs_from_database(session_id: str, supabase_client, limit: int = 100, level: str = 'INFO') -> Dict:
    """
    Get training logs from database
    
    Args:
        session_id: Session identifier (string or UUID)
        supabase_client: Supabase client instance
        limit: Maximum number of logs to return
        level: Log level filter
        
    Returns:
        Dict containing training logs
    """
    try:
        # Convert string session_id to UUID if needed
        try:
            import uuid
            uuid.UUID(session_id)
            uuid_session_id = session_id
        except (ValueError, TypeError):
            uuid_session_id = create_or_get_session_uuid(session_id)
            if not uuid_session_id:
                logger.warning(f"Could not get UUID for session {session_id}")
                return {'logs': [], 'total_logs': 0}
        
        response = supabase_client.table('training_logs').select('*').eq('session_id', uuid_session_id).eq('level', level).limit(limit).execute()
        
        return {
            'logs': response.data or [],
            'total_logs': len(response.data) if response.data else 0
        }
        
    except Exception as e:
        logger.error(f"Error getting logs from database: {str(e)}")
        raise


# Global storage for phase progress tracking
_phase_progress = {}


@training_api_bp.route('/start-complete-pipeline/<session_id>', methods=['POST'])
def start_complete_pipeline(session_id: str):
    """
    Start complete 7-phase training pipeline
    
    Args:
        session_id: Session identifier
        
    Returns:
        JSON response with pipeline start status
    """
    try:
        logger.info(f"Starting complete 7-phase pipeline for session {session_id}")
        
        # Get request data
        request_data = request.get_json() or {}
        model_parameters = request_data.get('model_parameters', {})
        
        # Initialize phase progress
        _phase_progress[session_id] = {
            'current_phase': 1,
            'phases': {
                1: {'name': 'Data Loading & Configuration', 'status': 'in_progress', 'progress': 0},
                2: {'name': 'Output Data Setup', 'status': 'pending', 'progress': 0},
                3: {'name': 'Dataset Creation - Time Features', 'status': 'pending', 'progress': 0},
                4: {'name': 'Data Preparation - Scaling & Splitting', 'status': 'pending', 'progress': 0},
                5: {'name': 'Model Training', 'status': 'pending', 'progress': 0},
                6: {'name': 'Model Testing - Predictions', 'status': 'pending', 'progress': 0},
                7: {'name': 'Re-scaling & Comprehensive Evaluation', 'status': 'pending', 'progress': 0}
            },
            'overall_progress': 0,
            'started_at': datetime.now().isoformat(),
            'status': 'running'
        }
        
        # Define progress callback
        def progress_callback(session_id, phase_num, phase_name, progress):
            if session_id in _phase_progress:
                _phase_progress[session_id]['current_phase'] = phase_num
                _phase_progress[session_id]['phases'][phase_num]['progress'] = progress
                _phase_progress[session_id]['phases'][phase_num]['status'] = 'completed' if progress >= 100 else 'in_progress'
                
                # Update overall progress
                total_progress = sum(p['progress'] for p in _phase_progress[session_id]['phases'].values())
                _phase_progress[session_id]['overall_progress'] = total_progress / 7
                
                logger.info(f"Phase {phase_num} ({phase_name}): {progress}%")
        
        # Start pipeline in background (in production, use Celery or similar)
        # For now, run synchronously
        try:
            results = run_complete_original_pipeline(
                session_id, 
                model_parameters, 
                progress_callback
            )
            
            # Update final status
            if session_id in _phase_progress:
                _phase_progress[session_id]['status'] = 'completed'
                _phase_progress[session_id]['completed_at'] = datetime.now().isoformat()
                _phase_progress[session_id]['results'] = results
            
            return jsonify({
                'success': True,
                'session_id': session_id,
                'status': 'pipeline_started',
                'message': 'Complete 7-phase pipeline started successfully',
                'phases_total': 7,
                'results': results
            })
            
        except Exception as pipeline_error:
            # Update error status
            if session_id in _phase_progress:
                _phase_progress[session_id]['status'] = 'failed'
                _phase_progress[session_id]['error'] = str(pipeline_error)
                _phase_progress[session_id]['failed_at'] = datetime.now().isoformat()
            
            return jsonify({
                'success': False,
                'session_id': session_id,
                'status': 'pipeline_failed',
                'error': str(pipeline_error)
            }), 500
        
    except Exception as e:
        logger.error(f"Error starting complete pipeline for session {session_id}: {str(e)}")
        return jsonify({
            'success': False,
            'session_id': session_id,
            'status': 'start_failed',
            'error': str(e)
        }), 500


@training_api_bp.route('/phase-status/<session_id>/<int:phase>', methods=['GET'])
def get_phase_status(session_id: str, phase: int):
    """
    Get status of a specific phase
    
    Args:
        session_id: Session identifier
        phase: Phase number (1-7)
        
    Returns:
        JSON response with phase status
    """
    try:
        if session_id not in _phase_progress:
            return jsonify({
                'session_id': session_id,
                'phase': phase,
                'status': 'not_found',
                'message': 'No pipeline found for this session'
            }), 404
        
        phase_data = _phase_progress[session_id]
        
        if phase < 1 or phase > 7:
            return jsonify({
                'session_id': session_id,
                'phase': phase,
                'status': 'invalid_phase',
                'message': 'Phase must be between 1 and 7'
            }), 400
        
        phase_info = phase_data['phases'].get(phase, {})
        
        return jsonify({
            'session_id': session_id,
            'phase': phase,
            'status': phase_info.get('status', 'unknown'),
            'progress': phase_info.get('progress', 0),
            'name': phase_info.get('name', f'Phase {phase}'),
            'current_phase': phase_data.get('current_phase', 1),
            'overall_progress': phase_data.get('overall_progress', 0),
            'pipeline_status': phase_data.get('status', 'unknown')
        })
        
    except Exception as e:
        logger.error(f"Error getting phase status for session {session_id}, phase {phase}: {str(e)}")
        return jsonify({
            'session_id': session_id,
            'phase': phase,
            'status': 'error',
            'error': str(e)
        }), 500


@training_api_bp.route('/comprehensive-evaluation/<session_id>', methods=['GET'])
def get_comprehensive_evaluation(session_id: str):
    """
    Get comprehensive evaluation results
    
    Args:
        session_id: Session identifier
        
    Returns:
        JSON response with comprehensive evaluation results
    """
    try:
        # Check if pipeline results are available
        if session_id not in _phase_progress:
            return jsonify({
                'session_id': session_id,
                'status': 'not_found',
                'message': 'No pipeline results found for this session'
            }), 404
        
        phase_data = _phase_progress[session_id]
        
        if phase_data.get('status') != 'completed':
            return jsonify({
                'session_id': session_id,
                'status': 'incomplete',
                'message': 'Pipeline not completed yet',
                'current_status': phase_data.get('status', 'unknown')
            }), 202  # Accepted but not ready
        
        # Get comprehensive results
        results = phase_data.get('results', {})
        final_results = results.get('final_results', {})
        
        # Extract evaluation metrics in the format expected by frontend
        evaluation_results = {
            'session_id': session_id,
            'status': 'completed',
            'evaluation_metrics': final_results.get('evaluation_results', {}).get('evaluation_metrics', {}),
            'model_comparison': final_results.get('evaluation_results', {}).get('model_comparison', {}),
            'training_metadata': {
                'total_models_trained': final_results.get('summary', {}).get('models_trained', 0),
                'datasets_processed': final_results.get('summary', {}).get('datasets_processed', 0),
                'total_phases': 7,
                'successful_phases': 7,
                'training_completed_at': phase_data.get('completed_at'),
                'training_duration': _calculate_duration(
                    phase_data.get('started_at'), 
                    phase_data.get('completed_at')
                )
            },
            'comprehensive_metrics': {
                # Include all metrics as shown in the image
                'mae': 'Mean Absolute Error',
                'rmse': 'Root Mean Squared Error', 
                'mse': 'Mean Squared Error',
                'mape': 'Mean Absolute Percentage Error',
                'mspe': 'Mean Squared Percentage Error',
                'wape': 'Weighted Absolute Percentage Error',
                'smape': 'Symmetric Mean Absolute Percentage Error',
                'mase': 'Mean Absolute Scaled Error'
            }
        }
        
        return jsonify(evaluation_results)
        
    except Exception as e:
        logger.error(f"Error getting comprehensive evaluation for session {session_id}: {str(e)}")
        return jsonify({
            'session_id': session_id,
            'status': 'error',
            'error': str(e)
        }), 500


@training_api_bp.route('/pipeline-overview/<session_id>', methods=['GET'])
def get_pipeline_overview(session_id: str):
    """
    Get complete overview of pipeline status
    
    Args:
        session_id: Session identifier
        
    Returns:
        JSON response with complete pipeline overview
    """
    try:
        if session_id not in _phase_progress:
            return jsonify({
                'session_id': session_id,
                'status': 'not_found',
                'message': 'No pipeline found for this session'
            }), 404
        
        phase_data = _phase_progress[session_id]
        
        return jsonify({
            'session_id': session_id,
            'status': phase_data.get('status', 'unknown'),
            'current_phase': phase_data.get('current_phase', 1),
            'overall_progress': phase_data.get('overall_progress', 0),
            'started_at': phase_data.get('started_at'),
            'completed_at': phase_data.get('completed_at'),
            'failed_at': phase_data.get('failed_at'),
            'error': phase_data.get('error'),
            'phases': phase_data.get('phases', {}),
            'total_phases': 7,
            'phases_summary': {
                'completed': len([p for p in phase_data.get('phases', {}).values() if p.get('status') == 'completed']),
                'in_progress': len([p for p in phase_data.get('phases', {}).values() if p.get('status') == 'in_progress']),
                'pending': len([p for p in phase_data.get('phases', {}).values() if p.get('status') == 'pending']),
                'failed': len([p for p in phase_data.get('phases', {}).values() if p.get('status') == 'failed'])
            }
        })
        
    except Exception as e:
        logger.error(f"Error getting pipeline overview for session {session_id}: {str(e)}")
        return jsonify({
            'session_id': session_id,
            'status': 'error',
            'error': str(e)
        }), 500


def _calculate_duration(start_time: str, end_time: str) -> Optional[str]:
    """
    Calculate duration between two ISO timestamp strings
    
    Args:
        start_time: Start time in ISO format
        end_time: End time in ISO format
        
    Returns:
        Duration string or None
    """
    try:
        if not start_time or not end_time:
            return None
        
        start = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
        end = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
        
        duration = end - start
        
        # Format duration nicely
        total_seconds = int(duration.total_seconds())
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        if hours > 0:
            return f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"
        
    except Exception as e:
        logger.error(f"Error calculating duration: {str(e)}")
        return None