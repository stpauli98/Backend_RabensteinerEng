"""
Training API module for training system
Provides API endpoints for training results and status
"""

from flask import Blueprint, jsonify, request
from typing import Dict, Optional
import logging
import sys
import os

# Import existing supabase client
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from supabase_client import get_supabase_client

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
        
        response_data = {
            'session_id': session_id,
            'status': results.get('status', 'unknown'),
            'evaluation_metrics': results.get('evaluation_metrics', {}),
            'model_performance': results.get('model_performance', {}),
            'best_model': results.get('best_model', {}),
            'summary': results.get('summary', {}),
            'created_at': results.get('created_at'),
            'completed_at': results.get('completed_at'),
            'message': 'Training results retrieved successfully'
        }
        
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
    Generate datasets for training
    
    Returns:
        JSON response with generation status
    """
    try:
        # TODO: Implement actual dataset generation logic
        # This is a placeholder implementation
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'message': 'Dataset generation started',
            'status': 'generating'
        }), 200
        
    except Exception as e:
        logger.error(f"Error generating datasets for {session_id}: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Failed to generate datasets',
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
        session_id: Session identifier
        supabase_client: Supabase client instance
        
    Returns:
        Dict containing training results
    """
    try:
        response = supabase_client.table('training_results').select('*').eq('session_id', session_id).order('created_at', desc=True).execute()
        
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
        session_id: Session identifier
        supabase_client: Supabase client instance
        
    Returns:
        Dict containing training status
    """
    try:
        response = supabase_client.table('training_logs').select('*').eq('session_id', session_id).order('created_at', desc=True).limit(1).execute()
        
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
        session_id: Session identifier
        supabase_client: Supabase client instance
        
    Returns:
        Dict containing training progress
    """
    try:
        response = supabase_client.table('training_logs').select('*').eq('session_id', session_id).order('created_at', desc=True).limit(1).execute()
        
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
        session_id: Session identifier
        supabase_client: Supabase client instance
        
    Returns:
        Dict containing training visualizations
    """
    try:
        response = supabase_client.table('training_visualizations').select('*').eq('session_id', session_id).execute()
        
        if response.data and len(response.data) > 0:
            # Organize plots by plot_name
            plots = {}
            metadata = {}
            created_at = None
            
            for viz in response.data:
                plot_name = viz.get('plot_name', 'unknown')
                plots[plot_name] = viz.get('plot_data_base64', '')
                
                if not metadata:
                    metadata = viz.get('metadata', {})
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
        session_id: Session identifier
        supabase_client: Supabase client instance
        
    Returns:
        Dict containing training metrics
    """
    try:
        response = supabase_client.table('training_results').select('*').eq('session_id', session_id).execute()
        
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
        session_id: Session identifier
        supabase_client: Supabase client instance
        limit: Maximum number of logs to return
        level: Log level filter
        
    Returns:
        Dict containing training logs
    """
    try:
        response = supabase_client.table('training_logs').select('*').eq('session_id', session_id).eq('level', level).limit(limit).execute()
        
        return {
            'logs': response.data or [],
            'total_logs': len(response.data) if response.data else 0
        }
        
    except Exception as e:
        logger.error(f"Error getting logs from database: {str(e)}")
        raise