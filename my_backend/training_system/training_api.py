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