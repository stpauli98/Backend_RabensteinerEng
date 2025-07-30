"""
Training API module for training system
Provides API endpoints for training results and status
"""

from flask import Blueprint, jsonify, request, send_file
from typing import Dict, Optional
import logging
import sys
import os
import io
import pickle
import tempfile
import zipfile
import threading
from datetime import datetime

# Import comprehensive error handling
from .error_handler import (
    get_error_handler, ErrorCategory, ErrorSeverity,
    TrainingSystemError, DatabaseError,
    handle_database_error
)

# Import existing supabase client
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from supabase_client import get_supabase_client, create_or_get_session_uuid

# Import the new pipeline function
from .pipeline_integration import run_complete_original_pipeline

# Import parameter conversion utilities
from .utils import convert_frontend_to_backend_params, sanitize_for_json, validate_frontend_model_parameters, validate_training_split_parameters, create_user_friendly_error_response, emit_training_progress, emit_training_metrics, emit_training_error, calculate_training_eta

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
    error_handler = get_error_handler()
    
    try:
        with error_handler.error_context("get_training_results", session_id):
            supabase = get_supabase_client()
            results = _get_results_from_database(session_id, supabase)
        
            if not results:
                return jsonify({
                    'session_id': session_id,
                    'status': 'not_found',
                    'message': 'No training results found for this session'
                }), 404
        
            # Also get visualizations from training_visualizations table
            try:
                visualizations = _get_visualizations_from_database(session_id, supabase)
                logger.info(f"Retrieved visualizations for {session_id}: {len(visualizations.get('plots', {}))} plots found")
            except Exception as viz_error:
                # Don't fail the entire request if visualizations can't be loaded
                error_details = handle_database_error(
                    viz_error, session_id,
                    operation="get_visualizations"
                )
                logger.warning(f"Failed to load visualizations: {error_details['error_code']}")
                visualizations = {'plots': {}}
        
            response_data = {
                'session_id': session_id,
                'status': results.get('status', 'unknown'),
                'evaluation_metrics': results.get('evaluation_metrics', {}),
                'model_performance': results.get('model_performance', {}),
                'best_model': results.get('best_model', {}),
                'summary': results.get('summary', {}),
                'visualizations': visualizations.get('plots', {}),  # Include violin plots
                'n_dat': results.get('n_dat', 0),  # Number of generated datasets
                'created_at': results.get('created_at'),
                'completed_at': results.get('completed_at'),
                'message': 'Training results retrieved successfully'
            }
            logger.info(f"API response includes visualizations: {'visualizations' in response_data}")
            
            return jsonify(response_data), 200
        
    except TrainingSystemError as ts_error:
        error_response = error_handler.handle_error(ts_error, session_id)
        return jsonify({
            'error': 'Failed to retrieve training results',
            'error_code': ts_error.error_code,
            'message': ts_error.message,
            'session_id': session_id,
            'recovery_suggestions': error_response.get('recovery_suggestions', [])
        }), 500
    except Exception as e:
        error_details = handle_database_error(e, session_id, operation="get_training_results")
        return jsonify({
            'error': 'Failed to retrieve training results',
            'error_code': error_details['error_details']['error_code'],
            'message': str(e),
            'session_id': session_id,
            'recovery_suggestions': error_details.get('recovery_suggestions', [])
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
        # Supabase client imported from parent directory
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
        # Training cancellation can be implemented via progress manager
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
    Generate datasets and violin plots - first step of the restructured workflow
    
    Expects JSON body with optional MTS parameters:
    {
        "mts_params": {
            "time_features": {"jahr": true, "monat": true, "woche": true, "feiertag": true, "zeitzone": "Europe/Vienna"},
            "zeitschritte": {"eingabe": 24, "ausgabe": 1, "zeitschrittweite": 1, "offset": 0},
            "preprocessing": {"interpolation": true, "outlier_removal": true, "scaling": true}
        }
    }
    
    Returns:
        JSON response with dataset generation status and violin plots
    """
    import threading
    
    try:
        logger.info(f"Starting dataset generation for session {session_id}")
        
        # Get request data for all parameters
        request_data = request.get_json() or {}
        mts_params = request_data.get('mts_parameters', {})  # Changed from 'mts_params'
        raw_model_params = request_data.get('model_parameters', {})  # NEW: Accept model parameters
        split_params = request_data.get('training_split', {})    # NEW: Accept training split
        
        logger.info(f"Received MTS parameters: {mts_params}")
        logger.info(f"Received raw model parameters: {list(raw_model_params.keys()) if raw_model_params else 'None'}")
        logger.info(f"Received split parameters: {split_params}")
        
        # 🔍 Validate parameters if provided (optional for dataset generation)
        validation_warnings = []
        validation_suggestions = []
        
        if raw_model_params:
            validation_result = validate_frontend_model_parameters(raw_model_params)
            
            # For dataset generation, parameter validation is non-blocking but informative
            if not validation_result['valid']:
                logger.warning(f"Model parameter validation failed: {validation_result['errors']}")
                validation_warnings.extend(validation_result['errors'])
            else:
                validation_warnings.extend(validation_result['warnings'])
                validation_suggestions.extend(validation_result['suggestions'])
        
        if split_params:
            split_validation = validate_training_split_parameters(split_params)
            
            if not split_validation['valid']:
                logger.warning(f"Split parameter validation failed: {split_validation['errors']}")
                validation_warnings.extend(split_validation['errors'])
            else:
                validation_warnings.extend(split_validation['warnings'])
                validation_suggestions.extend(split_validation['suggestions'])
        
        # Convert frontend flat structure to backend nested structure if model params provided
        model_params = {}
        if raw_model_params:
            try:
                model_params = convert_frontend_to_backend_params(raw_model_params)
                logger.info(f"Converted model parameters: {list(model_params.keys()) if model_params else 'None'}")
            except Exception as e:
                logger.error(f"Failed to convert frontend parameters: {str(e)}")
                # For dataset generation, parameter conversion failure is not critical
                logger.warning("Continuing dataset generation without model parameters")
                validation_warnings.append(f"Parameter conversion failed: {str(e)}")
                model_params = {}
        
        # Import the new dataset generation pipeline
        from .pipeline_integration import run_dataset_generation_pipeline
        
        # Start dataset generation in background thread
        def run_dataset_generation_async():
            try:
                logger.info(f"Running dataset generation pipeline for session {session_id}")
                
                # Use the new dataset generation pipeline
                result = run_dataset_generation_pipeline(
                    session_id=session_id,
                    supabase_client=get_supabase_client(),
                    socketio_instance=None,  # SocketIO will be added when Flask app is available
                    mts_params=mts_params if mts_params else None,
                    model_params=model_params if model_params else None,  # NEW: Pass model parameters
                    split_params=split_params if split_params else None   # NEW: Pass split parameters
                )
                
                # 🛑 CHECK FOR DATA VALIDATION ERRORS - STOP PROCESSING IF ERROR DETECTED
                if isinstance(result, dict) and result.get('status') == 'error':
                    logger.warning(f"Dataset generation stopped due to data validation error: {result.get('error_message')}")
                    logger.info(f"Error type: {result.get('error_type')} - Processing stopped, no results saved to database")
                    
                    # 🔔 NOTIFY FRONTEND VIA SOCKETIO ABOUT DATA INCOMPATIBILITY
                    try:
                        from flask import current_app
                        if hasattr(current_app, 'extensions') and 'socketio' in current_app.extensions:
                            socketio_instance = current_app.extensions['socketio']
                            room = f"training_{session_id}"
                            
                            error_notification = {
                                'session_id': session_id,
                                'status': 'data_validation_error',
                                'error_type': result.get('error_type'),
                                'error_message': result.get('error_message'),
                                'error_details': result.get('error_details', {}),
                                'timestamp': datetime.now().isoformat(),
                                'processing_stopped': True
                            }
                            
                            socketio_instance.emit('dataset_generation_error', error_notification, room=room)
                            logger.info(f"Sent data validation error notification to frontend via SocketIO room: {room}")
                        else:
                            logger.info("SocketIO not available, error notification not sent")
                    except Exception as socketio_error:
                        logger.error(f"Failed to send SocketIO error notification: {str(socketio_error)}")
                    
                    # 💾 SAVE ERROR STATUS TO DATABASE FOR FRONTEND STATUS POLLING
                    try:
                        supabase = get_supabase_client()
                        uuid_session_id = create_or_get_session_uuid(session_id)
                        if uuid_session_id:
                            error_status = {
                                'session_id': uuid_session_id,
                                'status': 'data_validation_error',
                                'error_message': result.get('error_message'),
                                'error_details': result.get('error_details', {}),
                                'completed_at': datetime.now().isoformat(),
                                'summary': {
                                    'error_type': result.get('error_type'),
                                    'processing_stopped': True,
                                    'input_length': result.get('error_details', {}).get('input_length'),
                                    'output_length': result.get('error_details', {}).get('output_length'),
                                    'difference': result.get('error_details', {}).get('difference')
                                }
                            }
                            response = supabase.table('training_results').insert(error_status).execute()
                            logger.info(f"Saved data validation error status to database for frontend polling")
                    except Exception as db_error:
                        logger.error(f"Failed to save error status to database: {str(db_error)}")
                    
                    # 🛑 STOP PROCESSING - Error status saved, processing terminated
                    return
                
                logger.info(f"Dataset generation completed successfully for session {session_id}")
                logger.info(f"Generated {result.get('dataset_count', 0)} datasets and {len(result.get('visualizations', {}))} visualizations")
                
                # Save violin plots to database
                try:
                    visualizations = result.get('visualizations', {})
                    if visualizations:
                        logger.info(f"Saving {len(visualizations)} visualizations to database")
                        
                        supabase = get_supabase_client()
                        for plot_name, plot_data in visualizations.items():
                            try:
                                # Convert session_id to UUID
                                uuid_session_id = create_or_get_session_uuid(session_id)
                                if uuid_session_id:
                                    viz_record = {
                                        'session_id': uuid_session_id,
                                        'plot_name': plot_name,
                                        'image_data': plot_data,
                                        'plot_type': 'violin_plot' if 'distribution' in plot_name else 'data_visualization',
                                        'plot_metadata': {
                                            'generated_by': 'dataset_generation_pipeline',
                                            'mts_params_applied': bool(mts_params),
                                            'dataset_count': result.get('dataset_count', 0)
                                        }
                                    }
                                    response = supabase.table('training_visualizations').insert(viz_record).execute()
                                    logger.info(f"Saved visualization {plot_name} for session {session_id}")
                            except Exception as viz_error:
                                logger.error(f"Error saving visualization {plot_name}: {str(viz_error)}")
                    
                    # Save dataset generation status to database
                    uuid_session_id = create_or_get_session_uuid(session_id)
                    if uuid_session_id:
                        dataset_status = {
                            'session_id': uuid_session_id,
                            'status': 'datasets_generated',
                            'dataset_count': result.get('dataset_count', 0),
                            # 'datasets_info': result.get('datasets_info', {}),  # Temporary: Remove until column exists
                            'mts_configuration': result.get('mts_configuration'),
                            'processing_summary': result.get('processing_summary', {}),
                            'completed_at': result.get('timestamp')
                        }
                        response = supabase.table('training_results').insert(dataset_status).execute()
                        logger.info(f"Saved dataset generation status for session {session_id}")
                        
                except Exception as save_error:
                    logger.error(f"Error saving dataset generation results: {str(save_error)}")
                    
            except Exception as e:
                logger.error(f"Error in async dataset generation: {str(e)}")
                # Save error status to database
                try:
                    supabase = get_supabase_client()
                    uuid_session_id = create_or_get_session_uuid(session_id)
                    if uuid_session_id:
                        error_status = {
                            'session_id': uuid_session_id,
                            'status': 'dataset_generation_failed',
                            'error_message': str(e),
                            'completed_at': datetime.now().isoformat()
                        }
                        supabase.table('training_results').insert(error_status).execute()
                except Exception:
                    pass  # Don't fail if we can't save error status
        
        # Start dataset generation in background thread
        generation_thread = threading.Thread(target=run_dataset_generation_async)
        generation_thread.daemon = True
        generation_thread.start()
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'message': 'Dataset generation started - violin plots will be generated',
            'validation_info': {
                'parameters_validated': bool(raw_model_params or split_params),
                'validation_warnings': validation_warnings,
                'validation_suggestions': validation_suggestions,
                'parameter_quality': 'good' if not validation_warnings else 'needs_attention'
            },
            'status': 'generating_datasets',
            'mts_params_provided': bool(mts_params),
            'workflow_step': 1,
            'next_step': 'Call /train-models endpoint with model parameters'
        }), 200
        
    except TrainingSystemError as ts_error:
        error_handler = get_error_handler()
        error_response = error_handler.handle_error(ts_error, session_id)
        return jsonify({
            'success': False,
            'error': 'Failed to start dataset generation',
            'error_code': ts_error.error_code,
            'message': ts_error.message,
            'session_id': session_id,
            'recovery_suggestions': error_response.get('recovery_suggestions', []),
            'workflow_step': 1
        }), 500
    except Exception as e:
        error_handler = get_error_handler()
        error_details = error_handler.handle_error(e, session_id, operation="start_dataset_generation")
        return jsonify({
            'success': False,
            'error': 'Failed to start dataset generation',
            'error_code': error_details['error_details']['error_code'],
            'message': str(e),
            'session_id': session_id,
            'recovery_suggestions': error_details.get('recovery_suggestions', []),
            'workflow_step': 1
        }), 500


@training_api_bp.route('/train-models/<session_id>', methods=['POST'])
def train_models(session_id: str):
    """
    Start model training for a session with user parameters - second step of restructured workflow
    
    Expects JSON body with:
    {
        "model_params": {
            "dense": {"layers": [64, 32], "epochs": 100, "batch_size": 32, "activation": "relu", ...},
            "cnn": {"filters": [32, 64], "kernel_size": [3, 3], "epochs": 100, ...},
            "lstm": {"units": [50, 50], "dropout": 0.2, "epochs": 100, ...},
            "svr": {"kernel": "rbf", "C": 1.0, "gamma": "scale", "epsilon": 0.1, ...},
            "linear": {"fit_intercept": true, "normalize": false, ...}
        },
        "split_params": {"train_ratio": 0.7, "validation_ratio": 0.15, "test_ratio": 0.15},
        "mts_params": {"time_features": {...}, "zeitschritte": {...}, "preprocessing": {...}}
    }
    
    Returns:
        JSON response with training status and results
    """
    import threading
    
    try:
        logger.info(f"Starting model training for session {session_id}")
        
        # Get request data (fixed parameter key names to match frontend)
        request_data = request.get_json() or {}
        raw_model_params = request_data.get('model_parameters', {})  # Changed from 'model_params'
        split_params = request_data.get('training_split', {})    # Changed from 'split_params'
        mts_params = request_data.get('mts_parameters', {})      # Changed from 'mts_params'
        
        logger.info(f"Received raw model parameters: {list(raw_model_params.keys()) if raw_model_params else 'None'}")
        logger.info(f"Received split parameters: {split_params}")
        logger.info(f"Received MTS parameters: {bool(mts_params)}")
        
        # 🔍 STEP 1: Validate model parameters
        if raw_model_params:
            validation_result = validate_frontend_model_parameters(raw_model_params)
            
            if not validation_result['valid']:
                logger.warning(f"Model parameter validation failed: {validation_result['errors']}")
                error_response = create_user_friendly_error_response(
                    'parameter_validation', 
                    f"Parameter validation errors: {', '.join(validation_result['errors'])}", 
                    session_id
                )
                error_response['validation_details'] = validation_result
                return jsonify(error_response), 400
            
            # Log warnings and suggestions
            if validation_result['warnings']:
                logger.warning(f"Model parameter warnings: {validation_result['warnings']}")
            if validation_result['suggestions']:
                logger.info(f"Model parameter suggestions: {validation_result['suggestions']}")
        
        # 🔍 STEP 2: Validate split parameters
        if split_params:
            split_validation = validate_training_split_parameters(split_params)
            
            if not split_validation['valid']:
                logger.warning(f"Split parameter validation failed: {split_validation['errors']}")
                error_response = create_user_friendly_error_response(
                    'parameter_validation', 
                    f"Split validation errors: {', '.join(split_validation['errors'])}", 
                    session_id
                )
                error_response['validation_details'] = split_validation
                return jsonify(error_response), 400
            
            # Log warnings and suggestions
            if split_validation['warnings']:
                logger.warning(f"Split parameter warnings: {split_validation['warnings']}")
        
        # 🔄 STEP 3: Convert frontend flat structure to backend nested structure
        model_params = {}
        if raw_model_params:
            try:
                model_params = convert_frontend_to_backend_params(raw_model_params)
                logger.info(f"Converted model parameters: {list(model_params.keys()) if model_params else 'None'}")
            except Exception as e:
                logger.error(f"Failed to convert frontend parameters: {str(e)}")
                return jsonify({
                    'success': False,
                    'error': 'Parameter conversion failed',
                    'message': f'Could not convert frontend parameters: {str(e)}',
                    'workflow_step': 2
                }), 400
        
        # Validate required parameters
        if not model_params:
            return jsonify({
                'success': False,
                'error': 'Model parameters are required',
                'message': 'Please provide model configuration parameters (dense, cnn, lstm, svr, linear)',
                'workflow_step': 2,
                'required_format': {
                    'model_params': {'dense': {}, 'cnn': {}, 'lstm': {}, 'svr': {}, 'linear': {}},
                    'split_params': {'train_ratio': 0.7, 'validation_ratio': 0.15, 'test_ratio': 0.15}
                }
            }), 400
        
        # Import the new model training pipeline
        from .pipeline_integration import run_model_training_pipeline
        
        # Start model training in background thread
        def run_model_training_async():
            import time
            training_start_time = time.time()
            
            try:
                logger.info(f"Running model training pipeline for session {session_id}")
                
                # 📡 Emit training start progress
                emit_training_progress(session_id, {
                    'status': 'training_started',
                    'message': 'Model training has begun',
                    'progress_percent': 0.0,
                    'phase': 'initialization',
                    'models_to_train': list(model_params.keys()) if model_params else [],
                    'estimated_duration': 'Calculating...'
                })
                
                # Prepare parameters for the new pipeline
                training_params = {
                    'model_params': model_params,
                    'split_params': split_params,
                    'mts_params': mts_params
                }
                
                # 📡 Emit data preparation progress
                emit_training_progress(session_id, {
                    'status': 'preparing_data',
                    'message': 'Preparing training data and splitting datasets',
                    'progress_percent': 15.0,
                    'phase': 'data_preparation'
                })
                
                # Use the new model training pipeline with parameter conversion
                result = run_model_training_pipeline(
                    session_id=session_id,
                    model_params=training_params,
                    supabase_client=get_supabase_client(),
                    socketio_instance=None  # SocketIO will be added when Flask app is available
                )
                
                # Calculate training duration
                training_duration = time.time() - training_start_time
                
                logger.info(f"Model training completed successfully for session {session_id}")
                logger.info(f"Trained {result.get('summary', {}).get('models_trained', 0)} models")
                logger.info(f"Training duration: {training_duration:.2f} seconds")
                
                # 📡 Emit training completion progress
                emit_training_progress(session_id, {
                    'status': 'training_completed',
                    'message': f"Successfully trained {result.get('summary', {}).get('models_trained', 0)} models",
                    'progress_percent': 100.0,
                    'phase': 'completed',
                    'training_duration': training_duration,
                    'models_trained': result.get('summary', {}).get('models_trained', 0),
                    'best_model': result.get('summary', {}).get('best_model', {})
                })
                
                # Save training results to database
                try:
                    supabase = get_supabase_client()
                    uuid_session_id = create_or_get_session_uuid(session_id)
                    
                    if uuid_session_id:
                        # Save comprehensive training results (sanitized for JSON)
                        # Map to existing database schema columns
                        raw_training_result_data = {
                            'session_id': uuid_session_id,
                            'status': 'training_completed',
                            'evaluation_metrics': result.get('evaluation_results', {}).get('evaluation_metrics', {}),
                            'model_performance': result.get('training_results', {}),
                            'best_model': result.get('summary', {}).get('best_model', {}),
                            'summary': result.get('summary', {}),
                            'training_metadata': {
                                'ui_parameters': result.get('ui_parameters', {}),
                                'converted_parameters': result.get('converted_parameters', {}),
                                'validation_warnings': result.get('validation_warnings', []),
                                'training_type': 'single_model_pipeline',  # Indicate this is from Phase 3.2
                                'framework_version': 'modular_system_v1'
                            },
                            'completed_at': result.get('timestamp')
                        }
                        
                        # Sanitize data to prevent JSON serialization errors
                        training_result_data = sanitize_for_json(raw_training_result_data)
                        logger.info(f"Sanitized training results for JSON serialization")
                        
                        response = supabase.table('training_results').upsert(training_result_data).execute()
                        logger.info(f"Saved training results for session {session_id}")
                        
                        # Save training visualizations
                        visualizations = result.get('visualizations', {})
                        if visualizations:
                            logger.info(f"Saving {len(visualizations)} training visualizations to database")
                            
                            for plot_name, plot_data in visualizations.items():
                                try:
                                    raw_viz_record = {
                                        'session_id': uuid_session_id,
                                        'plot_name': plot_name,
                                        'image_data': plot_data,
                                        'plot_type': 'training_result' if 'forecast' in plot_name or 'comparison' in plot_name else 'model_visualization',
                                        'plot_metadata': {
                                            'generated_by': 'model_training_pipeline',
                                            'models_trained': result.get('summary', {}).get('models_trained', 0),
                                            'training_completed': True
                                        }
                                    }
                                    
                                    # Sanitize visualization data for JSON serialization
                                    viz_record = sanitize_for_json(raw_viz_record)
                                    response = supabase.table('training_visualizations').upsert(viz_record).execute()
                                    logger.info(f"Saved training visualization {plot_name} for session {session_id}")
                                except Exception as viz_error:
                                    logger.error(f"Error saving training visualization {plot_name}: {str(viz_error)}")
                    
                except Exception as save_error:
                    logger.error(f"Error saving training results to database: {str(save_error)}")
                    
            except Exception as e:
                logger.error(f"Error in async model training: {str(e)}")
                
                # 📡 Emit training error progress
                emit_training_error(session_id, {
                    'message': f'Training failed: {str(e)}',
                    'error_details': str(e),
                    'phase': 'training_execution',
                    'progress_percent': 50.0  # Approximate progress when error occurred
                })
                
                # Create user-friendly error response
                error_type = 'training_failed'
                error_details = str(e)
                
                # Determine specific error type based on error message
                if 'insufficient data' in error_details.lower() or 'not enough' in error_details.lower():
                    error_type = 'insufficient_data'
                elif 'database' in error_details.lower() or 'supabase' in error_details.lower():
                    error_type = 'database_error'
                elif 'processing' in error_details.lower() or 'dataset' in error_details.lower():
                    error_type = 'data_processing'
                
                # Save enhanced error status to database
                try:
                    supabase = get_supabase_client()
                    uuid_session_id = create_or_get_session_uuid(session_id)
                    if uuid_session_id:
                        # Create comprehensive error record with user-friendly details
                        user_friendly_error = create_user_friendly_error_response(error_type, error_details, session_id)
                        
                        error_status = {
                            'session_id': uuid_session_id,
                            'status': 'training_failed',
                            'error_message': str(e),
                            'error_type': error_type,
                            'user_friendly_message': user_friendly_error['user_message'],
                            'recovery_suggestions': user_friendly_error['recovery_suggestions'],
                            'model_params_provided': list(model_params.keys()) if model_params else [],
                            'raw_parameters': raw_model_params,
                            'split_parameters': split_params,
                            'completed_at': datetime.now().isoformat()
                        }
                        
                        # Sanitize error status for JSON serialization
                        sanitized_error_status = sanitize_for_json(error_status)
                        supabase.table('training_results').upsert(sanitized_error_status).execute()
                        logger.info(f"Saved enhanced error status for session {session_id}")
                except Exception as db_error:
                    logger.error(f"Failed to save error status: {str(db_error)}")
                    pass  # Don't fail if we can't save error status
        
        # Start model training in background thread
        training_thread = threading.Thread(target=run_model_training_async)
        training_thread.daemon = True
        training_thread.start()
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'message': 'Model training started with user parameters',
            'status': 'training_models',
            'workflow_step': 2,
            'parameters_summary': {
                'models_selected': list(model_params.keys()) if model_params else [],
                'models_count': len(model_params) if model_params else 0,
                'split_provided': bool(split_params),
                'mts_params_provided': bool(mts_params),
                'parameter_validation': 'Will be validated during training'
            },
            'next_step': 'Monitor training progress via /status endpoint'
        }), 200
        
    except TrainingSystemError as ts_error:
        error_handler = get_error_handler()
        error_response = error_handler.handle_error(ts_error, session_id)
        return jsonify({
            'success': False,
            'error': 'Failed to start model training',
            'error_code': ts_error.error_code,
            'message': ts_error.message,
            'session_id': session_id,
            'recovery_suggestions': error_response.get('recovery_suggestions', []),
            'workflow_step': 2,
            'parameters_provided': {
                'models_selected': list(model_params.keys()) if model_params else [],
                'split_provided': bool(split_params),
                'mts_params_provided': bool(mts_params)
            }
        }), 500
    except Exception as e:
        error_handler = get_error_handler()
        error_details = error_handler.handle_error(e, session_id, operation="start_model_training")
        return jsonify({
            'success': False,
            'error': 'Failed to start model training',
            'error_code': error_details['error_details']['error_code'],
            'message': str(e),
            'session_id': session_id,
            'recovery_suggestions': error_details.get('recovery_suggestions', []),
            'workflow_step': 2
        }), 500


@training_api_bp.route('/save-model/<session_id>', methods=['POST'])
def save_model(session_id: str):
    """
    Save trained model for download
    
    Args:
        session_id: Session identifier
        
    Returns:
        JSON response with model save status and download information
    """
    error_handler = get_error_handler()
    
    try:
        logger.info(f"Starting model save for session {session_id}")
        
        # Get request data for model selection
        request_data = request.get_json() or {}
        model_name = request_data.get('model_name', 'best_model')  # Allow specific model selection
        
        logger.info(f"Saving model: {model_name} for session {session_id}")
        
        # Get session UUID
        supabase = get_supabase_client()
        uuid_session_id = create_or_get_session_uuid(session_id)
        
        if not uuid_session_id:
            return jsonify({
                'success': False,
                'error': 'Session not found',
                'message': f'No session found with ID: {session_id}'
            }), 404
        
        # Get training results from database to check if models exist
        results_response = supabase.table('training_results').select('*').eq('session_id', uuid_session_id).order('created_at', desc=True).limit(1).execute()
        
        if not results_response.data:
            return jsonify({
                'success': False,
                'error': 'No training results found',
                'message': f'No trained models found for session {session_id}. Please train a model first.'
            }), 404
        
        training_result = results_response.data[0]
        
        # Check if training was completed successfully
        if training_result.get('status') != 'training_completed':
            return jsonify({
                'success': False,
                'error': 'Training not completed',
                'message': f'Training has not completed successfully for session {session_id}. Current status: {training_result.get("status", "unknown")}'
            }), 400
        
        # For now, return model metadata since actual model files would be large
        # In a full implementation, this would create downloadable model files
        model_info = {
            'session_id': session_id,
            'model_name': model_name,
            'training_completed_at': training_result.get('completed_at'),
            'model_performance': training_result.get('model_performance', {}),
            'best_model': training_result.get('best_model', {}),
            'evaluation_metrics': training_result.get('evaluation_metrics', {}),
            'model_summary': training_result.get('summary', {}),
            'parameters_used': training_result.get('training_metadata', {}).get('converted_parameters', {}),
            'save_timestamp': datetime.now().isoformat(),
            'model_format': 'tensorflow_savedmodel',  # Indicate the format
            'download_instructions': 'Model metadata saved. In production, this would provide download links for model files.'
        }
        
        logger.info(f"Model save completed for session {session_id}")
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'message': f'Model {model_name} information saved successfully',
            'model_info': model_info,
            'saved_at': datetime.now().isoformat(),
            'next_steps': [
                'Model metadata has been prepared',
                'In production, downloadable model files would be available',
                'Use the model_info for model deployment or analysis'
            ]
        }), 200
        
    except Exception as e:
        error_handler = get_error_handler()
        error_details = error_handler.handle_error(e, session_id, operation="save_model")
        logger.error(f"Error saving model for {session_id}: {str(e)}")
        
        return jsonify({
            'success': False,
            'error': 'Failed to save model',
            'error_code': error_details['error_details']['error_code'],
            'message': str(e),
            'session_id': session_id,
            'recovery_suggestions': [
                'Ensure the session has completed training',
                'Check that training results are available',
                'Try again with a different model name'
            ]
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
        training_split = request_data.get('training_split', {})
        
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
                training_split,
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


@training_api_bp.route('/download-model/<session_id>', methods=['GET'])
def download_model(session_id: str):
    """
    Download trained model for a session as ZIP file
    
    Args:
        session_id: Session identifier
        
    Returns:
        ZIP file containing the trained model and metadata
    """
    error_handler = get_error_handler()
    
    try:
        with error_handler.error_context("download_model", session_id):
            logger.info(f"Starting model download for session {session_id}")
            
            # Get UUID session ID
            supabase = get_supabase_client()
            uuid_session_id = create_or_get_session_uuid(session_id)
            
            # Get training results from database
            logger.info(f"Fetching training results for UUID session {uuid_session_id}")
            results = supabase.table('training_results')\
                .select('*')\
                .eq('session_id', uuid_session_id)\
                .order('created_at', desc=True)\
                .limit(1)\
                .execute()
            
            if not results.data:
                return jsonify({
                    'success': False,
                    'error': 'No training results found for this session'
                }), 404
            
            training_result = results.data[0]
            logger.info(f"Found training results: {list(training_result.keys())}")
            
            # Create temporary ZIP file
            temp_dir = tempfile.mkdtemp()
            zip_path = os.path.join(temp_dir, f"model_{session_id}.zip")
            
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # Add training metadata as JSON
                import json
                metadata = {
                    'session_id': session_id,
                    'model_type': training_result.get('model_type', 'unknown'),
                    'training_duration': training_result.get('training_duration'),
                    'training_parameters': training_result.get('training_metadata', {}).get('ui_parameters', {}),
                    'model_config': training_result.get('training_metadata', {}).get('converted_parameters', {}),
                    'training_metrics': training_result.get('training_metrics', {}),
                    'evaluation_metrics': training_result.get('evaluation_metrics', {}),
                    'created_at': training_result.get('created_at'),
                    'status': training_result.get('status'),
                    'download_timestamp': datetime.now().isoformat()
                }
                
                zipf.writestr('model_metadata.json', json.dumps(metadata, indent=2))
                logger.info("Added metadata to ZIP")
                
                # Add training results summary
                summary = {
                    'session_info': {
                        'session_id': session_id,
                        'uuid_session_id': uuid_session_id,
                        'download_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    },
                    'model_performance': training_result.get('training_metrics', {}),
                    'training_config': training_result.get('training_metadata', {}),
                    'model_architecture': training_result.get('model_architecture', {}),
                    'data_info': training_result.get('processed_data_info', {})
                }
                
                zipf.writestr('training_summary.json', json.dumps(summary, indent=2))
                logger.info("Added training summary to ZIP")
                
                # Add model architecture info if available
                if 'model_architecture' in training_result:
                    zipf.writestr('model_architecture.json', 
                                json.dumps(training_result['model_architecture'], indent=2))
                
                # Add visualizations if available
                viz_results = supabase.table('training_visualizations')\
                    .select('*')\
                    .eq('session_id', uuid_session_id)\
                    .execute()
                    
                if viz_results.data:
                    for viz in viz_results.data:
                        if viz.get('plot_data'):
                            viz_filename = f"visualization_{viz.get('plot_type', 'unknown')}.json"
                            zipf.writestr(viz_filename, json.dumps(viz['plot_data'], indent=2))
                    logger.info(f"Added {len(viz_results.data)} visualizations to ZIP")
                
                # Add README with instructions
                readme_content = f"""# Model Download - Session {session_id}

## Contents
- `model_metadata.json`: Complete model training metadata
- `training_summary.json`: Training performance summary  
- `model_architecture.json`: Model architecture details (if available)
- `visualization_*.json`: Training visualizations (if available)

## Model Information
- Session ID: {session_id}
- Model Type: {training_result.get('model_type', 'unknown')}
- Training Status: {training_result.get('status', 'unknown')}
- Created: {training_result.get('created_at', 'unknown')}
- Downloaded: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Usage Notes
This ZIP contains the training metadata and results for your trained model.
The actual model weights are stored in the database and can be loaded 
using the provided session ID in your application.

For technical support, reference Session ID: {session_id}
"""
                zipf.writestr('README.md', readme_content)
                logger.info("Added README to ZIP")
            
            logger.info(f"Created ZIP file: {zip_path}")
            
            # Send the ZIP file for download
            return send_file(
                zip_path,
                as_attachment=True,
                download_name=f"model_{session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                mimetype='application/zip'
            )
            
    except Exception as e:
        logger.error(f"Error downloading model for session {session_id}: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Failed to download model: {str(e)}'
        }), 500


@training_api_bp.route('/evaluation-tables/<session_id>', methods=['GET'])
def get_evaluation_tables(session_id: str):
    """
    Get evaluation dataframes as tables for frontend display
    
    Args:
        session_id: Session identifier
        
    Returns:
        JSON response with evaluation tables (df_eval and df_eval_ts)
    """
    error_handler = get_error_handler()
    
    try:
        with error_handler.error_context("get_evaluation_tables", session_id):
            logger.info(f"Getting evaluation tables for session {session_id}")
            
            # Get UUID session ID
            supabase = get_supabase_client()
            uuid_session_id = create_or_get_session_uuid(session_id)
            
            # Get training results from database
            logger.info(f"Fetching training results for UUID session {uuid_session_id}")
            results = supabase.table('training_results')\
                .select('*')\
                .eq('session_id', uuid_session_id)\
                .order('created_at', desc=True)\
                .limit(1)\
                .execute()
            
            if not results.data:
                return jsonify({
                    'success': False,
                    'error': 'No training results found for this session'
                }), 404
            
            training_result = results.data[0]
            evaluation_metrics = training_result.get('evaluation_metrics', {})
            
            # Check if evaluation_metrics exists and is not None
            if not evaluation_metrics or evaluation_metrics is None:
                return jsonify({
                    'success': False,
                    'error': 'No evaluation metrics available. Model training must be completed first.'
                }), 404
            
            # Process evaluation metrics into table format
            evaluation_tables = {}
            
            for dataset_name, dataset_metrics in evaluation_metrics.items():
                # Create df_eval table (model performance comparison)
                df_eval_rows = []
                
                for model_name, model_data in dataset_metrics.items():
                    if 'metrics' in model_data:
                        metrics = model_data['metrics']
                        
                        eval_row = {
                            'Model': model_name.upper(),  # First column: model name
                            'delta [min]': round(model_data.get('delta_min', metrics.get('delta_min', 15.0)), 2),  # Time delta in minutes - exact match with original
                            'MAE': round(metrics.get('mae', 0.0), 5),
                            'MAPE': round(metrics.get('mape', 0.0), 5),
                            'MSE': round(metrics.get('mse', 0.0), 5),
                            'RMSE': round(metrics.get('rmse', 0.0), 5),
                            'NRMSE': round(metrics.get('nrmse', 0.0), 5),  # NRMSE as in original code
                            'WAPE': round(metrics.get('wape', 0.0), 5),  # WAPE missing from previous version
                            'sMAPE': round(metrics.get('smape', 0.0), 5),  # sMAPE as in original code
                            'MASE': round(metrics.get('mase', 0.0), 5)
                        }
                        
                        df_eval_rows.append(eval_row)
                
                # Create df_eval_ts table (time series evaluation) - simplified version
                df_eval_ts_rows = []
                
                for model_name, model_data in dataset_metrics.items():
                    if 'config' in model_data:
                        config = model_data['config']
                        metrics = model_data.get('metrics', {})
                        
                        ts_row = {
                            'Model': model_name.upper(),
                            'Type': model_data.get('model_type', 'unknown'),
                            'Config': f"Epochs: {config.get('epochs', 'N/A')}, "
                                    f"Activation: {config.get('activation', 'N/A')}, "
                                    f"Learning Rate: {config.get('learning_rate', 'N/A')}",
                            'Performance': f"MAE: {round(metrics.get('mae', 0.0), 4)}, "
                                         f"WAPE: {round(metrics.get('wape', 0.0), 2)}%",
                            'Status': 'Trained Successfully'
                        }
                        
                        df_eval_ts_rows.append(ts_row)
                
                evaluation_tables[dataset_name] = {
                    'df_eval': {
                        'title': f'Model Performance Evaluation - {dataset_name}',
                        'description': 'Comparison of different models performance metrics',
                        'columns': ['Model', 'MAE', 'MSE', 'RMSE', 'MAPE', 'NRMSE', 'WAPE', 'SMAPE', 'MASE'],
                        'data': df_eval_rows
                    },
                    'df_eval_ts': {
                        'title': f'Model Configuration & Training Info - {dataset_name}',
                        'description': 'Training configuration and status for each model',
                        'columns': ['Model', 'Type', 'Config', 'Performance', 'Status'],
                        'data': df_eval_ts_rows
                    }
                }
            
            logger.info(f"Generated evaluation tables for {len(evaluation_tables)} datasets")
            
            return jsonify({
                'success': True,
                'session_id': session_id,
                'tables': evaluation_tables,
                'total_datasets': len(evaluation_tables),
                'generated_at': datetime.now().isoformat()
            })
            
    except Exception as e:
        logger.error(f"Error getting evaluation tables for session {session_id}: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Failed to get evaluation tables: {str(e)}'
        }), 500