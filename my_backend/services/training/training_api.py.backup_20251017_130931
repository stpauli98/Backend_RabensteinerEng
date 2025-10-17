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
import pickle
import numpy as np
import pandas as pd

# Import existing supabase client
from utils.database import get_supabase_client, create_or_get_session_uuid

# Import the new pipeline function
from .pipeline_integration import run_complete_original_pipeline

logger = logging.getLogger(__name__)

# Create blueprint for training API
training_api_bp = Blueprint('training_api', __name__, url_prefix='/api/training')


def convert_training_split_params(training_split):
    """
    Convert frontend training split parameter names to backend expected names.
    
    Frontend sends: train_ratio, validation_ratio, test_ratio (0-1 scale)
    Backend expects: trainPercentage, valPercentage, testPercentage (0-100 scale)
    
    Args:
        training_split: Dict with frontend parameter names
        
    Returns:
        Dict with backend parameter names
    """
    if not training_split:
        return {}
    
    converted_split = {}
    
    # Map train_ratio to trainPercentage (converting from 0-1 to 0-100)
    if 'train_ratio' in training_split:
        converted_split['trainPercentage'] = training_split['train_ratio'] * 100
    elif 'trainPercentage' in training_split:
        converted_split['trainPercentage'] = training_split['trainPercentage']
    
    if 'validation_ratio' in training_split:
        converted_split['valPercentage'] = training_split['validation_ratio'] * 100
    elif 'valPercentage' in training_split:
        converted_split['valPercentage'] = training_split['valPercentage']
    
    if 'test_ratio' in training_split:
        converted_split['testPercentage'] = training_split['test_ratio'] * 100
    elif 'testPercentage' in training_split:
        converted_split['testPercentage'] = training_split['testPercentage']
    
    # Map other parameters
    if 'random_state' in training_split:
        converted_split['random_dat'] = training_split['random_state']
    elif 'shuffle' in training_split:
        # If shuffle is true, use a default random state, otherwise use None
        converted_split['random_dat'] = 42 if training_split.get('shuffle', True) else None
    elif 'random_dat' in training_split:
        converted_split['random_dat'] = training_split['random_dat']
    
    return converted_split


def cleanup_duplicate_visualizations(session_id: str):
    """
    Clean up duplicate visualizations for a session, keeping only the most recent ones.
    
    Args:
        session_id: Session identifier
    """
    try:
        supabase = get_supabase_client()
        uuid_session_id = create_or_get_session_uuid(session_id)
        
        # Get all visualizations for this session
        response = supabase.table('training_visualizations').select('*').eq('session_id', uuid_session_id).execute()
        
        if not response.data:
            return
        
        # Group by plot_name and keep only the most recent for each
        from collections import defaultdict
        plots_by_name = defaultdict(list)
        
        for viz in response.data:
            plots_by_name[viz['plot_name']].append(viz)
        
        # For each plot name, keep only the most recent and delete the rest
        for plot_name, visualizations in plots_by_name.items():
            if len(visualizations) > 1:
                # Sort by created_at desc to get most recent first
                visualizations.sort(key=lambda x: x['created_at'], reverse=True)
                
                # Keep the first (most recent), delete the rest
                for viz_to_delete in visualizations[1:]:
                    delete_response = supabase.table('training_visualizations').delete().eq('id', viz_to_delete['id']).execute()
                    logger.info(f"Deleted duplicate visualization {plot_name} with id {viz_to_delete['id']} for session {session_id}")
        
    except Exception as e:
        logger.error(f"Error cleaning up duplicate visualizations for session {session_id}: {str(e)}")


def save_visualization_to_database(session_id: str, viz_name: str, viz_data: str):
    """
    Save a visualization to the database (with duplicate check)

    Args:
        session_id: Session identifier
        viz_name: Name of the visualization
        viz_data: Base64 encoded visualization data
    """
    try:
        supabase = get_supabase_client()
        uuid_session_id = create_or_get_session_uuid(session_id)

        # Check if visualization already exists
        existing = supabase.table('training_visualizations').select('id').eq('session_id', uuid_session_id).eq('plot_name', viz_name).execute()

        if existing.data:
            # Update existing record instead of creating duplicate
            viz_record = {
                'image_data': viz_data,
                'plot_type': 'violin_plot' if 'distribution' in viz_name else 'other',
                'created_at': datetime.now().isoformat()
            }
            response = supabase.table('training_visualizations').update(viz_record).eq('session_id', uuid_session_id).eq('plot_name', viz_name).execute()
            logger.info(f"Updated existing visualization {viz_name} for session {session_id}")
        else:
            # Create new record
            viz_record = {
                'session_id': uuid_session_id,
                'plot_name': viz_name,
                'image_data': viz_data,
                'plot_type': 'violin_plot' if 'distribution' in viz_name else 'other',
                'created_at': datetime.now().isoformat()
            }
            response = supabase.table('training_visualizations').insert(viz_record).execute()
            logger.info(f"Created new visualization {viz_name} for session {session_id}")

        return True
        
    except Exception as e:
        logger.error(f"Error saving visualization {viz_name} for session {session_id}: {str(e)}")
        raise


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
        
        # Clean up any duplicate visualizations first
        cleanup_duplicate_visualizations(session_id)
        
        # Log how many visualizations we have after cleanup
        uuid_session_id = create_or_get_session_uuid(session_id)
        count_response = supabase.table('training_visualizations').select('id').eq('session_id', uuid_session_id).execute()
        logger.info(f"Found {len(count_response.data) if count_response.data else 0} visualizations for session {session_id} after cleanup")
        
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


@training_api_bp.route('/cleanup-duplicates/<session_id>', methods=['POST'])
def cleanup_duplicate_visualizations_endpoint(session_id: str):
    """
    Clean up duplicate visualizations for a session
    
    Args:
        session_id: Session identifier
        
    Returns:
        JSON response with cleanup status
    """
    try:
        logger.info(f"🧹 Starting duplicate cleanup for session {session_id}")
        
        # Clean up duplicates
        cleanup_duplicate_visualizations(session_id)
        
        # Get count after cleanup
        supabase = get_supabase_client()
        uuid_session_id = create_or_get_session_uuid(session_id)
        count_response = supabase.table('training_visualizations').select('id, plot_name').eq('session_id', uuid_session_id).execute()
        
        remaining_count = len(count_response.data) if count_response.data else 0
        plot_names = [viz['plot_name'] for viz in count_response.data] if count_response.data else []
        
        logger.info(f"✅ Cleanup completed. {remaining_count} visualizations remaining: {plot_names}")
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'remaining_visualizations': remaining_count,
            'plot_names': plot_names,
            'message': f'Duplicate cleanup completed for session {session_id}'
        }), 200
        
    except Exception as e:
        logger.error(f"Error cleaning up duplicates for {session_id}: {str(e)}")
        return jsonify({
            'error': 'Failed to cleanup duplicates',
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


@training_api_bp.route('/evaluation-tables/<session_id>', methods=['GET'])
def get_evaluation_tables(session_id: str):
    """
    Get evaluation tables (df_eval and df_eval_ts) for a training session
    
    Returns:
        JSON response with evaluation tables data formatted for frontend display
    """
    try:
        logger.info(f"📊 Getting evaluation tables for session: {session_id}")
        
        # Convert string session_id to UUID if needed
        try:
            import uuid
            uuid.UUID(session_id)
            uuid_session_id = session_id
        except (ValueError, TypeError):
            uuid_session_id = create_or_get_session_uuid(session_id)
            if not uuid_session_id:
                logger.warning(f"Could not get UUID for session {session_id}")
                return jsonify({
                    'success': False,
                    'error': 'Invalid session ID',
                    'session_id': session_id
                }), 400
        
        # Get training results from database
        supabase = get_supabase_client()
        results_response = supabase.table('training_results').select('*').eq('session_id', uuid_session_id).order('created_at', desc=True).execute()
        
        if not results_response.data or len(results_response.data) == 0:
            return jsonify({
                'success': False,
                'error': 'Model training must be completed first. Please train the model to generate evaluation tables.',
                'session_id': session_id
            }), 404
        
        training_result = results_response.data[0]
        
        # Check if training is completed
        if training_result.get('status') != 'completed':
            return jsonify({
                'success': False,
                'error': f'Training is {training_result.get("status", "unknown")}. Please wait for training to complete.',
                'session_id': session_id
            }), 400
        
        # Get evaluation metrics
        evaluation_metrics = training_result.get('evaluation_metrics', {})
        if not evaluation_metrics:
            return jsonify({
                'success': False,
                'error': 'No evaluation metrics found. Please ensure model training completed successfully.',
                'session_id': session_id
            }), 404
        
        # Format tables for each dataset
        tables = {}
        
        for dataset_name, dataset_metrics in evaluation_metrics.items():
            # Create df_eval table (overall metrics)
            df_eval_data = []
            df_eval_columns = ['Model', 'MAE', 'MSE', 'RMSE', 'MAPE', 'NRMSE', 'WAPE', 'sMAPE', 'MASE']
            
            for model_name, metrics in dataset_metrics.items():
                row = {
                    'Model': model_name,
                    'MAE': metrics.get('mae', 0.0),
                    'MSE': metrics.get('mse', 0.0),
                    'RMSE': metrics.get('rmse', 0.0),
                    'MAPE': metrics.get('mape', 0.0),
                    'NRMSE': metrics.get('nrmse', 0.0),
                    'WAPE': metrics.get('wape', 0.0),
                    'sMAPE': metrics.get('smape', 0.0),
                    'MASE': metrics.get('mase', 0.0)
                }
                df_eval_data.append(row)
            
            # Create df_eval_ts table (time series metrics - placeholder for now)
            # In the original file, this contains per-timestep metrics
            df_eval_ts_data = []
            df_eval_ts_columns = ['delta [min]', 'MAE', 'MSE', 'RMSE', 'MAPE', 'NRMSE', 'WAPE', 'sMAPE', 'MASE']
            
            # Generate time delta rows (similar to original file)
            time_deltas = [15, 30, 45, 60, 90, 120, 180, 240, 300, 360, 420, 480]
            for delta in time_deltas:
                # Get aggregated metrics for this time delta if available
                # For now, using placeholder values - should be computed from actual predictions
                row = {
                    'delta [min]': delta,
                    'MAE': 0.0,
                    'MSE': 0.0,
                    'RMSE': 0.0,
                    'MAPE': 0.0,
                    'NRMSE': 0.0,
                    'WAPE': 0.0,
                    'sMAPE': 0.0,
                    'MASE': 0.0
                }
                
                # If we have time-series specific metrics, use them
                if 'time_series_metrics' in dataset_metrics:
                    ts_metrics = dataset_metrics['time_series_metrics'].get(str(delta), {})
                    for metric_name in ['MAE', 'MSE', 'RMSE', 'MAPE', 'NRMSE', 'WAPE', 'sMAPE', 'MASE']:
                        row[metric_name] = ts_metrics.get(metric_name.lower(), row[metric_name])
                
                df_eval_ts_data.append(row)
            
            tables[dataset_name] = {
                'df_eval': {
                    'title': f'Model Performance Metrics - {dataset_name}',
                    'description': 'Overall evaluation metrics for each model',
                    'columns': df_eval_columns,
                    'data': df_eval_data
                },
                'df_eval_ts': {
                    'title': f'Time Series Evaluation - {dataset_name}',
                    'description': 'Metrics aggregated by time delta',
                    'columns': df_eval_ts_columns,
                    'data': df_eval_ts_data
                }
            }
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'tables': tables,
            'total_datasets': len(tables),
            'generated_at': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting evaluation tables: {str(e)}")
        logger.exception(e)
        return jsonify({
            'success': False,
            'error': 'Failed to get evaluation tables',
            'message': str(e),
            'session_id': session_id
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
        
        # Get request data for model parameters and training split
        request_data = request.get_json() or {}
        model_parameters = request_data.get('model_parameters', {})
        training_split = request_data.get('training_split', {})
        training_split = convert_training_split_params(training_split)
        
        
        # Start the complete 7-phase pipeline in background thread
        def run_pipeline_async():
            try:
                
                # Import and run the complete pipeline with user parameters
                result = run_complete_original_pipeline(
                    session_id=session_id,
                    model_parameters=model_parameters,
                    training_split=training_split,
                    progress_callback=lambda session_id, phase, msg, progress: None
                )
                
                if result.get('success'):
                    
                    # Save results to database
                    try:
                        from .results_generator import ResultsGenerator
                        results_gen = ResultsGenerator()
                        
                        # Extract and save evaluation results if they exist
                        if 'final_results' in result and 'evaluation_results' in result['final_results']:
                            results_gen.results = result['final_results']['evaluation_results']
                            success = results_gen.save_results_to_database(session_id, get_supabase_client())
                            if not success:
                                logger.warning(f"Failed to save results to database for session {session_id}")
                        
                        # Also save violin plots if they exist
                        if 'final_results' in result and 'visualizations' in result['final_results']:
                            visualizations = result['final_results']['visualizations']
                            
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
    Start model training for a session with user parameters
    
    Expects JSON body with:
    - model_parameters: Model configuration (MODE, LAY, N, EP, etc.)
    - training_split: Training data split parameters (trainPercentage, etc.)
    
    Returns:
        JSON response with training status
    """
    try:
        logger.info(f"📥 train_models endpoint called for session: {session_id}")
        
        # Get request data
        request_data = request.get_json() or {}
        logger.info(f"📋 Raw request data: {request_data}")
        
        model_parameters = request_data.get('model_parameters', {})
        logger.info(f"🔧 Model parameters received: {model_parameters}")
        
        training_split = request_data.get('training_split', {})
        logger.info(f"📊 Training split before conversion: {training_split}")
        
        training_split = convert_training_split_params(training_split)
        logger.info(f"📊 Training split after conversion: {training_split}")
        
        # Validate required parameters
        if not model_parameters:
            logger.error("❌ No model parameters provided")
            return jsonify({
                'success': False,
                'error': 'Model parameters are required',
                'message': 'Please provide model configuration parameters'
            }), 400
        
        if not training_split:
            logger.error("❌ No training split parameters provided")
            return jsonify({
                'success': False,
                'error': 'Training split parameters are required',
                'message': 'Please provide training data split parameters'
            }), 400
        
        # Start the complete pipeline with user parameters
        results = run_complete_original_pipeline(
            session_id, 
            model_parameters, 
            training_split
        )
        
        # Save results to database after pipeline completes
        if results and results.get('success'):
            # Import results generator to save results
            from .results_generator import ResultsGenerator
            
            # Create results generator instance and save results
            results_gen = ResultsGenerator()
            
            # Extract evaluation results from pipeline results
            if 'final_results' in results and 'evaluation_results' in results['final_results']:
                results_gen.results = results['final_results']['evaluation_results']
                success = results_gen.save_results_to_database(session_id, get_supabase_client())
                if not success:
                    logger.warning(f"Failed to save results to database for session {session_id}")
            
            # Also save visualizations if they exist
            if 'final_results' in results and 'visualizations' in results['final_results']:
                visualizations = results['final_results']['visualizations']
                
                for viz_name, viz_data in visualizations.items():
                    try:
                        save_visualization_to_database(session_id, viz_name, viz_data)
                    except Exception as viz_error:
                        logger.error(f"Failed to save visualization {viz_name}: {str(viz_error)}")
        
        # Don't include full results in response as it may contain non-serializable objects
        # Only return essential information
        response_data = {
            'success': True,
            'session_id': session_id,
            'message': 'Model training completed successfully',
            'status': 'completed'
        }
        
        # Safely extract only serializable parts from results
        if results and isinstance(results, dict):
            if 'summary' in results:
                response_data['summary'] = results['summary']
            if 'final_results' in results and 'summary' in results['final_results']:
                response_data['summary'] = results['final_results']['summary']
            if 'success' in results:
                response_data['training_success'] = results['success']
        
        return jsonify(response_data), 200
        
    except Exception as e:
        logger.error(f"Error training models for {session_id}: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Failed to train models',
            'message': str(e)
        }), 500


@training_api_bp.route('/list-models/<session_id>', methods=['GET'])
def list_models(session_id: str):
    """
    List all available trained models for a session
    
    Returns:
        JSON response with list of available models
    """
    try:
        import os
        import glob
        from datetime import datetime
        
        # Check if models directory exists
        models_dir = os.path.join('uploads', 'trained_models')
        session_models_dir = os.path.join(models_dir, session_id)
        
        # Find existing trained models for this session
        existing_models = []
        search_dirs = [models_dir, session_models_dir] if os.path.exists(session_models_dir) else [models_dir]
        
        for search_dir in search_dirs:
            if os.path.exists(search_dir):
                model_patterns = [
                    os.path.join(search_dir, '*.h5'),
                    os.path.join(search_dir, '*.pkl')
                ]
                
                for pattern in model_patterns:
                    found_files = glob.glob(pattern)
                    for file_path in found_files:
                        # Extract model info from filename
                        filename = os.path.basename(file_path)
                        file_size = os.path.getsize(file_path)
                        modified_time = os.path.getmtime(file_path)
                        
                        # Try to parse model type from filename
                        model_type = 'Unknown'
                        if 'dense' in filename.lower():
                            model_type = 'Dense'
                        elif 'cnn' in filename.lower():
                            model_type = 'CNN'
                        elif 'lstm' in filename.lower():
                            model_type = 'LSTM'
                        elif 'svr' in filename.lower():
                            model_type = 'SVR'
                        elif 'linear' in filename.lower():
                            model_type = 'Linear'
                        
                        existing_models.append({
                            'filename': filename,
                            'path': file_path,
                            'model_type': model_type,
                            'file_size': file_size,
                            'file_size_mb': round(file_size / (1024 * 1024), 2),
                            'modified_time': datetime.fromtimestamp(modified_time).isoformat(),
                            'format': 'h5' if file_path.endswith('.h5') else 'pkl'
                        })
        
        # Sort by modification time (newest first)
        existing_models.sort(key=lambda x: x['modified_time'], reverse=True)
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'models': existing_models,
            'count': len(existing_models)
        }), 200
        
    except Exception as e:
        logger.error(f"Error listing models for {session_id}: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Failed to list models',
            'message': str(e)
        }), 500


@training_api_bp.route('/download-model/<session_id>', methods=['GET'])
def download_model(session_id: str):
    """
    Download a trained model file for a session
    
    Returns:
        Model file as attachment or JSON error response
    """
    try:
        from flask import send_file
        import os
        import glob
        
        # Get optional model name from query params
        model_name = request.args.get('model_name')
        model_type = request.args.get('model_type')
        
        # Check if models directory exists
        models_dir = os.path.join('uploads', 'trained_models')
        session_models_dir = os.path.join(models_dir, session_id)
        
        # Find existing trained models for this session
        existing_models = []
        search_dirs = [models_dir, session_models_dir] if os.path.exists(session_models_dir) else [models_dir]
        
        for search_dir in search_dirs:
            if os.path.exists(search_dir):
                model_patterns = [
                    os.path.join(search_dir, '*.h5'),
                    os.path.join(search_dir, '*.pkl')
                ]
                
                for pattern in model_patterns:
                    found_files = glob.glob(pattern)
                    existing_models.extend(found_files)
        
        if not existing_models:
            return jsonify({
                'success': False,
                'error': 'No models found',
                'message': f'No trained models found for session {session_id}'
            }), 404
        
        # Sort by modification time and get the most recent or filter by name
        if model_name:
            # Filter by model name if provided
            filtered_models = [m for m in existing_models if model_name in os.path.basename(m)]
            if filtered_models:
                existing_models = filtered_models
        
        # Get the most recent model
        existing_models.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        model_path = existing_models[0]
        
        # Determine mime type based on file extension
        if model_path.endswith('.h5'):
            mimetype = 'application/x-hdf'
        elif model_path.endswith('.pkl'):
            mimetype = 'application/octet-stream'
        else:
            mimetype = 'application/octet-stream'
        
        # Ensure absolute path
        if not os.path.isabs(model_path):
            import pathlib
            base_dir = pathlib.Path(__file__).parent.parent.parent
            model_path = os.path.join(base_dir, model_path)
        
        # Send the file
        return send_file(
            model_path,
            mimetype=mimetype,
            as_attachment=True,
            download_name=os.path.basename(model_path)
        )
        
    except Exception as e:
        logger.error(f"Error downloading model for {session_id}: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Failed to download model',
            'message': str(e)
        }), 500


@training_api_bp.route('/save-model/<session_id>', methods=['POST'])
def save_model(session_id: str):
    """
    Save a trained model for a session
    
    Expects JSON body with:
    - model_name: Name of the model to save
    - model_type: Type of model (Dense, CNN, LSTM, etc.)
    - save_path: Optional custom save path
    
    Returns:
        JSON response with model save status
    """
    try:
        
        # Get request data
        request_data = request.get_json() or {}
        model_name = request_data.get('model_name')
        model_type = request_data.get('model_type', 'Dense')
        save_path = request_data.get('save_path')
        
        if not model_name:
            return jsonify({
                'success': False,
                'error': 'Model name is required',
                'message': 'Please provide a model name'
            }), 400
        
        # Import necessary modules for model saving
        import os
        import pickle
        import glob
        from datetime import datetime
        
        # Check if models directory exists (models should already be saved during training)
        models_dir = os.path.join('uploads', 'trained_models')
        session_models_dir = os.path.join(models_dir, session_id)
        
        # Find existing trained models for this session
        existing_models = []
        if os.path.exists(models_dir):
            # Look for models with any pattern (models are saved as modeltype_datasetname_timestamp)
            # Since we don't have session_id in the filename, we need to look for all models
            # and filter by creation time or look in session-specific directory
            model_patterns = [
                os.path.join(models_dir, '*.h5'),
                os.path.join(models_dir, '*.pkl'),
                # Also check session-specific directory if it exists
                os.path.join(session_models_dir, '*.h5'),
                os.path.join(session_models_dir, '*.pkl')
            ]
            
            for pattern in model_patterns:
                found_files = glob.glob(pattern)
                existing_models.extend(found_files)
                
            # Log what we found
            if existing_models:
                logger.info(f"Found {len(existing_models)} model files: {existing_models}")
        
        if not existing_models:
            logger.warning(f"No trained models found for session {session_id}")
            # Create directory for future use
            os.makedirs(session_models_dir, exist_ok=True)
            
        # Generate timestamp for unique filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # If we found existing models, use the most recent one
        if existing_models:
            # Sort by modification time and get the most recent
            existing_models.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            most_recent_model = existing_models[0]
            logger.info(f"Found existing model: {most_recent_model}")
            save_path = most_recent_model
            
            # Detect model type from file extension
            if most_recent_model.endswith('.h5'):
                model_type = model_type or 'Dense'  # Default to Dense for .h5 files
            elif most_recent_model.endswith('.pkl'):
                model_type = model_type or 'Linear'  # Default to Linear for .pkl files
        else:
            # Determine save path for metadata only (no actual model file)
            if not save_path:
                os.makedirs(session_models_dir, exist_ok=True)
                if model_type in ['Dense', 'CNN', 'LSTM', 'AR LSTM']:
                    # Save as .h5 for neural network models
                    save_path = os.path.join(session_models_dir, f'{model_name}_{timestamp}.h5')
                elif model_type in ['SVR_dir', 'SVR_MIMO', 'LIN', 'Linear']:
                    # Save as .pkl for sklearn models
                    save_path = os.path.join(session_models_dir, f'{model_name}_{timestamp}.pkl')
                else:
                    save_path = os.path.join(session_models_dir, f'{model_name}_{timestamp}.model')
        
        # Get the supabase client to save model metadata
        supabase = get_supabase_client()
        uuid_session_id = create_or_get_session_uuid(session_id)
        
        # Save model metadata to database
        model_metadata = {
            'session_id': uuid_session_id,
            'model_name': model_name,
            'model_type': model_type,
            'save_path': save_path,
            'created_at': datetime.now().isoformat()
        }
        
        # Insert into a models table (you may need to create this table)
        try:
            response = supabase.table('saved_models').insert(model_metadata).execute()
        except Exception as db_error:
            logger.warning(f"Could not save model metadata to database: {str(db_error)}")
        
        # Prepare response with model information
        response_data = {
            'success': True,
            'session_id': session_id,
            'model_name': model_name,
            'model_type': model_type,
            'save_path': save_path,
            'message': 'Model metadata saved successfully' if not existing_models else 'Model found and metadata saved successfully',
            'model_info': {
                'session_id': session_id,
                'model_name': model_name,
                'model_format': model_type,
                'save_timestamp': timestamp,
                'file_exists': os.path.exists(save_path) if save_path else False,
                'file_size': os.path.getsize(save_path) if save_path and os.path.exists(save_path) else 0
            }
        }
        
        # Add information about all found models
        if existing_models:
            response_data['found_models'] = [os.path.basename(m) for m in existing_models]
            response_data['models_count'] = len(existing_models)
        
        return jsonify(response_data), 200
        
    except Exception as e:
        logger.error(f"Error saving model for {session_id}: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Failed to save model',
            'message': str(e)
        }), 500


@training_api_bp.route('/download-model-h5/<session_id>', methods=['GET'])
def download_model_h5(session_id: str):
    """
    Download trained model in .h5 format from Supabase database
    
    Args:
        session_id: Training session ID
    
    Returns:
        Model file for download or error response
    """
    try:
        import pickle
        import base64
        import tempfile
        import os
        from flask import send_file
        from datetime import datetime
        
        # Get supabase client and convert session_id to UUID
        supabase = get_supabase_client()
        uuid_session_id = create_or_get_session_uuid(session_id)
        
        # Query training_results table for the model
        response = supabase.table('training_results')\
            .select('results')\
            .eq('session_id', uuid_session_id)\
            .eq('status', 'completed')\
            .order('created_at', desc=True)\
            .limit(1)\
            .execute()
        
        if not response.data:
            return jsonify({
                'success': False,
                'error': 'No trained model found',
                'message': f'No completed training found for session {session_id}'
            }), 404
        
        # Extract model data from results
        results = response.data[0]['results']
        trained_model = None
        model_type = None
        
        # Search for serialized model in results structure
        def find_serialized_model(obj, path=""):
            if isinstance(obj, dict):
                if '_model_type' in obj and obj.get('_model_type') == 'serialized_model':
                    return obj
                for key, value in obj.items():
                    result = find_serialized_model(value, f"{path}.{key}" if path else key)
                    if result:
                        return result
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    result = find_serialized_model(item, f"{path}[{i}]" if path else f"[{i}]")
                    if result:
                        return result
            return None
        
        serialized_model = find_serialized_model(results)
        
        if not serialized_model:
            return jsonify({
                'success': False,
                'error': 'No serialized model found',
                'message': 'Model data not found in training results'
            }), 404
        
        # Extract model information
        model_class = serialized_model.get('_model_class', 'Unknown')
        model_b64 = serialized_model.get('_model_data')
        
        if not model_b64:
            return jsonify({
                'success': False,
                'error': 'Model data not found',
                'message': 'Base64 model data is missing'
            }), 404
        
        # Deserialize model from base64
        try:
            model_bytes = base64.b64decode(model_b64)
            model = pickle.loads(model_bytes)
        except Exception as e:
            return jsonify({
                'success': False,
                'error': 'Failed to deserialize model',
                'message': f'Error deserializing model: {str(e)}'
            }), 500
        
        # Check if model has save method (Keras/TensorFlow models)
        if not hasattr(model, 'save'):
            return jsonify({
                'success': False,
                'error': 'Model format not supported',
                'message': f'Model class {model_class} does not support .h5 export'
            }), 400
        
        # Create temporary file for .h5 export
        temp_dir = tempfile.mkdtemp()
        temp_file = os.path.join(temp_dir, f'model_{session_id}.h5')
        
        try:
            # Save model as .h5 file
            model.save(temp_file)
            
            # Verify file was created
            if not os.path.exists(temp_file):
                raise Exception("Failed to create .h5 file")
            
            # Generate download filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            download_name = f'trained_model_{model_class}_{session_id}_{timestamp}.h5'
            
            # Return file for download
            return send_file(
                temp_file,
                as_attachment=True,
                download_name=download_name,
                mimetype='application/octet-stream'
            )
            
        except Exception as save_error:
            # Clean up temp file
            if os.path.exists(temp_file):
                os.remove(temp_file)
            if os.path.exists(temp_dir):
                os.rmdir(temp_dir)
                
            return jsonify({
                'success': False,
                'error': 'Failed to export model',
                'message': f'Error saving model as .h5: {str(save_error)}'
            }), 500
        
    except Exception as e:
        logger.error(f"Error downloading model for session {session_id}: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Internal server error',
            'message': str(e)
        }), 500


@training_api_bp.route('/list-models-database/<session_id>', methods=['GET'])
def list_models_database(session_id: str):
    """
    List available trained models from Supabase database
    
    Args:
        session_id: Training session ID
    
    Returns:
        JSON response with list of available models
    """
    try:
        # Get supabase client and convert session_id to UUID  
        supabase = get_supabase_client()
        uuid_session_id = create_or_get_session_uuid(session_id)
        
        # Query training_results table - get only the most recent completed training
        response = supabase.table('training_results')\
            .select('results, created_at')\
            .eq('session_id', uuid_session_id)\
            .eq('status', 'completed')\
            .order('created_at', desc=True)\
            .limit(1)\
            .execute()
        
        if not response.data:
            return jsonify({
                'success': True,
                'data': {'models': [], 'count': 0},
                'message': 'No trained models found'
            })
        
        models = []
        
        # Since we're getting only the latest record, process it directly
        if response.data:
            record = response.data[0]  # Get the most recent record
            results = record['results']
            created_at = record['created_at']
            
            # Search for serialized models in results
            def extract_models_info(obj, path=""):
                found_models = []
                if isinstance(obj, dict):
                    if '_model_type' in obj and obj.get('_model_type') == 'serialized_model':
                        model_class = obj.get('_model_class', 'Unknown')
                        model_data_size = len(obj.get('_model_data', ''))
                        
                        found_models.append({
                            'filename': f'model_{model_class}_{session_id}.h5',
                            'path': f'database://{path}',
                            'model_type': model_class,
                            'file_size': model_data_size,
                            'file_size_mb': round(model_data_size / (1024 * 1024), 2),
                            'modified_time': created_at,
                            'format': 'h5'
                        })
                    
                    for key, value in obj.items():
                        found_models.extend(extract_models_info(value, f"{path}.{key}" if path else key))
                elif isinstance(obj, list):
                    for i, item in enumerate(obj):
                        found_models.extend(extract_models_info(item, f"{path}[{i}]" if path else f"[{i}]"))
                
                return found_models
            
            models = extract_models_info(results)
        
        return jsonify({
            'success': True,
            'data': {
                'models': models,
                'count': len(models)
            }
        })
        
    except Exception as e:
        logger.error(f"Error listing models for session {session_id}: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Failed to list models',
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


@training_api_bp.route('/plot-variables/<session_id>', methods=['GET'])
def get_plot_variables(session_id: str):
    """
    Get available input and output variables for plotting
    
    Args:
        session_id: Session identifier
        
    Returns:
        JSON response with input and output variable names
    """
    try:
        supabase = get_supabase_client()
        
        # Try to get training results first
        results = _get_results_from_database(session_id, supabase)
        
        if results and 'results' in results:
            training_data = results['results']
            
            # Extract variable names from training data
            input_variables = []
            output_variables = []
            
            # Check for column information in results
            if 'data_info' in training_data:
                data_info = training_data['data_info']
                input_variables = data_info.get('input_columns', [])
                output_variables = data_info.get('output_columns', [])
            elif 'input_columns' in training_data and 'output_columns' in training_data:
                # Direct column names from training data
                input_variables = training_data.get('input_columns', [])
                output_variables = training_data.get('output_columns', [])
            elif 'columns' in training_data:
                # Fallback to columns if data_info not available
                columns = training_data.get('columns', {})
                input_variables = columns.get('input', [])
                output_variables = columns.get('output', [])
            elif 'model_info' in training_data:
                # Try to extract from model info
                model_info = training_data.get('model_info', {})
                input_variables = model_info.get('input_features', [])
                output_variables = model_info.get('output_features', [])
            
            # If still no variables, try to get from file metadata
            if not input_variables and not output_variables:
                try:
                    uuid_session_id = create_or_get_session_uuid(session_id)
                    
                    # Get file metadata from database
                    file_response = supabase.table('file_metadata').select('*').eq('session_id', uuid_session_id).execute()
                    
                    if file_response.data:
                        for file_data in file_response.data:
                            file_type = file_data.get('file_type', '')
                            columns = file_data.get('columns', [])
                            
                            if file_type == 'input' and not input_variables:
                                input_variables = [col for col in columns if col not in ['timestamp', 'UTC']]
                            elif file_type == 'output' and not output_variables:
                                output_variables = [col for col in columns if col not in ['timestamp', 'UTC']]
                                
                except Exception as e:
                    logger.debug(f"Could not get file metadata for session {session_id}: {str(e)}")
            
            # As a last resort, use some default variable names based on the original training
            if not input_variables and not output_variables:
                # Default names from training_original.py
                input_variables = ['Temperature', 'Load']  # Common input features
                output_variables = ['Predicted_Load']  # Common output feature
            
            return jsonify({
                'success': True,
                'session_id': session_id,
                'input_variables': input_variables,
                'output_variables': output_variables
            })
        else:
            # No training results yet, return empty arrays
            return jsonify({
                'success': True,
                'session_id': session_id,
                'input_variables': [],
                'output_variables': [],
                'message': 'No training data available yet'
            })
            
    except Exception as e:
        logger.error(f"Error getting plot variables for session {session_id}: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Failed to get plot variables',
            'message': str(e)
        }), 500


@training_api_bp.route('/generate-plot', methods=['POST'])
def generate_plot():
    """
    Generate plot based on user selections matching original training_original.py
    
    Expected request body:
    {
        'session_id': str,
        'plot_settings': {
            'num_sbpl': int,
            'x_sbpl': str ('UTC' or 'ts'),
            'y_sbpl_fmt': str ('original' or 'skaliert'),
            'y_sbpl_set': str ('gemeinsame Achse' or 'separate Achsen')
        },
        'df_plot_in': dict,  # {feature_name: bool, ...}
        'df_plot_out': dict,  # {feature_name: bool, ...}
        'df_plot_fcst': dict  # {feature_name: bool, ...}
    }
    """
    try:
        data = request.json
        session_id = data.get('session_id')
        
        if not session_id:
            return jsonify({
                'success': False,
                'error': 'Session ID is required'
            }), 400
            
        # Get plot settings
        plot_settings = data.get('plot_settings', {})
        num_sbpl = plot_settings.get('num_sbpl', 17)
        x_sbpl = plot_settings.get('x_sbpl', 'UTC')
        y_sbpl_fmt = plot_settings.get('y_sbpl_fmt', 'original')
        y_sbpl_set = plot_settings.get('y_sbpl_set', 'separate Achsen')
        
        # Get plot selections
        df_plot_in = data.get('df_plot_in', {})
        df_plot_out = data.get('df_plot_out', {})
        df_plot_fcst = data.get('df_plot_fcst', {})
        
        logger.info(f"Generate plot for session {session_id}")
        logger.info(f"Plot settings: num_sbpl={num_sbpl}, x_sbpl={x_sbpl}, y_sbpl_fmt={y_sbpl_fmt}, y_sbpl_set={y_sbpl_set}")
        logger.info(f"Input variables selected: {[k for k, v in df_plot_in.items() if v]}")
        logger.info(f"Output variables selected: {[k for k, v in df_plot_out.items() if v]}")
        logger.info(f"Forecast variables selected: {[k for k, v in df_plot_fcst.items() if v]}")
        
        # Load model data from database
        try:
            from utils.database import create_or_get_session_uuid
            from supabase import create_client
            import numpy as np
            import pandas as pd
            import os
            
            # Get UUID for session
            uuid_session_id = create_or_get_session_uuid(session_id)
            if not uuid_session_id:
                return jsonify({
                    'success': False,
                    'error': 'Session not found'
                }), 404
            
            # Get Supabase client
            supabase_url = os.getenv('SUPABASE_URL')
            supabase_key = os.getenv('SUPABASE_KEY')
            supabase = create_client(supabase_url, supabase_key)
            
            # Fetch training results from database
            response = supabase.table('training_results')\
                .select('results')\
                .eq('session_id', uuid_session_id)\
                .order('created_at', desc=True)\
                .limit(1)\
                .execute()
            
            if not response.data or len(response.data) == 0:
                return jsonify({
                    'success': False,
                    'error': 'Model not trained yet. Please train the model first.'
                }), 400
                
            # Extract model data from results
            results = response.data[0]['results']
            
            # Get model data from results and deserialize if needed
            model_data = results.get('trained_model')
            
            # Deserialize model if it's in serialized format
            trained_model = None
            if model_data:
                if isinstance(model_data, dict) and model_data.get('_model_type') == 'serialized_model':
                    try:
                        import pickle
                        import base64
                        # Decode base64 and deserialize model
                        model_bytes = base64.b64decode(model_data['_model_data'])
                        trained_model = pickle.loads(model_bytes)
                        logger.info(f"Successfully deserialized model of type {model_data.get('_model_class')}")
                    except Exception as e:
                        logger.error(f"Failed to deserialize model: {e}")
                        trained_model = None
                else:
                    # Legacy format - model stored as string or other format
                    trained_model = model_data
            
            test_data = results.get('test_data', {})
            metadata = results.get('metadata', {})
            scalers = results.get('scalers', {})
            
            # Check if we have test data - support both naming conventions
            has_test_data = False
            if test_data:
                # Check for both possible naming conventions
                if 'X' in test_data and 'y' in test_data:
                    # New format: X, y
                    tst_x = np.array(test_data.get('X'))
                    tst_y = np.array(test_data.get('y'))
                    has_test_data = True
                elif 'X_test' in test_data and 'y_test' in test_data:
                    # Old format: X_test, y_test
                    tst_x = np.array(test_data.get('X_test'))
                    tst_y = np.array(test_data.get('y_test'))
                    has_test_data = True
            
            if not has_test_data:
                logger.error(f"No test data found in database for session {session_id}")
                return jsonify({
                    'success': False,
                    'error': 'No training data available for this session',
                    'message': 'Please train a model first before generating plots'
                }), 400
            else:
                
                # Generate predictions using trained model if available
                # Note: Currently models are stored as string representations, not actual objects
                if trained_model and hasattr(trained_model, 'predict'):
                    tst_fcst = trained_model.predict(tst_x)
                else:
                    # Model is not available as object, cannot generate predictions
                    logger.error(f"Model is not available as an object for session {session_id} (stored as: {type(trained_model).__name__})")
                    return jsonify({
                        'success': False,
                        'error': 'Model not available for predictions',
                        'message': 'The trained model cannot be used for predictions. Please retrain the model.'
                    }), 400
            
            # Import matplotlib for plotting
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend
            import matplotlib.pyplot as plt
            import seaborn as sns
            from datetime import datetime
            import io
            import base64
            
            # Create figure based on settings
            num_sbpl = min(num_sbpl, len(tst_x))  # Limit to available test samples
            num_sbpl_x = int(np.ceil(np.sqrt(num_sbpl)))
            num_sbpl_y = int(np.ceil(num_sbpl / num_sbpl_x))
            
            fig, axs = plt.subplots(num_sbpl_y, num_sbpl_x, 
                                   figsize=(20, 13), 
                                   layout='constrained')
            
            # Flatten axs array for easier indexing
            if num_sbpl == 1:
                axs = [axs]
            else:
                axs = axs.flatten()
            
            # Color palette - make sure we have enough colors
            # Count total number of variables to plot
            total_vars = len([k for k, v in df_plot_in.items() if v]) + \
                        len([k for k, v in df_plot_out.items() if v]) + \
                        len([k for k, v in df_plot_fcst.items() if v])
            palette = sns.color_palette("tab20", max(20, total_vars))
            
            # Plot each subplot
            for i_sbpl in range(num_sbpl):
                ax = axs[i_sbpl] if num_sbpl > 1 else axs[0]
                
                # Create x-axis values
                if x_sbpl == 'UTC':
                    # Create UTC timestamps
                    x_values = pd.date_range(start='2024-01-01', 
                                            periods=tst_x.shape[1], 
                                            freq='1h')
                else:
                    # Use timestep indices
                    x_values = np.arange(tst_x.shape[1])
                
                # Plot selected input variables
                color_idx = 0
                for var_name, selected in df_plot_in.items():
                    if selected and color_idx < tst_x.shape[-1]:
                        if y_sbpl_fmt == 'original':
                            y_values = tst_x[i_sbpl, :, color_idx]
                        else:
                            y_values = tst_x[i_sbpl, :, color_idx]  # Already scaled
                        
                        ax.plot(x_values, y_values, 
                               label=f'IN: {var_name}',
                               color=palette[color_idx % len(palette)],
                               marker='o', markersize=2,
                               linewidth=1)
                        color_idx += 1
                
                # Plot selected output variables (ground truth)
                for i_out, (var_name, selected) in enumerate(df_plot_out.items()):
                    if selected and i_out < tst_y.shape[-1]:
                        if y_sbpl_fmt == 'original':
                            y_values = tst_y[i_sbpl, :, i_out]
                        else:
                            y_values = tst_y[i_sbpl, :, i_out]  # Already scaled
                        
                        ax.plot(x_values[:len(y_values)], y_values,
                               label=f'OUT: {var_name}',
                               color=palette[color_idx % len(palette)],
                               marker='s', markersize=2,
                               linewidth=1)
                        color_idx += 1
                
                # Plot selected forecast variables (predictions)
                for i_fcst, (var_name, selected) in enumerate(df_plot_fcst.items()):
                    if selected and i_fcst < tst_fcst.shape[-1] if len(tst_fcst.shape) > 2 else 1:
                        if len(tst_fcst.shape) == 3:
                            y_values = tst_fcst[i_sbpl, :, i_fcst]
                        elif len(tst_fcst.shape) == 2:
                            y_values = tst_fcst[i_sbpl, :]
                        else:
                            y_values = tst_fcst
                        
                        ax.plot(x_values[:len(y_values)], y_values,
                               label=f'FCST: {var_name}',
                               color=palette[color_idx % len(palette)],
                               marker='^', markersize=2,
                               linewidth=1, linestyle='--')
                        color_idx += 1
                
                # Configure subplot
                ax.set_title(f'Sample {i_sbpl + 1}', fontsize=10)
                ax.legend(loc='upper left', fontsize=8)
                ax.grid(True, alpha=0.3)
                
                if x_sbpl == 'UTC':
                    ax.set_xlabel('Time (UTC)', fontsize=9)
                    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
                else:
                    ax.set_xlabel('Timestep', fontsize=9)
                
                ax.set_ylabel('Value', fontsize=9)
                
                # Handle y-axis configuration
                if y_sbpl_set == 'separate Achsen':
                    # Each line gets its own y-axis scale
                    pass  # Already handled by matplotlib auto-scaling
            
            # Remove empty subplots
            for i in range(num_sbpl, len(axs)):
                fig.delaxes(axs[i])
            
            # Save plot to base64 string
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
            buffer.seek(0)
            plot_data = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            logger.info(f"Plot generated successfully for session {session_id}")
            
            return jsonify({
                'success': True,
                'session_id': session_id,
                'plot_data': f'data:image/png;base64,{plot_data}',
                'message': 'Plot generated successfully'
            })
            
        except Exception as e:
            logger.error(f"Error generating plot: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return jsonify({
                'success': False,
                'error': f'Failed to generate plot: {str(e)}'
            }), 500
            
    except Exception as e:
        logger.error(f"Error in generate_plot endpoint: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Failed to process plot request',
            'message': str(e)
        }), 500