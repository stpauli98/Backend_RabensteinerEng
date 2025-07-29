"""
Monitoring and system health API for training system
Provides endpoints for error tracking, system metrics, and health monitoring
"""

from flask import Blueprint, jsonify, request
from typing import Dict, List, Optional
import logging
import sys
import os
from datetime import datetime, timezone, timedelta

# Import existing supabase client
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from supabase_client import get_supabase_client

# Import error handling and logging
from .error_handler import get_error_handler, ErrorCategory, ErrorSeverity
from .logging_config import get_performance_logger

logger = logging.getLogger(__name__)

# Create blueprint for monitoring API
monitoring_api_bp = Blueprint('monitoring_api', __name__, url_prefix='/api/monitoring')


@monitoring_api_bp.route('/health', methods=['GET'])
def health_check():
    """
    System health check endpoint
    
    Returns:
        JSON response with system health status
    """
    try:
        health_status = {
            'status': 'healthy',
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'version': '1.0.0',
            'components': {}
        }
        
        # Check database connection
        try:
            supabase = get_supabase_client()
            response = supabase.table('sessions').select('id').limit(1).execute()
            health_status['components']['database'] = {
                'status': 'healthy',
                'response_time_ms': 'N/A'  # Could add timing here
            }
        except Exception as e:
            health_status['components']['database'] = {
                'status': 'unhealthy',
                'error': str(e)
            }
            health_status['status'] = 'degraded'
        
        # Check error handler
        try:
            error_handler = get_error_handler()
            error_stats = error_handler.get_error_statistics()
            health_status['components']['error_handler'] = {
                'status': 'healthy',
                'total_sessions_with_errors': error_stats.get('total_sessions_with_errors', 0),
                'total_cached_errors': error_stats.get('total_cached_errors', 0)
            }
        except Exception as e:
            health_status['components']['error_handler'] = {
                'status': 'unhealthy',
                'error': str(e)
            }
            health_status['status'] = 'degraded'
        
        # Check system resources
        try:
            import psutil
            health_status['components']['system_resources'] = {
                'status': 'healthy',
                'cpu_percent': psutil.cpu_percent(interval=1),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_percent': psutil.disk_usage('/').percent
            }
            
            # Mark as degraded if resources are high
            if (health_status['components']['system_resources']['cpu_percent'] > 90 or
                health_status['components']['system_resources']['memory_percent'] > 90 or
                health_status['components']['system_resources']['disk_percent'] > 90):
                health_status['components']['system_resources']['status'] = 'degraded'
                health_status['status'] = 'degraded'
                
        except ImportError:
            health_status['components']['system_resources'] = {
                'status': 'unknown',
                'error': 'psutil not available'
            }
        except Exception as e:
            health_status['components']['system_resources'] = {
                'status': 'unhealthy',
                'error': str(e)
            }
            health_status['status'] = 'degraded'
        
        # Determine overall status
        component_statuses = [comp['status'] for comp in health_status['components'].values()]
        if 'unhealthy' in component_statuses:
            health_status['status'] = 'unhealthy'
        elif 'degraded' in component_statuses:
            health_status['status'] = 'degraded'
        
        status_code = 200 if health_status['status'] == 'healthy' else 503
        return jsonify(health_status), status_code
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return jsonify({
            'status': 'unhealthy',
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'error': str(e)
        }), 503


@monitoring_api_bp.route('/errors', methods=['GET'])
def get_error_statistics():
    """
    Get error statistics and patterns
    
    Query parameters:
        - session_id: Filter by session ID
        - hours: Time window in hours (default: 24)
        - category: Filter by error category
        - severity: Filter by error severity
        
    Returns:
        JSON response with error statistics
    """
    try:
        # Get query parameters
        session_id = request.args.get('session_id')
        hours = int(request.args.get('hours', 24))
        category = request.args.get('category')
        severity = request.args.get('severity')
        
        error_handler = get_error_handler()
        
        # Get cached error statistics
        error_stats = error_handler.get_error_statistics()
        
        # Get session-specific errors if requested
        session_errors = []
        if session_id:
            session_errors = error_handler.get_session_errors(session_id, limit=50)
        
        # Get database errors for time window
        database_errors = []
        try:
            supabase = get_supabase_client()
            
            # Build query
            query = supabase.table('training_errors').select('*')
            
            # Filter by time window
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
            query = query.gte('timestamp', cutoff_time.isoformat())
            
            # Apply filters
            if session_id:
                query = query.eq('session_id', session_id)
            if category:
                query = query.eq('error_category', category)
            if severity:
                query = query.eq('error_severity', severity)
            
            # Execute query
            response = query.order('timestamp', desc=True).limit(100).execute()
            database_errors = response.data or []
            
        except Exception as db_error:
            logger.warning(f"Failed to fetch database errors: {str(db_error)}")
        
        # Prepare response
        response_data = {
            'time_window_hours': hours,
            'cached_statistics': error_stats,
            'session_errors': session_errors if session_id else [],
            'recent_database_errors': database_errors,
            'error_categories': [category.value for category in ErrorCategory],
            'error_severities': [severity.value for severity in ErrorSeverity],
            'filters_applied': {
                'session_id': session_id,
                'category': category,
                'severity': severity
            },
            'summary': {
                'total_cached_errors': error_stats.get('total_cached_errors', 0),
                'total_sessions_with_errors': error_stats.get('total_sessions_with_errors', 0),
                'recent_database_errors': len(database_errors),
                'error_patterns_detected': len(error_stats.get('error_patterns', {}))
            }
        }
        
        return jsonify(response_data), 200
        
    except Exception as e:
        logger.error(f"Error retrieving error statistics: {str(e)}")
        return jsonify({
            'error': 'Failed to retrieve error statistics',
            'message': str(e)
        }), 500


@monitoring_api_bp.route('/performance', methods=['GET'])
def get_performance_metrics():
    """
    Get performance metrics and statistics
    
    Query parameters:
        - operation: Filter by operation name
        - session_id: Filter by session ID
        
    Returns:
        JSON response with performance metrics
    """
    try:
        operation = request.args.get('operation')
        session_id = request.args.get('session_id')
        
        performance_logger = get_performance_logger()
        
        # Get operation metrics
        if operation:
            operation_metrics = performance_logger.get_operation_metrics(operation)
            if not operation_metrics:
                return jsonify({
                    'error': 'Operation not found',
                    'available_operations': list(performance_logger.get_operation_metrics().keys())
                }), 404
            
            response_data = {
                'operation': operation,
                'metrics': operation_metrics
            }
        else:
            # Get all metrics
            all_metrics = performance_logger.get_operation_metrics()
            
            # Calculate summary statistics
            summary = {
                'total_operations': len(all_metrics),
                'total_calls': sum(m.get('total_calls', 0) for m in all_metrics.values()),
                'total_successful_calls': sum(m.get('successful_calls', 0) for m in all_metrics.values()),
                'total_failed_calls': sum(m.get('failed_calls', 0) for m in all_metrics.values()),
                'average_success_rate': 0,
                'slowest_operation': None,
                'fastest_operation': None,
                'most_memory_intensive': None
            }
            
            if summary['total_calls'] > 0:
                summary['average_success_rate'] = (summary['total_successful_calls'] / summary['total_calls']) * 100
            
            # Find extremes
            slowest_duration = 0
            fastest_duration = float('inf')
            highest_memory = 0
            
            for op_name, metrics in all_metrics.items():
                avg_duration = metrics.get('avg_duration', 0)
                max_memory = metrics.get('max_memory', 0)
                
                if avg_duration > slowest_duration:
                    slowest_duration = avg_duration
                    summary['slowest_operation'] = {'name': op_name, 'avg_duration': avg_duration}
                
                if avg_duration < fastest_duration and avg_duration > 0:
                    fastest_duration = avg_duration
                    summary['fastest_operation'] = {'name': op_name, 'avg_duration': avg_duration}
                
                if max_memory > highest_memory:
                    highest_memory = max_memory
                    summary['most_memory_intensive'] = {'name': op_name, 'max_memory': max_memory}
            
            response_data = {
                'summary': summary,
                'operations': all_metrics
            }
        
        return jsonify(response_data), 200
        
    except Exception as e:
        logger.error(f"Error retrieving performance metrics: {str(e)}")
        return jsonify({
            'error': 'Failed to retrieve performance metrics',
            'message': str(e)
        }), 500


@monitoring_api_bp.route('/logs', methods=['GET'])
def get_system_logs():
    """
    Get system logs from database
    
    Query parameters:
        - session_id: Filter by session ID
        - level: Filter by log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        - hours: Time window in hours (default: 24)
        - operation: Filter by operation
        - limit: Maximum number of logs to return (default: 100, max: 1000)
        
    Returns:
        JSON response with system logs
    """
    try:
        # Get query parameters
        session_id = request.args.get('session_id')
        level = request.args.get('level', '').upper()
        hours = int(request.args.get('hours', 24))
        operation = request.args.get('operation')
        limit = min(int(request.args.get('limit', 100)), 1000)
        
        supabase = get_supabase_client()
        
        # Build query
        query = supabase.table('training_logs').select('*')
        
        # Filter by time window
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
        query = query.gte('timestamp', cutoff_time.isoformat())
        
        # Apply filters
        if session_id:
            query = query.eq('session_id', session_id)
        if level and level in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']:
            query = query.eq('level', level)
        if operation:
            query = query.eq('operation', operation)
        
        # Execute query
        response = query.order('timestamp', desc=True).limit(limit).execute()
        logs = response.data or []
        
        # Get log statistics
        stats_query = supabase.table('training_logs').select('level', options={'count': 'exact'})
        stats_query = stats_query.gte('timestamp', cutoff_time.isoformat())
        if session_id:
            stats_query = stats_query.eq('session_id', session_id)
        
        # Count by log level
        level_counts = {}
        for log_level in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']:
            level_response = stats_query.eq('level', log_level).execute()
            level_counts[log_level] = level_response.count or 0
        
        response_data = {
            'time_window_hours': hours,
            'total_logs_returned': len(logs),
            'logs': logs,
            'statistics': {
                'level_distribution': level_counts,
                'total_logs_in_window': sum(level_counts.values())
            },
            'filters_applied': {
                'session_id': session_id,
                'level': level if level else None,
                'operation': operation,
                'limit': limit
            }
        }
        
        return jsonify(response_data), 200
        
    except Exception as e:
        logger.error(f"Error retrieving system logs: {str(e)}")
        return jsonify({
            'error': 'Failed to retrieve system logs',
            'message': str(e)
        }), 500


@monitoring_api_bp.route('/sessions/<session_id>/health', methods=['GET'])
def get_session_health(session_id: str):
    """
    Get health status for a specific session
    
    Args:
        session_id: Session identifier
        
    Returns:
        JSON response with session health status
    """
    try:
        error_handler = get_error_handler()
        
        # Get session errors
        session_errors = error_handler.get_session_errors(session_id, limit=20)
        
        # Get recent database logs for session
        supabase = get_supabase_client()
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=24)
        
        logs_response = supabase.table('training_logs').select('*').eq('session_id', session_id).gte('timestamp', cutoff_time.isoformat()).order('timestamp', desc=True).limit(50).execute()
        recent_logs = logs_response.data or []
        
        # Count log levels
        log_counts = {'DEBUG': 0, 'INFO': 0, 'WARNING': 0, 'ERROR': 0, 'CRITICAL': 0}
        for log in recent_logs:
            level = log.get('level', 'INFO')
            if level in log_counts:
                log_counts[level] += 1
        
        # Get session training status
        training_status = 'unknown'
        try:
            training_response = supabase.table('training_results').select('status').eq('session_id', session_id).order('created_at', desc=True).limit(1).execute()
            if training_response.data:
                training_status = training_response.data[0].get('status', 'unknown')
        except Exception:
            pass
        
        # Determine health status
        health_status = 'healthy'
        health_issues = []
        
        # Check for errors
        error_count = len(session_errors)
        critical_errors = sum(1 for error in session_errors 
                            if error.get('severity') == 'critical')
        
        if critical_errors > 0:
            health_status = 'critical'
            health_issues.append(f"{critical_errors} critical errors detected")
        elif error_count > 5:
            health_status = 'degraded'
            health_issues.append(f"{error_count} errors in session")
        elif log_counts['ERROR'] > 10:
            health_status = 'degraded'
            health_issues.append(f"{log_counts['ERROR']} error logs in last 24 hours")
        
        # Check training status
        if training_status in ['failed', 'error']:
            health_status = 'unhealthy'
            health_issues.append(f"Training status: {training_status}")
        
        response_data = {
            'session_id': session_id,
            'health_status': health_status,
            'health_issues': health_issues,
            'training_status': training_status,
            'error_summary': {
                'total_errors': error_count,
                'critical_errors': critical_errors,
                'recent_errors': session_errors[:5]  # Last 5 errors
            },
            'log_summary': {
                'total_logs_24h': len(recent_logs),
                'level_distribution': log_counts,
                'recent_logs': recent_logs[:10]  # Last 10 logs
            },
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        return jsonify(response_data), 200
        
    except Exception as e:
        logger.error(f"Error retrieving session health for {session_id}: {str(e)}")
        return jsonify({
            'error': 'Failed to retrieve session health',
            'message': str(e),
            'session_id': session_id
        }), 500


@monitoring_api_bp.route('/sessions/<session_id>/errors/clear', methods=['POST'])
def clear_session_errors(session_id: str):
    """
    Clear cached errors for a session
    
    Args:
        session_id: Session identifier
        
    Returns:
        JSON response with operation status
    """
    try:
        error_handler = get_error_handler()
        error_handler.clear_session_errors(session_id)
        
        return jsonify({
            'success': True,
            'message': f'Cleared cached errors for session {session_id}',
            'session_id': session_id,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Error clearing session errors for {session_id}: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Failed to clear session errors',
            'message': str(e),
            'session_id': session_id
        }), 500


@monitoring_api_bp.route('/system/status', methods=['GET'])
def get_system_status():
    """
    Get comprehensive system status
    
    Returns:
        JSON response with system status overview
    """
    try:
        error_handler = get_error_handler()
        performance_logger = get_performance_logger()
        
        # Get error statistics
        error_stats = error_handler.get_error_statistics()
        
        # Get performance summary
        performance_stats = performance_logger.get_operation_metrics()
        
        # Calculate system metrics
        total_operations = len(performance_stats)
        total_calls = sum(m.get('total_calls', 0) for m in performance_stats.values())
        total_errors = error_stats.get('total_cached_errors', 0)
        
        # Determine system health
        system_health = 'healthy'
        if total_errors > 50:
            system_health = 'degraded'
        
        critical_error_count = sum(1 for severity_dist in error_stats.get('severity_distribution', {}).items() 
                                 if severity_dist[0] == 'critical')
        if critical_error_count > 0:
            system_health = 'critical'
        
        response_data = {
            'system_health': system_health,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'overview': {
                'total_operations_monitored': total_operations,
                'total_operation_calls': total_calls,
                'total_cached_errors': total_errors,
                'sessions_with_errors': error_stats.get('total_sessions_with_errors', 0),
                'error_patterns_detected': len(error_stats.get('error_patterns', {}))
            },
            'error_distribution': error_stats.get('severity_distribution', {}),
            'category_distribution': error_stats.get('category_distribution', {}),
            'top_operations': {
                name: {
                    'total_calls': metrics.get('total_calls', 0),
                    'success_rate': (metrics.get('successful_calls', 0) / metrics.get('total_calls', 1)) * 100,
                    'avg_duration': metrics.get('avg_duration', 0)
                }
                for name, metrics in sorted(
                    performance_stats.items(), 
                    key=lambda x: x[1].get('total_calls', 0), 
                    reverse=True
                )[:10]
            }
        }
        
        return jsonify(response_data), 200
        
    except Exception as e:
        logger.error(f"Error retrieving system status: {str(e)}")
        return jsonify({
            'error': 'Failed to retrieve system status',
            'message': str(e),
            'system_health': 'unknown'
        }), 500