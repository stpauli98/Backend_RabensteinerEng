"""
System API - Consolidated System Operations and Utilities

Consolidates system-level functionality including:
- Health checks and system monitoring
- Session management across all domains
- File cleanup and maintenance operations
- System status and diagnostics

API Endpoints:
    GET  /api/system/health - System health check
    GET  /api/system/status - Comprehensive system status
    GET  /api/system/sessions - List all active sessions
    POST /api/system/cleanup - Manual cleanup operations
    GET  /api/system/storage - Storage usage information
    GET  /api/system/diagnostics - System diagnostics
    POST /api/system/maintenance - Maintenance operations
"""

import os
import sys
import tempfile
import logging
import json
import time
import psutil
import threading
from datetime import datetime
from typing import Dict, Any, List, Optional
from threading import Lock

import pandas as pd
from flask import Blueprint, request, jsonify, current_app

# Import centralized storage and scheduler services
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.storage_config import storage_config
from services.scheduler import scheduler_service

# ============================================================================
# CONFIGURATION
# ============================================================================

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create Blueprint
bp = Blueprint('system', __name__)

# Helper function to get socketio instance
def get_socketio():
    return current_app.extensions.get('socketio')

# Helper function to get supabase client
def get_supabase_client():
    try:
        from services.supabase_client import get_supabase_client
        return get_supabase_client()
    except ImportError:
        logger.warning("Supabase client not available")
        return None

# Thread-safe storage for system metrics
system_metrics: Dict[str, Any] = {}
storage_lock = Lock()

# Configuration constants
SYSTEM_CHECK_INTERVAL = 300  # 5 minutes
MAX_LOG_ENTRIES = 1000

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_directory_size(path: str) -> int:
    """Calculate total size of a directory in bytes"""
    total_size = 0
    try:
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in filenames:
                file_path = os.path.join(dirpath, filename)
                if os.path.exists(file_path):
                    total_size += os.path.getsize(file_path)
    except (OSError, IOError) as e:
        logger.warning(f"Error calculating directory size for {path}: {e}")
    return total_size

def cleanup_directory(directory: str, max_age_hours: int = 24) -> Dict[str, Any]:
    """Cleanup old files in a directory"""
    cleanup_result = {
        'directory': directory,
        'files_removed': 0,
        'bytes_freed': 0,
        'errors': []
    }
    
    if not os.path.exists(directory):
        return cleanup_result
    
    current_time = time.time()
    max_age_seconds = max_age_hours * 3600
    
    try:
        for root, dirs, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    file_stat = os.stat(file_path)
                    if current_time - file_stat.st_mtime > max_age_seconds:
                        file_size = file_stat.st_size
                        os.remove(file_path)
                        cleanup_result['files_removed'] += 1
                        cleanup_result['bytes_freed'] += file_size
                except (OSError, IOError) as e:
                    cleanup_result['errors'].append(f"Error removing {file_path}: {str(e)}")
            
            # Remove empty directories
            for dir_name in dirs:
                dir_path = os.path.join(root, dir_name)
                try:
                    if not os.listdir(dir_path):  # Directory is empty
                        os.rmdir(dir_path)
                except (OSError, IOError):
                    pass  # Ignore errors removing directories
                    
    except Exception as e:
        cleanup_result['errors'].append(f"General cleanup error: {str(e)}")
    
    return cleanup_result

def get_system_info() -> Dict[str, Any]:
    """Get comprehensive system information"""
    try:
        # Basic system info
        info = {
            'timestamp': datetime.now().isoformat(),
            'uptime': time.time() - psutil.boot_time(),
            'cpu': {
                'percent': psutil.cpu_percent(interval=1),
                'cores': psutil.cpu_count(),
                'frequency': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None
            },
            'memory': {
                'total': psutil.virtual_memory().total,
                'available': psutil.virtual_memory().available,
                'percent': psutil.virtual_memory().percent,
                'used': psutil.virtual_memory().used
            },
            'disk': {
                'total': psutil.disk_usage('/').total,
                'used': psutil.disk_usage('/').used,
                'free': psutil.disk_usage('/').free,
                'percent': psutil.disk_usage('/').percent
            }
        }
        
        # Network info if available
        try:
            net_io = psutil.net_io_counters()
            info['network'] = {
                'bytes_sent': net_io.bytes_sent,
                'bytes_recv': net_io.bytes_recv,
                'packets_sent': net_io.packets_sent,
                'packets_recv': net_io.packets_recv
            }
        except Exception:
            info['network'] = None
        
        return info
        
    except Exception as e:
        logger.error(f"Error getting system info: {e}")
        return {
            'timestamp': datetime.now().isoformat(),
            'error': str(e)
        }

def collect_storage_info() -> Dict[str, Any]:
    """Collect storage usage information using centralized storage config"""
    try:
        # Get centralized storage statistics
        storage_stats = storage_config.get_storage_stats()
        
        # Convert to expected format
        storage_info = {}
        
        # Map directory statistics
        for dir_name, size_bytes in storage_stats.get('directories', {}).items():
            storage_info[dir_name] = {
                'path': str(getattr(storage_config, f'{dir_name}_dir', dir_name)),
                'exists': True,
                'size_bytes': size_bytes,
                'file_count': storage_stats.get('file_counts', {}).get(dir_name, 0),
                'oldest_file': storage_stats.get('oldest_files', {}).get(dir_name)
            }
        
        # Add legacy directories for backward compatibility
        legacy_dirs = {
            'uploads': 'uploads/',
            'temp_uploads': 'temp_uploads/',
            'chunk_uploads': 'chunk_uploads/',
            'temp_training_data': 'temp_training_data/',
            'models': 'models/'
        }
        
        for name, path in legacy_dirs.items():
            if name not in storage_info and os.path.exists(path):
                storage_info[name] = {
                    'path': path,
                    'size_bytes': get_directory_size(path),
                    'exists': True
                }
        
        return storage_info
        
    except Exception as e:
        logger.error(f"Error collecting storage info: {e}")
        return {'error': str(e)}

# ============================================================================
# HEALTH CHECK ENDPOINTS
# ============================================================================

@bp.route('/health', methods=['GET'])
def health_check():
    """Basic health check endpoint"""
    try:
        # Basic checks
        health_status = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'version': '1.0.0',
            'uptime': time.time(),
            'services': {}
        }
        
        # Check SocketIO
        socketio = get_socketio()
        health_status['services']['socketio'] = {
            'status': 'available' if socketio else 'unavailable'
        }
        
        # Check Supabase
        supabase = get_supabase_client()
        health_status['services']['database'] = {
            'status': 'available' if supabase else 'unavailable'
        }
        
        # Check essential directories
        essential_dirs = ['uploads', 'temp_uploads', 'chunk_uploads']
        health_status['services']['storage'] = {
            'status': 'healthy',
            'directories': {}
        }
        
        for dir_name in essential_dirs:
            exists = os.path.exists(dir_name)
            health_status['services']['storage']['directories'][dir_name] = {
                'exists': exists,
                'writable': os.access(dir_name, os.W_OK) if exists else False
            }
            if not exists:
                health_status['services']['storage']['status'] = 'degraded'
        
        return jsonify(health_status), 200
        
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return jsonify({
            'status': 'unhealthy',
            'timestamp': datetime.now().isoformat(),
            'error': str(e)
        }), 500

@bp.route('/status', methods=['GET'])
def system_status():
    """Comprehensive system status"""
    try:
        status = {
            'system': get_system_info(),
            'storage': collect_storage_info(),
            'services': {
                'socketio': get_socketio() is not None,
                'database': get_supabase_client() is not None
            },
            'application': {
                'name': 'Data Processing Backend',
                'version': '1.0.0',
                'environment': os.environ.get('FLASK_ENV', 'production'),
                'port': os.environ.get('PORT', '8080')
            }
        }
        
        return jsonify({
            'status': 'success',
            'data': status
        })
        
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        return jsonify({'error': str(e)}), 500

@bp.route('/diagnostics', methods=['GET'])
def system_diagnostics():
    """Detailed system diagnostics"""
    try:
        diagnostics = {
            'timestamp': datetime.now().isoformat(),
            'system_health': {},
            'storage_analysis': {},
            'service_status': {},
            'performance_metrics': {},
            'recommendations': []
        }
        
        # System health
        system_info = get_system_info()
        diagnostics['system_health'] = system_info
        
        # Storage analysis
        storage_info = collect_storage_info()
        diagnostics['storage_analysis'] = storage_info
        
        total_storage_mb = sum(info['size_bytes'] for info in storage_info.values()) / 1024 / 1024
        diagnostics['storage_analysis']['total_size_mb'] = round(total_storage_mb, 2)
        
        # Service status
        diagnostics['service_status'] = {
            'socketio': get_socketio() is not None,
            'database': get_supabase_client() is not None,
            'flask': True  # If we're responding, Flask is working
        }
        
        # Performance metrics
        if 'cpu' in system_info:
            cpu_percent = system_info['cpu']['percent']
            memory_percent = system_info['memory']['percent']
            disk_percent = system_info['disk']['percent']
            
            diagnostics['performance_metrics'] = {
                'cpu_usage': cpu_percent,
                'memory_usage': memory_percent,
                'disk_usage': disk_percent,
                'overall_health': 'good' if all([
                    cpu_percent < 80,
                    memory_percent < 85,
                    disk_percent < 90
                ]) else 'warning'
            }
            
            # Generate recommendations
            if cpu_percent > 80:
                diagnostics['recommendations'].append('High CPU usage detected. Consider optimizing background tasks.')
            if memory_percent > 85:
                diagnostics['recommendations'].append('High memory usage detected. Consider implementing memory cleanup.')
            if disk_percent > 90:
                diagnostics['recommendations'].append('Disk space is running low. Consider running cleanup operations.')
            if total_storage_mb > 1000:  # 1GB
                diagnostics['recommendations'].append('Application storage usage is high. Consider running cleanup operations.')
        
        return jsonify({
            'status': 'success',
            'diagnostics': diagnostics
        })
        
    except Exception as e:
        logger.error(f"Error getting diagnostics: {e}")
        return jsonify({'error': str(e)}), 500

# ============================================================================
# SESSION MANAGEMENT ENDPOINTS
# ============================================================================

@bp.route('/sessions', methods=['GET'])
def list_all_sessions():
    """List all active sessions across all services"""
    try:
        all_sessions = {
            'data_pipeline': [],
            'analytics': [],
            'machine_learning': [],
            'total_count': 0
        }
        
        # Try to get sessions from each domain
        try:
            from api.data_pipeline import chunk_storage as data_chunk_storage
            for session_id, info in data_chunk_storage.items():
                all_sessions['data_pipeline'].append({
                    'session_id': session_id,
                    'type': 'data_processing',
                    'filename': info.get('filename', 'unknown'),
                    'last_activity': info.get('last_activity', 0),
                    'status': 'active'
                })
        except ImportError:
            pass
        
        try:
            from api.analytics import chunk_uploads as analytics_chunk_uploads
            for session_id, info in analytics_chunk_uploads.items():
                all_sessions['analytics'].append({
                    'session_id': session_id,
                    'type': 'analytics',
                    'filename': info.get('filename', 'unknown'),
                    'last_activity': info.get('last_activity', 0),
                    'status': 'active'
                })
        except ImportError:
            pass
        
        try:
            from api.machine_learning import training_sessions
            for session_id, info in training_sessions.items():
                all_sessions['machine_learning'].append({
                    'session_id': session_id,
                    'type': 'training',
                    'name': info.get('name', session_id),
                    'status': info.get('status', 'unknown'),
                    'last_activity': info.get('last_activity', 0),
                    'uploaded_files': len(info.get('uploaded_files', []))
                })
        except ImportError:
            pass
        
        # Calculate totals
        all_sessions['total_count'] = (
            len(all_sessions['data_pipeline']) +
            len(all_sessions['analytics']) +
            len(all_sessions['machine_learning'])
        )
        
        return jsonify({
            'status': 'success',
            'sessions': all_sessions
        })
        
    except Exception as e:
        logger.error(f"Error listing sessions: {e}")
        return jsonify({'error': str(e)}), 500

# ============================================================================
# MAINTENANCE ENDPOINTS
# ============================================================================

@bp.route('/cleanup', methods=['POST'])
def manual_cleanup():
    """Manual cleanup operations"""
    try:
        data = request.get_json() or {}
        cleanup_config = data.get('config', {})
        
        # Default cleanup configuration
        default_config = {
            'temp_uploads': True,
            'chunk_uploads': True,
            'old_logs': True,
            'temp_training_data': True,
            'max_age_hours': 24
        }
        
        config = {**default_config, **cleanup_config}
        max_age_hours = config['max_age_hours']
        
        cleanup_results = {
            'started_at': datetime.now().isoformat(),
            'operations': [],
            'total_files_removed': 0,
            'total_bytes_freed': 0,
            'errors': []
        }
        
        # Cleanup directories based on configuration
        directories_to_clean = []
        
        if config['temp_uploads']:
            directories_to_clean.append('temp_uploads')
        if config['chunk_uploads']:
            directories_to_clean.append('chunk_uploads')
        if config['temp_training_data']:
            directories_to_clean.append('temp_training_data')
        if config['old_logs']:
            directories_to_clean.append('logs')
        
        # Perform cleanup
        for directory in directories_to_clean:
            if os.path.exists(directory):
                result = cleanup_directory(directory, max_age_hours)
                cleanup_results['operations'].append(result)
                cleanup_results['total_files_removed'] += result['files_removed']
                cleanup_results['total_bytes_freed'] += result['bytes_freed']
                cleanup_results['errors'].extend(result['errors'])
        
        # Additional cleanup: expired sessions
        try:
            from api.data_pipeline import cleanup_old_uploads
            cleanup_old_uploads()
            cleanup_results['operations'].append({
                'operation': 'data_pipeline_session_cleanup',
                'status': 'completed'
            })
        except ImportError:
            pass
        
        try:
            from api.machine_learning import cleanup_old_sessions
            cleanup_old_sessions()
            cleanup_results['operations'].append({
                'operation': 'ml_session_cleanup',
                'status': 'completed'
            })
        except ImportError:
            pass
        
        cleanup_results['completed_at'] = datetime.now().isoformat()
        cleanup_results['success'] = len(cleanup_results['errors']) == 0
        
        logger.info(f"Manual cleanup completed: {cleanup_results['total_files_removed']} files, {cleanup_results['total_bytes_freed']} bytes freed")
        
        return jsonify({
            'status': 'success',
            'cleanup_results': cleanup_results,
            'message': f'Cleanup completed: {cleanup_results["total_files_removed"]} files removed, {cleanup_results["total_bytes_freed"]} bytes freed'
        })
        
    except Exception as e:
        logger.error(f"Error in manual cleanup: {e}")
        return jsonify({'error': str(e)}), 500

@bp.route('/storage', methods=['GET'])
def storage_info():
    """Get detailed storage usage information"""
    try:
        storage_data = collect_storage_info()
        
        # Calculate totals and add analysis
        total_bytes = sum(info['size_bytes'] for info in storage_data.values())
        
        storage_analysis = {
            'directories': storage_data,
            'summary': {
                'total_bytes': total_bytes,
                'total_mb': round(total_bytes / 1024 / 1024, 2),
                'total_gb': round(total_bytes / 1024 / 1024 / 1024, 3),
                'largest_directory': max(storage_data.items(), key=lambda x: x[1]['size_bytes'])[0] if storage_data else None
            },
            'recommendations': []
        }
        
        # Generate storage recommendations
        for name, info in storage_data.items():
            size_mb = info['size_bytes'] / 1024 / 1024
            if size_mb > 100:  # More than 100MB
                storage_analysis['recommendations'].append(
                    f"Directory '{name}' is using {round(size_mb, 1)}MB - consider cleanup"
                )
        
        if total_bytes > 1024 * 1024 * 1024:  # More than 1GB
            storage_analysis['recommendations'].append(
                "Total storage usage exceeds 1GB - consider running cleanup operations"
            )
        
        return jsonify({
            'status': 'success',
            'storage': storage_analysis
        })
        
    except Exception as e:
        logger.error(f"Error getting storage info: {e}")
        return jsonify({'error': str(e)}), 500

@bp.route('/maintenance', methods=['POST'])
def maintenance_operations():
    """Perform various maintenance operations"""
    try:
        data = request.get_json() or {}
        operations = data.get('operations', [])
        
        if not operations:
            return jsonify({'error': 'No operations specified'}), 400
        
        results = {
            'started_at': datetime.now().isoformat(),
            'operations': {},
            'success': True,
            'errors': []
        }
        
        # Available maintenance operations
        available_ops = {
            'cleanup_temp_files': lambda: cleanup_directory('temp_uploads', 1),
            'cleanup_chunks': lambda: cleanup_directory('chunk_uploads', 1),
            'cleanup_logs': lambda: cleanup_directory('logs', 168),  # 1 week
            'system_check': get_system_info,
            'storage_analysis': collect_storage_info
        }
        
        # Execute requested operations
        for operation in operations:
            if operation in available_ops:
                try:
                    result = available_ops[operation]()
                    results['operations'][operation] = {
                        'status': 'completed',
                        'result': result
                    }
                except Exception as e:
                    results['operations'][operation] = {
                        'status': 'failed',
                        'error': str(e)
                    }
                    results['errors'].append(f"Operation '{operation}' failed: {str(e)}")
                    results['success'] = False
            else:
                results['operations'][operation] = {
                    'status': 'skipped',
                    'error': f"Unknown operation: {operation}"
                }
                results['errors'].append(f"Unknown operation: {operation}")
        
        results['completed_at'] = datetime.now().isoformat()
        
        return jsonify({
            'status': 'success',
            'maintenance_results': results
        })
        
    except Exception as e:
        logger.error(f"Error in maintenance operations: {e}")
        return jsonify({'error': str(e)}), 500

# ============================================================================
# UTILITY ENDPOINTS
# ============================================================================

@bp.route('/info', methods=['GET'])
def system_info():
    """Get basic system information"""
    try:
        info = {
            'application': {
                'name': 'Data Processing Backend',
                'version': '1.0.0',
                'description': 'Consolidated API for data processing, analytics, and machine learning',
                'environment': os.environ.get('FLASK_ENV', 'production'),
                'port': os.environ.get('PORT', '8080')
            },
            'api_domains': {
                'data_pipeline': '/api/data/*',
                'analytics': '/api/analytics/*', 
                'machine_learning': '/api/ml/*',
                'system': '/api/system/*'
            },
            'features': [
                'Chunked file uploads',
                'Real-time progress tracking',
                'Data processing and cleaning',
                'Statistical analysis',
                'Data visualization',
                'Machine learning training',
                'Session management',
                'System monitoring'
            ],
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify({
            'status': 'success',
            'info': info
        })
        
    except Exception as e:
        logger.error(f"Error getting system info: {e}")
        return jsonify({'error': str(e)}), 500

# ============================================================================
# SCHEDULER & STORAGE MANAGEMENT ENDPOINTS
# ============================================================================

@bp.route('/scheduler', methods=['GET'])
def scheduler_status():
    """Get scheduler service status and job statistics"""
    try:
        scheduler_stats = scheduler_service.get_job_stats()
        
        return jsonify({
            'status': 'success',
            'scheduler': scheduler_stats,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting scheduler status: {e}")
        return jsonify({'error': str(e)}), 500

@bp.route('/scheduler/trigger/<job_id>', methods=['POST'])
def trigger_scheduled_job(job_id):
    """Manually trigger a scheduled job"""
    try:
        success = scheduler_service.trigger_job(job_id)
        
        if success:
            return jsonify({
                'status': 'success',
                'message': f'Job {job_id} triggered successfully',
                'timestamp': datetime.now().isoformat()
            })
        else:
            return jsonify({
                'status': 'error',
                'message': f'Job {job_id} not found or could not be triggered'
            }), 404
            
    except Exception as e:
        logger.error(f"Error triggering job {job_id}: {e}")
        return jsonify({'error': str(e)}), 500

@bp.route('/storage/migrate', methods=['POST'])
def storage_migration():
    """Initiate storage migration from legacy locations"""
    try:
        data = request.get_json() or {}
        dry_run = data.get('dry_run', True)
        
        # Import migration utility
        from utils.storage_migration import StorageMigrator
        
        migrator = StorageMigrator(dry_run=dry_run)
        
        # Generate migration plan
        plan = migrator.create_migration_plan()
        
        if not dry_run:
            # Execute migration
            results = migrator.execute_migration(plan)
            
            return jsonify({
                'status': 'success',
                'migration_plan': plan,
                'migration_results': results,
                'timestamp': datetime.now().isoformat()
            })
        else:
            return jsonify({
                'status': 'success',
                'migration_plan': plan,
                'message': 'Dry run completed - no files were moved',
                'timestamp': datetime.now().isoformat()
            })
            
    except Exception as e:
        logger.error(f"Error in storage migration: {e}")
        return jsonify({'error': str(e)}), 500

@bp.route('/storage/cleanup', methods=['POST'])
def storage_cleanup():
    """Manually trigger centralized storage cleanup"""
    try:
        data = request.get_json() or {}
        force = data.get('force', False)
        
        cleanup_stats = storage_config.cleanup_expired_files(force=force)
        
        total_removed = sum([
            cleanup_stats.get('temp_files_removed', 0),
            cleanup_stats.get('session_files_removed', 0),
            cleanup_stats.get('processed_files_removed', 0),
            cleanup_stats.get('cache_files_removed', 0)
        ])
        
        return jsonify({
            'status': 'success',
            'cleanup_stats': cleanup_stats,
            'total_files_removed': total_removed,
            'force_mode': force,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error in storage cleanup: {e}")
        return jsonify({'error': str(e)}), 500

@bp.route('/storage/health', methods=['GET'])
def storage_health():
    """Get detailed storage health and usage statistics"""
    try:
        # Get centralized storage statistics
        storage_stats = storage_config.get_storage_stats()
        
        # Add health recommendations
        recommendations = []
        total_size_gb = storage_stats.get('total_size', 0) / (1024**3)
        
        if storage_stats.get('storage_health') == 'critical':
            recommendations.append('Critical: Storage usage >90%. Immediate cleanup required.')
        elif storage_stats.get('storage_health') == 'warning':
            recommendations.append('Warning: Storage usage >75%. Consider cleanup.')
        
        # Check for old files
        oldest_files = storage_stats.get('oldest_files', {})
        current_time = time.time()
        
        for dir_name, oldest_time in oldest_files.items():
            if oldest_time and (current_time - oldest_time) > (7 * 24 * 3600):  # 7 days
                recommendations.append(f'{dir_name}: Contains files older than 7 days')
        
        return jsonify({
            'status': 'success',
            'storage_health': {
                'overall_status': storage_stats.get('storage_health', 'unknown'),
                'total_size_gb': round(total_size_gb, 2),
                'directories': storage_stats.get('directories', {}),
                'file_counts': storage_stats.get('file_counts', {}),
                'oldest_files': storage_stats.get('oldest_files', {}),
                'recommendations': recommendations
            },
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting storage health: {e}")
        return jsonify({'error': str(e)}), 500