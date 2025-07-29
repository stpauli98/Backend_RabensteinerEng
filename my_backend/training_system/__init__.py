"""
Training System Package Initialization
Sets up comprehensive error handling, logging, and monitoring
"""

import os
import logging
from typing import Dict

# Initialize comprehensive logging first
from .logging_config import setup_training_system_logging, cleanup_logging, get_performance_logger

# Initialize error handling
from .error_handler import get_error_handler, shutdown_error_handler

# Package version
__version__ = "1.0.0"

# Global configuration
_initialized = False
_loggers = {}


def initialize_training_system(
    log_level: str = None,
    log_to_database: bool = None,
    log_to_file: bool = None,
    log_directory: str = None,
    enable_performance_monitoring: bool = True
) -> Dict:
    """
    Initialize the training system with comprehensive error handling and logging
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_to_database: Enable database logging
        log_to_file: Enable file logging
        log_directory: Directory for log files
        enable_performance_monitoring: Enable performance monitoring
        
    Returns:
        Dict containing initialization status and configured components
    """
    global _initialized, _loggers
    
    if _initialized:
        return {
            'status': 'already_initialized',
            'loggers': list(_loggers.keys()),
            'error_handler': 'active',
            'performance_monitoring': 'active' if enable_performance_monitoring else 'disabled'
        }
    
    try:
        # Get configuration from environment or use defaults
        log_level = log_level or os.getenv('TRAINING_LOG_LEVEL', 'INFO')
        log_to_database = log_to_database if log_to_database is not None else os.getenv('TRAINING_LOG_TO_DB', 'true').lower() == 'true'
        log_to_file = log_to_file if log_to_file is not None else os.getenv('TRAINING_LOG_TO_FILE', 'true').lower() == 'true'
        log_directory = log_directory or os.getenv('TRAINING_LOG_DIR')
        
        # Setup comprehensive logging
        _loggers = setup_training_system_logging(
            log_level=log_level,
            log_to_file=log_to_file,
            log_to_database=log_to_database,
            log_to_console=True,
            log_directory=log_directory,
            structured_logging=True
        )
        
        # Initialize error handler
        error_handler = get_error_handler(enable_database_logging=log_to_database)
        
        # Initialize performance monitoring if enabled
        performance_logger = None
        if enable_performance_monitoring:
            performance_logger = get_performance_logger()
        
        # Log successful initialization
        root_logger = _loggers['root']
        root_logger.info(f"Training system initialized successfully v{__version__}")
        root_logger.info(f"Configuration: log_level={log_level}, database_logging={log_to_database}, "
                        f"file_logging={log_to_file}, performance_monitoring={enable_performance_monitoring}")
        
        _initialized = True
        
        return {
            'status': 'initialized',
            'version': __version__,
            'configuration': {
                'log_level': log_level,
                'log_to_database': log_to_database,
                'log_to_file': log_to_file,
                'log_directory': log_directory,
                'performance_monitoring': enable_performance_monitoring
            },
            'components': {
                'loggers': list(_loggers.keys()),
                'error_handler': 'active',
                'performance_monitoring': 'active' if enable_performance_monitoring else 'disabled'
            }
        }
        
    except Exception as e:
        # Fallback logging if initialization fails
        print(f"Failed to initialize training system: {e}")
        return {
            'status': 'failed',
            'error': str(e),
            'fallback_logging': 'active'
        }


def shutdown_training_system():
    """
    Shutdown the training system and cleanup resources
    """
    global _initialized, _loggers
    
    if not _initialized:
        return
    
    try:
        # Log shutdown
        if _loggers and 'root' in _loggers:
            _loggers['root'].info("Shutting down training system...")
        
        # Shutdown error handler
        shutdown_error_handler()
        
        # Cleanup logging
        cleanup_logging()
        
        _initialized = False
        _loggers = {}
        
    except Exception as e:
        print(f"Error during training system shutdown: {e}")


def get_system_status() -> Dict:
    """
    Get current system status
    
    Returns:
        Dict containing system status information
    """
    try:
        if not _initialized:
            return {
                'status': 'not_initialized',
                'version': __version__
            }
        
        # Get error handler statistics
        error_handler = get_error_handler()
        error_stats = error_handler.get_error_statistics()
        
        # Get performance statistics
        performance_logger = get_performance_logger()
        performance_stats = performance_logger.get_operation_metrics()
        
        return {
            'status': 'active',
            'version': __version__,
            'initialized': _initialized,
            'components': {
                'loggers_active': len(_loggers),
                'error_handler': 'active',
                'performance_monitoring': 'active'
            },
            'statistics': {
                'total_cached_errors': error_stats.get('total_cached_errors', 0),
                'sessions_with_errors': error_stats.get('total_sessions_with_errors', 0),
                'monitored_operations': len(performance_stats),
                'total_operation_calls': sum(m.get('total_calls', 0) for m in performance_stats.values())
            }
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'version': __version__,
            'error': str(e)
        }


# Auto-initialize on import if environment variable is set
if os.getenv('TRAINING_AUTO_INIT', 'false').lower() == 'true':
    initialize_training_system()


# Import and expose key components
from .error_handler import (
    ErrorHandler,
    TrainingSystemError,
    DataProcessingError,
    ModelTrainingError,
    ParameterValidationError,
    DatabaseError,
    VisualizationError,
    ErrorCategory,
    ErrorSeverity,
    error_handler_decorator
)

from .logging_config import (
    StructuredFormatter,
    DatabaseLogHandler,
    PerformanceLogger,
    get_session_logger,
    performance_monitor
)

# Export public API
__all__ = [
    # Initialization
    'initialize_training_system',
    'shutdown_training_system',
    'get_system_status',
    
    # Error handling
    'ErrorHandler',
    'TrainingSystemError',
    'DataProcessingError', 
    'ModelTrainingError',
    'ParameterValidationError',
    'DatabaseError',
    'VisualizationError',
    'ErrorCategory',
    'ErrorSeverity',
    'error_handler_decorator',
    'get_error_handler',
    
    # Logging
    'StructuredFormatter',
    'DatabaseLogHandler', 
    'PerformanceLogger',
    'get_session_logger',
    'get_performance_logger',
    'performance_monitor',
    
    # Version
    '__version__'
]