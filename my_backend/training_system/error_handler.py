"""
Comprehensive error handling and logging module for training system
Provides centralized error handling, logging, and monitoring capabilities
"""

import logging
import traceback
import functools
import time
import threading
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Callable, Union
from enum import Enum
import json
import uuid
import sys
import os
from contextlib import contextmanager

# Import existing supabase client
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from supabase_client import get_supabase_client


class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error category types"""
    DATA_PROCESSING = "data_processing"
    MODEL_TRAINING = "model_training"
    PARAMETER_VALIDATION = "parameter_validation"
    DATABASE_OPERATION = "database_operation"
    FILE_OPERATION = "file_operation"
    VISUALIZATION = "visualization"
    API_REQUEST = "api_request"
    SYSTEM_ERROR = "system_error"
    CONFIGURATION = "configuration"
    MEMORY_ERROR = "memory_error"
    TIMEOUT_ERROR = "timeout_error"


class TrainingSystemError(Exception):
    """Base exception for training system errors"""
    
    def __init__(self, message: str, category: ErrorCategory = ErrorCategory.SYSTEM_ERROR, 
                 severity: ErrorSeverity = ErrorSeverity.MEDIUM, session_id: str = None,
                 error_code: str = None, details: Dict = None):
        super().__init__(message)
        self.message = message
        self.category = category
        self.severity = severity
        self.session_id = session_id
        self.error_code = error_code or self._generate_error_code()
        self.details = details or {}
        self.timestamp = datetime.now(timezone.utc).isoformat()
        
    def _generate_error_code(self) -> str:
        """Generate unique error code"""
        return f"TS-{self.category.value.upper()[:3]}-{uuid.uuid4().hex[:8].upper()}"
    
    def to_dict(self) -> Dict:
        """Convert error to dictionary"""
        return {
            'error_code': self.error_code,
            'message': self.message,
            'category': self.category.value,
            'severity': self.severity.value,
            'session_id': self.session_id,
            'timestamp': self.timestamp,
            'details': self.details,
            'traceback': traceback.format_exc()
        }


class DataProcessingError(TrainingSystemError):
    """Error in data processing operations"""
    def __init__(self, message: str, **kwargs):
        super().__init__(message, category=ErrorCategory.DATA_PROCESSING, **kwargs)


class ModelTrainingError(TrainingSystemError):
    """Error in model training operations"""
    def __init__(self, message: str, **kwargs):
        super().__init__(message, category=ErrorCategory.MODEL_TRAINING, **kwargs)


class ParameterValidationError(TrainingSystemError):
    """Error in parameter validation"""
    def __init__(self, message: str, **kwargs):
        super().__init__(message, category=ErrorCategory.PARAMETER_VALIDATION, **kwargs)


class DatabaseError(TrainingSystemError):
    """Error in database operations"""
    def __init__(self, message: str, **kwargs):
        super().__init__(message, category=ErrorCategory.DATABASE_OPERATION, **kwargs)


class VisualizationError(TrainingSystemError):
    """Error in visualization generation"""
    def __init__(self, message: str, **kwargs):
        super().__init__(message, category=ErrorCategory.VISUALIZATION, **kwargs)


class ErrorHandler:
    """
    Centralized error handling and logging system
    
    Features:
    - Structured error logging with metadata
    - Database error persistence
    - Error categorization and severity levels
    - Context-aware error handling
    - Performance monitoring
    - Error recovery suggestions
    """
    
    def __init__(self, supabase_client=None, enable_database_logging: bool = True):
        self.supabase = supabase_client or get_supabase_client()
        self.enable_database_logging = enable_database_logging
        self.logger = logging.getLogger(__name__)
        
        # Configure structured logging
        self._setup_structured_logging()
        
        # Error tracking
        self.error_cache = {}  # session_id -> List[errors]
        self.error_patterns = {}  # Track recurring error patterns
        self.performance_metrics = {}  # Track operation performance
        
        # Thread safety
        self._lock = threading.Lock()
        
        self.logger.info("ErrorHandler initialized with database logging enabled: %s", enable_database_logging)
    
    def _setup_structured_logging(self):
        """Setup structured logging configuration"""
        try:
            # Create custom formatter for structured logs
            formatter = logging.Formatter(
                '%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s'
            )
            
            # Get or create handler
            handler = None
            for h in self.logger.handlers:
                if isinstance(h, logging.StreamHandler):
                    handler = h
                    break
            
            if not handler:
                handler = logging.StreamHandler()
                self.logger.addHandler(handler)
            
            handler.setFormatter(formatter)
            
            # Set appropriate log level
            if not self.logger.level or self.logger.level == logging.NOTSET:
                self.logger.setLevel(logging.INFO)
                
        except Exception as e:
            print(f"Warning: Could not setup structured logging: {e}")
    
    def handle_error(self, error: Union[Exception, TrainingSystemError], 
                    session_id: str = None, context: Dict = None,
                    operation: str = None) -> Dict:
        """
        Handle error with comprehensive logging and tracking
        
        Args:
            error: Exception or TrainingSystemError to handle
            session_id: Session identifier for context
            context: Additional context information
            operation: Operation being performed when error occurred
            
        Returns:
            Dict containing error details and recovery suggestions
        """
        try:
            # Convert to TrainingSystemError if needed
            if not isinstance(error, TrainingSystemError):
                ts_error = TrainingSystemError(
                    message=str(error),
                    session_id=session_id,
                    details={
                        'original_exception': type(error).__name__,
                        'context': context or {},
                        'operation': operation
                    }
                )
            else:
                ts_error = error
                if session_id and not ts_error.session_id:
                    ts_error.session_id = session_id
                if context:
                    ts_error.details.update(context)
            
            # Log error with structured information
            self._log_error(ts_error, operation)
            
            # Cache error for session tracking
            self._cache_error(ts_error)
            
            # Save to database if enabled
            if self.enable_database_logging:
                self._save_error_to_database(ts_error)
            
            # Track error patterns
            self._track_error_patterns(ts_error)
            
            # Generate recovery suggestions
            recovery_suggestions = self._generate_recovery_suggestions(ts_error)
            
            # Prepare error response
            error_response = {
                'error_handled': True,
                'error_code': ts_error.error_code,
                'error_details': ts_error.to_dict(),
                'recovery_suggestions': recovery_suggestions,
                'session_id': ts_error.session_id,
                'timestamp': ts_error.timestamp
            }
            
            return error_response
            
        except Exception as handler_error:
            # Fallback error handling
            self.logger.error(f"Error in error handler: {str(handler_error)}")
            return {
                'error_handled': False,
                'handler_error': str(handler_error),
                'original_error': str(error),
                'session_id': session_id,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
    
    def _log_error(self, error: TrainingSystemError, operation: str = None):
        """Log error with structured information"""
        try:
            log_data = {
                'error_code': error.error_code,
                'category': error.category.value,
                'severity': error.severity.value,
                'session_id': error.session_id,
                'operation': operation,
                'details': error.details
            }
            
            # Log at appropriate level based on severity
            if error.severity == ErrorSeverity.CRITICAL:
                self.logger.critical("CRITICAL ERROR: %s | Code: %s | Session: %s | Details: %s", 
                                   error.message, error.error_code, error.session_id, json.dumps(log_data))
            elif error.severity == ErrorSeverity.HIGH:
                self.logger.error("HIGH SEVERITY ERROR: %s | Code: %s | Session: %s | Details: %s",
                                error.message, error.error_code, error.session_id, json.dumps(log_data))
            elif error.severity == ErrorSeverity.MEDIUM:
                self.logger.warning("MEDIUM SEVERITY ERROR: %s | Code: %s | Session: %s",
                                  error.message, error.error_code, error.session_id)
            else:
                self.logger.info("LOW SEVERITY ERROR: %s | Code: %s | Session: %s",
                               error.message, error.error_code, error.session_id)
                
        except Exception as e:
            print(f"Failed to log error: {e}")
    
    def _cache_error(self, error: TrainingSystemError):
        """Cache error for session tracking"""
        try:
            with self._lock:
                session_id = error.session_id or 'global'
                if session_id not in self.error_cache:
                    self.error_cache[session_id] = []
                
                self.error_cache[session_id].append(error)
                
                # Keep only last 50 errors per session
                if len(self.error_cache[session_id]) > 50:
                    self.error_cache[session_id] = self.error_cache[session_id][-50:]
                    
        except Exception as e:
            self.logger.error(f"Failed to cache error: {e}")
    
    def _save_error_to_database(self, error: TrainingSystemError):
        """Save error to database for persistence"""
        try:
            if not self.supabase:
                return
            
            error_record = {
                'session_id': error.session_id,
                'error_code': error.error_code,
                'error_message': error.message,
                'error_category': error.category.value,
                'error_severity': error.severity.value,
                'error_details': error.details,
                'traceback': traceback.format_exc(),
                'timestamp': error.timestamp,
                'created_at': error.timestamp
            }
            
            response = self.supabase.table('training_errors').insert(error_record).execute()
            
            if response.data:
                self.logger.debug(f"Saved error {error.error_code} to database")
            else:
                self.logger.warning(f"Failed to save error {error.error_code} to database")
                
        except Exception as e:
            self.logger.error(f"Database error logging failed: {e}")
    
    def _track_error_patterns(self, error: TrainingSystemError):
        """Track recurring error patterns"""
        try:
            with self._lock:
                pattern_key = f"{error.category.value}:{error.message[:100]}"
                
                if pattern_key not in self.error_patterns:
                    self.error_patterns[pattern_key] = {
                        'count': 0,
                        'first_seen': error.timestamp,
                        'last_seen': error.timestamp,
                        'sessions_affected': set(),
                        'severity_distribution': {}
                    }
                
                pattern = self.error_patterns[pattern_key]
                pattern['count'] += 1
                pattern['last_seen'] = error.timestamp
                
                if error.session_id:
                    pattern['sessions_affected'].add(error.session_id)
                
                severity = error.severity.value
                pattern['severity_distribution'][severity] = pattern['severity_distribution'].get(severity, 0) + 1
                
                # Log if this is becoming a pattern (3+ occurrences)
                if pattern['count'] >= 3 and pattern['count'] % 3 == 0:
                    self.logger.warning(
                        f"ERROR PATTERN DETECTED: {pattern_key} occurred {pattern['count']} times across {len(pattern['sessions_affected'])} sessions"
                    )
                    
        except Exception as e:
            self.logger.error(f"Failed to track error patterns: {e}")
    
    def _generate_recovery_suggestions(self, error: TrainingSystemError) -> List[str]:
        """Generate contextual recovery suggestions"""
        suggestions = []
        
        try:
            # Category-specific suggestions
            if error.category == ErrorCategory.DATA_PROCESSING:
                suggestions.extend([
                    "Verify input data format and structure",
                    "Check for missing or corrupted data files",
                    "Ensure data columns match expected schema",
                    "Validate datetime format consistency"
                ])
            
            elif error.category == ErrorCategory.MODEL_TRAINING:
                suggestions.extend([
                    "Check model parameters for valid ranges",
                    "Verify dataset size meets minimum requirements",
                    "Ensure sufficient system memory is available",
                    "Try reducing batch size or model complexity"
                ])
            
            elif error.category == ErrorCategory.PARAMETER_VALIDATION:
                suggestions.extend([
                    "Review parameter documentation for valid ranges",
                    "Check for required parameters that may be missing",
                    "Verify parameter data types match expectations",
                    "Use default values for optional parameters"
                ])
            
            elif error.category == ErrorCategory.DATABASE_OPERATION:
                suggestions.extend([
                    "Check database connection and credentials",
                    "Verify table schemas match expected structure",
                    "Retry operation after brief delay",
                    "Check for database service availability"
                ])
            
            elif error.category == ErrorCategory.MEMORY_ERROR:
                suggestions.extend([
                    "Reduce dataset size or use data chunking",
                    "Lower model complexity or batch size", 
                    "Clear cache and restart training process",
                    "Monitor system memory usage"
                ])
            
            # Severity-specific suggestions
            if error.severity == ErrorSeverity.CRITICAL:
                suggestions.insert(0, "CRITICAL: Stop all training operations and investigate immediately")
                suggestions.append("Contact system administrator if issue persists")
            
            elif error.severity == ErrorSeverity.HIGH:
                suggestions.insert(0, "HIGH PRIORITY: Address this issue before continuing")
            
            # Session-specific suggestions
            if error.session_id:
                session_errors = self.error_cache.get(error.session_id, [])
                if len(session_errors) > 5:
                    suggestions.append(f"Consider restarting session {error.session_id} - multiple errors detected")
            
            # Add generic suggestions if none specific
            if not suggestions:
                suggestions.extend([
                    "Review error details and logs for more information",
                    "Try the operation again after a brief wait",
                    "Check system resources and connectivity"
                ])
                
        except Exception as e:
            self.logger.error(f"Failed to generate recovery suggestions: {e}")
            suggestions = ["Contact technical support for assistance"]
        
        return suggestions
    
    def get_session_errors(self, session_id: str, limit: int = 20) -> List[Dict]:
        """Get recent errors for a session"""
        try:
            with self._lock:
                session_errors = self.error_cache.get(session_id, [])
                return [error.to_dict() for error in session_errors[-limit:]]
        except Exception as e:
            self.logger.error(f"Failed to get session errors: {e}")
            return []
    
    def get_error_statistics(self) -> Dict:
        """Get error statistics and patterns"""
        try:
            with self._lock:
                stats = {
                    'total_sessions_with_errors': len(self.error_cache),
                    'total_cached_errors': sum(len(errors) for errors in self.error_cache.values()),
                    'error_patterns': {},
                    'severity_distribution': {},
                    'category_distribution': {}
                }
                
                # Process error patterns
                for pattern_key, pattern_data in self.error_patterns.items():
                    if pattern_data['count'] >= 2:  # Only include patterns with 2+ occurrences
                        stats['error_patterns'][pattern_key] = {
                            'count': pattern_data['count'],
                            'sessions_affected': len(pattern_data['sessions_affected']),
                            'first_seen': pattern_data['first_seen'],
                            'last_seen': pattern_data['last_seen']
                        }
                
                # Process all cached errors for distributions
                for session_errors in self.error_cache.values():
                    for error in session_errors:
                        # Severity distribution
                        severity = error.severity.value
                        stats['severity_distribution'][severity] = stats['severity_distribution'].get(severity, 0) + 1
                        
                        # Category distribution
                        category = error.category.value
                        stats['category_distribution'][category] = stats['category_distribution'].get(category, 0) + 1
                
                return stats
                
        except Exception as e:
            self.logger.error(f"Failed to get error statistics: {e}")
            return {}
    
    def clear_session_errors(self, session_id: str):
        """Clear cached errors for a session"""
        try:
            with self._lock:
                if session_id in self.error_cache:
                    del self.error_cache[session_id]
                    self.logger.info(f"Cleared cached errors for session {session_id}")
        except Exception as e:
            self.logger.error(f"Failed to clear session errors: {e}")
    
    @contextmanager
    def error_context(self, operation: str, session_id: str = None, **context):
        """Context manager for automatic error handling"""
        start_time = time.time()
        try:
            self.logger.debug(f"Starting operation: {operation} | Session: {session_id}")
            yield
            
            # Record successful operation
            duration = time.time() - start_time
            self.logger.debug(f"Operation completed: {operation} | Duration: {duration:.2f}s | Session: {session_id}")
            
        except Exception as error:
            duration = time.time() - start_time
            context.update({
                'operation_duration': duration,
                'operation_failed': True
            })
            
            error_response = self.handle_error(error, session_id, context, operation)
            
            # Re-raise with additional context
            if isinstance(error, TrainingSystemError):
                raise error
            else:
                raise TrainingSystemError(
                    message=f"Operation '{operation}' failed: {str(error)}",
                    session_id=session_id,
                    details=context
                ) from error


def error_handler_decorator(category: ErrorCategory = ErrorCategory.SYSTEM_ERROR, 
                          severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                          operation: str = None):
    """
    Decorator for automatic error handling
    
    Usage:
        @error_handler_decorator(category=ErrorCategory.DATA_PROCESSING)
        def process_data(session_id):
            # Function implementation
            pass
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Extract session_id if available
            session_id = None
            if 'session_id' in kwargs:
                session_id = kwargs['session_id']
            elif args and isinstance(args[0], str):
                # Assume first string argument might be session_id
                session_id = args[0]
            
            op_name = operation or f"{func.__module__}.{func.__name__}"
            
            try:
                return func(*args, **kwargs)
            except TrainingSystemError:
                # Re-raise training system errors as-is
                raise
            except Exception as e:
                # Convert to training system error
                error_handler = get_error_handler()
                
                ts_error = TrainingSystemError(
                    message=f"Function '{func.__name__}' failed: {str(e)}",
                    category=category,
                    severity=severity,
                    session_id=session_id,
                    details={
                        'function': func.__name__,
                        'module': func.__module__,
                        'args_count': len(args),
                        'kwargs_keys': list(kwargs.keys())
                    }
                )
                
                error_handler.handle_error(ts_error, session_id, operation=op_name)
                raise ts_error from e
        
        return wrapper
    return decorator


# Global error handler instance
_error_handler_instance = None
_error_handler_lock = threading.Lock()


def get_error_handler(supabase_client=None, enable_database_logging: bool = True) -> ErrorHandler:
    """Get or create global ErrorHandler instance (singleton pattern)"""
    global _error_handler_instance
    
    with _error_handler_lock:
        if _error_handler_instance is None:
            _error_handler_instance = ErrorHandler(supabase_client, enable_database_logging)
        
        return _error_handler_instance


def shutdown_error_handler():
    """Shutdown the global error handler instance"""
    global _error_handler_instance
    
    with _error_handler_lock:
        if _error_handler_instance is not None:
            _error_handler_instance.logger.info("Shutting down ErrorHandler")
            _error_handler_instance = None


# Convenience functions for common error types
def handle_data_processing_error(error: Exception, session_id: str = None, **context) -> Dict:
    """Handle data processing errors"""
    error_handler = get_error_handler()
    return error_handler.handle_error(
        DataProcessingError(str(error), session_id=session_id, details=context),
        session_id, context, "data_processing"
    )


def handle_model_training_error(error: Exception, session_id: str = None, **context) -> Dict:
    """Handle model training errors"""
    error_handler = get_error_handler()
    return error_handler.handle_error(
        ModelTrainingError(str(error), session_id=session_id, details=context),
        session_id, context, "model_training"
    )


def handle_parameter_validation_error(error: Exception, session_id: str = None, **context) -> Dict:
    """Handle parameter validation errors"""
    error_handler = get_error_handler()
    return error_handler.handle_error(
        ParameterValidationError(str(error), session_id=session_id, details=context),
        session_id, context, "parameter_validation"
    )


def handle_database_error(error: Exception, session_id: str = None, **context) -> Dict:
    """Handle database errors"""
    error_handler = get_error_handler()
    return error_handler.handle_error(
        DatabaseError(str(error), session_id=session_id, details=context),
        session_id, context, "database_operation"
    )


def handle_visualization_error(error: Exception, session_id: str = None, **context) -> Dict:
    """Handle visualization errors"""
    error_handler = get_error_handler()
    return error_handler.handle_error(
        VisualizationError(str(error), session_id=session_id, details=context),
        session_id, context, "visualization"
    )