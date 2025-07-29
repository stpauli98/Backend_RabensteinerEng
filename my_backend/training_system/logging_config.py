"""
Comprehensive logging configuration for training system
Provides structured logging, performance monitoring, and centralized log management
"""

import logging
import logging.handlers
import json
import time
import threading
import os
import sys
from datetime import datetime, timezone
from typing import Dict, Optional, Any
from pathlib import Path

# Import existing supabase client
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from supabase_client import get_supabase_client


class StructuredFormatter(logging.Formatter):
    """
    Custom formatter for structured JSON logging
    """
    
    def __init__(self, include_extra: bool = True):
        super().__init__()
        self.include_extra = include_extra
    
    def format(self, record: logging.LogRecord) -> str:
        # Base log structure
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'message': record.getMessage(),
            'thread_id': record.thread,
            'thread_name': record.threadName,
            'process_id': record.process
        }
        
        # Add session context if available
        if hasattr(record, 'session_id'):
            log_entry['session_id'] = record.session_id
        
        # Add operation context if available
        if hasattr(record, 'operation'):
            log_entry['operation'] = record.operation
        
        # Add performance metrics if available
        if hasattr(record, 'duration'):
            log_entry['duration_seconds'] = record.duration
        
        if hasattr(record, 'memory_usage'):
            log_entry['memory_usage_mb'] = record.memory_usage
        
        # Add error context if available
        if hasattr(record, 'error_code'):
            log_entry['error_code'] = record.error_code
        
        if hasattr(record, 'error_category'):
            log_entry['error_category'] = record.error_category
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        # Add stack trace for errors
        if record.levelno >= logging.ERROR and record.stack_info:
            log_entry['stack_trace'] = record.stack_info
        
        # Add any extra fields if enabled
        if self.include_extra:
            for key, value in record.__dict__.items():
                if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 
                             'filename', 'module', 'lineno', 'funcName', 'created', 
                             'msecs', 'relativeCreated', 'thread', 'threadName', 'processName',
                             'process', 'message', 'exc_info', 'exc_text', 'stack_info']:
                    if key not in log_entry:  # Don't override existing fields
                        try:
                            # Ensure value is JSON serializable
                            json.dumps(value)
                            log_entry[key] = value
                        except (TypeError, ValueError):
                            log_entry[key] = str(value)
        
        return json.dumps(log_entry, ensure_ascii=False)


class DatabaseLogHandler(logging.Handler):
    """
    Custom log handler that writes logs to Supabase database
    """
    
    def __init__(self, supabase_client=None, table_name: str = 'training_logs', 
                 buffer_size: int = 10, flush_interval: int = 5):
        super().__init__()
        self.supabase = supabase_client or get_supabase_client()
        self.table_name = table_name
        self.buffer_size = buffer_size
        self.flush_interval = flush_interval
        
        # Buffer for batch writing
        self.buffer = []
        self.buffer_lock = threading.Lock()
        
        # Background flush timer
        self.flush_timer = None
        self._start_flush_timer()
    
    def emit(self, record: logging.LogRecord):
        """Emit a log record to the database buffer"""
        try:
            # Convert log record to database format
            log_entry = {
                'timestamp': datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
                'level': record.levelname,
                'logger_name': record.name,
                'module': record.module,
                'function_name': record.funcName,
                'line_number': record.lineno,
                'message': record.getMessage(),
                'thread_id': record.thread,
                'process_id': record.process,
                'session_id': getattr(record, 'session_id', None),
                'operation': getattr(record, 'operation', None),
                'error_code': getattr(record, 'error_code', None),
                'error_category': getattr(record, 'error_category', None),
                'duration_seconds': getattr(record, 'duration', None),
                'memory_usage_mb': getattr(record, 'memory_usage', None),
                'exception_info': self.format(record) if record.exc_info else None,
                'created_at': datetime.fromtimestamp(record.created, timezone.utc).isoformat()
            }
            
            # Add to buffer
            with self.buffer_lock:
                self.buffer.append(log_entry)
                
                # Flush if buffer is full
                if len(self.buffer) >= self.buffer_size:
                    self._flush_buffer()
                    
        except Exception:
            # Don't let logging errors break the application
            self.handleError(record)
    
    def _flush_buffer(self):
        """Flush buffered log entries to database"""
        try:
            if not self.buffer:
                return
                
            # Get current buffer and clear it
            entries_to_write = self.buffer.copy()
            self.buffer.clear()
            
            # Write to database
            if self.supabase and entries_to_write:
                response = self.supabase.table(self.table_name).insert(entries_to_write).execute()
                
                if not response.data:
                    print(f"Warning: Failed to write {len(entries_to_write)} log entries to database")
                    
        except Exception as e:
            print(f"Error flushing log buffer to database: {e}")
    
    def _start_flush_timer(self):
        """Start the periodic flush timer"""
        def flush_periodically():
            with self.buffer_lock:
                if self.buffer:
                    self._flush_buffer()
            
            # Schedule next flush
            self.flush_timer = threading.Timer(self.flush_interval, flush_periodically)
            self.flush_timer.daemon = True
            self.flush_timer.start()
        
        flush_periodically()
    
    def flush(self):
        """Manually flush the buffer"""
        with self.buffer_lock:
            self._flush_buffer()
    
    def close(self):
        """Close the handler and flush remaining logs"""
        if self.flush_timer:
            self.flush_timer.cancel()
        
        with self.buffer_lock:
            self._flush_buffer()
        
        super().close()


class PerformanceLogger:
    """
    Logger for performance monitoring and metrics
    """
    
    def __init__(self, logger_name: str = 'training_system.performance'):
        self.logger = logging.getLogger(logger_name)
        self.session_metrics = {}  # session_id -> metrics
        self.operation_metrics = {}  # operation -> metrics
        self._lock = threading.Lock()
    
    def log_operation_start(self, operation: str, session_id: str = None, **context):
        """Log the start of an operation"""
        start_time = time.time()
        
        log_extra = {
            'operation': operation,
            'session_id': session_id,
            'operation_status': 'started',
            'start_time': start_time,
            **context
        }
        
        self.logger.info(f"Operation started: {operation}", extra=log_extra)
        
        # Store start time for duration calculation
        key = f"{session_id or 'global'}:{operation}:{threading.current_thread().ident}"
        with self._lock:
            if 'operation_starts' not in self.session_metrics:
                self.session_metrics['operation_starts'] = {}
            self.session_metrics['operation_starts'][key] = start_time
        
        return start_time
    
    def log_operation_end(self, operation: str, session_id: str = None, 
                         success: bool = True, error: Exception = None, **context):
        """Log the end of an operation with performance metrics"""
        end_time = time.time()
        
        # Calculate duration
        key = f"{session_id or 'global'}:{operation}:{threading.current_thread().ident}"
        start_time = None
        
        with self._lock:
            if ('operation_starts' in self.session_metrics and 
                key in self.session_metrics['operation_starts']):
                start_time = self.session_metrics['operation_starts'].pop(key)
        
        duration = end_time - start_time if start_time else None
        
        # Get memory usage
        try:
            import psutil
            process = psutil.Process()
            memory_usage = process.memory_info().rss / 1024 / 1024  # MB
        except (ImportError, Exception):
            memory_usage = None
        
        log_extra = {
            'operation': operation,
            'session_id': session_id,
            'operation_status': 'completed' if success else 'failed',
            'duration': duration,
            'memory_usage': memory_usage,
            'success': success,
            **context
        }
        
        if error:
            log_extra['error'] = str(error)
            log_extra['error_type'] = type(error).__name__
        
        level = logging.INFO if success else logging.ERROR
        message = f"Operation {'completed' if success else 'failed'}: {operation}"
        
        if duration:
            message += f" (duration: {duration:.2f}s)"
        
        self.logger.log(level, message, extra=log_extra)
        
        # Update operation metrics
        self._update_operation_metrics(operation, duration, success, memory_usage)
    
    def _update_operation_metrics(self, operation: str, duration: float, 
                                success: bool, memory_usage: float):
        """Update aggregated operation metrics"""
        try:
            with self._lock:
                if operation not in self.operation_metrics:
                    self.operation_metrics[operation] = {
                        'total_calls': 0,
                        'successful_calls': 0,
                        'failed_calls': 0,
                        'total_duration': 0,
                        'min_duration': float('inf'),
                        'max_duration': 0,
                        'avg_duration': 0,
                        'total_memory': 0,
                        'max_memory': 0,
                        'avg_memory': 0
                    }
                
                metrics = self.operation_metrics[operation]
                metrics['total_calls'] += 1
                
                if success:
                    metrics['successful_calls'] += 1
                else:
                    metrics['failed_calls'] += 1
                
                if duration is not None:
                    metrics['total_duration'] += duration
                    metrics['min_duration'] = min(metrics['min_duration'], duration)
                    metrics['max_duration'] = max(metrics['max_duration'], duration)
                    metrics['avg_duration'] = metrics['total_duration'] / metrics['total_calls']
                
                if memory_usage is not None:
                    metrics['total_memory'] += memory_usage
                    metrics['max_memory'] = max(metrics['max_memory'], memory_usage)
                    metrics['avg_memory'] = metrics['total_memory'] / metrics['total_calls']
                    
        except Exception as e:
            self.logger.error(f"Failed to update operation metrics: {e}")
    
    def get_operation_metrics(self, operation: str = None) -> Dict:
        """Get operation performance metrics"""
        with self._lock:
            if operation:
                return self.operation_metrics.get(operation, {})
            else:
                return self.operation_metrics.copy()
    
    def log_performance_summary(self, session_id: str = None):
        """Log a performance summary"""
        with self._lock:
            summary = {
                'total_operations': len(self.operation_metrics),
                'operations': {}
            }
            
            for op, metrics in self.operation_metrics.items():
                summary['operations'][op] = {
                    'total_calls': metrics['total_calls'],
                    'success_rate': (metrics['successful_calls'] / metrics['total_calls'] * 100) if metrics['total_calls'] > 0 else 0,
                    'avg_duration': metrics['avg_duration'],
                    'max_duration': metrics['max_duration'],
                    'avg_memory': metrics['avg_memory'],
                    'max_memory': metrics['max_memory']
                }
            
            self.logger.info(f"Performance summary", extra={
                'session_id': session_id,
                'performance_summary': summary
            })
            
            return summary


def setup_training_system_logging(
    log_level: str = 'INFO',
    log_to_file: bool = True,
    log_to_database: bool = True,
    log_to_console: bool = True,
    log_directory: str = None,
    structured_logging: bool = True,
    max_file_size: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
) -> Dict[str, logging.Logger]:
    """
    Setup comprehensive logging for the training system
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_to_file: Enable file logging
        log_to_database: Enable database logging
        log_to_console: Enable console logging
        log_directory: Directory for log files (defaults to ./logs)
        structured_logging: Use structured JSON logging
        max_file_size: Maximum size of log files before rotation
        backup_count: Number of backup log files to keep
        
    Returns:
        Dictionary of configured loggers
    """
    
    # Set log level
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Create log directory if needed
    if log_to_file:
        if log_directory is None:
            log_directory = os.path.join(os.path.dirname(__file__), '..', 'logs')
        
        Path(log_directory).mkdir(parents=True, exist_ok=True)
    
    # Configure root logger for training system
    root_logger = logging.getLogger('training_system')
    root_logger.setLevel(numeric_level)
    
    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    handlers = []
    
    # Console handler
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        
        if structured_logging:
            console_handler.setFormatter(StructuredFormatter())
        else:
            console_formatter = logging.Formatter(
                '%(asctime)s | %(levelname)-8s | %(name)s.%(funcName)s:%(lineno)d | %(message)s'
            )
            console_handler.setFormatter(console_formatter)
        
        handlers.append(console_handler)
    
    # File handler with rotation
    if log_to_file:
        log_file = os.path.join(log_directory, 'training_system.log')
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=max_file_size, backupCount=backup_count
        )
        file_handler.setLevel(numeric_level)
        
        if structured_logging:
            file_handler.setFormatter(StructuredFormatter())
        else:
            file_formatter = logging.Formatter(
                '%(asctime)s | %(levelname)-8s | %(name)s.%(funcName)s:%(lineno)d | %(message)s'
            )
            file_handler.setFormatter(file_formatter)
        
        handlers.append(file_handler)
    
    # Database handler
    if log_to_database:
        try:
            db_handler = DatabaseLogHandler()
            db_handler.setLevel(logging.WARNING)  # Only log warnings and errors to DB
            handlers.append(db_handler)
        except Exception as e:
            print(f"Warning: Could not setup database logging: {e}")
    
    # Add handlers to root logger
    for handler in handlers:
        root_logger.addHandler(handler)
    
    # Create specialized loggers
    loggers = {
        'root': root_logger,
        'data_processing': logging.getLogger('training_system.data_processing'),
        'model_training': logging.getLogger('training_system.model_training'),
        'visualization': logging.getLogger('training_system.visualization'),
        'api': logging.getLogger('training_system.api'),
        'pipeline': logging.getLogger('training_system.pipeline'),
        'error_handler': logging.getLogger('training_system.error_handler'),
        'performance': logging.getLogger('training_system.performance')
    }
    
    # Set levels for specialized loggers
    for logger in loggers.values():
        logger.setLevel(numeric_level)
    
    # Log setup completion
    root_logger.info(f"Training system logging configured: level={log_level}, "
                    f"console={log_to_console}, file={log_to_file}, database={log_to_database}, "
                    f"structured={structured_logging}")
    
    return loggers


def get_session_logger(session_id: str, logger_name: str = 'training_system') -> logging.Logger:
    """
    Get a logger with session context
    
    Args:
        session_id: Session identifier to include in all log messages
        logger_name: Base logger name
        
    Returns:
        Logger with session context
    """
    logger = logging.getLogger(logger_name)
    
    # Create a logger adapter that adds session_id to all log records
    class SessionAdapter(logging.LoggerAdapter):
        def process(self, msg, kwargs):
            # Add session_id to extra fields
            if 'extra' not in kwargs:
                kwargs['extra'] = {}
            kwargs['extra']['session_id'] = self.extra['session_id']
            return msg, kwargs
    
    return SessionAdapter(logger, {'session_id': session_id})


def cleanup_logging():
    """Clean up logging handlers and flush any remaining logs"""
    try:
        # Get all training system loggers
        for name in list(logging.Logger.manager.loggerDict.keys()):
            if name.startswith('training_system'):
                logger = logging.getLogger(name)
                
                # Flush and close all handlers
                for handler in logger.handlers[:]:
                    try:
                        handler.flush()
                        handler.close()
                        logger.removeHandler(handler)
                    except Exception:
                        pass  # Ignore cleanup errors
                        
    except Exception as e:
        print(f"Error during logging cleanup: {e}")


# Performance monitoring context manager
class performance_monitor:
    """Context manager for performance monitoring"""
    
    def __init__(self, operation: str, session_id: str = None, 
                 logger: PerformanceLogger = None, **context):
        self.operation = operation
        self.session_id = session_id
        self.context = context
        self.logger = logger or PerformanceLogger()
        self.start_time = None
    
    def __enter__(self):
        self.start_time = self.logger.log_operation_start(
            self.operation, self.session_id, **self.context
        )
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        success = exc_type is None
        error = exc_val if exc_type else None
        
        self.logger.log_operation_end(
            self.operation, self.session_id, success, error, **self.context
        )


# Global performance logger instance
_performance_logger = None

def get_performance_logger() -> PerformanceLogger:
    """Get global performance logger instance"""
    global _performance_logger
    if _performance_logger is None:
        _performance_logger = PerformanceLogger()
    return _performance_logger