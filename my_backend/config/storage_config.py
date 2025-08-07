"""
Storage Configuration Module

Centralized configuration for all file storage locations and cleanup policies.
Provides unified paths and management for temporary files, sessions, and processed data.
"""

import os
import logging
import shutil
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

class StorageConfig:
    """Centralized storage configuration and management"""
    
    # Base storage directory
    BASE_STORAGE_DIR = "storage"
    
    # Storage subdirectories
    TEMP_DIR = "temp"           # All temporary files (auto-cleanup)
    SESSIONS_DIR = "sessions"   # Session-based data (organized cleanup) 
    PROCESSED_DIR = "processed" # Processed data cache (retention policy)
    UPLOADS_DIR = "uploads"     # Raw uploaded files (structured)
    LOGS_DIR = "logs"          # Application logs
    CACHE_DIR = "cache"        # Application cache files
    
    # Cleanup policies (in seconds)
    TEMP_FILE_LIFETIME = 3600      # 1 hour for temp files
    SESSION_FILE_LIFETIME = 86400  # 24 hours for session files  
    PROCESSED_FILE_LIFETIME = 259200  # 72 hours for processed files
    CACHE_FILE_LIFETIME = 604800   # 1 week for cache files
    
    # File size limits
    MAX_CHUNK_SIZE = 10 * 1024 * 1024    # 10MB per chunk
    MAX_FILE_SIZE = 500 * 1024 * 1024    # 500MB total
    MAX_STORAGE_SIZE = 10 * 1024 * 1024 * 1024  # 10GB total storage limit
    
    def __init__(self, base_dir: Optional[str] = None):
        """Initialize storage configuration
        
        Args:
            base_dir: Override base storage directory (for testing)
        """
        self.base_dir = Path(base_dir or self.BASE_STORAGE_DIR)
        self.lock = threading.Lock()
        self._initialize_directories()
    
    def _initialize_directories(self):
        """Create all required storage directories"""
        directories = [
            self.temp_dir,
            self.sessions_dir, 
            self.processed_dir,
            self.uploads_dir,
            self.logs_dir,
            self.cache_dir
        ]
        
        for directory in directories:
            try:
                directory.mkdir(parents=True, exist_ok=True)
                logger.debug(f"Ensured directory exists: {directory}")
            except OSError as e:
                logger.error(f"Failed to create directory {directory}: {e}")
                raise
    
    @property
    def temp_dir(self) -> Path:
        """Temporary files directory"""
        return self.base_dir / self.TEMP_DIR
    
    @property  
    def sessions_dir(self) -> Path:
        """Session files directory"""
        return self.base_dir / self.SESSIONS_DIR
    
    @property
    def processed_dir(self) -> Path:
        """Processed files directory"""
        return self.base_dir / self.PROCESSED_DIR
    
    @property
    def uploads_dir(self) -> Path:
        """Upload files directory"""
        return self.base_dir / self.UPLOADS_DIR
    
    @property
    def logs_dir(self) -> Path:
        """Logs directory"""
        return self.base_dir / self.LOGS_DIR
    
    @property
    def cache_dir(self) -> Path:
        """Cache files directory"""  
        return self.base_dir / self.CACHE_DIR
    
    def get_session_dir(self, session_id: str) -> Path:
        """Get directory for specific session
        
        Args:
            session_id: Session identifier
            
        Returns:
            Path to session directory
        """
        session_dir = self.sessions_dir / f"session_{session_id}"
        session_dir.mkdir(exist_ok=True)
        return session_dir
    
    def get_temp_file_path(self, prefix: str = "temp", suffix: str = ".tmp") -> Path:
        """Get unique temporary file path
        
        Args:
            prefix: File prefix
            suffix: File extension
            
        Returns:
            Path to temporary file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{prefix}_{timestamp}{suffix}"
        return self.temp_dir / filename
    
    def get_processed_file_path(self, category: str, filename: str) -> Path:
        """Get path for processed file
        
        Args:
            category: File category (e.g., 'csv', 'models', 'plots')
            filename: File name
            
        Returns:
            Path to processed file
        """
        category_dir = self.processed_dir / category
        category_dir.mkdir(exist_ok=True)
        return category_dir / filename
    
    def cleanup_expired_files(self, force: bool = False) -> Dict[str, int]:
        """Clean up expired files across all storage areas
        
        Args:
            force: Force cleanup regardless of age
            
        Returns:
            Dictionary with cleanup statistics
        """
        stats = {
            'temp_files_removed': 0,
            'session_files_removed': 0, 
            'processed_files_removed': 0,
            'cache_files_removed': 0,
            'errors': 0
        }
        
        current_time = time.time()
        
        with self.lock:
            # Clean temp files
            stats['temp_files_removed'] = self._cleanup_directory(
                self.temp_dir, 
                current_time - self.TEMP_FILE_LIFETIME if not force else 0
            )
            
            # Clean session files
            stats['session_files_removed'] = self._cleanup_directory(
                self.sessions_dir,
                current_time - self.SESSION_FILE_LIFETIME if not force else 0
            )
            
            # Clean processed files
            stats['processed_files_removed'] = self._cleanup_directory(
                self.processed_dir,
                current_time - self.PROCESSED_FILE_LIFETIME if not force else 0
            )
            
            # Clean cache files
            stats['cache_files_removed'] = self._cleanup_directory(
                self.cache_dir,
                current_time - self.CACHE_FILE_LIFETIME if not force else 0
            )
        
        logger.info(f"Cleanup completed: {stats}")
        return stats
    
    def _cleanup_directory(self, directory: Path, cutoff_time: float) -> int:
        """Clean up files older than cutoff time in directory
        
        Args:
            directory: Directory to clean
            cutoff_time: Unix timestamp cutoff
            
        Returns:
            Number of files removed
        """
        removed_count = 0
        
        if not directory.exists():
            return removed_count
            
        try:
            for item in directory.rglob("*"):
                if item.is_file():
                    try:
                        if item.stat().st_mtime < cutoff_time:
                            item.unlink()
                            removed_count += 1
                            logger.debug(f"Removed expired file: {item}")
                    except OSError as e:
                        logger.warning(f"Failed to remove file {item}: {e}")
                        
            # Remove empty directories
            for item in directory.rglob("*"):
                if item.is_dir() and not any(item.iterdir()):
                    try:
                        item.rmdir()
                        logger.debug(f"Removed empty directory: {item}")
                    except OSError:
                        pass  # Directory might not be empty due to hidden files
                        
        except Exception as e:
            logger.error(f"Error during directory cleanup of {directory}: {e}")
            
        return removed_count
    
    def get_storage_stats(self) -> Dict[str, any]:
        """Get storage usage statistics
        
        Returns:
            Dictionary with storage statistics
        """
        stats = {
            'total_size': 0,
            'directories': {},
            'file_counts': {},
            'oldest_files': {},
            'storage_health': 'good'
        }
        
        directories = {
            'temp': self.temp_dir,
            'sessions': self.sessions_dir,
            'processed': self.processed_dir,
            'uploads': self.uploads_dir,
            'logs': self.logs_dir,
            'cache': self.cache_dir
        }
        
        try:
            for name, directory in directories.items():
                if directory.exists():
                    dir_size = 0
                    file_count = 0
                    oldest_time = None
                    
                    for item in directory.rglob("*"):
                        if item.is_file():
                            size = item.stat().st_size
                            mtime = item.stat().st_mtime
                            
                            dir_size += size
                            file_count += 1
                            
                            if oldest_time is None or mtime < oldest_time:
                                oldest_time = mtime
                    
                    stats['directories'][name] = dir_size
                    stats['file_counts'][name] = file_count
                    stats['oldest_files'][name] = oldest_time
                    stats['total_size'] += dir_size
            
            # Determine storage health
            if stats['total_size'] > self.MAX_STORAGE_SIZE * 0.9:
                stats['storage_health'] = 'critical'
            elif stats['total_size'] > self.MAX_STORAGE_SIZE * 0.75:
                stats['storage_health'] = 'warning'
                
        except Exception as e:
            logger.error(f"Error calculating storage stats: {e}")
            stats['error'] = str(e)
            stats['storage_health'] = 'error'
        
        return stats
    
    def migrate_legacy_files(self) -> Dict[str, int]:
        """Migrate files from legacy storage locations
        
        Returns:
            Migration statistics
        """
        stats = {
            'files_migrated': 0,
            'directories_processed': 0,
            'errors': 0,
            'skipped': 0
        }
        
        # Legacy directory mappings
        legacy_mappings = [
            # (source, destination, description)
            (Path("chunk_uploads"), self.temp_dir / "chunks", "Chunk uploads"),
            (Path("api/temp_uploads"), self.temp_dir / "api", "API temp files"),
            (Path("temp_uploads"), self.temp_dir / "legacy", "Legacy temp files"),
            (Path("temp_training_data"), self.temp_dir / "training", "Training temp data"),
            (Path("uploads/file_uploads"), self.sessions_dir, "Session files"),
        ]
        
        logger.info("Starting legacy file migration...")
        
        for source, destination, description in legacy_mappings:
            if source.exists():
                logger.info(f"Migrating {description} from {source} to {destination}")
                
                try:
                    # Create destination directory
                    destination.mkdir(parents=True, exist_ok=True)
                    
                    # Move files and directories
                    for item in source.iterdir():
                        dest_path = destination / item.name
                        
                        if dest_path.exists():
                            logger.warning(f"Destination already exists, skipping: {dest_path}")
                            stats['skipped'] += 1
                            continue
                            
                        try:
                            shutil.move(str(item), str(dest_path))
                            stats['files_migrated'] += 1
                            logger.debug(f"Migrated: {item} -> {dest_path}")
                        except Exception as e:
                            logger.error(f"Failed to migrate {item}: {e}")
                            stats['errors'] += 1
                    
                    # Remove source directory if empty
                    try:
                        if not any(source.iterdir()):
                            source.rmdir()
                            logger.info(f"Removed empty legacy directory: {source}")
                    except OSError:
                        logger.warning(f"Could not remove legacy directory: {source}")
                    
                    stats['directories_processed'] += 1
                    
                except Exception as e:
                    logger.error(f"Failed to migrate {description}: {e}")
                    stats['errors'] += 1
            else:
                logger.debug(f"Legacy directory not found: {source}")
        
        logger.info(f"Migration completed: {stats}")
        return stats


# Global storage configuration instance
storage_config = StorageConfig()

# Convenience functions for backward compatibility
def get_temp_dir() -> str:
    """Get temporary files directory"""
    return str(storage_config.temp_dir)

def get_session_dir(session_id: str) -> str:
    """Get session directory"""
    return str(storage_config.get_session_dir(session_id))

def get_processed_dir() -> str:
    """Get processed files directory"""
    return str(storage_config.processed_dir)

def get_uploads_dir() -> str:
    """Get uploads directory"""
    return str(storage_config.uploads_dir)

def cleanup_expired_files() -> Dict[str, int]:
    """Clean up expired files"""
    return storage_config.cleanup_expired_files()

def get_storage_stats() -> Dict[str, any]:
    """Get storage statistics"""
    return storage_config.get_storage_stats()