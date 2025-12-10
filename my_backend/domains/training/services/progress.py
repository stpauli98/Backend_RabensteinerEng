"""
Progress Manager for Training System
Handles session isolation, progress tracking, and heartbeat monitoring
"""

import time
import os
import sys
import threading
import uuid
import logging
from datetime import datetime, timezone
from typing import Dict, Optional, Any, List
import json

from shared.database.operations import get_supabase_client

logger = logging.getLogger(__name__)


class ProgressManager:
    """
    Manages training progress with session isolation and heartbeat monitoring
    
    Key Features:
    - Session locking to prevent concurrent access
    - In-memory progress caching for real-time updates  
    - Batch database updates to reduce load
    - SocketIO integration for real-time frontend updates
    - Heartbeat monitoring for dead session detection
    - Automatic cleanup of abandoned sessions
    """
    
    def __init__(self, socketio_instance=None, supabase_client=None):
        self.socketio = socketio_instance
        self.supabase = supabase_client or get_supabase_client()
        
        self.progress_cache = {}
        self.session_locks = {}
        
        self.heartbeat_interval = 60
        self.db_update_interval = 30
        self.session_timeout = 300
        
        self.last_db_update = {}
        self.last_heartbeat = {}
        
        self.process_id = f"process_{os.getpid()}_{uuid.uuid4().hex[:8]}"
        self.process_info = {
            'pid': os.getpid(),
            'hostname': os.uname().nodename,
            'started_at': datetime.now(timezone.utc).isoformat(),
            'python_version': sys.version
        }
        
        self._heartbeat_thread = None
        self._cleanup_thread = None
        self._shutdown_event = threading.Event()
        
        self._start_background_services()
        
    
    def acquire_session_lock(self, session_id: str) -> bool:
        """
        Atomically acquire lock on a session for training
        
        Args:
            session_id: Session identifier (UUID)
            
        Returns:
            True if lock acquired successfully, False otherwise
        """
        try:
            
            response = self.supabase.rpc('acquire_session_lock', {
                'p_session_id': session_id,
                'p_process_id': self.process_id,
                'p_process_info': self.process_info
            }).execute()
            
            lock_acquired = response.data
            
            if lock_acquired:
                self.session_locks[session_id] = {
                    'process_id': self.process_id,
                    'acquired_at': time.time(),
                    'last_heartbeat': time.time()
                }
                
                self.progress_cache[session_id] = {
                    'overall_progress': 0,
                    'current_step': 'Initializing',
                    'total_steps': 7,
                    'completed_steps': 0,
                    'status': 'running',
                    'started_at': datetime.now(timezone.utc).isoformat(),
                    'step_details': {},
                    'model_progress': {}
                }
                
                self.last_heartbeat[session_id] = time.time()
                self.last_db_update[session_id] = time.time()
                
                
                self._emit_progress(session_id)
                
                return True
            else:
                logger.warning(f"❌ Failed to acquire session lock for {session_id} - session may be in use")
                return False
                
        except Exception as e:
            logger.error(f"Error acquiring session lock for {session_id}: {str(e)}")
            return False
    
    def release_session_lock(self, session_id: str, status: str = 'completed') -> bool:
        """
        Release session lock and update final status
        
        Args:
            session_id: Session identifier
            status: Final status ('completed', 'failed', 'cancelled')
            
        Returns:
            True if released successfully
        """
        try:
            
            if session_id in self.progress_cache:
                self.progress_cache[session_id].update({
                    'status': status,
                    'overall_progress': 100 if status == 'completed' else self.progress_cache[session_id].get('overall_progress', 0),
                    'current_step': f'Training {status}',
                    'completed_at': datetime.now(timezone.utc).isoformat()
                })
            
            self._save_progress_to_database(session_id, force=True)
            
            update_data = {
                'status': status,
                'completed_at': datetime.now(timezone.utc).isoformat(),
                'process_id': None,
                'updated_at': datetime.now(timezone.utc).isoformat()
            }
            
            self.supabase.table('training_progress').update(update_data).eq('session_id', session_id).execute()
            
            self._emit_progress(session_id)
            
            self.session_locks.pop(session_id, None)
            self.last_heartbeat.pop(session_id, None) 
            self.last_db_update.pop(session_id, None)
            
            
            return True
            
        except Exception as e:
            logger.error(f"Error releasing session lock for {session_id}: {str(e)}")
            return False
    
    def update_progress(self, session_id: str, progress_data: Dict) -> bool:
        """
        Update training progress for a session
        
        Args:
            session_id: Session identifier
            progress_data: Progress information
                - overall_progress: int (0-100)
                - current_step: str
                - completed_steps: int
                - step_details: dict
                - model_progress: dict
                
        Returns:
            True if updated successfully
        """
        try:
            if session_id not in self.session_locks:
                logger.warning(f"Cannot update progress for {session_id} - session not locked by this process")
                return False
            
            if session_id not in self.progress_cache:
                self.progress_cache[session_id] = {}
            
            self.progress_cache[session_id].update(progress_data)
            self.progress_cache[session_id]['updated_at'] = datetime.now(timezone.utc).isoformat()
            
            self._emit_progress(session_id)
            
            self.last_heartbeat[session_id] = time.time()
            
            should_update_db = self._should_update_database(session_id, progress_data)
            
            if should_update_db:
                self._save_progress_to_database(session_id)
                self.last_db_update[session_id] = time.time()
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating progress for {session_id}: {str(e)}")
            return False
    
    def get_progress(self, session_id: str) -> Optional[Dict]:
        """
        Get current progress for a session
        
        Args:
            session_id: Session identifier
            
        Returns:
            Progress data or None if not found
        """
        try:
            if session_id in self.progress_cache:
                return self.progress_cache[session_id].copy()
            
            response = self.supabase.table('training_progress').select('*').eq('session_id', session_id).execute()
            
            if response.data:
                progress_data = response.data[0]
                
                formatted_progress = {
                    'overall_progress': progress_data.get('overall_progress', 0),
                    'current_step': progress_data.get('current_step', 'Unknown'),
                    'total_steps': progress_data.get('total_steps', 7),
                    'completed_steps': progress_data.get('completed_steps', 0),
                    'status': progress_data.get('status', 'idle'),
                    'step_details': progress_data.get('step_details', {}),
                    'model_progress': progress_data.get('model_progress', {}),
                    'started_at': progress_data.get('started_at'),
                    'updated_at': progress_data.get('updated_at')
                }
                
                return formatted_progress
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting progress for {session_id}: {str(e)}")
            return None
    
    def is_session_locked(self, session_id: str) -> bool:
        """
        Check if a session is currently locked
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if session is locked
        """
        try:
            if session_id in self.session_locks:
                return True
            
            response = self.supabase.table('training_progress').select('status, process_id, last_heartbeat').eq('session_id', session_id).execute()
            
            if response.data:
                progress_data = response.data[0]
                status = progress_data.get('status')
                process_id = progress_data.get('process_id')
                last_heartbeat = progress_data.get('last_heartbeat')
                
                if status == 'running' and process_id and last_heartbeat:
                    from datetime import datetime
                    heartbeat_time = datetime.fromisoformat(last_heartbeat.replace('Z', '+00:00'))
                    age_seconds = (datetime.now(timezone.utc) - heartbeat_time).total_seconds()
                    
                    return age_seconds < self.session_timeout
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking session lock for {session_id}: {str(e)}")
            return False
    
    def _should_update_database(self, session_id: str, progress_data: Dict) -> bool:
        """
        Determine if we should update database based on various factors
        
        Args:
            session_id: Session identifier
            progress_data: Current progress data
            
        Returns:
            True if database should be updated
        """
        try:
            now = time.time()
            last_update = self.last_db_update.get(session_id, 0)
            
            if progress_data.get('overall_progress') == 100:
                return True
            
            if progress_data.get('status') in ['completed', 'failed', 'cancelled']:
                return True
            
            current_progress = progress_data.get('overall_progress', 0)
            if current_progress > 0 and current_progress % 14 == 0:
                return True
            
            if now - last_update > self.db_update_interval:
                return True
            
            completed_steps = progress_data.get('completed_steps', 0)
            if session_id in self.progress_cache:
                old_completed = self.progress_cache[session_id].get('completed_steps', 0)
                if completed_steps > old_completed:
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error determining database update need: {str(e)}")
            return False
    
    def _save_progress_to_database(self, session_id: str, force: bool = False) -> bool:
        """
        Save current progress to database
        
        Args:
            session_id: Session identifier
            force: Force update regardless of timing
            
        Returns:
            True if saved successfully
        """
        try:
            if session_id not in self.progress_cache:
                return False
            
            progress_data = self.progress_cache[session_id]
            
            db_data = {
                'overall_progress': progress_data.get('overall_progress', 0),
                'current_step': progress_data.get('current_step', ''),
                'completed_steps': progress_data.get('completed_steps', 0),
                'step_details': progress_data.get('step_details', {}),
                'model_progress': progress_data.get('model_progress', {}),
                'last_heartbeat': datetime.now(timezone.utc).isoformat(),
                'updated_at': datetime.now(timezone.utc).isoformat()
            }
            
            if 'status' in progress_data:
                db_data['status'] = progress_data['status']
            
            if 'completed_at' in progress_data:
                db_data['completed_at'] = progress_data['completed_at']
            
            db_data['session_id'] = session_id
            response = self.supabase.table('training_progress').upsert(db_data).execute()
            
            if response.data:
                return True
            else:
                logger.warning(f"No data returned when saving progress for {session_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error saving progress to database for {session_id}: {str(e)}")
            return False
    
    def _emit_progress(self, session_id: str):
        """
        Emit progress update via SocketIO
        
        Args:
            session_id: Session identifier
        """
        try:
            if self.socketio and session_id in self.progress_cache:
                progress_data = self.progress_cache[session_id].copy()
                progress_data['session_id'] = session_id
                
                self.socketio.emit('training_progress', progress_data, room=f"training_{session_id}")
                
                
        except Exception as e:
            logger.error(f"Error emitting progress for {session_id}: {str(e)}")
    
    def _start_background_services(self):
        """Start background threads for heartbeat and cleanup"""
        try:
            self._heartbeat_thread = threading.Thread(
                target=self._heartbeat_worker,
                daemon=True,
                name=f"ProgressManager-Heartbeat-{self.process_id}"
            )
            self._heartbeat_thread.start()
            
            self._cleanup_thread = threading.Thread(
                target=self._cleanup_worker,
                daemon=True,
                name=f"ProgressManager-Cleanup-{self.process_id}"
            )
            self._cleanup_thread.start()
            
            
        except Exception as e:
            logger.error(f"Error starting background services: {str(e)}")
    
    def _heartbeat_worker(self):
        """Background worker for sending heartbeats"""
        while not self._shutdown_event.is_set():
            try:
                current_time = time.time()
                
                for session_id in list(self.session_locks.keys()):
                    try:
                        response = self.supabase.rpc('update_session_heartbeat', {
                            'p_session_id': session_id,
                            'p_process_id': self.process_id
                        }).execute()
                        
                        if response.data:
                            self.last_heartbeat[session_id] = current_time
                        else:
                            logger.warning(f"Heartbeat failed for {session_id}")
                            
                    except Exception as e:
                        logger.error(f"Error sending heartbeat for {session_id}: {str(e)}")
                
                self._shutdown_event.wait(self.heartbeat_interval)
                
            except Exception as e:
                logger.error(f"Error in heartbeat worker: {str(e)}")
                self._shutdown_event.wait(5)
    
    def _cleanup_worker(self):
        """Background worker for cleaning up abandoned sessions and old cache"""
        while not self._shutdown_event.is_set():
            try:
                response = self.supabase.rpc('cleanup_abandoned_sessions').execute()
                cleanup_count = response.data
                
                if cleanup_count > 0:
                    pass
                
                current_time = time.time()
                sessions_to_remove = []
                
                for session_id, progress_data in self.progress_cache.items():
                    if progress_data.get('status') in ['completed', 'failed', 'cancelled']:
                        if 'completed_at' in progress_data:
                            try:
                                completed_time = datetime.fromisoformat(progress_data['completed_at'].replace('Z', '+00:00'))
                                age_seconds = (datetime.now(timezone.utc) - completed_time).total_seconds()
                                if age_seconds > 300:
                                    sessions_to_remove.append(session_id)
                            except Exception:
                                pass
                
                for session_id in sessions_to_remove:
                    self.progress_cache.pop(session_id, None)
                
                self._shutdown_event.wait(120)
                
            except Exception as e:
                logger.error(f"Error in cleanup worker: {str(e)}")
                self._shutdown_event.wait(60)
    
    def shutdown(self):
        """Gracefully shutdown the progress manager"""
        try:
            
            self._shutdown_event.set()
            
            for session_id in list(self.session_locks.keys()):
                self.release_session_lock(session_id, 'cancelled')
            
            if self._heartbeat_thread and self._heartbeat_thread.is_alive():
                self._heartbeat_thread.join(timeout=5)
            
            if self._cleanup_thread and self._cleanup_thread.is_alive():
                self._cleanup_thread.join(timeout=5)
            
            
        except Exception as e:
            logger.error(f"Error during shutdown: {str(e)}")
    
    def get_session_status(self, session_id: str) -> Dict:
        """
        Get comprehensive status information for a session
        
        Args:
            session_id: Session identifier
            
        Returns:
            Dict containing status information
        """
        try:
            progress_data = self.get_progress(session_id)
            
            is_locked = self.is_session_locked(session_id)
            
            lock_info = self.session_locks.get(session_id)
            
            status = {
                'session_id': session_id,
                'is_locked': is_locked,
                'owned_by_this_process': session_id in self.session_locks,
                'progress': progress_data,
                'lock_info': lock_info
            }
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting session status for {session_id}: {str(e)}")
            return {
                'session_id': session_id,
                'error': str(e)
            }


_progress_manager_instance = None
_progress_manager_lock = threading.Lock()


def get_progress_manager(socketio_instance=None, supabase_client=None) -> ProgressManager:
    """
    Get or create global ProgressManager instance (singleton pattern)
    
    Args:
        socketio_instance: SocketIO instance for real-time updates
        supabase_client: Supabase client instance
        
    Returns:
        ProgressManager instance
    """
    global _progress_manager_instance
    
    with _progress_manager_lock:
        if _progress_manager_instance is None:
            _progress_manager_instance = ProgressManager(socketio_instance, supabase_client)
        
        return _progress_manager_instance


def shutdown_progress_manager():
    """Shutdown the global progress manager instance"""
    global _progress_manager_instance
    
    with _progress_manager_lock:
        if _progress_manager_instance is not None:
            _progress_manager_instance.shutdown()
            _progress_manager_instance = None
