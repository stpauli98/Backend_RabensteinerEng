"""
In-memory repository za upload metadata - alternativa Redis-u
"""
import json
import time
from typing import Dict, Any, Optional, List, Set
from datetime import datetime
import logging
import threading
from ..config.settings import REDIS_TTL
from ..utils.exceptions import StorageError

logger = logging.getLogger(__name__)


class InMemoryRepository:
    """In-memory repository kao alternativa Redis-u"""
    
    def __init__(self):
        self._storage = {}
        self._locks = {}
        self._lock = threading.Lock()
        logger.info("Using in-memory storage (Redis not required)")
    
    def _make_key(self, key: str) -> str:
        """Kompatibilnost sa Redis repository"""
        return key
    
    def _cleanup_expired(self):
        """Briše istekle ključeve"""
        current_time = time.time()
        with self._lock:
            expired_keys = []
            for key, data in self._storage.items():
                if isinstance(data, dict) and '_expiry' in data:
                    if data['_expiry'] < current_time:
                        expired_keys.append(key)
            
            for key in expired_keys:
                del self._storage[key]
    
    def store_upload_metadata(self, upload_id: str, metadata: Dict[str, Any]) -> None:
        """Čuva metadata za upload"""
        try:
            key = self._make_key(f"upload:{upload_id}:metadata")
            
            # Dodaj timestamp i expiry
            metadata['created_at'] = datetime.utcnow().isoformat()
            metadata['last_activity'] = time.time()
            metadata['_expiry'] = time.time() + REDIS_TTL['upload_metadata']
            
            with self._lock:
                self._storage[key] = metadata
                
                # Dodaj u set aktivnih upload-ova
                active_key = self._make_key("active_uploads")
                if active_key not in self._storage:
                    self._storage[active_key] = set()
                self._storage[active_key].add(upload_id)
            
            logger.debug(f"Stored metadata for upload {upload_id}")
            
        except Exception as e:
            logger.error(f"Failed to store upload metadata: {str(e)}")
            raise StorageError(f"Failed to store metadata: {str(e)}", "memory")
    
    def get_upload_metadata(self, upload_id: str) -> Optional[Dict[str, Any]]:
        """Dohvata metadata za upload"""
        try:
            self._cleanup_expired()
            key = self._make_key(f"upload:{upload_id}:metadata")
            
            with self._lock:
                data = self._storage.get(key)
                
            if data:
                # Ukloni internal fields
                return {k: v for k, v in data.items() if not k.startswith('_')}
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get upload metadata: {str(e)}")
            raise StorageError(f"Failed to get metadata: {str(e)}", "memory")
    
    def update_upload_progress(self, upload_id: str, chunk_index: int, 
                             additional_data: Optional[Dict[str, Any]] = None) -> None:
        """Ažurira progress upload-a"""
        try:
            metadata = self.get_upload_metadata(upload_id)
            if not metadata:
                raise StorageError(f"Upload {upload_id} not found", "memory")
            
            # Ažuriraj received chunks
            received_chunks = metadata.get('received_chunks', 0)
            metadata['received_chunks'] = max(received_chunks, chunk_index + 1)
            metadata['last_activity'] = time.time()
            
            # Dodaj chunk u set primljenih
            chunks_key = self._make_key(f"upload:{upload_id}:chunks")
            with self._lock:
                if chunks_key not in self._storage:
                    self._storage[chunks_key] = set()
                self._storage[chunks_key].add(chunk_index)
            
            # Merge sa dodatnim podacima
            if additional_data:
                metadata.update(additional_data)
            
            # Sačuvaj ažurirane podatke
            self.store_upload_metadata(upload_id, metadata)
            
        except Exception as e:
            logger.error(f"Failed to update upload progress: {str(e)}")
            raise
    
    def store_chunk_info(self, upload_id: str, chunk_index: int, 
                        chunk_info: Dict[str, Any]) -> None:
        """Čuva informacije o pojedinačnom chunk-u"""
        try:
            key = self._make_key(f"upload:{upload_id}:chunk:{chunk_index}")
            
            chunk_info['_expiry'] = time.time() + REDIS_TTL['chunk_info']
            
            with self._lock:
                self._storage[key] = chunk_info
            
        except Exception as e:
            logger.error(f"Failed to store chunk info: {str(e)}")
            raise StorageError(f"Failed to store chunk info: {str(e)}", "memory")
    
    def get_chunk_info(self, upload_id: str, chunk_index: int) -> Optional[Dict[str, Any]]:
        """Dohvata informacije o chunk-u"""
        try:
            self._cleanup_expired()
            key = self._make_key(f"upload:{upload_id}:chunk:{chunk_index}")
            
            with self._lock:
                data = self._storage.get(key)
            
            if data:
                return {k: v for k, v in data.items() if not k.startswith('_')}
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get chunk info: {str(e)}")
            return None
    
    def get_received_chunks(self, upload_id: str) -> Set[int]:
        """Vraća set primljenih chunk index-a"""
        try:
            key = self._make_key(f"upload:{upload_id}:chunks")
            
            with self._lock:
                chunks = self._storage.get(key, set())
            
            return set(chunks)
            
        except Exception as e:
            logger.error(f"Failed to get received chunks: {str(e)}")
            return set()
    
    def delete_upload(self, upload_id: str) -> None:
        """Briše sve podatke vezane za upload"""
        try:
            with self._lock:
                # Pronađi sve ključeve vezane za ovaj upload
                keys_to_delete = []
                prefix = f"upload:{upload_id}:"
                
                for key in self._storage.keys():
                    if key.startswith(prefix):
                        keys_to_delete.append(key)
                
                # Briši sve ključeve
                for key in keys_to_delete:
                    del self._storage[key]
                
                # Ukloni iz aktivnih upload-ova
                active_key = self._make_key("active_uploads")
                if active_key in self._storage:
                    self._storage[active_key].discard(upload_id)
            
            logger.info(f"Deleted all data for upload {upload_id}")
            
        except Exception as e:
            logger.error(f"Failed to delete upload: {str(e)}")
    
    def get_expired_uploads(self, expiry_seconds: int) -> List[str]:
        """Pronalazi upload-ove koji su istekli"""
        try:
            expired = []
            current_time = time.time()
            
            with self._lock:
                active_key = self._make_key("active_uploads")
                active_uploads = self._storage.get(active_key, set())
                
                for upload_id in active_uploads:
                    metadata = self.get_upload_metadata(upload_id)
                    
                    if metadata:
                        last_activity = metadata.get('last_activity', 0)
                        if current_time - last_activity > expiry_seconds:
                            expired.append(upload_id)
                    else:
                        expired.append(upload_id)
            
            return expired
            
        except Exception as e:
            logger.error(f"Failed to get expired uploads: {str(e)}")
            return []
    
    def store_processing_result(self, upload_id: str, result: Dict[str, Any]) -> None:
        """Čuva rezultat procesiranja"""
        try:
            key = self._make_key(f"result:{upload_id}")
            
            result['processed_at'] = datetime.utcnow().isoformat()
            result['_expiry'] = time.time() + REDIS_TTL['processing_result']
            
            with self._lock:
                self._storage[key] = result
            
        except Exception as e:
            logger.error(f"Failed to store processing result: {str(e)}")
            raise StorageError(f"Failed to store result: {str(e)}", "memory")
    
    def get_processing_result(self, upload_id: str) -> Optional[Dict[str, Any]]:
        """Dohvata rezultat procesiranja"""
        try:
            self._cleanup_expired()
            key = self._make_key(f"result:{upload_id}")
            
            with self._lock:
                data = self._storage.get(key)
            
            if data:
                return {k: v for k, v in data.items() if not k.startswith('_')}
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get processing result: {str(e)}")
            return None
    
    def get_upload_statistics(self) -> Dict[str, Any]:
        """Vraća statistiku o upload-ovima"""
        try:
            with self._lock:
                active_key = self._make_key("active_uploads")
                active_count = len(self._storage.get(active_key, set()))
                
                stats = {
                    'active_uploads': active_count,
                    'total_keys': len(self._storage),
                    'memory_usage': 'In-memory storage'
                }
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get statistics: {str(e)}")
            return {}
    
    def acquire_upload_lock(self, upload_id: str, timeout: int = 30) -> bool:
        """Pokušava da dobije ekskluzivni lock za upload"""
        try:
            lock_key = self._make_key(f"lock:upload:{upload_id}")
            current_time = time.time()
            
            with self._lock:
                if lock_key in self._storage:
                    # Proveri da li je lock istekao
                    if self._storage[lock_key] > current_time:
                        return False
                
                # Postavi lock
                self._storage[lock_key] = current_time + timeout
                
            return True
            
        except Exception as e:
            logger.error(f"Failed to acquire lock: {str(e)}")
            return False
    
    def release_upload_lock(self, upload_id: str) -> None:
        """Oslobađa lock za upload"""
        try:
            lock_key = self._make_key(f"lock:upload:{upload_id}")
            
            with self._lock:
                if lock_key in self._storage:
                    del self._storage[lock_key]
            
        except Exception as e:
            logger.error(f"Failed to release lock: {str(e)}")