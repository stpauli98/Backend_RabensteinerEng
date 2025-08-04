"""
File-based repository za upload metadata - alternativa Redis-u
"""
import json
import os
import time
import fcntl
import threading
from typing import Dict, Any, Optional, List, Set
from datetime import datetime
import logging
from pathlib import Path
try:
    from ..config.settings import CHUNK_STORAGE_PATH, REDIS_TTL
    from ..utils.exceptions import StorageError
except ImportError:
    # Fallback za standalone testiranje
    CHUNK_STORAGE_PATH = '/tmp/row_data_uploads'
    REDIS_TTL = {
        'upload_metadata': 3600,
        'chunk_info': 1800,
        'processing_result': 7200
    }
    class StorageError(Exception):
        def __init__(self, message, code):
            super().__init__(message)
            self.code = code

logger = logging.getLogger(__name__)


class FileBasedRepository:
    """File-based repository koji koristi JSON fajlove umesto Redis-a"""
    
    def __init__(self, base_path: Optional[str] = None):
        self.base_path = Path(base_path or CHUNK_STORAGE_PATH)
        self.metadata_path = self.base_path / 'metadata'
        self.results_path = self.base_path / 'results'
        self.locks_path = self.base_path / 'locks'
        
        # Kreiraj direktorijume
        self.metadata_path.mkdir(parents=True, exist_ok=True)
        self.results_path.mkdir(parents=True, exist_ok=True)
        self.locks_path.mkdir(parents=True, exist_ok=True)
        
        # Thread lock za active_uploads.json
        self._active_uploads_lock = threading.Lock()
        
        logger.info(f"FileBasedRepository initialized at {self.base_path}")
    
    def _make_key(self, key: str) -> str:
        """Kompatibilnost sa Redis repository"""
        return key
    
    def _get_metadata_file(self, upload_id: str) -> Path:
        """Vraća putanju do metadata fajla"""
        return self.metadata_path / f"upload_{upload_id}.json"
    
    def _get_chunk_info_file(self, upload_id: str, chunk_index: int) -> Path:
        """Vraća putanju do chunk info fajla"""
        chunk_dir = self.metadata_path / f"chunks_{upload_id}"
        chunk_dir.mkdir(exist_ok=True)
        return chunk_dir / f"chunk_{chunk_index}.json"
    
    def _get_result_file(self, upload_id: str) -> Path:
        """Vraća putanju do result fajla"""
        return self.results_path / f"result_{upload_id}.json"
    
    def _get_lock_file(self, upload_id: str) -> Path:
        """Vraća putanju do lock fajla"""
        return self.locks_path / f"lock_{upload_id}"
    
    def _read_json_file(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Čita JSON fajl sa error handling"""
        if not file_path.exists():
            return None
        
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Failed to read {file_path}: {str(e)}")
            return None
    
    def _write_json_file(self, file_path: Path, data: Dict[str, Any]) -> None:
        """Piše JSON fajl sa atomskim write"""
        temp_file = file_path.with_suffix('.tmp')
        
        try:
            # Piši u temp fajl
            with open(temp_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            # Atomski rename
            temp_file.replace(file_path)
            
        except Exception as e:
            # Cleanup temp fajl ako postoji
            if temp_file.exists():
                temp_file.unlink()
            raise StorageError(f"Failed to write {file_path}: {str(e)}", "filesystem")
    
    def _update_active_uploads(self, upload_id: str, action: str = 'add') -> None:
        """Ažurira listu aktivnih upload-ova"""
        active_file = self.metadata_path / 'active_uploads.json'
        
        with self._active_uploads_lock:
            # Čitaj postojeće
            active_data = self._read_json_file(active_file) or {'uploads': [], 'updated': None}
            
            if action == 'add':
                if upload_id not in active_data['uploads']:
                    active_data['uploads'].append(upload_id)
            elif action == 'remove':
                active_data['uploads'] = [uid for uid in active_data['uploads'] if uid != upload_id]
            
            active_data['updated'] = datetime.utcnow().isoformat()
            
            # Sačuvaj
            self._write_json_file(active_file, active_data)
    
    def store_upload_metadata(self, upload_id: str, metadata: Dict[str, Any]) -> None:
        """Čuva metadata za upload"""
        try:
            # Dodaj timestamp
            metadata['created_at'] = datetime.utcnow().isoformat()
            metadata['last_activity'] = time.time()
            metadata['_expiry'] = time.time() + REDIS_TTL['upload_metadata']
            
            # Sačuvaj metadata
            metadata_file = self._get_metadata_file(upload_id)
            self._write_json_file(metadata_file, metadata)
            
            # Ažuriraj active uploads
            self._update_active_uploads(upload_id, 'add')
            
            logger.debug(f"Stored metadata for upload {upload_id}")
            
        except Exception as e:
            logger.error(f"Failed to store upload metadata: {str(e)}")
            raise StorageError(f"Failed to store metadata: {str(e)}", "filesystem")
    
    def get_upload_metadata(self, upload_id: str) -> Optional[Dict[str, Any]]:
        """Dohvata metadata za upload"""
        try:
            metadata_file = self._get_metadata_file(upload_id)
            metadata = self._read_json_file(metadata_file)
            
            if metadata:
                # Proveri expiry
                if metadata.get('_expiry', float('inf')) < time.time():
                    # Metadata je istekao, obriši
                    self.delete_upload(upload_id)
                    return None
                
                # Ukloni internal fields
                return {k: v for k, v in metadata.items() if not k.startswith('_')}
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get upload metadata: {str(e)}")
            raise StorageError(f"Failed to get metadata: {str(e)}", "filesystem")
    
    def update_upload_progress(self, upload_id: str, chunk_index: int, 
                             additional_data: Optional[Dict[str, Any]] = None) -> None:
        """Ažurira progress upload-a"""
        try:
            # Dohvati postojeći metadata
            metadata = self.get_upload_metadata(upload_id)
            if not metadata:
                # Ako metadata ne postoji, kreiraj ga sa prosleđenim podacima
                if additional_data and chunk_index == 0:
                    # Ovo je prvi chunk, kreiraj metadata
                    logger.info(f"Creating initial metadata for upload {upload_id}")
                    metadata = additional_data.copy()
                    metadata['received_chunks'] = 0
                    self.store_upload_metadata(upload_id, metadata)
                    metadata = self.get_upload_metadata(upload_id)
                else:
                    raise StorageError(f"Upload {upload_id} not found", "filesystem")
            
            # Ažuriraj
            received_chunks = metadata.get('received_chunks', 0)
            metadata['received_chunks'] = max(received_chunks, chunk_index + 1)
            metadata['last_activity'] = time.time()
            
            # Ažuriraj received chunks list
            chunks_file = self.metadata_path / f"chunks_{upload_id}_received.json"
            chunks_data = self._read_json_file(chunks_file) or {'chunks': []}
            
            if chunk_index not in chunks_data['chunks']:
                chunks_data['chunks'].append(chunk_index)
                self._write_json_file(chunks_file, chunks_data)
            
            # Merge additional data
            if additional_data:
                metadata.update(additional_data)
            
            # Sačuvaj
            self.store_upload_metadata(upload_id, metadata)
            
        except Exception as e:
            logger.error(f"Failed to update upload progress: {str(e)}")
            raise
    
    def store_chunk_info(self, upload_id: str, chunk_index: int, 
                        chunk_info: Dict[str, Any]) -> None:
        """Čuva informacije o pojedinačnom chunk-u"""
        try:
            chunk_file = self._get_chunk_info_file(upload_id, chunk_index)
            chunk_info['_expiry'] = time.time() + REDIS_TTL['chunk_info']
            
            self._write_json_file(chunk_file, chunk_info)
            
        except Exception as e:
            logger.error(f"Failed to store chunk info: {str(e)}")
            raise StorageError(f"Failed to store chunk info: {str(e)}", "filesystem")
    
    def get_chunk_info(self, upload_id: str, chunk_index: int) -> Optional[Dict[str, Any]]:
        """Dohvata informacije o chunk-u"""
        try:
            chunk_file = self._get_chunk_info_file(upload_id, chunk_index)
            chunk_info = self._read_json_file(chunk_file)
            
            if chunk_info:
                # Proveri expiry
                if chunk_info.get('_expiry', float('inf')) < time.time():
                    chunk_file.unlink()
                    return None
                
                return {k: v for k, v in chunk_info.items() if not k.startswith('_')}
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get chunk info: {str(e)}")
            return None
    
    def get_received_chunks(self, upload_id: str) -> Set[int]:
        """Vraća set primljenih chunk index-a"""
        try:
            chunks_file = self.metadata_path / f"chunks_{upload_id}_received.json"
            chunks_data = self._read_json_file(chunks_file)
            
            if chunks_data:
                return set(chunks_data.get('chunks', []))
            
            return set()
            
        except Exception as e:
            logger.error(f"Failed to get received chunks: {str(e)}")
            return set()
    
    def delete_upload(self, upload_id: str) -> None:
        """Briše sve podatke vezane za upload"""
        try:
            # Briši metadata fajl
            metadata_file = self._get_metadata_file(upload_id)
            if metadata_file.exists():
                metadata_file.unlink()
            
            # Briši chunk info direktorijum
            chunk_dir = self.metadata_path / f"chunks_{upload_id}"
            if chunk_dir.exists():
                for file in chunk_dir.iterdir():
                    file.unlink()
                chunk_dir.rmdir()
            
            # Briši received chunks fajl
            chunks_file = self.metadata_path / f"chunks_{upload_id}_received.json"
            if chunks_file.exists():
                chunks_file.unlink()
            
            # Briši result fajl
            result_file = self._get_result_file(upload_id)
            if result_file.exists():
                result_file.unlink()
            
            # Briši lock fajl
            lock_file = self._get_lock_file(upload_id)
            if lock_file.exists():
                lock_file.unlink()
            
            # Ažuriraj active uploads
            self._update_active_uploads(upload_id, 'remove')
            
            logger.info(f"Deleted all data for upload {upload_id}")
            
        except Exception as e:
            logger.error(f"Failed to delete upload: {str(e)}")
    
    def get_expired_uploads(self, expiry_seconds: int) -> List[str]:
        """Pronalazi upload-ove koji su istekli"""
        try:
            expired = []
            current_time = time.time()
            
            # Čitaj active uploads
            active_file = self.metadata_path / 'active_uploads.json'
            active_data = self._read_json_file(active_file)
            
            if not active_data:
                return []
            
            for upload_id in active_data.get('uploads', []):
                metadata = self.get_upload_metadata(upload_id)
                
                if metadata:
                    last_activity = metadata.get('last_activity', 0)
                    if current_time - last_activity > expiry_seconds:
                        expired.append(upload_id)
                else:
                    # Metadata ne postoji, dodaj u expired
                    expired.append(upload_id)
            
            return expired
            
        except Exception as e:
            logger.error(f"Failed to get expired uploads: {str(e)}")
            return []
    
    def store_processing_result(self, upload_id: str, result: Dict[str, Any]) -> None:
        """Čuva rezultat procesiranja"""
        try:
            result['processed_at'] = datetime.utcnow().isoformat()
            result['_expiry'] = time.time() + REDIS_TTL['processing_result']
            
            result_file = self._get_result_file(upload_id)
            self._write_json_file(result_file, result)
            
        except Exception as e:
            logger.error(f"Failed to store processing result: {str(e)}")
            raise StorageError(f"Failed to store result: {str(e)}", "filesystem")
    
    def get_processing_result(self, upload_id: str) -> Optional[Dict[str, Any]]:
        """Dohvata rezultat procesiranja"""
        try:
            result_file = self._get_result_file(upload_id)
            result = self._read_json_file(result_file)
            
            if result:
                # Proveri expiry
                if result.get('_expiry', float('inf')) < time.time():
                    result_file.unlink()
                    return None
                
                return {k: v for k, v in result.items() if not k.startswith('_')}
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get processing result: {str(e)}")
            return None
    
    def get_upload_statistics(self) -> Dict[str, Any]:
        """Vraća statistiku o upload-ovima"""
        try:
            # Broj aktivnih upload-ova
            active_file = self.metadata_path / 'active_uploads.json'
            active_data = self._read_json_file(active_file)
            active_count = len(active_data.get('uploads', [])) if active_data else 0
            
            # Veličina storage-a
            total_size = 0
            file_count = 0
            
            for path in [self.metadata_path, self.results_path]:
                for file in path.rglob('*.json'):
                    total_size += file.stat().st_size
                    file_count += 1
            
            return {
                'active_uploads': active_count,
                'total_files': file_count,
                'storage_size': f"{total_size / 1024 / 1024:.2f} MB",
                'storage_path': str(self.base_path)
            }
            
        except Exception as e:
            logger.error(f"Failed to get statistics: {str(e)}")
            return {}
    
    def acquire_upload_lock(self, upload_id: str, timeout: int = 30) -> bool:
        """Pokušava da dobije ekskluzivni lock za upload"""
        try:
            lock_file = self._get_lock_file(upload_id)
            
            # Proveri da li lock već postoji
            if lock_file.exists():
                # Čitaj lock info
                lock_info = self._read_json_file(lock_file)
                if lock_info:
                    expiry = lock_info.get('expiry', 0)
                    if expiry > time.time():
                        return False  # Lock je još aktivan
            
            # Kreiraj novi lock
            lock_info = {
                'locked_at': datetime.utcnow().isoformat(),
                'expiry': time.time() + timeout
            }
            
            self._write_json_file(lock_file, lock_info)
            return True
            
        except Exception as e:
            logger.error(f"Failed to acquire lock: {str(e)}")
            return False
    
    def release_upload_lock(self, upload_id: str) -> None:
        """Oslobađa lock za upload"""
        try:
            lock_file = self._get_lock_file(upload_id)
            if lock_file.exists():
                lock_file.unlink()
            
        except Exception as e:
            logger.error(f"Failed to release lock: {str(e)}")
    
    def cleanup_expired_files(self) -> int:
        """Briše sve expired fajlove"""
        try:
            cleaned_count = 0
            current_time = time.time()
            
            # Cleanup metadata
            for file in self.metadata_path.glob('upload_*.json'):
                data = self._read_json_file(file)
                if data and data.get('_expiry', float('inf')) < current_time:
                    upload_id = file.stem.replace('upload_', '')
                    self.delete_upload(upload_id)
                    cleaned_count += 1
            
            # Cleanup results
            for file in self.results_path.glob('result_*.json'):
                data = self._read_json_file(file)
                if data and data.get('_expiry', float('inf')) < current_time:
                    file.unlink()
                    cleaned_count += 1
            
            # Cleanup old locks
            for file in self.locks_path.glob('lock_*'):
                data = self._read_json_file(file)
                if data and data.get('expiry', 0) < current_time:
                    file.unlink()
                    cleaned_count += 1
            
            logger.info(f"Cleaned up {cleaned_count} expired files")
            return cleaned_count
            
        except Exception as e:
            logger.error(f"Failed to cleanup expired files: {str(e)}")
            return 0