"""
Repository za upravljanje upload metadata sa Redis backend-om
"""
import redis
import json
import time
from typing import Dict, Any, Optional, List, Set
from datetime import datetime, timedelta
import logging
from ..config.settings import REDIS_CONFIG, REDIS_KEY_PREFIX, REDIS_TTL
from ..utils.exceptions import RedisError, StorageError

logger = logging.getLogger(__name__)


class UploadRepository:
    """Repository za skladištenje upload metadata u Redis"""
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        if redis_client:
            self.redis = redis_client
        else:
            try:
                # Kreiraj Redis connection pool
                pool = redis.ConnectionPool(**REDIS_CONFIG)
                self.redis = redis.Redis(connection_pool=pool)
                
                # Test konekcije
                self.redis.ping()
                logger.info("Successfully connected to Redis")
                
            except redis.ConnectionError as e:
                logger.error(f"Failed to connect to Redis: {str(e)}")
                raise RedisError(f"Redis connection failed: {str(e)}")
    
    def _make_key(self, key: str) -> str:
        """Dodaje RowData prefiks ključu"""
        return f"{REDIS_KEY_PREFIX}{key}"
    
    def store_upload_metadata(self, upload_id: str, metadata: Dict[str, Any]) -> None:
        """
        Čuva metadata za upload
        
        Args:
            upload_id: Jedinstveni ID upload-a
            metadata: Metadata dictionary
        """
        try:
            key = self._make_key(f"upload:{upload_id}:metadata")
            
            # Dodaj timestamp
            metadata['created_at'] = datetime.utcnow().isoformat()
            metadata['last_activity'] = time.time()
            
            # Serijalizuj i sačuvaj sa TTL
            self.redis.setex(
                key,
                REDIS_TTL['upload_metadata'],
                json.dumps(metadata)
            )
            
            # Dodaj u set aktivnih upload-ova
            self.redis.sadd(self._make_key("active_uploads"), upload_id)
            
            logger.debug(f"Stored metadata for upload {upload_id}")
            
        except Exception as e:
            logger.error(f"Failed to store upload metadata: {str(e)}")
            raise StorageError(f"Failed to store metadata: {str(e)}", "redis")
    
    def get_upload_metadata(self, upload_id: str) -> Optional[Dict[str, Any]]:
        """
        Dohvata metadata za upload
        
        Returns:
            Metadata dict ili None ako ne postoji
        """
        try:
            key = self._make_key(f"upload:{upload_id}:metadata")
            data = self.redis.get(key)
            
            if data:
                return json.loads(data)
            
            return None
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode metadata for {upload_id}: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Failed to get upload metadata: {str(e)}")
            raise StorageError(f"Failed to get metadata: {str(e)}", "redis")
    
    def update_upload_progress(self, upload_id: str, chunk_index: int, 
                             additional_data: Optional[Dict[str, Any]] = None) -> None:
        """
        Ažurira progress upload-a
        
        Args:
            upload_id: ID upload-a
            chunk_index: Index primljenog chunk-a
            additional_data: Dodatni podaci za ažuriranje
        """
        try:
            metadata = self.get_upload_metadata(upload_id)
            if not metadata:
                raise StorageError(f"Upload {upload_id} not found", "redis")
            
            # Ažuriraj received chunks
            received_chunks = metadata.get('received_chunks', 0)
            metadata['received_chunks'] = max(received_chunks, chunk_index + 1)
            metadata['last_activity'] = time.time()
            
            # Dodaj chunk u set primljenih
            chunks_key = self._make_key(f"upload:{upload_id}:chunks")
            self.redis.sadd(chunks_key, chunk_index)
            
            # Postavi TTL na chunks set
            self.redis.expire(chunks_key, REDIS_TTL['chunk_info'])
            
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
        """
        Čuva informacije o pojedinačnom chunk-u
        
        Args:
            upload_id: ID upload-a
            chunk_index: Index chunk-a
            chunk_info: Informacije o chunk-u
        """
        try:
            key = self._make_key(f"upload:{upload_id}:chunk:{chunk_index}")
            
            self.redis.setex(
                key,
                REDIS_TTL['chunk_info'],
                json.dumps(chunk_info)
            )
            
        except Exception as e:
            logger.error(f"Failed to store chunk info: {str(e)}")
            raise StorageError(f"Failed to store chunk info: {str(e)}", "redis")
    
    def get_chunk_info(self, upload_id: str, chunk_index: int) -> Optional[Dict[str, Any]]:
        """Dohvata informacije o chunk-u"""
        try:
            key = self._make_key(f"upload:{upload_id}:chunk:{chunk_index}")
            data = self.redis.get(key)
            
            if data:
                return json.loads(data)
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get chunk info: {str(e)}")
            return None
    
    def get_received_chunks(self, upload_id: str) -> Set[int]:
        """Vraća set primljenih chunk index-a"""
        try:
            key = self._make_key(f"upload:{upload_id}:chunks")
            chunks = self.redis.smembers(key)
            
            return {int(chunk) for chunk in chunks}
            
        except Exception as e:
            logger.error(f"Failed to get received chunks: {str(e)}")
            return set()
    
    def delete_upload(self, upload_id: str) -> None:
        """
        Briše sve podatke vezane za upload
        
        Args:
            upload_id: ID upload-a za brisanje
        """
        try:
            # Pronađi sve ključeve vezane za ovaj upload
            pattern = self._make_key(f"upload:{upload_id}:*")
            keys = list(self.redis.scan_iter(match=pattern))
            
            # Dodaj glavni metadata ključ
            keys.append(self._make_key(f"upload:{upload_id}:metadata"))
            
            # Briši sve ključeve
            if keys:
                self.redis.delete(*keys)
            
            # Ukloni iz aktivnih upload-ova
            self.redis.srem(self._make_key("active_uploads"), upload_id)
            
            logger.info(f"Deleted all data for upload {upload_id}")
            
        except Exception as e:
            logger.error(f"Failed to delete upload: {str(e)}")
            # Ne bacaj exception, samo loguj
    
    def get_expired_uploads(self, expiry_seconds: int) -> List[str]:
        """
        Pronalazi upload-ove koji su istekli
        
        Args:
            expiry_seconds: Broj sekundi nakon kojih upload ističe
            
        Returns:
            Lista upload ID-jeva koji su istekli
        """
        try:
            expired = []
            current_time = time.time()
            
            # Dohvati sve aktivne upload-ove
            active_uploads = self.redis.smembers(self._make_key("active_uploads"))
            
            for upload_id in active_uploads:
                upload_id = upload_id.decode() if isinstance(upload_id, bytes) else upload_id
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
        """
        Čuva rezultat procesiranja
        
        Args:
            upload_id: ID upload-a
            result: Rezultat procesiranja
        """
        try:
            key = self._make_key(f"result:{upload_id}")
            
            result['processed_at'] = datetime.utcnow().isoformat()
            
            self.redis.setex(
                key,
                REDIS_TTL['processing_result'],
                json.dumps(result)
            )
            
        except Exception as e:
            logger.error(f"Failed to store processing result: {str(e)}")
            raise StorageError(f"Failed to store result: {str(e)}", "redis")
    
    def get_processing_result(self, upload_id: str) -> Optional[Dict[str, Any]]:
        """Dohvata rezultat procesiranja"""
        try:
            key = self._make_key(f"result:{upload_id}")
            data = self.redis.get(key)
            
            if data:
                return json.loads(data)
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get processing result: {str(e)}")
            return None
    
    def get_upload_statistics(self) -> Dict[str, Any]:
        """Vraća statistiku o upload-ovima"""
        try:
            active_count = self.redis.scard(self._make_key("active_uploads"))
            
            # Prebrojava po statusu
            stats = {
                'active_uploads': active_count,
                'total_keys': 0,
                'memory_usage': 0
            }
            
            # Ukupan broj ključeva
            pattern = self._make_key("*")
            stats['total_keys'] = len(list(self.redis.scan_iter(match=pattern, count=100)))
            
            # Memorijska potrošnja (ako je dostupno)
            try:
                info = self.redis.info('memory')
                stats['memory_usage'] = info.get('used_memory_human', 'N/A')
            except:
                pass
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get statistics: {str(e)}")
            return {}
    
    def acquire_upload_lock(self, upload_id: str, timeout: int = 30) -> bool:
        """
        Pokušava da dobije ekskluzivni lock za upload
        
        Args:
            upload_id: ID upload-a
            timeout: Lock timeout u sekundama
            
        Returns:
            True ako je lock dobijen
        """
        try:
            key = self._make_key(f"lock:upload:{upload_id}")
            
            # SET NX EX - postavi samo ako ne postoji, sa expiry
            acquired = self.redis.set(key, "1", nx=True, ex=timeout)
            
            return bool(acquired)
            
        except Exception as e:
            logger.error(f"Failed to acquire lock: {str(e)}")
            return False
    
    def release_upload_lock(self, upload_id: str) -> None:
        """Oslobađa lock za upload"""
        try:
            key = self._make_key(f"lock:upload:{upload_id}")
            self.redis.delete(key)
            
        except Exception as e:
            logger.error(f"Failed to release lock: {str(e)}")
            # Ne bacaj exception