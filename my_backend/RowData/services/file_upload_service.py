"""
Servis za upload fajlova sa streaming podrškom
"""
import os
import tempfile
import shutil
import hashlib
from typing import Iterator, Dict, Any, Optional, List, Tuple
from datetime import datetime
import logging
from ..config.settings import CHUNK_STORAGE_PATH, MAX_CHUNK_SIZE, UPLOAD_EXPIRY_TIME
from ..utils.exceptions import (
    UploadError, ChunkError, FileSystemError, ValidationError
)
# Repository će biti prosleđen preko konstruktora

logger = logging.getLogger(__name__)


class FileUploadService:
    """Servis za upravljanje upload-om fajlova"""
    
    def __init__(self, repository=None):
        self.storage_path = os.path.join(CHUNK_STORAGE_PATH, 'chunks')
        self.temp_path = os.path.join(CHUNK_STORAGE_PATH, 'temp')
        # Repository mora biti prosleđen eksplicitno
        if repository is None:
            raise ValueError("Repository must be provided to FileUploadService")
        self.repository = repository
        
        # Kreiraj direktorijume ako ne postoje
        os.makedirs(self.storage_path, exist_ok=True)
        os.makedirs(self.temp_path, exist_ok=True)
        
        logger.info(f"FileUploadService initialized with storage path: {self.storage_path}")
    
    def _get_chunk_path(self, upload_id: str, chunk_index: int) -> str:
        """Generiše putanju za specifičan chunk"""
        # Dodaj folder za upload_id radi bolje organizacije
        upload_dir = os.path.join(self.storage_path, upload_id)
        os.makedirs(upload_dir, exist_ok=True)
        
        return os.path.join(upload_dir, f"chunk_{chunk_index:05d}.part")
    
    def _get_temp_file_path(self, upload_id: str) -> str:
        """Generiše putanju za privremeni spojeni fajl"""
        return os.path.join(self.temp_path, f"{upload_id}.tmp")
    
    def store_chunk(self, upload_id: str, chunk_index: int, 
                   chunk_data: bytes, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Čuva chunk na disk sa streaming write-om
        
        Returns:
            Dict sa informacijama o sačuvanom chunk-u
        """
        try:
            chunk_path = self._get_chunk_path(upload_id, chunk_index)
            
            # Proveri da li chunk već postoji
            if os.path.exists(chunk_path):
                logger.warning(f"Chunk {chunk_index} for upload {upload_id} already exists")
                # Možda je retry, proveri hash
                existing_hash = self._calculate_file_hash(chunk_path)
                new_hash = hashlib.md5(chunk_data).hexdigest()
                
                if existing_hash != new_hash:
                    raise ChunkError(
                        f"Chunk {chunk_index} already exists with different content",
                        upload_id, chunk_index
                    )
                else:
                    logger.info(f"Chunk {chunk_index} is duplicate, skipping")
                    return {
                        'chunk_index': chunk_index,
                        'size': len(chunk_data),
                        'hash': new_hash,
                        'duplicate': True
                    }
            
            # Streaming write sa buffer-om
            chunk_size_written = 0
            with open(chunk_path, 'wb') as f:
                # Piši u blokovima od 64KB
                block_size = 65536
                for i in range(0, len(chunk_data), block_size):
                    block = chunk_data[i:i + block_size]
                    f.write(block)
                    chunk_size_written += len(block)
            
            # Verifikuj da je ceo chunk upisan
            if chunk_size_written != len(chunk_data):
                os.unlink(chunk_path)
                raise ChunkError(
                    f"Failed to write complete chunk. Written: {chunk_size_written}, Expected: {len(chunk_data)}",
                    upload_id, chunk_index
                )
            
            # Sačuvaj metadata u repository
            chunk_info = {
                'chunk_index': chunk_index,
                'size': chunk_size_written,
                'hash': hashlib.md5(chunk_data).hexdigest(),
                'stored_at': datetime.utcnow().isoformat(),
                'path': chunk_path
            }
            
            self.repository.store_chunk_info(upload_id, chunk_index, chunk_info)
            
            # Ažuriraj upload metadata
            self.repository.update_upload_progress(upload_id, chunk_index, metadata)
            
            logger.info(f"Successfully stored chunk {chunk_index} for upload {upload_id}")
            return chunk_info
            
        except Exception as e:
            logger.error(f"Error storing chunk {chunk_index} for upload {upload_id}: {str(e)}")
            if isinstance(e, (ChunkError, UploadError)):
                raise
            raise FileSystemError(f"Failed to store chunk: {str(e)}", chunk_path)
    
    def get_chunk_stream(self, upload_id: str, chunk_index: int, 
                        buffer_size: int = 65536) -> Iterator[bytes]:
        """
        Streaming read chunk-a sa disk-a
        
        Yields:
            bytes: Blokovi podataka
        """
        chunk_path = self._get_chunk_path(upload_id, chunk_index)
        
        if not os.path.exists(chunk_path):
            raise ChunkError(f"Chunk {chunk_index} not found", upload_id, chunk_index)
        
        try:
            with open(chunk_path, 'rb') as f:
                while True:
                    data = f.read(buffer_size)
                    if not data:
                        break
                    yield data
        except IOError as e:
            raise FileSystemError(f"Failed to read chunk: {str(e)}", chunk_path)
    
    def combine_chunks_streaming(self, upload_id: str) -> Iterator[str]:
        """
        Kombinuje chunk-ove u streaming način bez učitavanja u memoriju
        
        Yields:
            str: Linije iz kombinovanog fajla
        """
        metadata = self.repository.get_upload_metadata(upload_id)
        if not metadata:
            raise UploadError(f"Upload {upload_id} not found", upload_id)
        
        total_chunks = metadata['total_chunks']
        received_chunks = metadata.get('received_chunks', 0)
        
        if received_chunks != total_chunks:
            raise UploadError(
                f"Not all chunks received. Expected: {total_chunks}, Received: {received_chunks}",
                upload_id
            )
        
        buffer = ""
        
        try:
            for chunk_index in range(total_chunks):
                logger.info(f"Processing chunk {chunk_index + 1}/{total_chunks} for upload {upload_id}")
                
                # Streaming read chunk-a
                for data_block in self.get_chunk_stream(upload_id, chunk_index):
                    # Dekodiraj blok
                    decoded = self._safe_decode(data_block)
                    buffer += decoded
                    
                    # Procesuj kompletne linije
                    lines = buffer.split('\n')
                    buffer = lines[-1]  # Zadrži poslednju (možda nepotpunu) liniju
                    
                    for line in lines[:-1]:
                        yield line
            
            # Yield poslednju liniju ako postoji
            if buffer:
                yield buffer
                
        except Exception as e:
            logger.error(f"Error combining chunks for upload {upload_id}: {str(e)}")
            raise
    
    def _safe_decode(self, data: bytes) -> str:
        """Sigurno dekodiranje sa podrškom za različite encoding-e"""
        encodings = ['utf-8', 'utf-16', 'utf-16le', 'utf-16be', 'latin1', 'cp1252']
        
        for encoding in encodings:
            try:
                return data.decode(encoding)
            except UnicodeDecodeError:
                continue
        
        # Ako ništa ne radi, koristi utf-8 sa replace
        return data.decode('utf-8', errors='replace')
    
    def cleanup_upload(self, upload_id: str) -> None:
        """Briše sve chunk-ove i metadata za upload"""
        try:
            # Briši folder sa chunk-ovima
            upload_dir = os.path.join(self.storage_path, upload_id)
            if os.path.exists(upload_dir):
                shutil.rmtree(upload_dir)
                logger.info(f"Deleted chunks for upload {upload_id}")
            
            # Briši temp fajl ako postoji
            temp_file = self._get_temp_file_path(upload_id)
            if os.path.exists(temp_file):
                os.unlink(temp_file)
                logger.info(f"Deleted temp file for upload {upload_id}")
            
            # Briši metadata iz repository-ja
            self.repository.delete_upload(upload_id)
            
        except Exception as e:
            logger.error(f"Error cleaning up upload {upload_id}: {str(e)}")
            # Ne bacaj exception, samo loguj
    
    def cleanup_old_uploads(self) -> int:
        """
        Briše stare upload-ove koji nisu završeni
        
        Returns:
            Broj obrisanih upload-ova
        """
        try:
            old_uploads = self.repository.get_expired_uploads(UPLOAD_EXPIRY_TIME)
            
            for upload_id in old_uploads:
                logger.info(f"Cleaning up expired upload: {upload_id}")
                self.cleanup_upload(upload_id)
            
            # Ako repository ima cleanup_expired_files metod, pozovi ga
            if hasattr(self.repository, 'cleanup_expired_files'):
                self.repository.cleanup_expired_files()
            
            return len(old_uploads)
            
        except Exception as e:
            logger.error(f"Error cleaning up old uploads: {str(e)}")
            return 0
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """Kalkuliše MD5 hash fajla"""
        hash_md5 = hashlib.md5()
        
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        
        return hash_md5.hexdigest()
    
    def verify_upload_integrity(self, upload_id: str) -> Tuple[bool, Optional[str]]:
        """
        Verifikuje integritet upload-a
        
        Returns:
            Tuple: (da li je valjan, poruka o grešci)
        """
        try:
            metadata = self.repository.get_upload_metadata(upload_id)
            if not metadata:
                return False, "Upload not found"
            
            total_chunks = metadata['total_chunks']
            
            # Proveri da li svi chunk-ovi postoje
            for i in range(total_chunks):
                chunk_path = self._get_chunk_path(upload_id, i)
                if not os.path.exists(chunk_path):
                    return False, f"Missing chunk {i}"
                
                # Proveri hash ako postoji u metadata
                chunk_info = self.repository.get_chunk_info(upload_id, i)
                if chunk_info and 'hash' in chunk_info:
                    actual_hash = self._calculate_file_hash(chunk_path)
                    if actual_hash != chunk_info['hash']:
                        return False, f"Chunk {i} hash mismatch"
            
            return True, None
            
        except Exception as e:
            logger.error(f"Error verifying upload {upload_id}: {str(e)}")
            return False, str(e)
    
    def get_upload_size(self, upload_id: str) -> int:
        """Vraća ukupnu veličinu upload-a"""
        total_size = 0
        metadata = self.repository.get_upload_metadata(upload_id)
        
        if not metadata:
            raise UploadError(f"Upload {upload_id} not found", upload_id)
        
        for i in range(metadata['total_chunks']):
            chunk_info = self.repository.get_chunk_info(upload_id, i)
            if chunk_info:
                total_size += chunk_info.get('size', 0)
        
        return total_size