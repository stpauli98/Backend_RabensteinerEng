"""
Factory pattern za kreiranje repository instance
"""
import os
import logging
from typing import Union
from ..config.settings import REDIS_CONFIG

logger = logging.getLogger(__name__)

# Cache za repository instancu
_repository_instance = None


def get_repository() -> Union['UploadRepository', 'FileBasedRepository', 'InMemoryRepository']:
    """
    Factory funkcija koja vraća odgovarajući repository na osnovu konfiguracije
    
    Priority:
    1. Environment varijabla ROWDATA_STORAGE_BACKEND
    2. Pokušaj Redis ako je dostupan
    3. File-based storage kao fallback
    4. In-memory kao poslednja opcija
    
    Returns:
        Repository instanca
    """
    global _repository_instance
    
    # Vrati postojeću instancu ako postoji
    if _repository_instance is not None:
        return _repository_instance
    
    # Proveri environment varijablu
    storage_backend = os.environ.get('ROWDATA_STORAGE_BACKEND', 'auto').lower()
    
    if storage_backend == 'redis':
        # Forsiraj Redis
        try:
            from .upload_repository import UploadRepository
            _repository_instance = UploadRepository()
            logger.info("Using Redis storage backend")
            return _repository_instance
        except Exception as e:
            logger.error(f"Failed to initialize Redis repository: {str(e)}")
            raise
    
    elif storage_backend == 'file':
        # Forsiraj file-based
        from .file_based_repository import FileBasedRepository
        _repository_instance = FileBasedRepository()
        logger.info("Using file-based storage backend")
        return _repository_instance
    
    elif storage_backend == 'memory':
        # Forsiraj in-memory
        from .in_memory_repository import InMemoryRepository
        _repository_instance = InMemoryRepository()
        logger.info("Using in-memory storage backend")
        return _repository_instance
    
    else:
        # Auto-detect mode
        # 1. Pokušaj Redis
        try:
            import redis
            # Test Redis konekcije
            test_client = redis.Redis(**REDIS_CONFIG)
            test_client.ping()
            
            from .upload_repository import UploadRepository
            _repository_instance = UploadRepository()
            logger.info("Auto-detected and using Redis storage backend")
            return _repository_instance
            
        except Exception as e:
            logger.debug(f"Redis not available: {str(e)}")
        
        # 2. Koristi file-based kao primarni fallback
        try:
            from .file_based_repository import FileBasedRepository
            _repository_instance = FileBasedRepository()
            logger.info("Using file-based storage backend (Redis not available)")
            return _repository_instance
            
        except Exception as e:
            logger.error(f"Failed to initialize file-based repository: {str(e)}")
        
        # 3. In-memory kao poslednja opcija
        from .in_memory_repository import InMemoryRepository
        _repository_instance = InMemoryRepository()
        logger.warning("Using in-memory storage backend (no persistent storage available)")
        return _repository_instance


def reset_repository():
    """Reset repository instance (uglavnom za testiranje)"""
    global _repository_instance
    _repository_instance = None
    logger.debug("Repository instance reset")


def get_storage_info() -> dict:
    """Vraća informacije o trenutnom storage backend-u"""
    repository = get_repository()
    
    info = {
        'backend': repository.__class__.__name__,
        'persistent': True,
        'distributed': False,
        'description': 'Unknown storage backend'
    }
    
    if 'UploadRepository' in info['backend']:
        info.update({
            'backend': 'Redis',
            'distributed': True,
            'description': 'Redis-based distributed storage'
        })
    elif 'FileBasedRepository' in info['backend']:
        info.update({
            'backend': 'File',
            'description': 'File-based persistent storage'
        })
    elif 'InMemoryRepository' in info['backend']:
        info.update({
            'backend': 'Memory',
            'persistent': False,
            'description': 'In-memory storage (non-persistent)'
        })
    
    return info