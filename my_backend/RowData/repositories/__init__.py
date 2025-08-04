"""
Repository sloj za RowData modul
"""

# Factory je primarni način pristupa
from .repository_factory import get_repository, get_storage_info, reset_repository

# Individual repository klase (za direktnu upotrebu ako je potrebno)
try:
    from .upload_repository import UploadRepository
except ImportError:
    UploadRepository = None

from .file_based_repository import FileBasedRepository
from .in_memory_repository import InMemoryRepository

__all__ = [
    'get_repository',
    'get_storage_info', 
    'reset_repository',
    'UploadRepository',
    'FileBasedRepository',
    'InMemoryRepository'
]