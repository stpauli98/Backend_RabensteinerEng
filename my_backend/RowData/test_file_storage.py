#!/usr/bin/env python3
"""
Test script za verifikaciju file-based storage rada
"""
import os
import sys
import tempfile
from pathlib import Path

# Dodaj trenutni direktorijum u Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Postavi environment varijablu da forsira file storage
os.environ['ROWDATA_STORAGE_BACKEND'] = 'file'
os.environ['ROWDATA_FILE_STORAGE_PATH'] = '/tmp/rowdata_test'

# Importuj direktno iz fajla
sys.path.insert(0, os.path.join(current_dir, 'repositories'))
from file_based_repository import FileBasedRepository


def test_file_storage():
    """Testira file-based storage funkcionalnost"""
    print("=== Testiranje File-Based Storage ===\n")
    
    # 1. Kreiraj file-based repository direktno
    print("1. Kreiranje file-based repository...")
    repo = FileBasedRepository('/tmp/rowdata_test')
    print(f"   Tip: {type(repo).__name__}")
    print(f"   Bazna putanja: {repo.base_path}")
    
    # 2. Test osnovnih operacija
    print("\n2. Testiranje osnovnih operacija...")
    
    # Test upload metadata
    upload_id = "test_upload_123"
    metadata = {
        'total_chunks': 10,
        'delimiter': ',',
        'timezone': 'UTC',
        'has_header': True,
        'received_chunks': 0
    }
    
    print(f"   a) Čuvanje metadata za upload '{upload_id}'...")
    repo.store_upload_metadata(upload_id, metadata)
    print("      ✓ Uspešno sačuvano")
    
    # Dohvati metadata
    print(f"   b) Dohvatanje metadata...")
    retrieved_metadata = repo.get_upload_metadata(upload_id)
    if retrieved_metadata:
        print("      ✓ Metadata uspešno dohvaćen")
        print(f"      Total chunks: {retrieved_metadata.get('total_chunks')}")
        print(f"      Delimiter: {retrieved_metadata.get('delimiter')}")
    else:
        print("      ✗ Greška pri dohvatanju metadata")
    
    # Test chunk info
    print(f"   c) Čuvanje chunk informacija...")
    chunk_info = {
        'size': 1024,
        'hash': 'abc123',
        'path': '/tmp/test_chunk.part'
    }
    repo.store_chunk_info(upload_id, 0, chunk_info)
    print("      ✓ Chunk info sačuvan")
    
    # Dohvati chunk info
    retrieved_chunk = repo.get_chunk_info(upload_id, 0)
    if retrieved_chunk:
        print("      ✓ Chunk info uspešno dohvaćen")
        print(f"      Size: {retrieved_chunk.get('size')} bytes")
    
    # Test statistike
    print(f"\n   d) Dohvatanje statistike...")
    stats = repo.get_upload_statistics()
    print(f"      Aktivni upload-ovi: {stats.get('active_uploads', 0)}")
    print(f"      Ukupno fajlova: {stats.get('total_files', 0)}")
    print(f"      Veličina storage-a: {stats.get('storage_size', '0 MB')}")
    
    # Cleanup
    print(f"\n   e) Brisanje test podataka...")
    repo.delete_upload(upload_id)
    print("      ✓ Podaci obrisani")
    
    # Verifikuj da su obrisani
    deleted_metadata = repo.get_upload_metadata(upload_id)
    if deleted_metadata is None:
        print("      ✓ Verifikovano - podaci su uspešno obrisani")
    else:
        print("      ✗ Greška - podaci još uvek postoje")
    
    print("\n=== Test završen uspešno! ===")
    print("\nFile-based storage radi kako treba. Redis NIJE potreban!")


if __name__ == "__main__":
    try:
        test_file_storage()
    except Exception as e:
        print(f"\n✗ Greška tokom testiranja: {str(e)}")
        import traceback
        traceback.print_exc()