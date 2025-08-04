#!/usr/bin/env python3
"""
Test script za verifikaciju da RowData modul može da se importuje bez Redis-a
"""
import os
import sys

# Postavi environment da koristi file storage
os.environ['ROWDATA_STORAGE_BACKEND'] = 'file'

print("Testing RowData module import...")
print("-" * 50)

try:
    # Test importovanje modula
    from RowData import rowdata_blueprint
    print("✅ Successfully imported rowdata_blueprint")
    
    # Test factory import
    from RowData.repositories.repository_factory import get_repository, get_storage_info
    print("✅ Successfully imported repository factory")
    
    # Test storage info
    storage_info = get_storage_info()
    print(f"✅ Storage backend: {storage_info['backend']} - {storage_info['description']}")
    
    # Test repository creation
    repo = get_repository()
    print(f"✅ Repository created: {type(repo).__name__}")
    
    # Test da li je file-based
    if hasattr(repo, 'base_path'):
        print(f"✅ Using file storage at: {repo.base_path}")
    
    print("\n" + "=" * 50)
    print("🎉 SUCCESS! RowData module works without Redis!")
    print("=" * 50)
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    import traceback
    traceback.print_exc()
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()