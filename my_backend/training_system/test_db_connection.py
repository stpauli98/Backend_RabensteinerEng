#!/usr/bin/env python3
"""
Test script for database connections and new training tables
Tests all new tables and functions we just created
"""

import sys
import os
import uuid
import json
from datetime import datetime

# Add parent directory to path to import supabase_client
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from supabase_client import get_supabase_client
    print("✅ Successfully imported supabase_client")
except ImportError as e:
    print(f"❌ Failed to import supabase_client: {e}")
    sys.exit(1)


def test_database_connection():
    """Test basic database connection"""
    print("\n🔧 Testing database connection...")
    
    try:
        supabase = get_supabase_client()
        if not supabase:
            print("❌ Failed to get Supabase client")
            return False
        
        print("✅ Supabase client created successfully")
        return True
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return False


def test_base_tables():
    """Test that base tables exist and work"""
    print("\n📋 Testing base tables...")
    
    try:
        supabase = get_supabase_client()
        
        # Test sessions table
        print("  Testing sessions table...")
        response = supabase.table('sessions').select('*').limit(1).execute()
        print(f"  ✅ sessions table accessible, found {len(response.data)} records")
        
        # Test session_mappings table
        print("  Testing session_mappings table...")
        response = supabase.table('session_mappings').select('*').limit(1).execute()
        print(f"  ✅ session_mappings table accessible, found {len(response.data)} records")
        
        # Test time_info table
        print("  Testing time_info table...")
        response = supabase.table('time_info').select('*').limit(1).execute()
        print(f"  ✅ time_info table accessible, found {len(response.data)} records")
        
        # Test zeitschritte table
        print("  Testing zeitschritte table...")
        response = supabase.table('zeitschritte').select('*').limit(1).execute()
        print(f"  ✅ zeitschritte table accessible, found {len(response.data)} records")
        
        # Test files table
        print("  Testing files table...")
        response = supabase.table('files').select('*').limit(1).execute()
        print(f"  ✅ files table accessible, found {len(response.data)} records")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Base tables test failed: {e}")
        return False


def test_training_tables():
    """Test that new training tables exist and work"""
    print("\n🚀 Testing training tables...")
    
    try:
        supabase = get_supabase_client()
        
        # Test training_results table
        print("  Testing training_results table...")
        response = supabase.table('training_results').select('*').limit(1).execute()
        print(f"  ✅ training_results table accessible, found {len(response.data)} records")
        
        # Test training_progress table
        print("  Testing training_progress table...")
        response = supabase.table('training_progress').select('*').limit(1).execute()
        print(f"  ✅ training_progress table accessible, found {len(response.data)} records")
        
        # Test training_logs table
        print("  Testing training_logs table...")
        response = supabase.table('training_logs').select('*').limit(1).execute()
        print(f"  ✅ training_logs table accessible, found {len(response.data)} records")
        
        # Test training_visualizations table
        print("  Testing training_visualizations table...")
        response = supabase.table('training_visualizations').select('*').limit(1).execute()
        print(f"  ✅ training_visualizations table accessible, found {len(response.data)} records")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Training tables test failed: {e}")
        return False


def test_session_functions():
    """Test PostgreSQL functions for session management"""
    print("\n⚙️ Testing PostgreSQL functions...")
    
    try:
        supabase = get_supabase_client()
        
        # Test cleanup_abandoned_sessions function
        print("  Testing cleanup_abandoned_sessions function...")
        response = supabase.rpc('cleanup_abandoned_sessions').execute()
        cleanup_count = response.data
        print(f"  ✅ cleanup_abandoned_sessions executed, cleaned up {cleanup_count} sessions")
        
        # Create a test session for function testing
        test_session_response = supabase.table('sessions').insert({}).execute()
        if not test_session_response.data:
            raise Exception("Failed to create test session")
        
        test_session_id = test_session_response.data[0]['id']
        test_process_id = f"test_process_{uuid.uuid4().hex[:8]}"
        
        print(f"  Created test session: {test_session_id}")
        
        # Test acquire_session_lock function
        print("  Testing acquire_session_lock function...")
        response = supabase.rpc('acquire_session_lock', {
            'p_session_id': test_session_id,
            'p_process_id': test_process_id,
            'p_process_info': {'test': True, 'timestamp': datetime.now().isoformat()}
        }).execute()
        
        lock_acquired = response.data
        print(f"  ✅ acquire_session_lock executed, lock acquired: {lock_acquired}")
        
        # Test update_session_heartbeat function
        print("  Testing update_session_heartbeat function...")
        response = supabase.rpc('update_session_heartbeat', {
            'p_session_id': test_session_id,
            'p_process_id': test_process_id
        }).execute()
        
        heartbeat_updated = response.data
        print(f"  ✅ update_session_heartbeat executed, heartbeat updated: {heartbeat_updated}")
        
        # Cleanup test session
        supabase.table('sessions').delete().eq('id', test_session_id).execute()
        print(f"  🧹 Cleaned up test session: {test_session_id}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ PostgreSQL functions test failed: {e}")
        return False


def test_data_loader_integration():
    """Test data_loader.py integration with database"""
    print("\n🔗 Testing data_loader integration...")
    
    try:
        from data_loader import create_data_loader
        
        # Create data loader
        data_loader = create_data_loader()
        print("  ✅ DataLoader created successfully")
        
        # Test if supabase client is working
        if not data_loader.supabase:
            raise Exception("DataLoader has no supabase client")
        
        print("  ✅ DataLoader has valid supabase client")
        
        # Test basic operations (without actual session data)
        try:
            # This will fail gracefully since we don't have test data
            session_data = data_loader.load_session_data("test-session-id")
            print("  ⚠️  load_session_data executed (may have returned empty data)")
        except Exception as e:
            print(f"  ⚠️  load_session_data failed as expected (no test data): {e}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ DataLoader integration test failed: {e}")
        return False


def test_existing_supabase_functions():
    """Test existing supabase_client.py functions"""
    print("\n🔄 Testing existing supabase_client functions...")
    
    try:
        from supabase_client import create_or_get_session_uuid, save_time_info, save_zeitschritte
        
        # Test create_or_get_session_uuid
        print("  Testing create_or_get_session_uuid...")
        test_string_session = f"test_session_{uuid.uuid4().hex[:8]}"
        session_uuid = create_or_get_session_uuid(test_string_session)
        
        if session_uuid:
            print(f"  ✅ create_or_get_session_uuid successful: {session_uuid}")
            
            # Test save_time_info
            print("  Testing save_time_info...")
            test_time_info = {
                "jahr": True,
                "monat": False,
                "woche": True,
                "feiertag": False,
                "zeitzone": "UTC",
                "category_data": {"test": "data"}
            }
            
            time_info_saved = save_time_info(session_uuid, test_time_info)
            print(f"  ✅ save_time_info result: {time_info_saved}")
            
            # Test save_zeitschritte
            print("  Testing save_zeitschritte...")
            test_zeitschritte = {
                "eingabe": "24",
                "ausgabe": "1",
                "zeitschrittweite": "1",
                "offset": "0"
            }
            
            zeitschritte_saved = save_zeitschritte(session_uuid, test_zeitschritte)
            print(f"  ✅ save_zeitschritte result: {zeitschritte_saved}")
            
            # Cleanup test data
            supabase = get_supabase_client()
            supabase.table('sessions').delete().eq('id', session_uuid).execute()
            print(f"  🧹 Cleaned up test session: {session_uuid}")
            
        else:
            print("  ❌ create_or_get_session_uuid failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"  ❌ Existing functions test failed: {e}")
        return False


def main():
    """Run all database tests"""
    print("🧪 Starting Database Connection Tests")
    print("=" * 50)
    
    tests = [
        ("Database Connection", test_database_connection),
        ("Base Tables", test_base_tables),
        ("Training Tables", test_training_tables),
        ("PostgreSQL Functions", test_session_functions),
        ("DataLoader Integration", test_data_loader_integration),
        ("Existing Functions", test_existing_supabase_functions),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("🏁 TEST SUMMARY")
    print("=" * 50)
    
    passed = 0
    failed = 0
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status}: {test_name}")
        if result:
            passed += 1
        else:
            failed += 1
    
    print(f"\nTotal: {passed + failed} tests")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    
    if failed == 0:
        print("\n🎉 All tests passed! Database is ready for training system.")
    else:
        print(f"\n⚠️  {failed} test(s) failed. Please check the issues above.")
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)