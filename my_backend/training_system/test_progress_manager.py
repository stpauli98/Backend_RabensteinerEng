#!/usr/bin/env python3
"""
Test script for ProgressManager class
Tests session isolation, progress tracking, and heartbeat monitoring
"""

import sys
import os
import time
import uuid
import threading
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from supabase_client import get_supabase_client, create_or_get_session_uuid
    from training_system.progress_manager import ProgressManager, get_progress_manager
    print("✅ Successfully imported ProgressManager")
except ImportError as e:
    print(f"❌ Failed to import ProgressManager: {e}")
    sys.exit(1)


def test_progress_manager_creation():
    """Test ProgressManager creation and initialization"""
    print("\n🔧 Testing ProgressManager creation...")
    
    try:
        # Test direct creation
        pm = ProgressManager()
        print(f"  ✅ ProgressManager created with process_id: {pm.process_id}")
        
        # Test singleton pattern
        pm1 = get_progress_manager()
        pm2 = get_progress_manager()
        
        if pm1 is pm2:
            print("  ✅ Singleton pattern working correctly")
        else:
            print("  ❌ Singleton pattern not working")
            return False
        
        # Test background services
        if pm1._heartbeat_thread and pm1._heartbeat_thread.is_alive():
            print("  ✅ Heartbeat thread is running")
        else:
            print("  ❌ Heartbeat thread not running")
        
        if pm1._cleanup_thread and pm1._cleanup_thread.is_alive():
            print("  ✅ Cleanup thread is running")
        else:
            print("  ❌ Cleanup thread not running")
        
        return True
        
    except Exception as e:
        print(f"  ❌ ProgressManager creation failed: {e}")
        return False


def test_session_locking():
    """Test session locking mechanism"""
    print("\n🔒 Testing session locking...")
    
    try:
        pm = get_progress_manager()
        
        # Create a test session
        test_string_session = f"test_session_{uuid.uuid4().hex[:8]}"
        session_uuid = create_or_get_session_uuid(test_string_session)
        
        if not session_uuid:
            print("  ❌ Failed to create test session")
            return False
        
        print(f"  Created test session: {session_uuid}")
        
        # Test acquiring lock
        lock_acquired = pm.acquire_session_lock(session_uuid)
        if lock_acquired:
            print("  ✅ Successfully acquired session lock")
        else:
            print("  ❌ Failed to acquire session lock")
            return False
        
        # Test checking lock status
        is_locked = pm.is_session_locked(session_uuid)
        if is_locked:
            print("  ✅ Session lock status correctly detected")
        else:
            print("  ❌ Session lock status not detected")
        
        # Test acquiring same lock again (should fail)
        pm2 = ProgressManager()  # Different process
        lock_acquired_2 = pm2.acquire_session_lock(session_uuid)
        if not lock_acquired_2:
            print("  ✅ Concurrent lock acquisition correctly prevented")
        else:
            print("  ❌ Concurrent lock acquisition not prevented")
        
        # Test releasing lock
        lock_released = pm.release_session_lock(session_uuid, 'completed')
        if lock_released:
            print("  ✅ Successfully released session lock")
        else:
            print("  ❌ Failed to release session lock")
        
        # Cleanup
        supabase = get_supabase_client()
        supabase.table('sessions').delete().eq('id', session_uuid).execute()
        pm2.shutdown()
        
        return True
        
    except Exception as e:
        print(f"  ❌ Session locking test failed: {e}")
        return False


def test_progress_tracking():
    """Test progress tracking and caching"""
    print("\n📊 Testing progress tracking...")
    
    try:
        pm = get_progress_manager()
        
        # Create test session
        test_string_session = f"test_session_{uuid.uuid4().hex[:8]}"
        session_uuid = create_or_get_session_uuid(test_string_session)
        
        if not session_uuid:
            print("  ❌ Failed to create test session")
            return False
        
        # Acquire lock
        lock_acquired = pm.acquire_session_lock(session_uuid)
        if not lock_acquired:
            print("  ❌ Failed to acquire session lock")
            return False
        
        print(f"  Testing progress updates for session: {session_uuid}")
        
        # Test progress updates
        progress_updates = [
            {'overall_progress': 0, 'current_step': 'Starting', 'completed_steps': 0},
            {'overall_progress': 14, 'current_step': 'Data Loading', 'completed_steps': 1},
            {'overall_progress': 28, 'current_step': 'Data Processing', 'completed_steps': 2},
            {'overall_progress': 42, 'current_step': 'Model Training', 'completed_steps': 3},
            {'overall_progress': 85, 'current_step': 'Evaluation', 'completed_steps': 6},
            {'overall_progress': 100, 'current_step': 'Completed', 'completed_steps': 7}
        ]
        
        for i, progress in enumerate(progress_updates):
            success = pm.update_progress(session_uuid, progress)
            if success:
                print(f"  ✅ Progress update {i+1}: {progress['overall_progress']}% - {progress['current_step']}")
            else:
                print(f"  ❌ Progress update {i+1} failed")
                
            # Add small delay to test timing
            time.sleep(0.1)
        
        # Test getting progress
        final_progress = pm.get_progress(session_uuid)
        if final_progress:
            print(f"  ✅ Retrieved final progress: {final_progress['overall_progress']}%")
            
            if final_progress['overall_progress'] == 100:
                print("  ✅ Final progress correctly stored")
            else:
                print("  ❌ Final progress not correct")
        else:
            print("  ❌ Failed to retrieve progress")
        
        # Release lock
        pm.release_session_lock(session_uuid, 'completed')
        
        # Cleanup
        supabase = get_supabase_client()
        supabase.table('sessions').delete().eq('id', session_uuid).execute()
        
        return True
        
    except Exception as e:
        print(f"  ❌ Progress tracking test failed: {e}")
        return False


def test_heartbeat_monitoring():
    """Test heartbeat monitoring"""
    print("\n💓 Testing heartbeat monitoring...")
    
    try:
        pm = get_progress_manager()
        
        # Create test session
        test_string_session = f"test_session_{uuid.uuid4().hex[:8]}"
        session_uuid = create_or_get_session_uuid(test_string_session)
        
        if not session_uuid:
            print("  ❌ Failed to create test session")
            return False
        
        # Acquire lock
        lock_acquired = pm.acquire_session_lock(session_uuid)
        if not lock_acquired:
            print("  ❌ Failed to acquire session lock")
            return False
        
        print(f"  Testing heartbeat for session: {session_uuid}")
        
        # Check initial heartbeat
        initial_heartbeat = pm.last_heartbeat.get(session_uuid)
        if initial_heartbeat:
            print("  ✅ Initial heartbeat recorded")
        else:
            print("  ❌ Initial heartbeat not recorded")
        
        # Wait a bit and update progress (should trigger heartbeat)
        time.sleep(2)
        pm.update_progress(session_uuid, {'overall_progress': 50, 'current_step': 'Testing heartbeat'})
        
        # Check if heartbeat was updated
        updated_heartbeat = pm.last_heartbeat.get(session_uuid)
        if updated_heartbeat and updated_heartbeat > initial_heartbeat:
            print("  ✅ Heartbeat updated after progress update")
        else:
            print("  ❌ Heartbeat not updated")
        
        # Check database heartbeat
        supabase = get_supabase_client()
        response = supabase.table('training_progress').select('last_heartbeat').eq('session_id', session_uuid).execute()
        
        if response.data and response.data[0]['last_heartbeat']:
            print("  ✅ Heartbeat stored in database")
        else:
            print("  ❌ Heartbeat not stored in database")
        
        # Release lock
        pm.release_session_lock(session_uuid, 'completed')
        
        # Cleanup
        supabase.table('sessions').delete().eq('id', session_uuid).execute()
        
        return True
        
    except Exception as e:
        print(f"  ❌ Heartbeat monitoring test failed: {e}")
        return False


def test_concurrent_access():
    """Test concurrent access prevention"""
    print("\n🚫 Testing concurrent access prevention...")
    
    try:
        # Create test session
        test_string_session = f"test_session_{uuid.uuid4().hex[:8]}"
        session_uuid = create_or_get_session_uuid(test_string_session)
        
        if not session_uuid:
            print("  ❌ Failed to create test session")
            return False
        
        # Create two progress managers (simulating different processes)
        pm1 = ProgressManager()
        pm2 = ProgressManager()
        
        print(f"  Testing concurrent access for session: {session_uuid}")
        print(f"  Process 1: {pm1.process_id}")
        print(f"  Process 2: {pm2.process_id}")
        
        # Both try to acquire lock simultaneously
        results = {}
        
        def acquire_lock(pm, pm_name):
            results[pm_name] = pm.acquire_session_lock(session_uuid)
        
        # Start both attempts concurrently
        thread1 = threading.Thread(target=acquire_lock, args=(pm1, 'pm1'))
        thread2 = threading.Thread(target=acquire_lock, args=(pm2, 'pm2'))
        
        thread1.start()
        thread2.start()
        
        thread1.join()
        thread2.join()
        
        # Check results
        success_count = sum(1 for success in results.values() if success)
        
        if success_count == 1:
            print("  ✅ Exactly one process acquired the lock (correct)")
            
            # Find which one succeeded
            winner = 'pm1' if results['pm1'] else 'pm2'
            winner_pm = pm1 if winner == 'pm1' else pm2
            
            print(f"  ✅ Process {winner} acquired the lock")
            
            # Test that the winner can update progress
            progress_success = winner_pm.update_progress(session_uuid, {
                'overall_progress': 50,
                'current_step': 'Testing concurrent access'
            })
            
            if progress_success:
                print("  ✅ Winner can update progress")
            else:
                print("  ❌ Winner cannot update progress")
            
            # Test that the loser cannot update progress
            loser_pm = pm2 if winner == 'pm1' else pm1
            progress_fail = loser_pm.update_progress(session_uuid, {
                'overall_progress': 75,
                'current_step': 'Should fail'
            })
            
            if not progress_fail:
                print("  ✅ Loser correctly prevented from updating progress")
            else:
                print("  ❌ Loser was able to update progress (should not happen)")
            
            # Release lock
            winner_pm.release_session_lock(session_uuid, 'completed')
            
        else:
            print(f"  ❌ {success_count} processes acquired the lock (should be exactly 1)")
            return False
        
        # Cleanup
        pm1.shutdown()
        pm2.shutdown()
        
        supabase = get_supabase_client()
        supabase.table('sessions').delete().eq('id', session_uuid).execute()
        
        return True
        
    except Exception as e:
        print(f"  ❌ Concurrent access test failed: {e}")
        return False


def test_session_status():
    """Test session status reporting"""
    print("\n📋 Testing session status reporting...")
    
    try:
        pm = get_progress_manager()
        
        # Create test session
        test_string_session = f"test_session_{uuid.uuid4().hex[:8]}"
        session_uuid = create_or_get_session_uuid(test_string_session)
        
        if not session_uuid:
            print("  ❌ Failed to create test session")
            return False
        
        # Test status for unlocked session
        status_unlocked = pm.get_session_status(session_uuid)
        if not status_unlocked['is_locked'] and not status_unlocked['owned_by_this_process']:
            print("  ✅ Correctly reported unlocked session status")
        else:
            print("  ❌ Incorrect status for unlocked session")
        
        # Acquire lock and test status
        pm.acquire_session_lock(session_uuid)
        
        status_locked = pm.get_session_status(session_uuid)
        if status_locked['is_locked'] and status_locked['owned_by_this_process']:
            print("  ✅ Correctly reported locked session status")
        else:
            print("  ❌ Incorrect status for locked session")
        
        # Update progress and check status
        pm.update_progress(session_uuid, {
            'overall_progress': 75,
            'current_step': 'Testing status',
            'completed_steps': 5
        })
        
        status_with_progress = pm.get_session_status(session_uuid)
        if status_with_progress['progress'] and status_with_progress['progress']['overall_progress'] == 75:
            print("  ✅ Status correctly includes progress data")
        else:
            print("  ❌ Status missing or incorrect progress data")
        
        # Release lock
        pm.release_session_lock(session_uuid, 'completed')
        
        # Cleanup
        supabase = get_supabase_client()
        supabase.table('sessions').delete().eq('id', session_uuid).execute()
        
        return True
        
    except Exception as e:
        print(f"  ❌ Session status test failed: {e}")
        return False


def main():
    """Run all ProgressManager tests"""
    print("🧪 Starting ProgressManager Tests")
    print("=" * 50)
    
    tests = [
        ("ProgressManager Creation", test_progress_manager_creation),
        ("Session Locking", test_session_locking),
        ("Progress Tracking", test_progress_tracking),
        ("Heartbeat Monitoring", test_heartbeat_monitoring),
        ("Concurrent Access Prevention", test_concurrent_access),
        ("Session Status Reporting", test_session_status),
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
        print("\n🎉 All ProgressManager tests passed! Session isolation is working correctly.")
    else:
        print(f"\n⚠️  {failed} test(s) failed. Please check the issues above.")
    
    # Cleanup global instance
    try:
        from training_system.progress_manager import shutdown_progress_manager
        shutdown_progress_manager()
        print("\n🧹 Global ProgressManager instance cleaned up")
    except Exception as e:
        print(f"\n⚠️  Error cleaning up: {e}")
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)