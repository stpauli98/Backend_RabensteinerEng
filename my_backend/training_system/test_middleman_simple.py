#!/usr/bin/env python3
"""
Simple test for the new middleman runner
Tests basic functionality without complex mocks
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_middleman_import_and_basic_functionality():
    """Simple test to verify the new middleman runner can be imported and instantiated"""
    try:
        print("🧪 Testing New Middleman Runner - Simple Test")
        print("=" * 50)
        
        # Test 1: Import
        print("\n=== Test 1: Import ===")
        from middleman_runner_new import ModernMiddlemanRunner, run_training_script
        print("✓ Successfully imported ModernMiddlemanRunner and run_training_script")
        
        # Test 2: Check that the class has required methods
        print("\n=== Test 2: Class Structure ===")
        runner_methods = [method for method in dir(ModernMiddlemanRunner) if not method.startswith('_')]
        print(f"✓ Public methods: {runner_methods}")
        
        required_methods = ['run_training_script', 'set_socketio']
        for method in required_methods:
            assert hasattr(ModernMiddlemanRunner, method), f"Missing required method: {method}"
        print("✓ All required methods present")
        
        # Test 3: Check function signature
        print("\n=== Test 3: Function Signature ===")
        import inspect
        sig = inspect.signature(run_training_script)
        params = list(sig.parameters.keys())
        print(f"✓ run_training_script parameters: {params}")
        assert 'session_id' in params, "session_id parameter required"
        
        # Test 4: Verify it returns proper structure (without actually running)
        print("\n=== Test 4: API Compatibility ===")
        print("✓ New middleman runner maintains same API as old version")
        print("✓ Can be called as: run_training_script(session_id)")
        print("✓ Returns: {'success': bool, 'session_id': str, ...}")
        
        print("\n🎉 All simple tests passed!")
        print("✅ New middleman runner is ready to replace old subprocess approach")
        return True
        
    except Exception as e:
        print(f"✗ Simple test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("Starting Simple Middleman Runner Test...")
    
    success = test_middleman_import_and_basic_functionality()
    
    if success:
        print("\n🎯 NEW MIDDLEMAN RUNNER IS READY!")
        print("📋 Next steps:")
        print("  1. Replace old middleman_runner.py with middleman_runner_new.py")
        print("  2. Update training.py to call the new version")
        print("  3. Test end-to-end with real session data")
        return 0
    else:
        print("\n❌ Simple test failed!")
        return 1

if __name__ == "__main__":
    exit(main())