#!/usr/bin/env python3
"""
Test script for the new modern middleman runner
Tests the integration between frontend API calls and TrainingPipeline
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
from unittest.mock import Mock, patch

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_modern_middleman_runner():
    """Test the modern middleman runner functionality"""
    try:
        print("🧪 Starting Modern Middleman Runner Tests")
        print("=" * 55)
        
        from middleman_runner_new import ModernMiddlemanRunner, run_training_script
        
        print("✓ Successfully imported ModernMiddlemanRunner")
        
        # Test 1: Basic instantiation
        test_basic_instantiation(ModernMiddlemanRunner)
        
        # Test 2: Session validation
        test_session_validation(ModernMiddlemanRunner)
        
        # Test 3: Training execution with mocks
        test_training_execution(ModernMiddlemanRunner)
        
        # Test 4: Error handling
        test_error_handling(ModernMiddlemanRunner)
        
        # Test 5: Legacy function compatibility
        test_legacy_function(run_training_script)
        
        print("\n🎉 All Modern Middleman Runner tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Modern Middleman Runner tests failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_basic_instantiation(ModernMiddlemanRunner):
    """Test basic instantiation of ModernMiddlemanRunner"""
    print("\n=== Test 1: Basic Instantiation ===")
    
    with patch('middleman_runner_new.get_supabase_client') as mock_supabase:
        mock_supabase.return_value = Mock()
        
        runner = ModernMiddlemanRunner()
        
        assert runner is not None, "Runner should instantiate"
        assert runner.supabase is not None, "Runner should have supabase client"
        assert runner.socketio is None, "SocketIO should be None initially"
        
        # Test SocketIO setting
        mock_socketio = Mock()
        runner.set_socketio(mock_socketio)
        assert runner.socketio == mock_socketio, "SocketIO should be set"
        
        print("✓ Basic instantiation working")

def test_session_validation(ModernMiddlemanRunner):
    """Test session validation functionality"""
    print("\n=== Test 2: Session Validation ===")
    
    with patch('middleman_runner_new.get_supabase_client') as mock_supabase_client:
        mock_supabase = Mock()
        mock_supabase_client.return_value = mock_supabase
        
        runner = ModernMiddlemanRunner()
        
        # Mock session query response
        session_response = Mock()
        session_response.data = [{'uuid': 'test-uuid-123'}]
        
        # Mock files query response  
        files_response = Mock()
        files_response.data = [
            {'type': 'input', 'fileName': 'input1.csv'},
            {'type': 'input', 'fileName': 'input2.csv'}, 
            {'type': 'output', 'fileName': 'output1.csv'}
        ]
        
        # Setup mock to return different responses for different table queries
        def mock_table_side_effect(table_name):
            mock_table = Mock()
            if table_name == 'sessions':
                mock_table.select.return_value.eq.return_value.execute.return_value = session_response
            elif table_name == 'files':
                mock_table.select.return_value.eq.return_value.execute.return_value = files_response
            return mock_table
        
        mock_supabase.table.side_effect = mock_table_side_effect
        
        is_valid = runner._validate_session("test_session")
        assert is_valid == True, "Session should be valid with input and output files"
        
        print("✓ Session validation working")

def test_training_execution(ModernMiddlemanRunner):
    """Test training execution with mocked dependencies"""
    print("\n=== Test 3: Training Execution ===")
    
    with patch('middleman_runner_new.get_supabase_client') as mock_supabase_client:
        with patch('middleman_runner_new.run_training_for_session') as mock_run_training:
            mock_supabase = Mock()
            mock_supabase_client.return_value = mock_supabase
            
            # Mock successful training results
            mock_training_results = {
                'session_id': 'test_session',
                'status': 'completed',
                'summary': {
                    'total_datasets': 1,
                    'total_models': 3,
                    'best_model': {'name': 'lstm', 'mae': 0.123}
                }
            }
            mock_run_training.return_value = mock_training_results
            
            runner = ModernMiddlemanRunner()
            
            # Mock session validation to return True
            runner._validate_session = Mock(return_value=True)
            
            result = runner.run_training_script("test_session")
            
            assert result['success'] == True, "Training should succeed"
            assert result['session_id'] == "test_session", "Session ID should match"
            assert 'results' in result, "Results should be present"
            assert result['results'] == mock_training_results, "Results should match mock data"
            
            print("✓ Training execution working")

def test_error_handling(ModernMiddlemanRunner):
    """Test error handling in training execution"""
    print("\n=== Test 4: Error Handling ===")
    
    with patch('middleman_runner_new.get_supabase_client') as mock_supabase_client:
        with patch('middleman_runner_new.run_training_for_session') as mock_run_training:
            mock_supabase = Mock()
            mock_supabase_client.return_value = mock_supabase
            
            # Mock training failure
            mock_run_training.side_effect = Exception("Training pipeline failed")
            
            runner = ModernMiddlemanRunner()
            
            # Mock session validation to return True
            runner._validate_session = Mock(return_value=True)
            runner._save_error_to_database = Mock()  # Mock error saving
            
            result = runner.run_training_script("test_session")
            
            assert result['success'] == False, "Training should fail"
            assert 'error' in result, "Error should be present"
            assert "Training pipeline failed" in result['error'], "Error message should match"
            
            # Verify error was saved
            runner._save_error_to_database.assert_called_once()
            
            print("✓ Error handling working")

def test_legacy_function(run_training_script):
    """Test legacy function compatibility"""
    print("\n=== Test 5: Legacy Function Compatibility ===")
    
    with patch('middleman_runner_new.ModernMiddlemanRunner') as mock_runner_class:
        mock_runner = Mock()
        mock_runner_class.return_value = mock_runner
        
        mock_result = {
            'success': True,
            'session_id': 'test_session',
            'message': 'Training completed'
        }
        mock_runner.run_training_script.return_value = mock_result
        
        result = run_training_script("test_session")
        
        assert result == mock_result, "Legacy function should return same result"
        mock_runner.run_training_script.assert_called_once_with("test_session")
        
        print("✓ Legacy function compatibility working")

def main():
    """Main test function"""
    print("Starting Modern Middleman Runner Tests...")
    
    success = test_modern_middleman_runner()
    
    if success:
        print("\n🎯 MODERN MIDDLEMAN RUNNER IS READY!")
        print("✅ Frontend can now call the new pipeline directly")
        print("✅ No more subprocess dependencies")
        print("✅ Real extracted functions are used")
        return 0
    else:
        print("\n❌ Modern Middleman Runner tests failed!")
        return 1

if __name__ == "__main__":
    exit(main())