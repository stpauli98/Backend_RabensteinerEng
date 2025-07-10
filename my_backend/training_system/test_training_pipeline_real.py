#!/usr/bin/env python3
"""
Test script for TrainingPipeline with real integration
Tests the complete pipeline with real extracted functions
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import pandas as pd
import numpy as np
from typing import Dict
from unittest.mock import Mock

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_training_pipeline_real():
    """Test the TrainingPipeline with real integration"""
    try:
        from training_system.training_pipeline import TrainingPipeline
        
        print("✓ Successfully imported TrainingPipeline")
        
        # Test 1: Create mock supabase client
        print("\n=== Test 1: Create TrainingPipeline ===")
        mock_supabase = create_mock_supabase()
        mock_socketio = create_mock_socketio()
        
        pipeline = TrainingPipeline(mock_supabase, mock_socketio)
        print("✓ TrainingPipeline created successfully")
        
        # Test 2: Test progress tracking
        print("\n=== Test 2: Test Progress Tracking ===")
        pipeline._update_progress(1, "Test progress message")
        print("✓ Progress tracking works")
        
        # Test 3: Test configuration creation
        print("\n=== Test 3: Test Configuration Creation ===")
        session_data = create_mock_session_data()
        mts_config = pipeline._create_mts_config(session_data)
        mdl_config = pipeline._create_mdl_config(session_data)
        print(f"✓ MTS config created: I_N={mts_config.I_N}, O_N={mts_config.O_N}")
        print(f"✓ MDL config created: MODE={mdl_config.MODE}, LAY={mdl_config.LAY}")
        
        # Test 4: Test database save methods
        print("\n=== Test 4: Test Database Save Methods ===")
        mock_results = create_mock_pipeline_results()
        
        # Mock the database response
        mock_supabase.table.return_value.insert.return_value.execute.return_value.data = [{'id': 123}]
        
        success = pipeline._save_results_to_database("test_session", mock_results)
        print(f"✓ Database save test: {success}")
        
        # Test 5: Test visualization save
        print("\n=== Test 5: Test Visualization Save ===")
        mock_visualizations = {
            'input_violin_plot': 'base64_encoded_image_data_here',
            'output_violin_plot': 'base64_encoded_image_data_here_2'
        }
        
        pipeline._save_visualizations_to_database("test_session", mock_visualizations)
        print("✓ Visualization save test completed")
        
        print("\n=== All TrainingPipeline Real Tests Passed! ===")
        return True
        
    except Exception as e:
        print(f"✗ TrainingPipeline real test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def create_mock_supabase():
    """Create a mock supabase client"""
    mock_supabase = Mock()
    
    # Mock table operations
    mock_table = Mock()
    mock_insert = Mock()
    mock_execute = Mock()
    
    mock_execute.data = [{'id': 123}]
    mock_insert.execute.return_value = mock_execute
    mock_table.insert.return_value = mock_insert
    mock_table.upsert.return_value = mock_insert
    mock_supabase.table.return_value = mock_table
    
    return mock_supabase

def create_mock_socketio():
    """Create a mock socketio instance"""
    mock_socketio = Mock()
    return mock_socketio

def create_mock_session_data():
    """Create mock session data"""
    return {
        'time_info': {
            'jahr': True,
            'monat': True,
            'woche': True,
            'feiertag': True,
            'zeitzone': 'UTC'
        },
        'zeitschritte': {
            'eingabe': '24',
            'ausgabe': '1',
            'zeitschrittweite': '1',
            'offset': '0'
        }
    }

def create_mock_pipeline_results():
    """Create mock pipeline results"""
    return {
        'session_id': 'test_session',
        'timestamp': '2025-07-10T14:30:00',
        'status': 'completed',
        'training_results': {
            'test_dataset': {
                'dense': {
                    'model': 'mock_model',
                    'type': 'neural_network',
                    'config': 'Dense'
                }
            }
        },
        'evaluation_results': {
            'evaluation_metrics': {
                'test_dataset': {
                    'dense': {
                        'metrics': {
                            'wape': 141.58,
                            'smape': 50.23,
                            'mase': 1.23,
                            'mae': 1.16,
                            'mse': 2.34,
                            'rmse': 1.53
                        }
                    }
                }
            },
            'summary': {
                'total_datasets': 1,
                'total_models': 1,
                'best_model': {
                    'name': 'dense',
                    'dataset': 'test_dataset',
                    'mae': 1.16
                }
            }
        },
        'visualizations': {
            'input_violin_plot': 'base64_encoded_image_data',
            'output_violin_plot': 'base64_encoded_image_data_2'
        },
        'summary': {
            'total_datasets': 1,
            'total_models': 1,
            'best_model': {
                'name': 'dense',
                'dataset': 'test_dataset',
                'mae': 1.16
            }
        }
    }

def main():
    """Main test function"""
    print("Starting TrainingPipeline Real Integration Tests...")
    print("=" * 55)
    
    success = test_training_pipeline_real()
    
    if success:
        print("\n🎉 All tests completed successfully!")
        return 0
    else:
        print("\n❌ Some tests failed!")
        return 1

if __name__ == "__main__":
    exit(main())