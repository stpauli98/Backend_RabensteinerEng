#!/usr/bin/env python3
"""
Test script for training API endpoints
Tests all API endpoints with mock database responses
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from unittest.mock import Mock, patch
import pytest
from flask import Flask

# Set up logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_training_api_endpoints():
    """Test all training API endpoints"""
    try:
        print("🧪 Starting Training API Tests")
        print("=" * 50)
        
        # Import and create Flask app
        from training_system.training_api import training_api_bp
        
        app = Flask(__name__)
        app.register_blueprint(training_api_bp)
        app.config['TESTING'] = True
        
        print("✓ Flask app created and blueprint registered")
        
        with app.test_client() as client:
            # Test 1: Results endpoint
            test_results_endpoint(client)
            
            # Test 2: Status endpoint  
            test_status_endpoint(client)
            
            # Test 3: Visualizations endpoint
            test_visualizations_endpoint(client)
            
            # Test 4: Metrics endpoint
            test_metrics_endpoint(client)
            
            # Test 5: Progress endpoint
            test_progress_endpoint(client)
            
            # Test 6: Logs endpoint
            test_logs_endpoint(client)
        
        print("\n🎉 All API endpoint tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ API endpoint tests failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_results_endpoint(client):
    """Test /api/training/results/<session_id> endpoint"""
    print("\n=== Test 1: Results Endpoint ===")
    
    session_id = "test_session_123"
    
    # Mock successful response
    mock_results = {
        'session_id': session_id,
        'status': 'completed',
        'evaluation_metrics': {
            'test_dataset': {
                'dense': {
                    'metrics': {
                        'wape': 141.58,
                        'smape': 50.23,
                        'mase': 1.23,
                        'mae': 1.16
                    }
                }
            }
        },
        'model_performance': {},
        'best_model': {'name': 'dense', 'dataset': 'test_dataset', 'mae': 1.16},
        'summary': {},
        'created_at': '2025-07-10T14:30:00',
        'completed_at': '2025-07-10T14:45:00'
    }
    
    with patch('training_system.training_api.get_supabase_client') as mock_supabase_client:
        with patch('training_system.training_api._get_results_from_database') as mock_get_results:
            mock_get_results.return_value = mock_results
            
            response = client.get(f'/api/training/results/{session_id}')
            
            assert response.status_code == 200, f"Expected 200, got {response.status_code}"
            
            data = json.loads(response.data)
            assert data['session_id'] == session_id
            assert data['status'] == 'completed'
            assert 'evaluation_metrics' in data
            assert 'best_model' in data
            
            print("✓ Results endpoint working correctly")

def test_status_endpoint(client):
    """Test /api/training/status/<session_id> endpoint"""
    print("\n=== Test 2: Status Endpoint ===")
    
    session_id = "test_session_123"
    
    # Mock completed training
    mock_results = {
        'status': 'completed',
        'created_at': '2025-07-10T14:30:00',
        'completed_at': '2025-07-10T14:45:00'
    }
    
    with patch('training_system.training_api.get_supabase_client') as mock_supabase_client:
        with patch('training_system.training_api._get_results_from_database') as mock_get_results:
            with patch('training_system.training_api._get_status_from_database') as mock_get_status:
                mock_get_results.return_value = mock_results
                mock_get_status.return_value = {}
                
                response = client.get(f'/api/training/status/{session_id}')
                
                assert response.status_code == 200, f"Expected 200, got {response.status_code}"
                
                data = json.loads(response.data)
                assert data['session_id'] == session_id
                assert data['status'] == 'completed'
                assert data['progress'] == 100
                assert 'current_step' in data
                
                print("✓ Status endpoint working correctly")

def test_visualizations_endpoint(client):
    """Test /api/training/visualizations/<session_id> endpoint"""
    print("\n=== Test 3: Visualizations Endpoint ===")
    
    session_id = "test_session_123"
    
    # Mock visualizations
    mock_visualizations = {
        'plots': {
            'input_violin_plot': 'base64_encoded_image_data_here',
            'output_violin_plot': 'base64_encoded_image_data_here_2'
        },
        'metadata': {'generated_by': 'real_pipeline'},
        'created_at': '2025-07-10T14:35:00'
    }
    
    with patch('training_system.training_api.get_supabase_client') as mock_supabase_client:
        with patch('training_system.training_api._get_visualizations_from_database') as mock_get_viz:
            mock_get_viz.return_value = mock_visualizations
            
            response = client.get(f'/api/training/visualizations/{session_id}')
            
            assert response.status_code == 200, f"Expected 200, got {response.status_code}"
            
            data = json.loads(response.data)
            assert data['session_id'] == session_id
            assert 'plots' in data
            assert len(data['plots']) == 2
            assert 'input_violin_plot' in data['plots']
            assert 'output_violin_plot' in data['plots']
            
            print("✓ Visualizations endpoint working correctly")

def test_metrics_endpoint(client):
    """Test /api/training/metrics/<session_id> endpoint"""
    print("\n=== Test 4: Metrics Endpoint ===")
    
    session_id = "test_session_123"
    
    # Mock metrics
    mock_results = {
        'evaluation_metrics': {
            'test_dataset': {
                'dense': {'mae': 0.123, 'mse': 0.456},
                'lstm': {'mae': 0.234, 'mse': 0.567}
            }
        }
    }
    
    with patch('training_system.training_api.get_supabase_client') as mock_supabase_client:
        with patch('training_system.training_api._get_results_from_database') as mock_get_results:
            mock_get_results.return_value = mock_results
            
            response = client.get(f'/api/training/metrics/{session_id}')
            
            assert response.status_code == 200, f"Expected 200, got {response.status_code}"
            
            data = json.loads(response.data)
            assert data['session_id'] == session_id
            
            print("✓ Metrics endpoint working correctly")

def test_progress_endpoint(client):
    """Test /api/training/progress/<session_id> endpoint"""
    print("\n=== Test 5: Progress Endpoint ===")
    
    session_id = "test_session_123"
    
    # Mock progress
    mock_progress = {
        'progress': {
            'overall': 75,
            'current_step': 'Model Training',
            'total_steps': 7,
            'completed_steps': 5
        },
        'created_at': '2025-07-10T14:30:00'
    }
    
    with patch('training_system.training_api.get_supabase_client') as mock_supabase_client:
        with patch('training_system.training_api._get_progress_from_database') as mock_get_progress:
            mock_get_progress.return_value = mock_progress
            
            response = client.get(f'/api/training/progress/{session_id}')
            
            assert response.status_code == 200, f"Expected 200, got {response.status_code}"
            
            data = json.loads(response.data)
            assert data['session_id'] == session_id
            
            print("✓ Progress endpoint working correctly")

def test_logs_endpoint(client):
    """Test /api/training/logs/<session_id> endpoint"""
    print("\n=== Test 6: Logs Endpoint ===")
    
    session_id = "test_session_123"
    
    # Mock logs
    mock_logs = {
        'logs': [
            {
                'timestamp': '2025-07-10T14:30:00',
                'level': 'INFO',
                'message': 'Training started',
                'step': 'initialization'
            },
            {
                'timestamp': '2025-07-10T14:35:00',
                'level': 'INFO',
                'message': 'Model training completed',
                'step': 'training'
            }
        ],
        'total_logs': 2
    }
    
    with patch('training_system.training_api.get_supabase_client') as mock_supabase_client:
        with patch('training_system.training_api._get_logs_from_database') as mock_get_logs:
            mock_get_logs.return_value = mock_logs
            
            response = client.get(f'/api/training/logs/{session_id}')
            
            assert response.status_code == 200, f"Expected 200, got {response.status_code}"
            
            data = json.loads(response.data)
            print(f"Debug - Logs response data: {data}")
            assert data['session_id'] == session_id
            assert 'logs' in data
            assert len(data['logs']) == 2, f"Expected 2 logs, got {len(data['logs'])}: {data['logs']}"
            
            print("✓ Logs endpoint working correctly")

def main():
    """Main test function"""
    print("Starting Training API Tests...")
    
    success = test_training_api_endpoints()
    
    if success:
        print("\n🎯 API ENDPOINTS ARE READY FOR FRONTEND!")
        return 0
    else:
        print("\n❌ API endpoint tests failed!")
        return 1

if __name__ == "__main__":
    exit(main())