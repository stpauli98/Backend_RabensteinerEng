#!/usr/bin/env python3
"""
Test script for pipeline integration
Tests the connection between extracted modules and TrainingPipeline
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import pandas as pd
import numpy as np
from typing import Dict

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_pipeline_integration():
    """Test the real pipeline integration functions"""
    try:
        from training_system.pipeline_integration import (
            RealDataProcessor, RealModelTrainer, RealResultsGenerator, 
            RealVisualizationGenerator, run_real_training_pipeline
        )
        from training_system.config import MTS, MDL
        
        print("✓ Successfully imported pipeline integration modules")
        
        # Test 1: RealDataProcessor initialization
        print("\n=== Test 1: RealDataProcessor ===")
        data_processor = RealDataProcessor()
        print("✓ RealDataProcessor created successfully")
        
        # Test 2: RealModelTrainer initialization
        print("\n=== Test 2: RealModelTrainer ===")
        model_trainer = RealModelTrainer()
        print("✓ RealModelTrainer created successfully")
        
        # Test 3: RealResultsGenerator initialization
        print("\n=== Test 3: RealResultsGenerator ===")
        results_generator = RealResultsGenerator()
        print("✓ RealResultsGenerator created successfully")
        
        # Test 4: RealVisualizationGenerator initialization
        print("\n=== Test 4: RealVisualizationGenerator ===")
        viz_generator = RealVisualizationGenerator()
        print("✓ RealVisualizationGenerator created successfully")
        
        # Test 5: Test data creation for model training
        print("\n=== Test 5: Create sample datasets ===")
        sample_datasets = create_sample_datasets()
        print(f"✓ Created sample datasets: {list(sample_datasets.keys())}")
        
        # Test 6: Test model training
        print("\n=== Test 6: Test model training ===")
        config = MDL()
        config.MODE = "Dense"  # Test with Dense mode
        model_trainer_with_config = RealModelTrainer(config)
        
        session_data = {
            'zeitschritte': {'eingabe': 13, 'ausgabe': 13}
        }
        
        training_results = model_trainer_with_config.train_all_models(sample_datasets, session_data)
        print(f"✓ Model training completed: {len(training_results)} datasets")
        
        # Test 7: Test results generation
        print("\n=== Test 7: Test results generation ===")
        evaluation_results = results_generator.generate_results(training_results, session_data)
        print(f"✓ Results generation completed")
        print(f"  - Evaluation metrics: {len(evaluation_results.get('evaluation_metrics', {}))}")
        print(f"  - Summary: {evaluation_results.get('summary', {})}")
        
        # Test 8: Test visualization creation
        print("\n=== Test 8: Test visualization creation ===")
        processed_data = create_sample_processed_data()
        visualizations = viz_generator.create_visualizations(training_results, evaluation_results, processed_data)
        print(f"✓ Visualization creation completed: {len(visualizations)} plots")
        
        print("\n=== All Pipeline Integration Tests Passed! ===")
        return True
        
    except Exception as e:
        print(f"✗ Pipeline integration test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def create_sample_datasets() -> Dict:
    """Create sample datasets for testing"""
    np.random.seed(42)
    
    # Create sample time series data
    samples = 100
    time_steps_in = 13
    time_steps_out = 13
    input_features = 3
    output_features = 2
    
    X = np.random.randn(samples, time_steps_in, input_features)
    y = np.random.randn(samples, time_steps_out, output_features)
    
    datasets = {
        'test_dataset': {
            'X': X,
            'y': y,
            'time_steps_in': time_steps_in,
            'time_steps_out': time_steps_out,
            'input_features': input_features,
            'output_features': output_features
        }
    }
    
    return datasets

def create_sample_processed_data() -> Dict:
    """Create sample processed data for visualization testing"""
    # Create sample input and output DataFrames
    np.random.seed(42)
    
    input_df = pd.DataFrame({
        'feature1': np.random.randn(100),
        'feature2': np.random.randn(100),
        'feature3': np.random.randn(100)
    })
    
    output_df = pd.DataFrame({
        'target1': np.random.randn(100),
        'target2': np.random.randn(100)
    })
    
    processed_data = {
        'input_data': {'input_file': input_df},
        'output_data': {'output_file': output_df}
    }
    
    return processed_data

def main():
    """Main test function"""
    print("Starting Pipeline Integration Tests...")
    print("=" * 50)
    
    success = test_pipeline_integration()
    
    if success:
        print("\n🎉 All tests completed successfully!")
        return 0
    else:
        print("\n❌ Some tests failed!")
        return 1

if __name__ == "__main__":
    exit(main())