#!/usr/bin/env python3
"""
MODEL EXECUTION TEST - Test actual model training with sample data
================================================================

This script tests if all models can actually train with sample data,
not just if the functions exist.
"""

import numpy as np
import sys
import logging
from typing import Dict, Any, Tuple

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_sample_data(samples=1000, features=3, time_steps=24, outputs=2):
    """Create sample time series data for testing"""
    np.random.seed(42)  # For reproducible tests
    
    # Create sample input data (samples, time_steps, features)
    X = np.random.randn(samples, time_steps, features).astype(np.float32)
    
    # Create sample output data (samples, outputs)  
    y = np.random.randn(samples, time_steps, outputs).astype(np.float32)
    
    # Split into train/val
    split_idx = int(0.8 * samples)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    logger.info(f"Created sample data: X_train{X_train.shape}, y_train{y_train.shape}")
    logger.info(f"                     X_val{X_val.shape}, y_val{y_val.shape}")
    
    return X_train, y_train, X_val, y_val

def create_mdl_config(model_type: str, **kwargs):
    """Create MDL configuration for different model types"""
    sys.path.append('.')
    from training_system.config import MDL
    
    config = MDL()
    
    if model_type.lower() in ['dense', 'cnn', 'lstm']:
        # Neural network models
        config.LAY = kwargs.get('layers', 2)
        config.N = kwargs.get('neurons', 50)
        config.EP = kwargs.get('epochs', 2)  # Small for testing
        config.ACTF = kwargs.get('activation', 'relu')
        
        if model_type.lower() == 'cnn':
            config.K = kwargs.get('kernel_size', 1)  # Adjusted for testing
            
    elif model_type.lower() in ['svr_dir', 'svr_mimo']:
        # SVR models
        config.KERNEL = kwargs.get('kernel', 'rbf')
        config.C = kwargs.get('C', 1.0)
        config.EPSILON = kwargs.get('epsilon', 0.1)
    
    return config

def test_model_training(model_name: str, train_func, X_train, y_train, X_val, y_val, **config_kwargs):
    """Test if a model can actually train with sample data"""
    try:
        logger.info(f"🧪 Testing {model_name} model training...")
        
        if model_name.lower() in ['dense', 'cnn', 'lstm', 'ar_lstm']:
            # Neural network models need MDL config
            config = create_mdl_config(model_name, **config_kwargs)
            
            if model_name.lower() in ['ar_lstm']:
                # AR LSTM might need special handling
                model = train_func(X_train, y_train, X_val, y_val, config)
            else:
                model = train_func(X_train, y_train, X_val, y_val, config)
                
        elif model_name.lower() in ['svr_dir', 'svr_mimo']:
            # SVR models need MDL config
            config = create_mdl_config(model_name, **config_kwargs)
            model = train_func(X_train, y_train, config)
            
        elif model_name.lower() == 'linear':
            # Linear model has simple interface
            model = train_func(X_train, y_train)
        
        if model is not None:
            logger.info(f"  ✅ {model_name}: Training completed successfully")
            return True, None
        else:
            logger.error(f"  ❌ {model_name}: Training returned None")
            return False, "Training returned None"
            
    except Exception as e:
        error_msg = str(e)
        if "TensorFlow" in error_msg or "tensorflow" in error_msg:
            logger.error(f"  ⚠️ {model_name}: TensorFlow not available - {error_msg}")
            return False, f"TensorFlow not available: {error_msg}"
        else:
            logger.error(f"  ❌ {model_name}: Training failed - {error_msg}")
            return False, f"Training error: {error_msg}"

def test_all_activation_functions():
    """Test different activation functions with Dense model"""
    logger.info("🔍 TESTING ALL ACTIVATION FUNCTIONS WITH DENSE MODEL")
    
    X_train, y_train, X_val, y_val = create_sample_data(samples=100, features=2, time_steps=10, outputs=1)
    
    activations_to_test = ['relu', 'sigmoid', 'tanh', 'linear', 'softmax']
    results = {}
    
    try:
        sys.path.append('.')
        from training_system.model_trainer import train_dense
        
        for activation in activations_to_test:
            success, error = test_model_training(
                'dense', train_dense, X_train, y_train, X_val, y_val,
                activation=activation, epochs=1, neurons=16, layers=1
            )
            results[activation] = {'success': success, 'error': error}
            
    except ImportError as e:
        logger.error(f"❌ Could not import Dense training function: {e}")
        for activation in activations_to_test:
            results[activation] = {'success': False, 'error': f'Import error: {e}'}
    
    return results

def main():
    """Main test execution"""
    logger.info("🚀 STARTING MODEL EXECUTION TESTS")
    logger.info("=" * 60)
    
    # Create sample data for testing
    X_train, y_train, X_val, y_val = create_sample_data(samples=200, features=2, time_steps=10, outputs=1)
    
    # Test all models
    models_to_test = [
        {
            'name': 'Dense',
            'import_name': 'train_dense',
            'config': {'epochs': 1, 'neurons': 16, 'layers': 1, 'activation': 'relu'}
        },
        {
            'name': 'CNN', 
            'import_name': 'train_cnn',
            'config': {'epochs': 1, 'neurons': 16, 'layers': 1, 'activation': 'relu', 'kernel_size': 1}
        },
        {
            'name': 'LSTM',
            'import_name': 'train_lstm', 
            'config': {'epochs': 1, 'neurons': 16, 'layers': 1, 'activation': 'tanh'}
        },
        {
            'name': 'AR_LSTM',
            'import_name': 'train_ar_lstm',
            'config': {'epochs': 1, 'neurons': 16, 'layers': 1, 'activation': 'tanh'}
        },
        {
            'name': 'SVR_dir',
            'import_name': 'train_svr_dir',
            'config': {'kernel': 'linear', 'C': 1.0, 'epsilon': 0.1}
        },
        {
            'name': 'SVR_MIMO', 
            'import_name': 'train_svr_mimo',
            'config': {'kernel': 'linear', 'C': 1.0, 'epsilon': 0.1}
        },
        {
            'name': 'Linear',
            'import_name': 'train_linear_model',
            'config': {}
        }
    ]
    
    results = {}
    
    try:
        sys.path.append('.')
        from training_system.model_trainer import (
            train_dense, train_cnn, train_lstm, train_ar_lstm,
            train_svr_dir, train_svr_mimo, train_linear_model
        )
        
        training_functions = {
            'train_dense': train_dense,
            'train_cnn': train_cnn, 
            'train_lstm': train_lstm,
            'train_ar_lstm': train_ar_lstm,
            'train_svr_dir': train_svr_dir,
            'train_svr_mimo': train_svr_mimo,
            'train_linear_model': train_linear_model
        }
        
        for model_info in models_to_test:
            model_name = model_info['name']
            import_name = model_info['import_name']
            config = model_info['config']
            
            if import_name in training_functions:
                train_func = training_functions[import_name]
                success, error = test_model_training(
                    model_name, train_func, X_train, y_train, X_val, y_val, **config
                )
                results[model_name] = {'success': success, 'error': error}
            else:
                results[model_name] = {'success': False, 'error': 'Function not found'}
                
    except ImportError as e:
        logger.error(f"❌ Could not import training functions: {e}")
        for model_info in models_to_test:
            results[model_info['name']] = {'success': False, 'error': f'Import error: {e}'}
    
    # Test activation functions
    logger.info("")
    activation_results = test_all_activation_functions()
    
    # Summary
    logger.info("")
    logger.info("📊 MODEL EXECUTION TEST SUMMARY")
    logger.info("=" * 60)
    
    # Model results
    model_passed = sum(1 for r in results.values() if r['success'])
    model_failed = len(results) - model_passed
    
    logger.info(f"🏗️  MODEL TRAINING TESTS:")
    logger.info(f"  ✅ PASSED: {model_passed}")
    logger.info(f"  ❌ FAILED: {model_failed}")
    
    if model_failed > 0:
        logger.info("  Failed models:")
        for name, result in results.items():
            if not result['success']:
                logger.info(f"    ❌ {name}: {result['error']}")
    
    # Activation results
    activation_passed = sum(1 for r in activation_results.values() if r['success'])
    activation_failed = len(activation_results) - activation_passed
    
    logger.info(f"\n⚡ ACTIVATION FUNCTION TESTS:")
    logger.info(f"  ✅ PASSED: {activation_passed}")
    logger.info(f"  ❌ FAILED: {activation_failed}")
    
    if activation_failed > 0:
        logger.info("  Failed activations:")
        for name, result in activation_results.items():
            if not result['success']:
                logger.info(f"    ❌ {name}: {result['error']}")
    
    # Overall success rate
    total_passed = model_passed + activation_passed
    total_tests = len(results) + len(activation_results)
    success_rate = (total_passed / total_tests) * 100 if total_tests > 0 else 0
    
    logger.info(f"\n🎯 OVERALL SUCCESS RATE: {success_rate:.1f}%")
    
    if success_rate == 100:
        logger.info("🎉 ALL MODELS AND ACTIVATIONS CAN EXECUTE SUCCESSFULLY!")
        return 0
    elif success_rate >= 70:
        logger.info("⚠️  MOST MODELS WORKING - SOME ISSUES (likely TensorFlow)")
        return 1
    else:
        logger.info("🚨 MAJOR EXECUTION ISSUES - SYSTEM NEEDS ATTENTION")
        return 2

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except Exception as e:
        logger.error(f"💥 CRITICAL ERROR: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(3)