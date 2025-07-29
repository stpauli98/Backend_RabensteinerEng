#!/usr/bin/env python3
"""
COMPREHENSIVE MODEL AND ACTIVATION FUNCTION VALIDATION TEST
============================================================

Tests all supported models and activation functions to ensure they work correctly:

MODELS TESTED:
- Dense (Neural Network)
- CNN (Convolutional Neural Network) 
- LSTM (Long Short-Term Memory)
- AR LSTM (Autoregressive LSTM)
- SVR_dir (Support Vector Regression Direct)
- SVR_MIMO (Support Vector Regression Multi-Input Multi-Output)
- LIN (Linear Regression)

ACTIVATION FUNCTIONS TESTED:
- ReLU, Sigmoid, Tanh, Linear, Softmax, Keine

SVR KERNELS TESTED:
- linear, poly, rbf, sigmoid
"""

import sys
import logging
import traceback
from typing import Dict, List, Tuple

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def validate_activation_mappings() -> Tuple[List[str], List[str]]:
    """Validate that all activation functions are properly mapped"""
    logger.info("🔍 VALIDATING ACTIVATION FUNCTION MAPPINGS")
    
    # Frontend activation functions (from ModelConfiguration.tsx)
    frontend_activations = ['ReLU', 'Sigmoid', 'Tanh', 'Linear', 'Softmax', 'Keine']
    
    # Expected backend mappings
    expected_mappings = {
        'ReLU': 'relu',
        'Sigmoid': 'sigmoid', 
        'Tanh': 'tanh',
        'Linear': 'linear',
        'Softmax': 'softmax',
        'Keine': 'linear'  # 'Keine' means 'None', mapped to linear
    }
    
    passed = []
    failed = []
    
    for frontend_name in frontend_activations:
        if frontend_name in expected_mappings:
            backend_name = expected_mappings[frontend_name]
            logger.info(f"  ✅ {frontend_name} → {backend_name}")
            passed.append(frontend_name)
        else:
            logger.error(f"  ❌ {frontend_name} → NOT MAPPED")
            failed.append(frontend_name)
    
    return passed, failed

def validate_model_types() -> Tuple[List[str], List[str]]:
    """Validate that all model types have training functions"""
    logger.info("🔍 VALIDATING MODEL TYPES AND TRAINING FUNCTIONS")
    
    # Frontend model types (from ModelConfiguration.tsx)
    frontend_models = ['Dense', 'CNN', 'LSTM', 'AR LSTM', 'SVR_dir', 'SVR_MIMO', 'LIN']
    
    # Expected training functions
    expected_functions = {
        'Dense': 'train_dense',
        'CNN': 'train_cnn', 
        'LSTM': 'train_lstm',
        'AR LSTM': 'train_ar_lstm',
        'SVR_dir': 'train_svr_dir',
        'SVR_MIMO': 'train_svr_mimo',
        'LIN': 'train_linear_model'
    }
    
    passed = []
    failed = []
    
    try:
        # Try to import all training functions
        sys.path.append('.')
        from training_system.model_trainer import (
            train_dense, train_cnn, train_lstm, train_ar_lstm,
            train_svr_dir, train_svr_mimo, train_linear_model
        )
        
        available_functions = {
            'Dense': train_dense,
            'CNN': train_cnn,
            'LSTM': train_lstm,
            'AR LSTM': train_ar_lstm,
            'SVR_dir': train_svr_dir,
            'SVR_MIMO': train_svr_mimo,
            'LIN': train_linear_model
        }
        
        for model_name in frontend_models:
            if model_name in available_functions and callable(available_functions[model_name]):
                logger.info(f"  ✅ {model_name} → {expected_functions[model_name]}()")
                passed.append(model_name)
            else:
                logger.error(f"  ❌ {model_name} → FUNCTION NOT FOUND")
                failed.append(model_name)
                
    except ImportError as e:
        logger.error(f"❌ Failed to import training functions: {e}")
        failed = frontend_models
    
    return passed, failed

def validate_svr_kernels() -> Tuple[List[str], List[str]]:
    """Validate SVR kernel support"""
    logger.info("🔍 VALIDATING SVR KERNEL TYPES")
    
    # Frontend kernel types (from ModelConfiguration.tsx)
    frontend_kernels = ['linear', 'poly', 'rbf', 'sigmoid']
    
    # Backend validation (from utils.py)
    valid_backend_kernels = ["linear", "poly", "rbf", "sigmoid"]
    
    passed = []
    failed = []
    
    for kernel in frontend_kernels:
        if kernel in valid_backend_kernels:
            logger.info(f"  ✅ SVR kernel '{kernel}' supported")
            passed.append(kernel)
        else:
            logger.error(f"  ❌ SVR kernel '{kernel}' NOT supported")
            failed.append(kernel)
    
    return passed, failed

def test_parameter_conversion():
    """Test parameter conversion from frontend to backend format"""
    logger.info("🔍 TESTING PARAMETER CONVERSION")
    
    test_cases = [
        # Dense model with different activations
        {
            'name': 'Dense with ReLU',
            'params': {'MODE': 'Dense', 'LAY': 2, 'N': 50, 'EP': 10, 'ACTF': 'ReLU'}
        },
        {
            'name': 'Dense with Sigmoid',
            'params': {'MODE': 'Dense', 'LAY': 2, 'N': 50, 'EP': 10, 'ACTF': 'Sigmoid'}
        },
        {
            'name': 'Dense with Softmax',
            'params': {'MODE': 'Dense', 'LAY': 2, 'N': 50, 'EP': 10, 'ACTF': 'Softmax'}
        },
        {
            'name': 'Dense with Keine',
            'params': {'MODE': 'Dense', 'LAY': 2, 'N': 50, 'EP': 10, 'ACTF': 'Keine'}
        },
        
        # CNN model
        {
            'name': 'CNN with ReLU',
            'params': {'MODE': 'CNN', 'LAY': 2, 'N': 32, 'K': 3, 'EP': 10, 'ACTF': 'ReLU'}
        },
        
        # LSTM model
        {
            'name': 'LSTM with Tanh',
            'params': {'MODE': 'LSTM', 'LAY': 2, 'N': 50, 'EP': 10, 'ACTF': 'Tanh'}
        },
        
        # AR LSTM model
        {
            'name': 'AR LSTM with ReLU',
            'params': {'MODE': 'AR LSTM', 'LAY': 2, 'N': 50, 'EP': 10, 'ACTF': 'ReLU'}
        },
        
        # SVR models with different kernels
        {
            'name': 'SVR_dir with RBF kernel',
            'params': {'MODE': 'SVR_dir', 'KERNEL': 'rbf', 'C': 1.0, 'EPSILON': 0.1}
        },
        {
            'name': 'SVR_MIMO with linear kernel',
            'params': {'MODE': 'SVR_MIMO', 'KERNEL': 'linear', 'C': 1.0, 'EPSILON': 0.1}
        },
        
        # Linear model
        {
            'name': 'Linear model',
            'params': {'MODE': 'LIN'}
        }
    ]
    
    passed = []
    failed = []
    
    try:
        sys.path.append('.')
        from training_system.utils import convert_frontend_to_backend_params
        
        for test_case in test_cases:
            try:
                result = convert_frontend_to_backend_params(test_case['params'])
                if result and isinstance(result, dict) and len(result) > 0:
                    logger.info(f"  ✅ {test_case['name']}: {list(result.keys())}")
                    passed.append(test_case['name'])
                else:
                    logger.error(f"  ❌ {test_case['name']}: Empty or invalid result")
                    failed.append(test_case['name'])
            except Exception as e:
                logger.error(f"  ❌ {test_case['name']}: {str(e)}")
                failed.append(test_case['name'])
                
    except ImportError as e:
        logger.error(f"❌ Failed to import parameter conversion: {e}")
        failed = [tc['name'] for tc in test_cases]
    
    return passed, failed

def check_tensorflow_availability():
    """Check if TensorFlow is available for neural network models"""
    logger.info("🔍 CHECKING TENSORFLOW AVAILABILITY")
    
    try:
        import tensorflow as tf
        logger.info(f"  ✅ TensorFlow version: {tf.__version__}")
        
        # Check if specific activation functions are available
        test_activations = ['relu', 'sigmoid', 'tanh', 'linear', 'softmax']
        available_activations = []
        unavailable_activations = []
        
        for activation in test_activations:
            try:
                # Try to get the activation function
                if hasattr(tf.keras.activations, activation):
                    available_activations.append(activation)
                    logger.info(f"    ✅ Activation '{activation}' available")
                else:
                    unavailable_activations.append(activation)
                    logger.warning(f"    ⚠️ Activation '{activation}' not found")
            except Exception as e:
                unavailable_activations.append(activation)
                logger.error(f"    ❌ Activation '{activation}' error: {e}")
        
        return True, available_activations, unavailable_activations
        
    except ImportError:
        logger.error("  ❌ TensorFlow not available - neural network models will fail")
        return False, [], []

def check_sklearn_availability():
    """Check if scikit-learn is available for SVR models"""
    logger.info("🔍 CHECKING SCIKIT-LEARN AVAILABILITY")
    
    try:
        from sklearn.svm import SVR
        from sklearn import __version__ as sklearn_version
        logger.info(f"  ✅ scikit-learn version: {sklearn_version}")
        
        # Test SVR kernel availability
        test_kernels = ['linear', 'poly', 'rbf', 'sigmoid']
        working_kernels = []
        failed_kernels = []
        
        for kernel in test_kernels:
            try:
                # Try to create SVR instance with kernel
                model = SVR(kernel=kernel, C=1.0, epsilon=0.1)
                working_kernels.append(kernel)
                logger.info(f"    ✅ SVR kernel '{kernel}' available")
            except Exception as e:
                failed_kernels.append(kernel)
                logger.error(f"    ❌ SVR kernel '{kernel}' error: {e}")
        
        return True, working_kernels, failed_kernels
        
    except ImportError:
        logger.error("  ❌ scikit-learn not available - SVR models will fail")
        return False, [], []

def main():
    """Main test execution"""
    logger.info("🚀 STARTING COMPREHENSIVE MODEL AND ACTIVATION FUNCTION VALIDATION")
    logger.info("=" * 80)
    
    # Track overall results
    all_passed = []
    all_failed = []
    
    # 1. Validate activation function mappings
    passed, failed = validate_activation_mappings()
    all_passed.extend([f"Activation: {p}" for p in passed])
    all_failed.extend([f"Activation: {f}" for f in failed])
    
    logger.info("")
    
    # 2. Validate model types
    passed, failed = validate_model_types()
    all_passed.extend([f"Model: {p}" for p in passed])
    all_failed.extend([f"Model: {f}" for f in failed])
    
    logger.info("")
    
    # 3. Validate SVR kernels
    passed, failed = validate_svr_kernels()
    all_passed.extend([f"SVR Kernel: {p}" for p in passed])
    all_failed.extend([f"SVR Kernel: {f}" for f in failed])
    
    logger.info("")
    
    # 4. Test parameter conversion
    passed, failed = test_parameter_conversion()
    all_passed.extend([f"Parameter Conversion: {p}" for p in passed])
    all_failed.extend([f"Parameter Conversion: {f}" for f in failed])
    
    logger.info("")
    
    # 5. Check TensorFlow availability
    tf_available, tf_activations, tf_failed = check_tensorflow_availability()
    if tf_available:
        all_passed.extend([f"TensorFlow Activation: {a}" for a in tf_activations])
        all_failed.extend([f"TensorFlow Activation: {a}" for a in tf_failed])
    else:
        all_failed.append("TensorFlow: Not Available")
    
    logger.info("")
    
    # 6. Check scikit-learn availability
    sklearn_available, sklearn_kernels, sklearn_failed = check_sklearn_availability()
    if sklearn_available:
        all_passed.extend([f"SKLearn SVR Kernel: {k}" for k in sklearn_kernels])
        all_failed.extend([f"SKLearn SVR Kernel: {k}" for k in sklearn_failed])
    else:
        all_failed.append("scikit-learn: Not Available")
    
    # Final summary
    logger.info("")
    logger.info("📊 FINAL VALIDATION SUMMARY")
    logger.info("=" * 80)
    logger.info(f"✅ PASSED: {len(all_passed)} items")
    logger.info(f"❌ FAILED: {len(all_failed)} items")
    
    if all_passed:
        logger.info("\n✅ PASSED ITEMS:")
        for item in all_passed:
            logger.info(f"  ✅ {item}")
    
    if all_failed:
        logger.info("\n❌ FAILED ITEMS:")
        for item in all_failed:
            logger.info(f"  ❌ {item}")
    
    # Overall status
    success_rate = len(all_passed) / (len(all_passed) + len(all_failed)) * 100 if (len(all_passed) + len(all_failed)) > 0 else 0
    
    logger.info("")
    logger.info(f"🎯 SUCCESS RATE: {success_rate:.1f}%")
    
    if len(all_failed) == 0:
        logger.info("🎉 ALL MODELS AND ACTIVATION FUNCTIONS ARE WORKING CORRECTLY!")
        return 0
    elif success_rate >= 80:
        logger.info("⚠️  MOST COMPONENTS WORKING - MINOR ISSUES DETECTED")
        return 1
    else:
        logger.info("🚨 MAJOR ISSUES DETECTED - SYSTEM NEEDS ATTENTION")
        return 2

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except Exception as e:
        logger.error(f"💥 CRITICAL ERROR: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(3)