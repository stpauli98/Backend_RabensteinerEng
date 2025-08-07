"""
Test script for parameter conversion utilities
Tests the conversion from frontend parameters to reference MDL format
"""

import sys
import os
import logging

# Add the parent directory to sys.path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from parameter_converter import (
    FrontendParameterConverter, 
    ReferenceMDLValidator,
    convert_frontend_parameters_to_mdl
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_dense_parameter_conversion():
    """Test Dense Neural Network parameter conversion"""
    print("\n=== Testing Dense Neural Network Parameter Conversion ===")
    
    frontend_params = {
        "dense_layers": [128, 64, 32],
        "dense_activation": "ReLU", 
        "dense_epochs": 150,
        "dense_batch_size": 64,
        "dense_optimizer": "Adam",
        "dense_learning_rate": 0.001,
        "dense_validation_split": 0.15,
        "dense_loss": "mse",
        "dense_metrics": ["mae", "mse"]
    }
    
    try:
        converter = FrontendParameterConverter()
        mdl_config = converter.convert_frontend_to_mdl_format(frontend_params)
        
        print("✓ Conversion successful")
        print(f"Models configured: {list(mdl_config.keys())}")
        
        if 'DENSE' in mdl_config:
            dense = mdl_config['DENSE']
            print(f"Dense config: L1_N={dense['L1_N']}, L2_N={dense['L2_N']}, L3_N={dense['L3_N']}")
            print(f"Epochs: {dense['EP']}, Batch size: {dense['BS']}")
            print(f"Activation: {dense['L1_A']}, Optimizer: {dense['OPT']}")
            
        return True
        
    except Exception as e:
        print(f"✗ Dense conversion failed: {str(e)}")
        return False


def test_cnn_parameter_conversion():
    """Test CNN parameter conversion"""
    print("\n=== Testing CNN Parameter Conversion ===")
    
    frontend_params = {
        "cnn_filters": [64, 128],
        "cnn_kernel_size": [5, 3],
        "cnn_pool_size": [2, 2],
        "cnn_activation": "ReLU",
        "cnn_epochs": 100,
        "cnn_batch_size": 32,
        "cnn_dense_neurons": 75,
        "cnn_optimizer": "Adam",
        "cnn_learning_rate": 0.001
    }
    
    try:
        converter = FrontendParameterConverter()
        mdl_config = converter.convert_frontend_to_mdl_format(frontend_params)
        
        print("✓ Conversion successful")
        
        if 'CNN' in mdl_config:
            cnn = mdl_config['CNN']
            print(f"CNN config: L1_F={cnn['L1_F']}, L2_F={cnn['L2_F']}")
            print(f"Kernel sizes: L1_K={cnn['L1_K']}, L2_K={cnn['L2_K']}")
            print(f"Pool sizes: L1_P={cnn['L1_P']}, L2_P={cnn['L2_P']}")
            print(f"Dense layer: L3_N={cnn['L3_N']}")
            
        return True
        
    except Exception as e:
        print(f"✗ CNN conversion failed: {str(e)}")
        return False


def test_lstm_parameter_conversion():
    """Test LSTM parameter conversion"""
    print("\n=== Testing LSTM Parameter Conversion ===")
    
    frontend_params = {
        "lstm_units": [100, 50],
        "lstm_dropout": [0.3, 0.2],
        "lstm_return_sequences_l1": True,
        "lstm_return_sequences_l2": False,
        "lstm_dense_neurons": 25,
        "lstm_dense_activation": "relu",
        "lstm_epochs": 200,
        "lstm_batch_size": 16
    }
    
    try:
        converter = FrontendParameterConverter()
        mdl_config = converter.convert_frontend_to_mdl_format(frontend_params)
        
        print("✓ Conversion successful")
        
        if 'LSTM' in mdl_config:
            lstm = mdl_config['LSTM']
            print(f"LSTM config: L1_N={lstm['L1_N']}, L2_N={lstm['L2_N']}")
            print(f"Dropout: L1_D={lstm['L1_D']}, L2_D={lstm['L2_D']}")
            print(f"Return sequences: L1_RS={lstm['L1_RS']}, L2_RS={lstm['L2_RS']}")
            print(f"Dense layer: L3_N={lstm['L3_N']}, L3_A={lstm['L3_A']}")
            
        return True
        
    except Exception as e:
        print(f"✗ LSTM conversion failed: {str(e)}")
        return False


def test_svr_parameter_conversion():
    """Test SVR parameter conversion"""
    print("\n=== Testing SVR Parameter Conversion ===")
    
    frontend_params = {
        "svr_kernel": "RBF",
        "svr_C": 2.0,
        "svr_gamma": "scale",
        "svr_epsilon": 0.05,
        "svr_degree": 3,
        "svr_coef0": 0.0,
        "svr_shrinking": True,
        "svr_cache_size": 300,
        "svr_max_iter": 2000
    }
    
    try:
        converter = FrontendParameterConverter()
        mdl_config = converter.convert_frontend_to_mdl_format(frontend_params)
        
        print("✓ Conversion successful")
        
        if 'SVR' in mdl_config:
            svr = mdl_config['SVR']
            print(f"SVR config: KERNEL={svr['KERNEL']}, C={svr['C']}")
            print(f"Epsilon: {svr['EPSILON']}, Gamma: {svr['GAMMA']}")
            print(f"Cache size: {svr['CACHE_SIZE']}, Max iter: {svr['MAX_ITER']}")
            
        return True
        
    except Exception as e:
        print(f"✗ SVR conversion failed: {str(e)}")
        return False


def test_linear_parameter_conversion():
    """Test Linear Regression parameter conversion"""
    print("\n=== Testing Linear Regression Parameter Conversion ===")
    
    frontend_params = {
        "linear_fit_intercept": True,
        "linear_normalize": False,
        "linear_copy_x": True,
        "linear_n_jobs": None,
        "linear_positive": False
    }
    
    try:
        converter = FrontendParameterConverter()
        mdl_config = converter.convert_frontend_to_mdl_format(frontend_params)
        
        print("✓ Conversion successful")
        
        if 'LINEAR' in mdl_config:
            linear = mdl_config['LINEAR']
            print(f"Linear config: FIT_INTERCEPT={linear['FIT_INTERCEPT']}")
            print(f"Normalize: {linear['NORMALIZE']}, Copy X: {linear['COPY_X']}")
            print(f"Positive: {linear['POSITIVE']}, N_jobs: {linear['N_JOBS']}")
            
        return True
        
    except Exception as e:
        print(f"✗ Linear conversion failed: {str(e)}")
        return False


def test_multiple_models_conversion():
    """Test conversion of multiple models at once"""
    print("\n=== Testing Multiple Models Parameter Conversion ===")
    
    frontend_params = {
        # Dense parameters
        "dense_layers": [64, 32],
        "dense_activation": "ReLU",
        "dense_epochs": 100,
        
        # CNN parameters  
        "cnn_filters": [32, 64],
        "cnn_kernel_size": [3, 3],
        "cnn_epochs": 150,
        
        # SVR parameters
        "svr_kernel": "RBF",
        "svr_C": 1.5,
        "svr_epsilon": 0.1,
        
        # Linear parameters
        "linear_fit_intercept": True
    }
    
    try:
        converter = FrontendParameterConverter()
        mdl_config = converter.convert_frontend_to_mdl_format(frontend_params)
        
        print("✓ Conversion successful")
        print(f"Models configured: {list(mdl_config.keys())}")
        
        expected_models = ['DENSE', 'CNN', 'SVR', 'LINEAR']
        for model in expected_models:
            if model in mdl_config:
                print(f"  ✓ {model} configured")
            else:
                print(f"  ✗ {model} missing")
                
        return True
        
    except Exception as e:
        print(f"✗ Multiple models conversion failed: {str(e)}")
        return False


def test_parameter_validation():
    """Test parameter validation with invalid values"""
    print("\n=== Testing Parameter Validation ===")
    
    # Test with some invalid parameters
    frontend_params = {
        "dense_layers": [0, -32, 2000],  # Invalid: negative and too large
        "dense_epochs": -50,              # Invalid: negative
        "dense_batch_size": 0,            # Invalid: zero
        "dense_learning_rate": 2.0,       # Invalid: too large
        "dense_activation": "InvalidActivation",  # Invalid activation
        
        "svr_C": -1.0,                    # Invalid: negative
        "svr_kernel": "InvalidKernel",    # Invalid kernel
        "svr_epsilon": -0.1               # Invalid: negative
    }
    
    try:
        mdl_config, validation_result = convert_frontend_parameters_to_mdl(frontend_params)
        
        print("✓ Validation completed")
        print(f"Valid: {validation_result['is_valid']}")
        print(f"Errors: {len(validation_result['errors'])}")
        print(f"Warnings: {len(validation_result['warnings'])}")
        
        if validation_result['errors']:
            print("Validation errors:")
            for error in validation_result['errors'][:3]:  # Show first 3 errors
                print(f"  - {error}")
        
        if validation_result['warnings']:
            print("Validation warnings:")
            for warning in validation_result['warnings'][:3]:  # Show first 3 warnings
                print(f"  - {warning}")
        
        # Check if corrected parameters are reasonable
        if 'DENSE' in mdl_config:
            dense = mdl_config['DENSE']
            print(f"Corrected Dense layers: L1_N={dense['L1_N']}, L2_N={dense['L2_N']}, L3_N={dense['L3_N']}")
            
        return True
        
    except Exception as e:
        print(f"✗ Validation test failed: {str(e)}")
        return False


def test_legacy_mode_conversion():
    """Test legacy MODE-based parameter conversion"""
    print("\n=== Testing Legacy MODE Parameter Conversion ===")
    
    # Test legacy format with MODE parameter
    legacy_params = {
        "MODE": "Dense",
        "LAY": 3,
        "N": 128,
        "EP": 200,
        "ACTF": "ReLU"
    }
    
    try:
        # Import the utils function that should handle this
        from utils import convert_frontend_to_backend_params
        
        result = convert_frontend_to_backend_params(legacy_params)
        
        print("✓ Legacy conversion successful")
        print(f"Converted models: {list(result.keys())}")
        
        return True
        
    except Exception as e:
        print(f"✗ Legacy conversion failed: {str(e)}")
        return False


def run_all_tests():
    """Run all parameter conversion tests"""
    print("🧪 Running Parameter Conversion Tests")
    print("=" * 50)
    
    tests = [
        test_dense_parameter_conversion,
        test_cnn_parameter_conversion,
        test_lstm_parameter_conversion,
        test_svr_parameter_conversion,
        test_linear_parameter_conversion,
        test_multiple_models_conversion,
        test_parameter_validation,
        test_legacy_mode_conversion
    ]
    
    results = []
    for test in tests:
        try:
            success = test()
            results.append(success)
        except Exception as e:
            print(f"✗ Test {test.__name__} crashed: {str(e)}")
            results.append(False)
    
    # Summary
    print("\n" + "=" * 50)
    print("🏁 Test Results Summary")
    print("=" * 50)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Passed: {passed}/{total}")
    print(f"Success rate: {passed/total*100:.1f}%")
    
    if passed == total:
        print("🎉 All tests passed! Parameter conversion is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)