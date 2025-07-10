#!/usr/bin/env python3
"""
Test script for model_trainer.py real functions
Tests the extracted ML training functions from training_backend_test_2.py
"""

import sys
import os
import numpy as np
import warnings

# Suppress TensorFlow warnings for cleaner output
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from training_system.model_trainer import (
        train_dense, train_cnn, train_lstm, train_ar_lstm, 
        train_svr_dir, train_svr_mimo, train_linear_model,
        ModelTrainer, create_model_trainer
    )
    from training_system.config import MDL
    print("✅ Successfully imported all ML training functions")
except ImportError as e:
    print(f"❌ Failed to import ML functions: {e}")
    sys.exit(1)


def create_test_data():
    """Create synthetic test data for ML models"""
    try:
        # Create synthetic time series data similar to what the models expect
        n_samples = 100
        n_timesteps_in = 13  # MTS.I_N
        n_timesteps_out = 13  # MTS.O_N
        n_features_in = 2
        n_features_out = 1
        
        # Generate synthetic input data (time series with some pattern)
        np.random.seed(42)  # For reproducible tests
        
        # Create input data with some temporal patterns
        X = np.random.randn(n_samples, n_timesteps_in, n_features_in)
        for i in range(n_samples):
            # Add some temporal correlation
            for t in range(1, n_timesteps_in):
                X[i, t] = 0.7 * X[i, t-1] + 0.3 * X[i, t]
        
        # Create output data that somewhat depends on input (for realistic testing)
        y = np.random.randn(n_samples, n_timesteps_out, n_features_out)
        for i in range(n_samples):
            # Make output somewhat dependent on input mean
            input_mean = np.mean(X[i])
            y[i] = y[i] + 0.1 * input_mean
        
        # Split into train/validation
        split_idx = int(0.8 * n_samples)
        
        train_x = X[:split_idx]
        train_y = y[:split_idx]
        val_x = X[split_idx:]
        val_y = y[split_idx:]
        
        print(f"  ✅ Created test data:")
        print(f"    - Train: X{train_x.shape}, y{train_y.shape}")
        print(f"    - Val: X{val_x.shape}, y{val_y.shape}")
        
        return train_x, train_y, val_x, val_y
        
    except Exception as e:
        print(f"  ❌ Failed to create test data: {e}")
        return None, None, None, None


def test_configuration_setup():
    """Test that MDL configuration is set up correctly for testing"""
    print("\n🔧 Testing MDL configuration setup...")
    
    try:
        # Check current MDL.MODE
        print(f"  Current MDL.MODE: '{MDL.MODE}'")
        
        # Since MDL.MODE = "LIN" by default, check linear mode attributes
        if MDL.MODE == "LIN":
            if hasattr(MDL, 'a') and MDL.a == 5:
                print("  ✅ LIN mode configuration correct (a=5)")
            else:
                print(f"  ❌ LIN mode configuration incorrect: a={getattr(MDL, 'a', 'missing')}")
                return False
        
        # For comprehensive testing, we need to temporarily change MODE to test other models
        # But we'll start with what we have
        
        return True
        
    except Exception as e:
        print(f"  ❌ Configuration setup test failed: {e}")
        return False


def test_linear_model():
    """Test the train_linear_model function"""
    print("\n📈 Testing train_linear_model function...")
    
    try:
        train_x, train_y, val_x, val_y = create_test_data()
        if train_x is None:
            return False
        
        # Test linear model (this should work regardless of MDL.MODE)
        print("  Training linear model...")
        models = train_linear_model(train_x, train_y)
        
        # Validate results
        if isinstance(models, list):
            print(f"  ✅ Returned list of {len(models)} models")
        else:
            print(f"  ❌ Expected list, got {type(models)}")
            return False
        
        # Check that we have the right number of models (one per output feature)
        expected_models = train_y.shape[2]  # n_features_out
        if len(models) == expected_models:
            print(f"  ✅ Correct number of models: {len(models)}")
        else:
            print(f"  ❌ Expected {expected_models} models, got {len(models)}")
            return False
        
        # Test that models can make predictions
        print("  Testing model predictions...")
        for i, model in enumerate(models):
            # Reshape test data for linear model
            X_test = train_x.reshape(train_x.shape[0] * train_x.shape[1], train_x.shape[2])
            
            try:
                predictions = model.predict(X_test[:10])  # Test with first 10 samples
                print(f"    ✅ Model {i} prediction shape: {predictions.shape}")
            except Exception as e:
                print(f"    ❌ Model {i} prediction failed: {e}")
                return False
        
        return True
        
    except Exception as e:
        print(f"  ❌ Linear model test failed: {e}")
        return False


def test_svr_models():
    """Test SVR model functions"""
    print("\n🤖 Testing SVR model functions...")
    
    try:
        train_x, train_y, val_x, val_y = create_test_data()
        if train_x is None:
            return False
        
        # Create mock MDL config for SVR
        class MockMDL:
            KERNEL = "poly"
            C = 1
            EPSILON = 0.1
        
        mock_mdl = MockMDL()
        
        # Test SVR_dir
        print("  Testing train_svr_dir...")
        try:
            svr_dir_models = train_svr_dir(train_x, train_y, mock_mdl)
            
            if isinstance(svr_dir_models, list) and len(svr_dir_models) > 0:
                print(f"  ✅ SVR_dir returned {len(svr_dir_models)} models")
            else:
                print(f"  ❌ SVR_dir failed: {type(svr_dir_models)}")
                return False
        except Exception as e:
            print(f"  ❌ SVR_dir test failed: {e}")
            return False
        
        # Test SVR_MIMO
        print("  Testing train_svr_mimo...")
        try:
            svr_mimo_models = train_svr_mimo(train_x, train_y, mock_mdl)
            
            if isinstance(svr_mimo_models, list) and len(svr_mimo_models) > 0:
                print(f"  ✅ SVR_MIMO returned {len(svr_mimo_models)} models")
            else:
                print(f"  ❌ SVR_MIMO failed: {type(svr_mimo_models)}")
                return False
        except Exception as e:
            print(f"  ❌ SVR_MIMO test failed: {e}")
            return False
        
        return True
        
    except Exception as e:
        print(f"  ❌ SVR models test failed: {e}")
        return False


def test_neural_network_models():
    """Test TensorFlow/Keras based models"""
    print("\n🧠 Testing Neural Network models...")
    
    try:
        train_x, train_y, val_x, val_y = create_test_data()
        if train_x is None:
            return False
        
        # Create mock MDL config for neural networks
        class MockMDL:
            LAY = 2      # 2 layers (small for testing)
            N = 32       # 32 neurons (small for testing)
            EP = 1       # 1 epoch (fast for testing)
            ACTF = "relu"
            K = 3        # For CNN kernel size
        
        mock_mdl = MockMDL()
        
        print("  Testing train_dense...")
        try:
            # Suppress model training output for cleaner test results
            import io
            import contextlib
            
            f = io.StringIO()
            with contextlib.redirect_stdout(f):
                dense_model = train_dense(train_x, train_y, val_x, val_y, mock_mdl)
            
            if dense_model is not None:
                print(f"  ✅ Dense model created successfully")
                
                # Test model prediction
                predictions = dense_model.predict(val_x[:5], verbose=0)
                print(f"    ✅ Dense model prediction shape: {predictions.shape}")
            else:
                print(f"  ❌ Dense model creation failed")
                return False
        except Exception as e:
            print(f"  ❌ Dense model test failed: {e}")
            return False
        
        print("  Testing train_cnn...")
        try:
            f = io.StringIO()
            with contextlib.redirect_stdout(f):
                cnn_model = train_cnn(train_x, train_y, val_x, val_y, mock_mdl)
            
            if cnn_model is not None:
                print(f"  ✅ CNN model created successfully")
                
                # Test model prediction
                predictions = cnn_model.predict(val_x[:5], verbose=0)
                print(f"    ✅ CNN model prediction shape: {predictions.shape}")
            else:
                print(f"  ❌ CNN model creation failed")
                return False
        except Exception as e:
            print(f"  ❌ CNN model test failed: {e}")
            return False
        
        print("  Testing train_lstm...")
        try:
            f = io.StringIO()
            with contextlib.redirect_stdout(f):
                lstm_model = train_lstm(train_x, train_y, val_x, val_y, mock_mdl)
            
            if lstm_model is not None:
                print(f"  ✅ LSTM model created successfully")
                
                # Test model prediction
                predictions = lstm_model.predict(val_x[:5], verbose=0)
                print(f"    ✅ LSTM model prediction shape: {predictions.shape}")
            else:
                print(f"  ❌ LSTM model creation failed")
                return False
        except Exception as e:
            print(f"  ❌ LSTM model test failed: {e}")
            return False
        
        return True
        
    except Exception as e:
        print(f"  ❌ Neural network models test failed: {e}")
        return False


def test_model_trainer_class():
    """Test the ModelTrainer wrapper class"""
    print("\n🏭 Testing ModelTrainer class...")
    
    try:
        # Create model trainer
        trainer = create_model_trainer()
        
        if trainer is not None:
            print("  ✅ ModelTrainer created successfully")
        else:
            print("  ❌ ModelTrainer creation failed")
            return False
        
        # Check that it has the expected attributes
        if hasattr(trainer, 'config') and hasattr(trainer, 'trained_models'):
            print("  ✅ ModelTrainer has expected attributes")
        else:
            print("  ❌ ModelTrainer missing expected attributes")
            return False
        
        # Test data splitting
        train_x, train_y, val_x, val_y = create_test_data()
        if train_x is None:
            return False
        
        # Combine data for splitting test
        X = np.concatenate([train_x, val_x], axis=0)
        y = np.concatenate([train_y, val_y], axis=0)
        
        X_train, X_test, y_train, y_test = trainer._split_data(X, y)
        
        print(f"  ✅ Data splitting works: train{X_train.shape}, test{X_test.shape}")
        
        # Test train_all_models with mock data
        # Since MDL.MODE = "LIN", it should use linear model
        datasets = {
            'test_dataset': {
                'X': X_train,
                'y': y_train,
                'time_steps_in': 13,
                'time_steps_out': 13
            }
        }
        
        session_data = {}
        
        print("  Testing train_all_models...")
        try:
            import io
            import contextlib
            
            f = io.StringIO()
            with contextlib.redirect_stdout(f):
                results = trainer.train_all_models(datasets, session_data)
            
            if results and 'test_dataset' in results:
                print("  ✅ train_all_models executed successfully")
                print(f"    Results keys: {list(results['test_dataset'].keys())}")
            else:
                print(f"  ❌ train_all_models failed: {results}")
                return False
        except Exception as e:
            print(f"  ❌ train_all_models test failed: {e}")
            return False
        
        return True
        
    except Exception as e:
        print(f"  ❌ ModelTrainer class test failed: {e}")
        return False


def test_function_signatures():
    """Test that all functions have the expected signatures from original"""
    print("\n✍️ Testing function signatures...")
    
    try:
        import inspect
        
        # Test train_dense signature
        sig = inspect.signature(train_dense)
        params = list(sig.parameters.keys())
        expected_params = ['train_x', 'train_y', 'val_x', 'val_y', 'MDL']
        
        if params == expected_params:
            print("  ✅ train_dense signature correct")
        else:
            print(f"  ❌ train_dense signature incorrect: {params} vs {expected_params}")
            return False
        
        # Test train_linear_model signature
        sig = inspect.signature(train_linear_model)
        params = list(sig.parameters.keys())
        expected_params = ['trn_x', 'trn_y']
        
        if params == expected_params:
            print("  ✅ train_linear_model signature correct")
        else:
            print(f"  ❌ train_linear_model signature incorrect: {params} vs {expected_params}")
            return False
        
        # Test SVR functions
        sig = inspect.signature(train_svr_dir)
        params = list(sig.parameters.keys())
        expected_params = ['train_x', 'train_y', 'MDL']
        
        if params == expected_params:
            print("  ✅ train_svr_dir signature correct")
        else:
            print(f"  ❌ train_svr_dir signature incorrect: {params} vs {expected_params}")
            return False
        
        return True
        
    except Exception as e:
        print(f"  ❌ Function signatures test failed: {e}")
        return False


def main():
    """Run all model trainer tests"""
    print("🧪 Starting Model Trainer Real Functions Tests")
    print("=" * 60)
    
    tests = [
        ("Configuration Setup", test_configuration_setup),
        ("Linear Model", test_linear_model),
        ("SVR Models", test_svr_models),
        ("Neural Network Models", test_neural_network_models),
        ("ModelTrainer Class", test_model_trainer_class),
        ("Function Signatures", test_function_signatures),
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
    print("\n" + "=" * 60)
    print("🏁 MODEL TRAINER TEST SUMMARY")
    print("=" * 60)
    
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
        print("\n🎉 All model trainer tests passed! Real ML functions are working correctly.")
    else:
        print(f"\n⚠️  {failed} test(s) failed. Please check the issues above.")
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)