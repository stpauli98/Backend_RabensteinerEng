#!/usr/bin/env python3
"""
Test script for results_generator.py real evaluation functions
Tests the extracted evaluation functions (wape, smape, mase) from training_backend_test_2.py
"""

import sys
import os
import numpy as np
import warnings

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from training_system.results_generator import (
        wape, smape, mase,
        ResultsGenerator, create_results_generator
    )
    print("✅ Successfully imported all evaluation functions")
except ImportError as e:
    print(f"❌ Failed to import evaluation functions: {e}")
    sys.exit(1)


def create_test_data():
    """Create synthetic test data for evaluation metrics"""
    try:
        # Create synthetic evaluation data
        np.random.seed(42)  # For reproducible tests
        
        # Create true values with some pattern
        n_samples = 100
        y_true = np.random.randn(n_samples) * 10 + 50  # Values around 50
        
        # Create predicted values with some correlation to true values
        y_pred = y_true + np.random.randn(n_samples) * 2  # Some prediction error
        
        print(f"  ✅ Created test data:")
        print(f"    - y_true: shape {y_true.shape}, mean {y_true.mean():.2f}, std {y_true.std():.2f}")
        print(f"    - y_pred: shape {y_pred.shape}, mean {y_pred.mean():.2f}, std {y_pred.std():.2f}")
        
        return y_true, y_pred
        
    except Exception as e:
        print(f"  ❌ Failed to create test data: {e}")
        return None, None


def test_wape_function():
    """Test the wape function"""
    print("\\n📊 Testing wape function...")
    
    try:
        y_true, y_pred = create_test_data()
        if y_true is None:
            return False
        
        # Test basic WAPE calculation
        print("  Testing basic WAPE calculation...")
        wape_result = wape(y_true, y_pred)
        
        if isinstance(wape_result, (int, float)) and not np.isnan(wape_result):
            print(f"  ✅ WAPE calculated successfully: {wape_result:.4f}%")
        else:
            print(f"  ❌ WAPE calculation failed: {wape_result}")
            return False
        
        # Test edge case: zero true values
        print("  Testing edge case: zero denominator...")
        y_true_zero = np.zeros(10)
        y_pred_zero = np.ones(10)
        
        wape_zero = wape(y_true_zero, y_pred_zero)
        if np.isnan(wape_zero):
            print("  ✅ WAPE correctly handles zero denominator (returns NaN)")
        else:
            print(f"  ❌ WAPE should return NaN for zero denominator, got {wape_zero}")
            return False
        
        # Test perfect prediction
        print("  Testing perfect prediction...")
        wape_perfect = wape(y_true, y_true)
        if abs(wape_perfect) < 1e-10:  # Should be very close to 0
            print(f"  ✅ WAPE for perfect prediction: {wape_perfect:.10f}%")
        else:
            print(f"  ❌ WAPE for perfect prediction should be ~0, got {wape_perfect}")
            return False
        
        return True
        
    except Exception as e:
        print(f"  ❌ WAPE function test failed: {e}")
        return False


def test_smape_function():
    """Test the smape function"""
    print("\\n📈 Testing smape function...")
    
    try:
        y_true, y_pred = create_test_data()
        if y_true is None:
            return False
        
        # Test basic SMAPE calculation
        print("  Testing basic SMAPE calculation...")
        smape_result = smape(y_true, y_pred)
        
        if isinstance(smape_result, (int, float)) and not np.isnan(smape_result):
            print(f"  ✅ SMAPE calculated successfully: {smape_result:.4f}%")
        else:
            print(f"  ❌ SMAPE calculation failed: {smape_result}")
            return False
        
        # Test edge case: both true and predicted are zero
        print("  Testing edge case: both zero...")
        y_true_zero = np.zeros(10)
        y_pred_zero = np.zeros(10)
        
        smape_zero = smape(y_true_zero, y_pred_zero)
        if isinstance(smape_zero, (int, float)):
            print(f"  ✅ SMAPE handles both zero case: {smape_zero:.4f}%")
        else:
            print(f"  ❌ SMAPE failed for both zero case: {smape_zero}")
            return False
        
        # Test perfect prediction
        print("  Testing perfect prediction...")
        smape_perfect = smape(y_true, y_true)
        if abs(smape_perfect) < 1e-10:  # Should be very close to 0
            print(f"  ✅ SMAPE for perfect prediction: {smape_perfect:.10f}%")
        else:
            print(f"  ❌ SMAPE for perfect prediction should be ~0, got {smape_perfect}")
            return False
        
        return True
        
    except Exception as e:
        print(f"  ❌ SMAPE function test failed: {e}")
        return False


def test_mase_function():
    """Test the mase function"""
    print("\\n📉 Testing mase function...")
    
    try:
        y_true, y_pred = create_test_data()
        if y_true is None:
            return False
        
        # Test basic MASE calculation with m=1
        print("  Testing basic MASE calculation (m=1)...")
        mase_result = mase(y_true, y_pred, m=1)
        
        if isinstance(mase_result, (int, float)) and not np.isnan(mase_result):
            print(f"  ✅ MASE calculated successfully: {mase_result:.4f}")
        else:
            print(f"  ❌ MASE calculation failed: {mase_result}")
            return False
        
        # Test with different seasonality
        print("  Testing MASE with m=5...")
        if len(y_true) > 5:
            mase_m5 = mase(y_true, y_pred, m=5)
            if isinstance(mase_m5, (int, float)) and not np.isnan(mase_m5):
                print(f"  ✅ MASE with m=5: {mase_m5:.4f}")
            else:
                print(f"  ❌ MASE with m=5 failed: {mase_m5}")
                return False
        
        # Test edge case: insufficient data
        print("  Testing edge case: insufficient data...")
        y_short = np.array([1, 2, 3])
        y_pred_short = np.array([1.1, 2.1, 3.1])
        
        try:
            mase_short = mase(y_short, y_pred_short, m=5)  # m > n
            print(f"  ❌ MASE should raise ValueError for insufficient data, got {mase_short}")
            return False
        except ValueError as e:
            print(f"  ✅ MASE correctly raises ValueError: {str(e)}")
        
        # Test edge case: naive MAE is zero
        print("  Testing edge case: constant true values...")
        y_constant = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
        y_pred_varied = np.array([5.1, 5.2, 5.0, 4.9, 5.1])
        
        try:
            mase_constant = mase(y_constant, y_pred_varied, m=1)
            print(f"  ❌ MASE should raise ZeroDivisionError for constant values, got {mase_constant}")
            return False
        except ZeroDivisionError as e:
            print(f"  ✅ MASE correctly raises ZeroDivisionError: {str(e)}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ MASE function test failed: {e}")
        return False


def test_results_generator_class():
    """Test the ResultsGenerator class"""
    print("\\n🏭 Testing ResultsGenerator class...")
    
    try:
        # Create results generator
        generator = create_results_generator()
        
        if generator is not None:
            print("  ✅ ResultsGenerator created successfully")
        else:
            print("  ❌ ResultsGenerator creation failed")
            return False
        
        # Test class methods
        y_true, y_pred = create_test_data()
        if y_true is None:
            return False
        
        # Test class wape method
        print("  Testing class wape method...")
        class_wape = generator.wape(y_true, y_pred)
        standalone_wape = wape(y_true, y_pred)
        
        if abs(class_wape - standalone_wape) < 1e-10:
            print(f"  ✅ Class wape matches standalone: {class_wape:.4f}%")
        else:
            print(f"  ❌ Class wape differs: {class_wape:.4f}% vs {standalone_wape:.4f}%")
            return False
        
        # Test class smape method
        print("  Testing class smape method...")
        class_smape = generator.smape(y_true, y_pred)
        standalone_smape = smape(y_true, y_pred)
        
        if abs(class_smape - standalone_smape) < 1e-10:
            print(f"  ✅ Class smape matches standalone: {class_smape:.4f}%")
        else:
            print(f"  ❌ Class smape differs: {class_smape:.4f}% vs {standalone_smape:.4f}%")
            return False
        
        # Test class mase method
        print("  Testing class mase method...")
        class_mase = generator.mase(y_true, y_pred, m=1)
        standalone_mase = mase(y_true, y_pred, m=1)
        
        if abs(class_mase - standalone_mase) < 1e-10:
            print(f"  ✅ Class mase matches standalone: {class_mase:.4f}")
        else:
            print(f"  ❌ Class mase differs: {class_mase:.4f} vs {standalone_mase:.4f}")
            return False
        
        return True
        
    except Exception as e:
        print(f"  ❌ ResultsGenerator class test failed: {e}")
        return False


def test_function_signatures():
    """Test that all functions have the expected signatures from original"""
    print("\\n✍️ Testing function signatures...")
    
    try:
        import inspect
        
        # Test wape signature
        sig = inspect.signature(wape)
        params = list(sig.parameters.keys())
        expected_params = ['y_true', 'y_pred']
        
        if params == expected_params:
            print("  ✅ wape signature correct")
        else:
            print(f"  ❌ wape signature incorrect: {params} vs {expected_params}")
            return False
        
        # Test smape signature
        sig = inspect.signature(smape)
        params = list(sig.parameters.keys())
        expected_params = ['y_true', 'y_pred']
        
        if params == expected_params:
            print("  ✅ smape signature correct")
        else:
            print(f"  ❌ smape signature incorrect: {params} vs {expected_params}")
            return False
        
        # Test mase signature
        sig = inspect.signature(mase)
        params = list(sig.parameters.keys())
        expected_params = ['y_true', 'y_pred', 'm']
        
        if params == expected_params:
            print("  ✅ mase signature correct")
        else:
            print(f"  ❌ mase signature incorrect: {params} vs {expected_params}")
            return False
        
        return True
        
    except Exception as e:
        print(f"  ❌ Function signatures test failed: {e}")
        return False


def test_numerical_accuracy():
    """Test numerical accuracy against known values"""
    print("\\n🔢 Testing numerical accuracy...")
    
    try:
        # Test with known values
        y_true = np.array([100, 200, 300, 400, 500])
        y_pred = np.array([110, 190, 310, 390, 510])
        
        # Expected WAPE: |10+10+10+10+10| / |100+200+300+400+500| = 50/1500 = 0.0333... = 3.33%
        expected_wape = 3.333333333333333
        actual_wape = wape(y_true, y_pred)
        
        if abs(actual_wape - expected_wape) < 1e-10:
            print(f"  ✅ WAPE numerical accuracy: {actual_wape:.10f}% (expected: {expected_wape:.10f}%)")
        else:
            print(f"  ❌ WAPE numerical error: {actual_wape:.10f}% vs {expected_wape:.10f}%")
            return False
        
        # Test SMAPE
        # Manual calculation for verification:
        # For each pair: |y_pred - y_true| / ((|y_true| + |y_pred|) / 2)
        # Expected: should be reasonable positive percentage
        smape_result = smape(y_true, y_pred)
        
        if 0.0 < smape_result < 20.0:  # Reasonable range for this test data
            print(f"  ✅ SMAPE numerical result reasonable: {smape_result:.4f}%")
        else:
            print(f"  ❌ SMAPE numerical result unexpected: {smape_result:.4f}%")
            return False
        
        # Test MASE
        mase_result = mase(y_true, y_pred, m=1)
        
        if 0.05 < mase_result < 2.0:  # Should be reasonable range (adjusted)
            print(f"  ✅ MASE numerical result reasonable: {mase_result:.4f}")
        else:
            print(f"  ❌ MASE numerical result unexpected: {mase_result:.4f}")
            return False
        
        return True
        
    except Exception as e:
        print(f"  ❌ Numerical accuracy test failed: {e}")
        return False


def main():
    """Run all results generator tests"""
    print("🧪 Starting Results Generator Real Functions Tests")
    print("=" * 60)
    
    tests = [
        ("WAPE Function", test_wape_function),
        ("SMAPE Function", test_smape_function),
        ("MASE Function", test_mase_function),
        ("ResultsGenerator Class", test_results_generator_class),
        ("Function Signatures", test_function_signatures),
        ("Numerical Accuracy", test_numerical_accuracy),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\\n❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\\n" + "=" * 60)
    print("🏁 RESULTS GENERATOR TEST SUMMARY")
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
    
    print(f"\\nTotal: {passed + failed} tests")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    
    if failed == 0:
        print("\\n🎉 All results generator tests passed! Real evaluation functions are working correctly.")
    else:
        print(f"\\n⚠️  {failed} test(s) failed. Please check the issues above.")
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)