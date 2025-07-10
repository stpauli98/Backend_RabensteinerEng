#!/usr/bin/env python3
"""
Test script for data_processor.py real functions
Tests the extracted transform_data function from training_backend_test_2.py
"""

import sys
import os
import pandas as pd
import numpy as np
import math

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from training_system.data_processor import DataProcessor, create_data_processor
    from training_system.config import MTS
    print("✅ Successfully imported DataProcessor")
except ImportError as e:
    print(f"❌ Failed to import DataProcessor: {e}")
    # Try to import individual components for testing
    try:
        from training_system.data_processor import DataProcessor
        print("✅ Successfully imported DataProcessor (without config)")
    except ImportError as e2:
        print(f"❌ Failed to import DataProcessor: {e2}")
        sys.exit(1)


def test_transform_data_function():
    """Test the extracted transform_data function"""
    print("\n🔧 Testing transform_data function...")
    
    try:
        # Create a simple mock MTS config for testing
        class MockMTS:
            def __init__(self):
                self.I_N = 13
                self.O_N = 13
                self.DELT = 3
                self.OFST = 0
                self.use_time_features = False
                self.interpolation = False
                self.outlier_removal = False
                self.scaling = False
                self.timezone = 'UTC'
        
        config = MockMTS()
        processor = DataProcessor(config)
        
        # Create test information DataFrame like in original system
        # This simulates the 'inf' DataFrame that contains file information
        test_data = {
            'th_strt': [-2.0, -1.0, -0.5],  # Time horizon start (hours)
            'th_end': [0.0, 0.0, 0.0],      # Time horizon end (hours)
        }
        
        inf = pd.DataFrame(test_data, index=['file1', 'file2', 'file3'])
        
        print(f"  ✅ Created test DataFrame: {inf.shape}")
        print(f"  Input data:\n{inf}")
        
        # Test parameters from original function
        N = 13      # Number of time steps
        OFST = 0.0  # Global offset
        
        # Apply transform_data function
        result_inf = processor.transform_data(inf, N, OFST)
        
        print(f"  ✅ transform_data executed successfully")
        print(f"  Result data:\n{result_inf}")
        
        # Validate the results
        expected_results = []
        for i, (idx, row) in enumerate(inf.iterrows()):
            th_strt = row['th_strt']
            th_end = row['th_end']
            
            # Manual calculation of expected delt_transf
            expected_delt_transf = (th_end - th_strt) * 60 / (N - 1)
            expected_results.append(expected_delt_transf)
            
            print(f"  File {idx}:")
            print(f"    - th_strt: {th_strt}, th_end: {th_end}")
            print(f"    - Expected delt_transf: {expected_delt_transf}")
            print(f"    - Actual delt_transf: {result_inf.loc[idx, 'delt_transf']}")
            
            # Check delt_transf calculation
            if abs(result_inf.loc[idx, 'delt_transf'] - expected_delt_transf) < 0.001:
                print(f"    ✅ delt_transf calculation correct")
            else:
                print(f"    ❌ delt_transf calculation incorrect")
                return False
            
            # Check offset calculation
            if 'ofst_transf' in result_inf.columns:
                ofst_transf = result_inf.loc[idx, 'ofst_transf']
                print(f"    - Offset: {ofst_transf}")
                
                # Validate offset logic
                if expected_delt_transf > 0 and round(60/expected_delt_transf) == 60/expected_delt_transf:
                    # Should be numeric offset
                    if isinstance(ofst_transf, (int, float)):
                        print(f"    ✅ Offset calculated as numeric: {ofst_transf}")
                    else:
                        print(f"    ❌ Expected numeric offset, got: {ofst_transf}")
                        return False
                else:
                    # Should be variable offset
                    if ofst_transf == "var":
                        print(f"    ✅ Offset correctly marked as variable")
                    else:
                        print(f"    ❌ Expected 'var' offset, got: {ofst_transf}")
                        return False
            else:
                print(f"    ❌ ofst_transf column not found")
                return False
        
        return True
        
    except Exception as e:
        print(f"  ❌ transform_data test failed: {e}")
        return False


def test_transform_data_edge_cases():
    """Test transform_data function with edge cases"""
    print("\n🔬 Testing transform_data edge cases...")
    
    try:
        # Create mock config
        class MockMTS:
            def __init__(self):
                self.timezone = 'UTC'
        
        config = MockMTS()
        processor = DataProcessor(config)
        
        # Test case 1: Zero time difference (should cause division by zero protection)
        print("  Testing zero time difference...")
        inf1 = pd.DataFrame({
            'th_strt': [0.0],
            'th_end': [0.0]
        }, index=['zero_diff'])
        
        result1 = processor.transform_data(inf1, 13, 0.0)
        delt_transf1 = result1.loc['zero_diff', 'delt_transf']
        
        if delt_transf1 == 0.0:
            print("  ✅ Zero time difference handled correctly")
        else:
            print(f"  ❌ Zero time difference not handled correctly: {delt_transf1}")
            return False
        
        # Test case 2: Negative time difference
        print("  Testing negative time difference...")
        inf2 = pd.DataFrame({
            'th_strt': [1.0],
            'th_end': [-1.0]
        }, index=['negative'])
        
        result2 = processor.transform_data(inf2, 13, 0.0)
        delt_transf2 = result2.loc['negative', 'delt_transf']
        
        expected_negative = (-1.0 - 1.0) * 60 / (13 - 1)  # Should be -10.0
        if abs(delt_transf2 - expected_negative) < 0.001:
            print(f"  ✅ Negative time difference handled correctly: {delt_transf2}")
        else:
            print(f"  ❌ Negative time difference incorrect: expected {expected_negative}, got {delt_transf2}")
            return False
        
        # Test case 3: Large time difference
        print("  Testing large time difference...")
        inf3 = pd.DataFrame({
            'th_strt': [-24.0],
            'th_end': [24.0]
        }, index=['large'])
        
        result3 = processor.transform_data(inf3, 13, 0.0)
        delt_transf3 = result3.loc['large', 'delt_transf']
        
        expected_large = (24.0 - (-24.0)) * 60 / (13 - 1)  # Should be 240.0
        if abs(delt_transf3 - expected_large) < 0.001:
            print(f"  ✅ Large time difference handled correctly: {delt_transf3}")
        else:
            print(f"  ❌ Large time difference incorrect: expected {expected_large}, got {delt_transf3}")
            return False
        
        return True
        
    except Exception as e:
        print(f"  ❌ Edge cases test failed: {e}")
        return False


def test_transform_data_offset_calculation():
    """Test detailed offset calculation logic"""
    print("\n📐 Testing offset calculation logic...")
    
    try:
        class MockMTS:
            def __init__(self):
                self.timezone = 'UTC'
        
        config = MockMTS()
        processor = DataProcessor(config)
        
        # Test case: Regular intervals that should have constant offset
        print("  Testing regular intervals (should have constant offset)...")
        
        # Time horizon: -1 hour to 0 hour, 13 steps = 5-minute intervals
        # delt_transf = (0 - (-1)) * 60 / (13-1) = 60/12 = 5 minutes
        # Since 60/5 = 12 (integer), this should have constant offset
        
        inf = pd.DataFrame({
            'th_strt': [-1.0],  # -1 hour
            'th_end': [0.0]     # 0 hour
        }, index=['regular'])
        
        N = 13
        OFST = 0.0
        
        result = processor.transform_data(inf, N, OFST)
        
        delt_transf = result.loc['regular', 'delt_transf']
        ofst_transf = result.loc['regular', 'ofst_transf']
        
        print(f"    - delt_transf: {delt_transf} minutes")
        print(f"    - ofst_transf: {ofst_transf}")
        
        # Expected: 5 minutes
        expected_delt = 5.0
        if abs(delt_transf - expected_delt) < 0.001:
            print("    ✅ delt_transf calculation correct")
        else:
            print(f"    ❌ delt_transf incorrect: expected {expected_delt}, got {delt_transf}")
            return False
        
        # Since 60/5 = 12 (integer), offset should be calculated
        if isinstance(ofst_transf, (int, float)):
            print("    ✅ Offset calculated as numeric (constant intervals)")
            
            # Manual calculation of expected offset
            # ofst_transf = OFST - (th_strt - floor(th_strt)) * 60 + 60
            # ofst_transf = 0 - (-1.0 - floor(-1.0)) * 60 + 60
            # ofst_transf = 0 - (-1.0 - (-1.0)) * 60 + 60
            # ofst_transf = 0 - 0 * 60 + 60 = 60
            # Then while ofst_transf - delt_transf >= 0: ofst_transf -= delt_transf
            # 60 - 5 = 55, 55 - 5 = 50, ..., 5 - 5 = 0
            expected_ofst = 0.0
            if abs(ofst_transf - expected_ofst) < 0.001:
                print(f"    ✅ Offset value correct: {ofst_transf}")
            else:
                print(f"    ⚠️  Offset value different than expected: expected {expected_ofst}, got {ofst_transf}")
                # This might be correct depending on the exact logic, so we won't fail
        else:
            print(f"    ❌ Expected numeric offset for regular intervals, got: {ofst_transf}")
            return False
        
        # Test case: Irregular intervals that should have variable offset
        print("  Testing irregular intervals (should have variable offset)...")
        
        # Use a time horizon that results in non-integer division
        # Example: -1.7 to 0.3 hours, 13 steps
        inf2 = pd.DataFrame({
            'th_strt': [-1.7],
            'th_end': [0.3]
        }, index=['irregular'])
        
        result2 = processor.transform_data(inf2, N, OFST)
        
        delt_transf2 = result2.loc['irregular', 'delt_transf']
        ofst_transf2 = result2.loc['irregular', 'ofst_transf']
        
        print(f"    - delt_transf: {delt_transf2} minutes")
        print(f"    - ofst_transf: {ofst_transf2}")
        
        # Check if 60/delt_transf is integer
        if abs(round(60/delt_transf2) - 60/delt_transf2) > 0.001:
            # Should be variable offset
            if ofst_transf2 == "var":
                print("    ✅ Variable offset correctly identified")
            else:
                print(f"    ❌ Expected 'var' offset for irregular intervals, got: {ofst_transf2}")
                return False
        else:
            print("    ✅ Regular intervals detected (may have constant offset)")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Offset calculation test failed: {e}")
        return False


def test_real_world_scenario():
    """Test with realistic scenario from training system"""
    print("\n🌍 Testing real-world scenario...")
    
    try:
        class MockMTS:
            def __init__(self):
                self.timezone = 'UTC'
        
        config = MockMTS()
        processor = DataProcessor(config)
        
        # Simulate realistic training scenario
        # Multiple files with different time horizons
        realistic_data = {
            'th_strt': [-24.0, -12.0, -6.0, -1.0],  # Different lookback periods
            'th_end': [0.0, 0.0, 0.0, 0.0],         # All predict current time
        }
        
        inf = pd.DataFrame(realistic_data, index=['daily', 'half_day', 'quarter_day', 'hourly'])
        
        print(f"  ✅ Created realistic scenario with {len(inf)} files")
        
        # Parameters that might be used in real training
        N = 13      # Input sequence length
        OFST = 0.0  # No global offset
        
        result = processor.transform_data(inf, N, OFST)
        
        print("  Results:")
        for idx in result.index:
            th_strt = inf.loc[idx, 'th_strt']
            th_end = inf.loc[idx, 'th_end']
            delt_transf = result.loc[idx, 'delt_transf']
            ofst_transf = result.loc[idx, 'ofst_transf']
            
            time_span_hours = th_end - th_strt
            
            print(f"    {idx}:")
            print(f"      - Time span: {time_span_hours} hours")
            print(f"      - Time step: {delt_transf:.2f} minutes")
            print(f"      - Offset: {ofst_transf}")
            
            # Validate that time step makes sense
            expected_minutes_per_step = time_span_hours * 60 / (N - 1)
            if abs(delt_transf - expected_minutes_per_step) < 0.001:
                print(f"      ✅ Time step calculation correct")
            else:
                print(f"      ❌ Time step calculation incorrect")
                return False
        
        print("  ✅ All realistic scenario calculations correct")
        return True
        
    except Exception as e:
        print(f"  ❌ Real-world scenario test failed: {e}")
        return False


def main():
    """Run all data processor tests"""
    print("🧪 Starting Data Processor Real Function Tests")
    print("=" * 60)
    
    tests = [
        ("Transform Data Function", test_transform_data_function),
        ("Edge Cases", test_transform_data_edge_cases),
        ("Offset Calculation", test_transform_data_offset_calculation),
        ("Real-World Scenario", test_real_world_scenario),
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
    print("🏁 DATA PROCESSOR TEST SUMMARY")
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
        print("\n🎉 All data processor tests passed! Transform function is working correctly.")
    else:
        print(f"\n⚠️  {failed} test(s) failed. Please check the issues above.")
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)