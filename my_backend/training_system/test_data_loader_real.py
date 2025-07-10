#!/usr/bin/env python3
"""
Test script for data_loader.py real functions
Tests the extracted functions from training_backend_test_2.py
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
import math

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from training_system.data_loader import DataLoader, create_data_loader
    print("✅ Successfully imported DataLoader")
except ImportError as e:
    print(f"❌ Failed to import DataLoader: {e}")
    sys.exit(1)


def test_load_csv_data():
    """Test the enhanced load_csv_data function"""
    print("\n🔧 Testing load_csv_data function...")
    
    try:
        data_loader = create_data_loader()
        
        # Test CSV path
        csv_path = "/Users/posao/Downloads/Test.csv"
        
        # Load CSV data
        df = data_loader.load_csv_data(csv_path, delimiter=";")
        
        print(f"  ✅ Successfully loaded CSV: {df.shape}")
        print(f"  ✅ Columns: {list(df.columns)}")
        print(f"  ✅ Data types: {df.dtypes.to_dict()}")
        
        # Check that we have expected columns
        if 'UTC' in df.columns and 'Leistung [kW]' in df.columns:
            print("  ✅ Expected columns found")
        else:
            print("  ❌ Expected columns not found")
            return False
            
        # Check data is not empty
        if not df.empty:
            print("  ✅ Data is not empty")
        else:
            print("  ❌ Data is empty")
            return False
            
        return True
        
    except Exception as e:
        print(f"  ❌ load_csv_data test failed: {e}")
        return False


def test_process_csv_data():
    """Test the extracted process_csv_data function"""
    print("\n📊 Testing process_csv_data function...")
    
    try:
        data_loader = create_data_loader()
        
        # Load test CSV
        csv_path = "/Users/posao/Downloads/Test.csv"
        df = data_loader.load_csv_data(csv_path, delimiter=";")
        
        # Prepare data structures like in original function
        dat = {"test_file": df}
        inf = pd.DataFrame()
        
        # Test the process_csv_data function
        updated_dat, updated_inf = data_loader.process_csv_data(dat, inf)
        
        print(f"  ✅ process_csv_data executed successfully")
        
        # Check that data was processed
        if "test_file" in updated_dat:
            processed_df = updated_dat["test_file"]
            print(f"  ✅ DataFrame updated: {processed_df.shape}")
            
            # Check UTC conversion
            if pd.api.types.is_datetime64_any_dtype(processed_df['UTC']):
                print("  ✅ UTC column converted to datetime")
            else:
                print("  ❌ UTC column not converted properly")
                return False
                
        # Check information DataFrame
        if "test_file" in updated_inf.index:
            info_row = updated_inf.loc["test_file"]
            print(f"  ✅ Information extracted:")
            print(f"    - UTC min: {info_row['utc_min']}")
            print(f"    - UTC max: {info_row['utc_max']}")
            print(f"    - Time step (delt): {info_row['delt']} minutes")
            print(f"    - Offset: {info_row['ofst']}")
            print(f"    - Total points (n_all): {info_row['n_all']}")
            print(f"    - Numeric points (n_num): {info_row['n_num']}")
            print(f"    - Numeric percentage: {info_row['rate_num']}%")
            print(f"    - Min value: {info_row['val_min']}")
            print(f"    - Max value: {info_row['val_max']}")
            
            # Validate expected results for our test data
            expected_delt = 3.0  # 3 minutes between data points
            if abs(info_row['delt'] - expected_delt) < 0.1:
                print(f"  ✅ Time step calculation correct: {info_row['delt']} ≈ {expected_delt}")
            else:
                print(f"  ❌ Time step calculation incorrect: {info_row['delt']} ≠ {expected_delt}")
                return False
            
            # Check offset calculation (should be constant for regular intervals)
            if isinstance(info_row['ofst'], (int, float)):
                print(f"  ✅ Offset calculated as numeric: {info_row['ofst']}")
            elif info_row['ofst'] == "var":
                print(f"  ✅ Offset marked as variable: {info_row['ofst']}")
            else:
                print(f"  ❌ Unexpected offset value: {info_row['ofst']}")
            
            # Check that all points are numeric (should be 100%)
            if info_row['rate_num'] == 100.0:
                print("  ✅ All data points are numeric (100%)")
            else:
                print(f"  ⚠️  Some data points are not numeric: {info_row['rate_num']}%")
            
        else:
            print("  ❌ Information not extracted properly")
            return False
            
        return True
        
    except Exception as e:
        print(f"  ❌ process_csv_data test failed: {e}")
        return False


def test_utc_index_functions():
    """Test UTC index utility functions"""
    print("\n🕐 Testing UTC index functions...")
    
    try:
        data_loader = create_data_loader()
        
        # Load test CSV and process it
        csv_path = "/Users/posao/Downloads/Test.csv"
        df = data_loader.load_csv_data(csv_path, delimiter=";")
        
        # Convert UTC to datetime manually for testing
        df['UTC'] = pd.to_datetime(df['UTC'], format="%Y-%m-%d %H:%M:%S")
        
        # Test utc_idx_pre
        print("  Testing utc_idx_pre...")
        
        # Test with a time that exists
        test_time = df['UTC'].iloc[5]  # Pick 6th timestamp
        idx_pre = data_loader.utc_idx_pre(df, test_time)
        
        if idx_pre is not None:
            print(f"  ✅ utc_idx_pre found index: {idx_pre} for time {test_time}")
            
            # Should return the exact index or the one before
            if idx_pre <= 5:
                print("  ✅ utc_idx_pre returned correct index")
            else:
                print("  ❌ utc_idx_pre returned incorrect index")
                return False
        else:
            print("  ❌ utc_idx_pre returned None")
            return False
        
        # Test utc_idx_post
        print("  Testing utc_idx_post...")
        
        # Test with a time that exists
        idx_post = data_loader.utc_idx_post(df, test_time)
        
        if idx_post is not None:
            print(f"  ✅ utc_idx_post found index: {idx_post} for time {test_time}")
            
            # Should return the exact index or the one after
            if idx_post >= 5:
                print("  ✅ utc_idx_post returned correct index")
            else:
                print("  ❌ utc_idx_post returned incorrect index")
                return False
        else:
            print("  ❌ utc_idx_post returned None")
            return False
        
        # Test with time between data points
        print("  Testing with time between data points...")
        
        time_between = df['UTC'].iloc[0] + pd.Timedelta(minutes=1.5)  # 1.5 min after first point
        
        idx_pre_between = data_loader.utc_idx_pre(df, time_between)
        idx_post_between = data_loader.utc_idx_post(df, time_between)
        
        if idx_pre_between == 0 and idx_post_between == 1:
            print("  ✅ Index functions work correctly for times between data points")
        else:
            print(f"  ❌ Index functions incorrect for between times: pre={idx_pre_between}, post={idx_post_between}")
            return False
        
        return True
        
    except Exception as e:
        print(f"  ❌ UTC index functions test failed: {e}")
        return False


def test_data_validation():
    """Test data validation with known values"""
    print("\n✅ Testing data validation with known values...")
    
    try:
        data_loader = create_data_loader()
        
        # Load and process test data
        csv_path = "/Users/posao/Downloads/Test.csv"
        df = data_loader.load_csv_data(csv_path, delimiter=";")
        
        dat = {"test_file": df}
        inf = pd.DataFrame()
        
        updated_dat, updated_inf = data_loader.process_csv_data(dat, inf)
        
        # Get processed info
        info = updated_inf.loc["test_file"]
        processed_df = updated_dat["test_file"]
        
        print("  Validating calculated values:")
        
        # Check time span calculation
        time_span_hours = (info['utc_max'] - info['utc_min']).total_seconds() / 3600
        print(f"  - Time span: {time_span_hours:.2f} hours")
        
        # Check that min/max values are from the data
        actual_min = processed_df['Leistung [kW]'].min()
        actual_max = processed_df['Leistung [kW]'].max()
        
        if info['val_min'] == actual_min and info['val_max'] == actual_max:
            print(f"  ✅ Min/Max values correct: {actual_min} / {actual_max}")
        else:
            print(f"  ❌ Min/Max values incorrect: expected {actual_min}/{actual_max}, got {info['val_min']}/{info['val_max']}")
            return False
        
        # Validate time step calculation manually
        first_time = processed_df['UTC'].iloc[0]
        second_time = processed_df['UTC'].iloc[1]
        manual_delt = (second_time - first_time).total_seconds() / 60
        
        if abs(info['delt'] - manual_delt) < 0.01:
            print(f"  ✅ Time step calculation validated: {manual_delt} minutes")
        else:
            print(f"  ❌ Time step calculation error: calculated {info['delt']}, manual {manual_delt}")
            return False
        
        print("  ✅ All validation checks passed!")
        return True
        
    except Exception as e:
        print(f"  ❌ Data validation test failed: {e}")
        return False


def main():
    """Run all data loader tests"""
    print("🧪 Starting Data Loader Real Function Tests")
    print("=" * 60)
    
    tests = [
        ("CSV Loading", test_load_csv_data),
        ("CSV Data Processing", test_process_csv_data),
        ("UTC Index Functions", test_utc_index_functions),
        ("Data Validation", test_data_validation),
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
    print("🏁 DATA LOADER TEST SUMMARY")
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
        print("\n🎉 All data loader tests passed! Real functions are working correctly.")
    else:
        print(f"\n⚠️  {failed} test(s) failed. Please check the issues above.")
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)