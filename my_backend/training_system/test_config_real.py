#!/usr/bin/env python3
"""
Test script for config.py real classes
Tests the extracted MTS, T, MDL classes and HOL dictionary from training_backend_test_2.py
"""

import sys
import os
import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from training_system.config import MTS, T, MDL, HOL
    print("✅ Successfully imported MTS, T, MDL, HOL")
except ImportError as e:
    print(f"❌ Failed to import config classes: {e}")
    sys.exit(1)


def test_mts_class():
    """Test MTS class configuration"""
    print("\n🔧 Testing MTS class...")
    
    try:
        # Test basic attributes
        print(f"  MTS.I_N: {MTS.I_N}")
        print(f"  MTS.O_N: {MTS.O_N}")
        print(f"  MTS.DELT: {MTS.DELT}")
        print(f"  MTS.OFST: {MTS.OFST}")
        
        # Validate expected values from original
        if MTS.I_N == 13:
            print("  ✅ I_N value correct (13)")
        else:
            print(f"  ❌ I_N value incorrect: expected 13, got {MTS.I_N}")
            return False
            
        if MTS.O_N == 13:
            print("  ✅ O_N value correct (13)")
        else:
            print(f"  ❌ O_N value incorrect: expected 13, got {MTS.O_N}")
            return False
            
        if MTS.DELT == 3:
            print("  ✅ DELT value correct (3)")
        else:
            print(f"  ❌ DELT value incorrect: expected 3, got {MTS.DELT}")
            return False
            
        if MTS.OFST == 0:
            print("  ✅ OFST value correct (0)")
        else:
            print(f"  ❌ OFST value incorrect: expected 0, got {MTS.OFST}")
            return False
        
        return True
        
    except Exception as e:
        print(f"  ❌ MTS class test failed: {e}")
        return False


def test_t_class():
    """Test T class time features configuration"""
    print("\n🕐 Testing T class...")
    
    try:
        # Test nested classes exist
        subclasses = ['Y', 'M', 'W', 'D', 'H']
        for subclass_name in subclasses:
            if hasattr(T, subclass_name):
                print(f"  ✅ T.{subclass_name} class exists")
            else:
                print(f"  ❌ T.{subclass_name} class missing")
                return False
        
        # Test Y (Year) class attributes
        print("  Testing T.Y (Year) class...")
        if hasattr(T.Y, 'IMP') and T.Y.IMP == False:
            print("    ✅ T.Y.IMP = False")
        else:
            print(f"    ❌ T.Y.IMP incorrect: {getattr(T.Y, 'IMP', 'missing')}")
            return False
            
        if hasattr(T.Y, 'TH_STRT') and T.Y.TH_STRT == -24:
            print("    ✅ T.Y.TH_STRT = -24")
        else:
            print(f"    ❌ T.Y.TH_STRT incorrect: {getattr(T.Y, 'TH_STRT', 'missing')}")
            return False
            
        if hasattr(T.Y, 'TH_END') and T.Y.TH_END == 0:
            print("    ✅ T.Y.TH_END = 0")
        else:
            print(f"    ❌ T.Y.TH_END incorrect: {getattr(T.Y, 'TH_END', 'missing')}")
            return False
        
        # Test calculated DELT value
        expected_delt_y = (T.Y.TH_END - T.Y.TH_STRT) * 60 / (MTS.I_N - 1)
        if hasattr(T.Y, 'DELT') and abs(T.Y.DELT - expected_delt_y) < 0.001:
            print(f"    ✅ T.Y.DELT calculated correctly: {T.Y.DELT}")
        else:
            print(f"    ❌ T.Y.DELT calculation incorrect: expected {expected_delt_y}, got {getattr(T.Y, 'DELT', 'missing')}")
            return False
        
        # Test H (Holiday) class specific attributes
        print("  Testing T.H (Holiday) class...")
        if hasattr(T.H, 'CNTRY') and T.H.CNTRY == "Österreich":
            print("    ✅ T.H.CNTRY = 'Österreich'")
        else:
            print(f"    ❌ T.H.CNTRY incorrect: {getattr(T.H, 'CNTRY', 'missing')}")
            return False
        
        # Test timezone
        if hasattr(T, 'TZ') and T.TZ == "Europe/Vienna":
            print("  ✅ T.TZ = 'Europe/Vienna'")
        else:
            print(f"  ❌ T.TZ incorrect: {getattr(T, 'TZ', 'missing')}")
            return False
        
        return True
        
    except Exception as e:
        print(f"  ❌ T class test failed: {e}")
        return False


def test_mdl_class():
    """Test MDL class model configuration"""
    print("\n🤖 Testing MDL class...")
    
    try:
        # Test MODE attribute
        if hasattr(MDL, 'MODE'):
            print(f"  ✅ MDL.MODE = '{MDL.MODE}'")
            
            # In original, MODE = "LIN" so we should see that
            if MDL.MODE == "LIN":
                print("  ✅ Default MODE is 'LIN' as expected")
            else:
                print(f"  ⚠️  MODE is '{MDL.MODE}', expected 'LIN' from original")
        else:
            print("  ❌ MDL.MODE attribute missing")
            return False
        
        # Test conditional attributes based on MODE
        # Since MODE = "LIN", we should have the 'a' attribute
        if MDL.MODE == "LIN":
            if hasattr(MDL, 'a') and MDL.a == 5:
                print("  ✅ MDL.a = 5 for LIN mode")
            else:
                print(f"  ❌ MDL.a incorrect for LIN mode: {getattr(MDL, 'a', 'missing')}")
                return False
        
        # Test that other mode attributes are not set (since they're in if/elif blocks)
        dense_attrs = ['LAY', 'N', 'EP', 'ACTF']
        svr_attrs = ['KERNEL', 'C', 'EPSILON']
        
        # These should not exist when MODE != their respective mode
        mode_specific_found = []
        for attr in dense_attrs + svr_attrs:
            if hasattr(MDL, attr):
                mode_specific_found.append(attr)
        
        if len(mode_specific_found) == 0:
            print("  ✅ No mode-specific attributes found (correct for LIN mode)")
        else:
            print(f"  ⚠️  Found mode-specific attributes: {mode_specific_found}")
            # This might be correct depending on how Python handles the class definition
        
        return True
        
    except Exception as e:
        print(f"  ❌ MDL class test failed: {e}")
        return False


def test_hol_dictionary():
    """Test HOL holiday dictionary"""
    print("\n🎄 Testing HOL dictionary...")
    
    try:
        # Test that HOL is a dictionary
        if isinstance(HOL, dict):
            print("  ✅ HOL is a dictionary")
        else:
            print(f"  ❌ HOL is not a dictionary: {type(HOL)}")
            return False
        
        # Test expected countries
        expected_countries = ["Österreich", "Deutschland", "Schweiz"]
        for country in expected_countries:
            if country in HOL:
                print(f"  ✅ Country '{country}' found in HOL")
            else:
                print(f"  ❌ Country '{country}' missing from HOL")
                return False
        
        # Test that values are lists of datetime objects
        for country, holidays in HOL.items():
            if isinstance(holidays, list):
                print(f"  ✅ HOL['{country}'] is a list")
            else:
                print(f"  ❌ HOL['{country}'] is not a list: {type(holidays)}")
                return False
            
            # Check first few holidays are datetime objects (if any exist)
            if len(holidays) > 0:
                if isinstance(holidays[0], datetime.datetime):
                    print(f"  ✅ HOL['{country}'] contains datetime objects")
                else:
                    print(f"  ❌ HOL['{country}'] contains {type(holidays[0])}, expected datetime")
                    return False
            else:
                print(f"  ⚠️  HOL['{country}'] is empty (may be expected for {country})")
        
        # Test specific holidays for Austria
        austria_holidays = HOL.get("Österreich", [])
        if len(austria_holidays) > 0:
            print(f"  ✅ Austria has {len(austria_holidays)} holidays")
            
            # Check for specific known holidays
            holiday_dates = [h.strftime("%Y-%m-%d") for h in austria_holidays]
            
            # Check for New Year's Day 2022
            if "2022-01-01" in holiday_dates:
                print("  ✅ Found New Year's Day 2022")
            else:
                print("  ❌ New Year's Day 2022 not found")
                return False
            
            # Check for National Day 2024
            if "2024-10-26" in holiday_dates:
                print("  ✅ Found National Day 2024")
            else:
                print("  ❌ National Day 2024 not found")
                return False
        else:
            print("  ❌ Austria has no holidays")
            return False
        
        return True
        
    except Exception as e:
        print(f"  ❌ HOL dictionary test failed: {e}")
        return False


def test_cross_references():
    """Test cross-references between classes work correctly"""
    print("\n🔗 Testing cross-references between classes...")
    
    try:
        # Test that T class DELT calculations use MTS.I_N
        print("  Testing T class DELT calculations use MTS.I_N...")
        
        # Manual calculation for T.Y.DELT
        expected_y_delt = (T.Y.TH_END - T.Y.TH_STRT) * 60 / (MTS.I_N - 1)
        if abs(T.Y.DELT - expected_y_delt) < 0.001:
            print(f"  ✅ T.Y.DELT uses MTS.I_N correctly: {T.Y.DELT}")
        else:
            print(f"  ❌ T.Y.DELT calculation wrong: expected {expected_y_delt}, got {T.Y.DELT}")
            return False
        
        # Manual calculation for T.M.DELT
        expected_m_delt = (T.M.TH_END - T.M.TH_STRT) * 60 / (MTS.I_N - 1)
        if abs(T.M.DELT - expected_m_delt) < 0.001:
            print(f"  ✅ T.M.DELT uses MTS.I_N correctly: {T.M.DELT}")
        else:
            print(f"  ❌ T.M.DELT calculation wrong: expected {expected_m_delt}, got {T.M.DELT}")
            return False
        
        # Test that different time features have different DELT values
        time_features = [T.Y, T.M, T.W, T.D, T.H]
        delt_values = [feature.DELT for feature in time_features]
        
        print(f"  DELT values: Y={T.Y.DELT}, M={T.M.DELT}, W={T.W.DELT}, D={T.D.DELT}, H={T.H.DELT}")
        
        # Check that they're different where expected
        if T.Y.DELT != T.M.DELT:  # Different time horizons
            print("  ✅ T.Y.DELT != T.M.DELT (different time horizons)")
        else:
            print("  ❌ T.Y.DELT == T.M.DELT (should be different)")
            return False
        
        return True
        
    except Exception as e:
        print(f"  ❌ Cross-references test failed: {e}")
        return False


def test_configuration_values():
    """Test that configuration values match original system"""
    print("\n📊 Testing configuration values match original...")
    
    try:
        # Test key values that should match original exactly
        config_checks = [
            ("MTS.I_N", MTS.I_N, 13),
            ("MTS.O_N", MTS.O_N, 13),
            ("MTS.DELT", MTS.DELT, 3),
            ("MTS.OFST", MTS.OFST, 0),
            ("T.Y.TH_STRT", T.Y.TH_STRT, -24),
            ("T.Y.TH_END", T.Y.TH_END, 0),
            ("T.M.TH_STRT", T.M.TH_STRT, -1),
            ("T.H.TH_STRT", T.H.TH_STRT, -100),
            ("T.H.CNTRY", T.H.CNTRY, "Österreich"),
            ("T.TZ", T.TZ, "Europe/Vienna"),
            ("MDL.MODE", MDL.MODE, "LIN"),
        ]
        
        all_correct = True
        for name, actual, expected in config_checks:
            if actual == expected:
                print(f"  ✅ {name} = {actual}")
            else:
                print(f"  ❌ {name} = {actual}, expected {expected}")
                all_correct = False
        
        return all_correct
        
    except Exception as e:
        print(f"  ❌ Configuration values test failed: {e}")
        return False


def main():
    """Run all config tests"""
    print("🧪 Starting Config Real Classes Tests")
    print("=" * 60)
    
    tests = [
        ("MTS Class", test_mts_class),
        ("T Class", test_t_class),
        ("MDL Class", test_mdl_class),
        ("HOL Dictionary", test_hol_dictionary),
        ("Cross References", test_cross_references),
        ("Configuration Values", test_configuration_values),
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
    print("🏁 CONFIG TEST SUMMARY")
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
        print("\n🎉 All config tests passed! Real configuration classes are working correctly.")
    else:
        print(f"\n⚠️  {failed} test(s) failed. Please check the issues above.")
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)