#!/usr/bin/env python3
"""
Test script for visualization.py real functions
Tests the extracted visualization functions from training_backend_test_2.py
"""

import sys
import os
import numpy as np
import pandas as pd
import warnings

# Suppress matplotlib warnings for cleaner output
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from training_system.visualization import (
        Visualizer, create_visualizer
    )
    from training_system.config import PLOT_SETTINGS
    print("✅ Successfully imported all visualization functions")
except ImportError as e:
    print(f"❌ Failed to import visualization functions: {e}")
    sys.exit(1)


def create_test_data():
    """Create synthetic test data for visualization functions"""
    try:
        np.random.seed(42)  # For reproducible tests
        
        # Create synthetic combined arrays similar to what the original expects
        n_samples = 1000
        n_input_features = 3
        n_output_features = 2
        
        # Input combined array (similar to i_combined_array)
        i_combined_array = np.random.randn(n_samples, n_input_features) * 10 + 50
        
        # Output combined array (similar to o_combined_array)  
        o_combined_array = np.random.randn(n_samples, n_output_features) * 5 + 25
        
        # Create data arrays dict
        data_arrays = {
            'i_combined_array': i_combined_array,
            'o_combined_array': o_combined_array
        }
        
        print(f"  ✅ Created test data:")
        print(f"    - Input array: shape {i_combined_array.shape}")
        print(f"    - Output array: shape {o_combined_array.shape}")
        
        return data_arrays
        
    except Exception as e:
        print(f"  ❌ Failed to create test data: {e}")
        return None


def test_visualizer_creation():
    """Test creating visualizer instance"""
    print("\\n🏭 Testing Visualizer creation...")
    
    try:
        # Test factory function
        visualizer = create_visualizer()
        
        if visualizer is not None:
            print("  ✅ Visualizer created successfully")
        else:
            print("  ❌ Visualizer creation failed")
            return False
        
        # Test that it has expected attributes
        if hasattr(visualizer, 'plots') and hasattr(visualizer, 'setup_plot_style'):
            print("  ✅ Visualizer has expected attributes")
        else:
            print("  ❌ Visualizer missing expected attributes")
            return False
        
        # Test plot style setup
        try:
            visualizer.setup_plot_style()
            print("  ✅ Plot style setup completed")
        except Exception as e:
            print(f"  ⚠️  Plot style setup warning: {e}")
            # This is not critical for functionality
        
        return True
        
    except Exception as e:
        print(f"  ❌ Visualizer creation test failed: {e}")
        return False


def test_config_import():
    """Test PLOT_SETTINGS configuration"""
    print("\\n🔧 Testing PLOT_SETTINGS configuration...")
    
    try:
        # Test that PLOT_SETTINGS exists and has expected structure
        if isinstance(PLOT_SETTINGS, dict):
            print("  ✅ PLOT_SETTINGS is a dictionary")
        else:
            print(f"  ❌ PLOT_SETTINGS is not a dictionary: {type(PLOT_SETTINGS)}")
            return False
        
        # Test expected keys
        expected_keys = ['figure_size', 'dpi', 'font_size', 'color_palette', 'violin_plot']
        for key in expected_keys:
            if key in PLOT_SETTINGS:
                print(f"  ✅ Found key: {key}")
            else:
                print(f"  ❌ Missing key: {key}")
                return False
        
        # Test violin_plot settings
        violin_settings = PLOT_SETTINGS.get('violin_plot', {})
        if 'inner' in violin_settings and 'linewidth' in violin_settings:
            print("  ✅ Violin plot settings complete")
        else:
            print("  ❌ Violin plot settings incomplete")
            return False
        
        return True
        
    except Exception as e:
        print(f"  ❌ PLOT_SETTINGS configuration test failed: {e}")
        return False


def test_input_distribution_plot():
    """Test input data distribution plot creation"""
    print("\\n📊 Testing input distribution plot...")
    
    try:
        data_arrays = create_test_data()
        if data_arrays is None:
            return False
        
        visualizer = create_visualizer()
        
        # Test input distribution plot
        print("  Testing _create_input_distribution_plot...")
        input_plot = visualizer._create_input_distribution_plot(data_arrays['i_combined_array'])
        
        if isinstance(input_plot, str) and input_plot.startswith('data:image/png;base64,'):
            print("  ✅ Input distribution plot created successfully")
            print(f"    Plot data length: {len(input_plot)} characters")
        else:
            print(f"  ❌ Input distribution plot failed: {type(input_plot)}")
            return False
        
        # Test that plot contains expected base64 data
        if len(input_plot) > 1000:  # Should be substantial base64 data
            print("  ✅ Plot contains substantial data")
        else:
            print(f"  ❌ Plot data too small: {len(input_plot)} characters")
            return False
        
        return True
        
    except Exception as e:
        print(f"  ❌ Input distribution plot test failed: {e}")
        return False


def test_output_distribution_plot():
    """Test output data distribution plot creation"""
    print("\\n📈 Testing output distribution plot...")
    
    try:
        data_arrays = create_test_data()
        if data_arrays is None:
            return False
        
        visualizer = create_visualizer()
        
        # Test output distribution plot with input data for color offset
        print("  Testing _create_output_distribution_plot...")
        output_plot = visualizer._create_output_distribution_plot(
            data_arrays['o_combined_array'], 
            data_arrays['i_combined_array']
        )
        
        if isinstance(output_plot, str) and output_plot.startswith('data:image/png;base64,'):
            print("  ✅ Output distribution plot created successfully")
            print(f"    Plot data length: {len(output_plot)} characters")
        else:
            print(f"  ❌ Output distribution plot failed: {type(output_plot)}")
            return False
        
        # Test output plot without input data (no color offset)
        print("  Testing output plot without input data...")
        output_plot_no_input = visualizer._create_output_distribution_plot(
            data_arrays['o_combined_array']
        )
        
        if isinstance(output_plot_no_input, str) and output_plot_no_input.startswith('data:image/png;base64,'):
            print("  ✅ Output distribution plot (no input) created successfully")
        else:
            print(f"  ❌ Output distribution plot (no input) failed: {type(output_plot_no_input)}")
            return False
        
        return True
        
    except Exception as e:
        print(f"  ❌ Output distribution plot test failed: {e}")
        return False


def test_violin_plots_function():
    """Test main violin plots function"""
    print("\\n🎻 Testing create_violin_plots function...")
    
    try:
        data_arrays = create_test_data()
        if data_arrays is None:
            return False
        
        visualizer = create_visualizer()
        
        # Test violin plots creation
        print("  Testing create_violin_plots...")
        violin_plots = visualizer.create_violin_plots(data_arrays)
        
        if isinstance(violin_plots, dict):
            print(f"  ✅ Violin plots function returned dict with {len(violin_plots)} plots")
        else:
            print(f"  ❌ Violin plots function failed: {type(violin_plots)}")
            return False
        
        # Test expected plot keys
        expected_keys = ['input_distribution', 'output_distribution']
        for key in expected_keys:
            if key in violin_plots:
                print(f"    ✅ Found plot: {key}")
                
                # Validate plot data
                plot_data = violin_plots[key]
                if isinstance(plot_data, str) and plot_data.startswith('data:image/png;base64,'):
                    print(f"      ✅ Plot data valid ({len(plot_data)} chars)")
                else:
                    print(f"      ❌ Plot data invalid: {type(plot_data)}")
                    return False
            else:
                print(f"    ❌ Missing plot: {key}")
                return False
        
        # Test with partial data
        print("  Testing with only input data...")
        partial_data = {'i_combined_array': data_arrays['i_combined_array']}
        partial_plots = visualizer.create_violin_plots(partial_data)
        
        if 'input_distribution' in partial_plots and 'output_distribution' not in partial_plots:
            print("  ✅ Partial data handling works correctly")
        else:
            print("  ❌ Partial data handling failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"  ❌ Violin plots function test failed: {e}")
        return False


def test_figure_to_base64():
    """Test figure to base64 conversion"""
    print("\\n🔄 Testing figure to base64 conversion...")
    
    try:
        import matplotlib.pyplot as plt
        
        visualizer = create_visualizer()
        
        # Create a simple test figure
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot([1, 2, 3, 4], [1, 4, 2, 3])
        ax.set_title("Test Plot")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        
        # Test conversion
        print("  Testing _figure_to_base64...")
        base64_data = visualizer._figure_to_base64(fig)
        
        plt.close(fig)  # Clean up
        
        if isinstance(base64_data, str) and base64_data.startswith('data:image/png;base64,'):
            print("  ✅ Figure to base64 conversion successful")
            print(f"    Base64 data length: {len(base64_data)} characters")
        else:
            print(f"  ❌ Figure to base64 conversion failed: {type(base64_data)}")
            return False
        
        # Test that base64 data is substantial
        if len(base64_data) > 1000:
            print("  ✅ Base64 data contains substantial content")
        else:
            print(f"  ❌ Base64 data too small: {len(base64_data)} characters")
            return False
        
        return True
        
    except Exception as e:
        print(f"  ❌ Figure to base64 test failed: {e}")
        return False


def test_edge_cases():
    """Test edge cases and error handling"""
    print("\\n🧪 Testing edge cases...")
    
    try:
        visualizer = create_visualizer()
        
        # Test with empty arrays
        print("  Testing with empty arrays...")
        try:
            empty_array = np.array([]).reshape(0, 2)
            result = visualizer._create_input_distribution_plot(empty_array)
            print("  ⚠️  Empty array test - this should handle gracefully")
        except Exception as e:
            print(f"  ✅ Empty array correctly raises exception: {type(e).__name__}")
        
        # Test with single feature
        print("  Testing with single feature...")
        single_feature = np.random.randn(100, 1)
        single_plot = visualizer._create_input_distribution_plot(single_feature)
        
        if isinstance(single_plot, str) and single_plot.startswith('data:image/png;base64,'):
            print("  ✅ Single feature plot created successfully")
        else:
            print(f"  ❌ Single feature plot failed: {type(single_plot)}")
            return False
        
        # Test with many features
        print("  Testing with many features...")
        many_features = np.random.randn(50, 10)
        many_plot = visualizer._create_input_distribution_plot(many_features)
        
        if isinstance(many_plot, str) and many_plot.startswith('data:image/png;base64,'):
            print("  ✅ Many features plot created successfully")
        else:
            print(f"  ❌ Many features plot failed: {type(many_plot)}")
            return False
        
        return True
        
    except Exception as e:
        print(f"  ❌ Edge cases test failed: {e}")
        return False


def main():
    """Run all visualization tests"""
    print("🧪 Starting Visualization Real Functions Tests")
    print("=" * 60)
    
    tests = [
        ("Visualizer Creation", test_visualizer_creation),
        ("Configuration Import", test_config_import),
        ("Input Distribution Plot", test_input_distribution_plot),
        ("Output Distribution Plot", test_output_distribution_plot),
        ("Violin Plots Function", test_violin_plots_function),
        ("Figure to Base64", test_figure_to_base64),
        ("Edge Cases", test_edge_cases),
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
    print("🏁 VISUALIZATION TEST SUMMARY")
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
        print("\\n🎉 All visualization tests passed! Real visualization functions are working correctly.")
    else:
        print(f"\\n⚠️  {failed} test(s) failed. Please check the issues above.")
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)