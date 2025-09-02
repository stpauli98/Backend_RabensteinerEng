#!/usr/bin/env python3
"""
Standalone test script for backend training system
Tests the complete pipeline without frontend
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import logging

# Add parent directories to path to access training modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import training modules
from services.training.data_loader import load, transf
from services.training.pipeline_exact import run_exact_training_pipeline
from services.training.config import MDL, MTS, T

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_backend_training():
    """
    Test the backend training pipeline with test data
    """
    print("\n" + "="*60)
    print("BACKEND TRAINING TEST - BEZ FRONTENDA")
    print("="*60 + "\n")
    
    try:
        # Step 1: Load test data
        print("1. UČITAVANJE TEST PODATAKA...")
        print("-" * 40)
        
        # Create data dictionaries
        i_dat = {}
        o_dat = {}
        
        # Load input CSV
        input_path = os.path.join(os.path.dirname(__file__), "data", "input_test.csv")
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Test input file not found: {input_path}")
            
        df_input = pd.read_csv(input_path, delimiter=';')
        df_input.columns = ['UTC', 'value']  # Ensure correct column names
        i_dat['input_test'] = df_input
        print(f"✓ Učitan input fajl: {input_path}")
        print(f"  Shape: {df_input.shape}, Columns: {list(df_input.columns)}")
        
        # Load output CSV
        output_path = os.path.join(os.path.dirname(__file__), "data", "output_test.csv")
        if not os.path.exists(output_path):
            raise FileNotFoundError(f"Test output file not found: {output_path}")
            
        df_output = pd.read_csv(output_path, delimiter=';')
        df_output.columns = ['UTC', 'value']  # Ensure correct column names
        o_dat['output_test'] = df_output
        print(f"✓ Učitan output fajl: {output_path}")
        print(f"  Shape: {df_output.shape}, Columns: {list(df_output.columns)}")
        
        # Step 2: Create info DataFrames using load() function
        print("\n2. KREIRANJE INFO DATAFRAMES...")
        print("-" * 40)
        
        i_dat_inf = pd.DataFrame()
        o_dat_inf = pd.DataFrame()
        
        # Process input data
        i_dat, i_dat_inf = load(i_dat, i_dat_inf)
        print(f"✓ Procesiran input data")
        print(f"  i_dat_inf shape: {i_dat_inf.shape}")
        print(f"  UTC range: {i_dat_inf['utc_min'].iloc[0]} to {i_dat_inf['utc_max'].iloc[0]}")
        
        # Process output data
        o_dat, o_dat_inf = load(o_dat, o_dat_inf)
        print(f"✓ Procesiran output data")
        print(f"  o_dat_inf shape: {o_dat_inf.shape}")
        
        # Step 3: Apply transformations using transf()
        print("\n3. PRIMENA TRANSFORMACIJA...")
        print("-" * 40)
        
        # Use default MTS configuration
        mts = MTS()
        print(f"MTS Config: I_N={mts.I_N}, O_N={mts.O_N}, DELT={mts.DELT}, OFST={mts.OFST}")
        
        # Add required columns for transf function
        i_dat_inf['th_strt'] = -2  # Default from original
        i_dat_inf['th_end'] = 0    # Default from original
        o_dat_inf['th_strt'] = -2
        o_dat_inf['th_end'] = 0
        
        i_dat_inf = transf(i_dat_inf, mts.I_N, mts.OFST)
        o_dat_inf = transf(o_dat_inf, mts.O_N, mts.OFST)
        print("✓ Transformacije primenjene")
        
        # Add more required columns
        i_dat_inf['spec'] = 'Historische Daten'
        i_dat_inf['meth'] = 'Lineare Interpolation'
        i_dat_inf['avg'] = False
        i_dat_inf['scal'] = True
        i_dat_inf['scal_min'] = 0
        i_dat_inf['scal_max'] = 1
        
        o_dat_inf['spec'] = 'Historische Daten'
        o_dat_inf['meth'] = 'Lineare Interpolation'
        o_dat_inf['avg'] = False
        o_dat_inf['scal'] = True
        o_dat_inf['scal_min'] = 0
        o_dat_inf['scal_max'] = 1
        
        # Step 4: Determine time boundaries
        print("\n4. ODREĐIVANJE VREMENSKIH GRANICA...")
        print("-" * 40)
        
        utc_strt = i_dat_inf['utc_min'].min()
        utc_end = i_dat_inf['utc_max'].max()
        
        if not o_dat_inf.empty:
            utc_strt = max(utc_strt, o_dat_inf['utc_min'].min())
            utc_end = min(utc_end, o_dat_inf['utc_max'].max())
        
        print(f"✓ UTC Start: {utc_strt}")
        print(f"✓ UTC End: {utc_end}")
        
        # Step 5: Test with different model configurations
        print("\n5. TESTIRANJE MODELA...")
        print("-" * 40)
        
        # Test Linear model (simplest and fastest)
        print("\nTestiranje LINEAR modela...")
        mdl_config = MDL(mode="LIN")
        
        try:
            result = run_exact_training_pipeline(
                i_dat=i_dat,
                o_dat=o_dat,
                i_dat_inf=i_dat_inf,
                o_dat_inf=o_dat_inf,
                utc_strt=utc_strt,
                utc_end=utc_end,
                random_dat=False,
                mdl_config=mdl_config
            )
            
            print("\n" + "="*60)
            print("✓✓✓ TEST USPEŠAN! ✓✓✓")
            print("="*60)
            
            # Display results
            print("\n6. REZULTATI:")
            print("-" * 40)
            print(f"✓ Model treniran: {result['trained_model'] is not None}")
            print(f"✓ Training data shape: {result['train_data']['X'].shape}")
            print(f"✓ Validation data shape: {result['val_data']['X'].shape}")
            print(f"✓ Test data shape: {result['test_data']['X'].shape}")
            print(f"✓ Number of input scalers: {len([s for s in result['scalers']['input'].values() if s is not None])}")
            print(f"✓ Number of output scalers: {len([s for s in result['scalers']['output'].values() if s is not None])}")
            
            # Metadata
            meta = result['metadata']
            print(f"\nMetadata:")
            print(f"  - Total datasets: {meta['n_dat']}")
            print(f"  - Training sets: {meta['n_train']}")
            print(f"  - Validation sets: {meta['n_val']}")
            print(f"  - Test sets: {meta['n_test']}")
            print(f"  - Model type: {meta['model_config'].MODE}")
            
            # Make a test prediction (if model supports it)
            if hasattr(result['trained_model'], 'predict'):
                test_input = result['test_data']['X'][:1]  # First test sample
                prediction = result['trained_model'].predict(test_input)
                print(f"\nTest predikcija:")
                print(f"  - Input shape: {test_input.shape}")
                print(f"  - Prediction shape: {prediction.shape}")
                print(f"  - First predicted value: {prediction[0, 0, 0]:.4f}")
            
            return True
            
        except Exception as e:
            print(f"\n✗ GREŠKA pri treniranju: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
            
    except Exception as e:
        print(f"\n✗ GREŠKA: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_backend_training()
    
    if success:
        print("\n" + "="*60)
        print("BACKEND RADI ISPRAVNO BEZ FRONTENDA! ✓")
        print("="*60 + "\n")
        sys.exit(0)
    else:
        print("\n" + "="*60)
        print("TEST NEUSPEŠAN! ✗")
        print("="*60 + "\n")
        sys.exit(1)