"""Test that all models return the same output structure for EvaluationTables.tsx"""

import numpy as np
import pandas as pd
from datetime import datetime
from services.training.config import MDL, MTS
from services.training.pipeline_exact import run_exact_training_pipeline

# Create dummy test data
np.random.seed(42)
n_samples = 50
n_timesteps = 13
n_features_in = 3
n_features_out = 2

# Create test data structures with UTC column as expected by the pipeline
i_dat = {
    'TestInput': pd.DataFrame({
        'UTC': pd.date_range('2024-01-01', periods=1000, freq='1h'),
        'Value': np.random.randn(1000)
    })
}

o_dat = {
    'TestOutput': pd.DataFrame({
        'UTC': pd.date_range('2024-01-01', periods=1000, freq='1h'),
        'Value': np.random.randn(1000)
    })
}

# Create info DataFrames with all required columns including delt_transf
i_dat_inf = pd.DataFrame({
    'file': ['TestInput'],
    'utc_min': [pd.Timestamp('2024-01-01')],
    'utc_max': [pd.Timestamp('2024-02-01')],
    'min': [0],
    'max': [100],
    'spec': ['Historische Daten'],
    'th_strt': [-1],
    'th_end': [0],
    'meth': ['Lineare Interpolation'],
    'avg': [False],
    'scal': [True],
    'scal_max': [1],
    'scal_min': [0],
    'delt': [60],
    'delt_transf': [60.0],  # Added required column
    'ofst_transf': [0.0]     # Added required column
}).set_index('file')

o_dat_inf = pd.DataFrame({
    'file': ['TestOutput'],
    'utc_min': [pd.Timestamp('2024-01-01')],
    'utc_max': [pd.Timestamp('2024-02-01')],
    'min': [0],
    'max': [100],
    'spec': ['Historische Daten'],
    'th_strt': [0],
    'th_end': [1],
    'meth': ['Lineare Interpolation'],
    'avg': [False],
    'scal': [True],
    'scal_max': [1],
    'scal_min': [0],
    'delt': [60],
    'delt_transf': [60.0],  # Added required column
    'ofst_transf': [0.0]     # Added required column  
}).set_index('file')

utc_strt = pd.Timestamp('2024-01-02')
utc_end = pd.Timestamp('2024-01-20')

print("=" * 80)
print("TESTING OUTPUT STRUCTURE FOR ALL MODELS")
print("=" * 80)

# Test all model types
model_types = ['LIN', 'Dense', 'CNN', 'LSTM', 'AR LSTM', 'SVR_dir', 'SVR_MIMO']
results = {}

for model_type in model_types:
    print(f"\nTesting {model_type} model...")
    
    try:
        # Create model config
        mdl_config = MDL(mode=model_type)
        
        # Set minimal parameters for faster testing
        if model_type in ['Dense', 'CNN', 'LSTM', 'AR LSTM']:
            mdl_config.LAY = 1
            mdl_config.N = 8
            mdl_config.EP = 1
            mdl_config.ACTF = 'ReLU'
            if model_type == 'CNN':
                mdl_config.K = 3
        elif model_type in ['SVR_dir', 'SVR_MIMO']:
            mdl_config.KERNEL = 'linear'
            mdl_config.C = 1.0
            mdl_config.EPSILON = 0.1
        
        # Run pipeline (with minimal data for speed)
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
        
        results[model_type] = result
        print(f"✅ {model_type} completed")
        
    except Exception as e:
        print(f"❌ {model_type} failed: {str(e)[:100]}")
        results[model_type] = None

print("\n" + "=" * 80)
print("CHECKING OUTPUT STRUCTURE CONSISTENCY")
print("=" * 80)

# Check that all models return the same structure
required_keys = [
    'trained_model',
    'train_data', 
    'val_data',
    'test_data',
    'scalers',
    'metadata',
    'evaluation_metrics'
]

eval_metrics_keys = [
    'test_metrics_scaled',
    'test_metrics_original',
    'val_metrics_scaled',
    'model_type'
]

all_consistent = True

for model_type, result in results.items():
    if result is None:
        print(f"\n❌ {model_type}: Failed to run")
        all_consistent = False
        continue
        
    print(f"\n{model_type}:")
    
    # Check main structure
    for key in required_keys:
        if key in result:
            print(f"  ✅ Has '{key}'")
        else:
            print(f"  ❌ Missing '{key}'")
            all_consistent = False
    
    # Check evaluation_metrics structure
    if 'evaluation_metrics' in result:
        eval_metrics = result['evaluation_metrics']
        for key in eval_metrics_keys:
            if key in eval_metrics:
                print(f"    ✅ evaluation_metrics has '{key}'")
            else:
                print(f"    ❌ evaluation_metrics missing '{key}'")
                all_consistent = False
        
        # Check model_type matches
        if eval_metrics.get('model_type') == model_type:
            print(f"    ✅ model_type correctly set to '{model_type}'")
        else:
            print(f"    ❌ model_type mismatch: expected '{model_type}', got '{eval_metrics.get('model_type')}'")
            all_consistent = False

print("\n" + "=" * 80)
print("WHAT EVALUATIONTABLES.TSX EXPECTS:")
print("=" * 80)

print("""
EvaluationTables.tsx expects from API endpoint /api/training/evaluation-tables/<session_id>:
{
    success: boolean,
    session_id: string,
    df_eval?: Record<string, any>,    // Evaluation metrics table
    df_eval_ts?: Record<string, any>, // Time series evaluation table
    model_type?: string,               // Model type used
    error?: string
}

The backend endpoint (training.py:1870-2000) formats the data from evaluation_metrics into:
- df_eval: Dictionary with output features as keys, containing metrics
- df_eval_ts: Dictionary with output features as keys, containing time series metrics
""")

print("\n" + "=" * 80)
print("VERIFICATION RESULT:")
print("=" * 80)

if all_consistent:
    print("✅ ALL MODELS RETURN CONSISTENT STRUCTURE!")
    print("\nAll models provide:")
    print("1. evaluation_metrics with test/val metrics")
    print("2. model_type field identifying the model")
    print("3. Same data structure that can be formatted into df_eval/df_eval_ts")
    print("\n✅ SVI MODELI VRAĆAJU ISTI FORMAT PODATAKA KOJI EVALUATIONTABLES.TSX MOŽE PRIKAZATI!")
else:
    print("❌ Some models have inconsistent output structure")

print("=" * 80)