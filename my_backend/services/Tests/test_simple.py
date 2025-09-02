#!/usr/bin/env python3
"""
Simple test to verify backend works without frontend
"""
import sys
import os
# Add parent directories to path to access training modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

print("Testing backend modules...")

# Test 1: Import modules
print("\n1. Testing imports...")
try:
    from services.training.config import MDL, MTS, T
    from services.training.data_loader import load, transf
    from services.training.model_trainer import train_linear_model
    print("✓ All imports successful")
except Exception as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Create config objects
print("\n2. Testing config objects...")
try:
    mts = MTS()
    mdl = MDL(mode="LIN")
    print(f"✓ MTS: I_N={mts.I_N}, O_N={mts.O_N}")
    print(f"✓ MDL: MODE={mdl.MODE}")
except Exception as e:
    print(f"✗ Config failed: {e}")
    sys.exit(1)

# Test 3: Test with simple numpy arrays
print("\n3. Testing linear model training...")
try:
    import numpy as np
    
    # Create simple test data
    n_samples = 10
    n_timesteps = mts.I_N
    n_features = 2
    
    train_x = np.random.randn(n_samples, n_timesteps, n_features)
    train_y = np.random.randn(n_samples, mts.O_N, 1)
    
    print(f"✓ Train X shape: {train_x.shape}")
    print(f"✓ Train Y shape: {train_y.shape}")
    
    # Train simple linear model
    models = train_linear_model(train_x, train_y)
    
    if models is not None and len(models) > 0:
        print(f"✓ Linear model trained successfully ({len(models)} models)")
        
        # Make prediction with first model
        X_test = train_x[:1].reshape(1 * n_timesteps, n_features)
        pred = models[0].predict(X_test)
        print(f"✓ Prediction shape: {pred.shape}")
        print(f"✓ First prediction value: {pred[0]:.4f}")
    else:
        print("✗ Model training returned None")
        
except Exception as e:
    print(f"✗ Model training failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*50)
print("✓✓✓ BACKEND WORKS WITHOUT FRONTEND! ✓✓✓")
print("="*50)