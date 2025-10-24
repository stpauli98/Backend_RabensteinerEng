#!/usr/bin/env python3
"""
Test ALL training models to verify they work
Tests: Dense, CNN, LSTM, AR-LSTM, SVR_dir, SVR_MIMO, Linear
"""
import sys
import os
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from services.training.config import MDL, MTS
from services.training.model_trainer import (
    train_dense, train_cnn, train_lstm, train_ar_lstm,
    train_svr_dir, train_svr_mimo, train_linear_model
)

print("="*60)
print("TESTING ALL MODEL TYPES")
print("="*60)

mts = MTS()
n_samples = 20
n_timesteps = mts.I_N
n_features = 2

train_x = np.random.randn(n_samples, n_timesteps, n_features)
train_y = np.random.randn(n_samples, mts.O_N, 1)

val_x = np.random.randn(5, n_timesteps, n_features)
val_y = np.random.randn(5, mts.O_N, 1)

print(f"\nTest data shapes:")
print(f"Train X: {train_x.shape}, Train Y: {train_y.shape}")
print(f"Val X: {val_x.shape}, Val Y: {val_y.shape}")

results = {}

print("\n" + "-"*40)
print("1. Testing LINEAR Model...")
print("-"*40)
try:
    models = train_linear_model(train_x, train_y)
    if models and len(models) > 0:
        print(f"✅ LINEAR: Success! Trained {len(models)} models")
        results['LINEAR'] = 'SUCCESS'
    else:
        print("❌ LINEAR: Failed - No models returned")
        results['LINEAR'] = 'FAILED'
except Exception as e:
    print(f"❌ LINEAR: Error - {str(e)}")
    results['LINEAR'] = f'ERROR: {str(e)}'

print("\n" + "-"*40)
print("2. Testing DENSE Neural Network...")
print("-"*40)
try:
    mdl = MDL(mode="Dense")
    print(f"Config: LAY={mdl.LAY}, N={mdl.N}, EP={mdl.EP}, ACTF={mdl.ACTF}")
    
    mdl.EP = 1
    
    model = train_dense(train_x, train_y, val_x, val_y, mdl)
    if model is not None:
        print("✅ DENSE: Success! Model trained")
        pred = model.predict(val_x[:1])
        print(f"   Prediction shape: {pred.shape}")
        results['DENSE'] = 'SUCCESS'
    else:
        print("❌ DENSE: Failed - No model returned")
        results['DENSE'] = 'FAILED'
except Exception as e:
    print(f"❌ DENSE: Error - {str(e)}")
    results['DENSE'] = f'ERROR: {str(e)}'

print("\n" + "-"*40)
print("3. Testing CNN Model...")
print("-"*40)
try:
    mdl = MDL(mode="CNN")
    print(f"Config: LAY={mdl.LAY}, N={mdl.N}, K={mdl.K}, EP={mdl.EP}, ACTF={mdl.ACTF}")
    
    mdl.EP = 1
    
    model = train_cnn(train_x, train_y, val_x, val_y, mdl)
    if model is not None:
        test_input = val_x[:1].reshape(1, val_x.shape[1], val_x.shape[2], 1)
        pred = model.predict(test_input)
        print("✅ CNN: Success! Model trained")
        print(f"   Prediction shape: {pred.shape}")
        results['CNN'] = 'SUCCESS'
    else:
        print("❌ CNN: Failed - No model returned")
        results['CNN'] = 'FAILED'
except Exception as e:
    print(f"❌ CNN: Error - {str(e)}")
    results['CNN'] = f'ERROR: {str(e)}'

print("\n" + "-"*40)
print("4. Testing LSTM Model...")
print("-"*40)
try:
    mdl = MDL(mode="LSTM")
    print(f"Config: LAY={mdl.LAY}, N={mdl.N}, EP={mdl.EP}, ACTF={mdl.ACTF}")
    
    mdl.EP = 1
    
    model = train_lstm(train_x, train_y, val_x, val_y, mdl)
    if model is not None:
        pred = model.predict(val_x[:1])
        print("✅ LSTM: Success! Model trained")
        print(f"   Prediction shape: {pred.shape}")
        results['LSTM'] = 'SUCCESS'
    else:
        print("❌ LSTM: Failed - No model returned")
        results['LSTM'] = 'FAILED'
except Exception as e:
    print(f"❌ LSTM: Error - {str(e)}")
    results['LSTM'] = f'ERROR: {str(e)}'

print("\n" + "-"*40)
print("5. Testing AR-LSTM Model...")
print("-"*40)
try:
    mdl = MDL(mode="AR LSTM")
    print(f"Config: LAY={mdl.LAY}, N={mdl.N}, EP={mdl.EP}, ACTF={mdl.ACTF}")
    
    mdl.EP = 1
    
    model = train_ar_lstm(train_x, train_y, val_x, val_y, mdl)
    if model is not None:
        pred = model.predict(val_x[:1])
        print("✅ AR-LSTM: Success! Model trained")
        print(f"   Prediction shape: {pred.shape}")
        results['AR-LSTM'] = 'SUCCESS'
    else:
        print("❌ AR-LSTM: Failed - No model returned")
        results['AR-LSTM'] = 'FAILED'
except Exception as e:
    print(f"❌ AR-LSTM: Error - {str(e)}")
    results['AR-LSTM'] = f'ERROR: {str(e)}'

print("\n" + "-"*40)
print("6. Testing SVR_dir Model...")
print("-"*40)
try:
    mdl = MDL(mode="SVR_dir")
    print(f"Config: C={mdl.C}, EPSILON={mdl.EPSILON}, KERNEL={mdl.KERNEL}")
    
    models = train_svr_dir(train_x, train_y, mdl)
    if models and len(models) > 0:
        print(f"✅ SVR_dir: Success! Trained {len(models)} SVR models")
        X_test = val_x[:1].reshape(1 * n_timesteps, n_features)
        pred = models[0].predict(X_test)
        print(f"   Prediction shape: {pred.shape}")
        results['SVR_dir'] = 'SUCCESS'
    else:
        print("❌ SVR_dir: Failed - No models returned")
        results['SVR_dir'] = 'FAILED'
except Exception as e:
    print(f"❌ SVR_dir: Error - {str(e)}")
    results['SVR_dir'] = f'ERROR: {str(e)}'

print("\n" + "-"*40)
print("7. Testing SVR_MIMO Model...")
print("-"*40)
try:
    mdl = MDL(mode="SVR_MIMO")
    print(f"Config: C={mdl.C}, EPSILON={mdl.EPSILON}, KERNEL={mdl.KERNEL}")
    
    models = train_svr_mimo(train_x, train_y, mdl)
    if models is not None and len(models) > 0:
        X_test = val_x[:1].reshape(1, -1)
        predictions = [m.predict(X_test) for m in models]
        print(f"✅ SVR_MIMO: Success! Trained {len(models)} SVR models")
        print(f"   First prediction shape: {predictions[0].shape}")
        results['SVR_MIMO'] = 'SUCCESS'
    else:
        print("❌ SVR_MIMO: Failed - No model returned")
        results['SVR_MIMO'] = 'FAILED'
except Exception as e:
    print(f"❌ SVR_MIMO: Error - {str(e)}")
    results['SVR_MIMO'] = f'ERROR: {str(e)}'

print("\n" + "="*60)
print("TEST SUMMARY")
print("="*60)

success_count = 0
fail_count = 0

for model_name, status in results.items():
    if status == 'SUCCESS':
        print(f"✅ {model_name:10} - SUCCESS")
        success_count += 1
    else:
        print(f"❌ {model_name:10} - {status}")
        fail_count += 1

print("\n" + "-"*40)
print(f"Total: {success_count} SUCCESS, {fail_count} FAILED")
print("-"*40)

if fail_count == 0:
    print("\n🎉 ALL MODELS WORK! 🎉")
else:
    print(f"\n⚠️  {fail_count} models have issues")

print("\n" + "="*60)
print("TESTING ACTIVATION FUNCTIONS")
print("="*60)

from services.training.model_trainer import ACTIVATION_FUNCTIONS

print("\nSupported activation functions:")
for key, value in ACTIVATION_FUNCTIONS.items():
    print(f"  '{key}' → '{value}'")

print("\n" + "-"*40)
print("Testing Dense with ReLU activation...")
print("-"*40)
try:
    mdl = MDL(mode="Dense")
    mdl.ACTF = "ReLU"
    mdl.EP = 1
    
    model = train_dense(train_x, train_y, val_x, val_y, mdl)
    if model:
        print("✅ ReLU activation works!")
    else:
        print("❌ ReLU activation failed")
except Exception as e:
    print(f"❌ ReLU test error: {e}")

print("\n" + "="*60)
