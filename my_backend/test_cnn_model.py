#!/usr/bin/env python3
"""
Test script to verify CNN model training works correctly
"""

import requests
import json
import time

# Test configuration
BASE_URL = "http://localhost:8080/api/training"
SESSION_ID = "session_1756836972548_5uroo1q"

def test_cnn_model():
    """Test CNN model training"""
    
    print("Testing CNN model training...")
    
    # Prepare training request for CNN model
    payload = {
        "model_parameters": {
            "MODE": "CNN",  # This should map to "CNN" in backend
            "LAY": 3,       # Number of layers
            "N": 64,        # Number of filters
            "K": 3,         # Kernel size
            "ACTF": "ReLU", # Activation function
            "EP": 20        # Epochs
        },
        "training_split": {
            "split": 0.8,
            "validation_split": 0.1
        }
    }
    
    # Send training request
    url = f"{BASE_URL}/train-models/{SESSION_ID}"
    print(f"\nSending POST request to: {url}")
    print(f"Payload: {json.dumps(payload, indent=2)}")
    
    try:
        response = requests.post(url, json=payload)
        print(f"\nResponse status: {response.status_code}")
        print(f"Response: {response.json()}")
        
        if response.status_code == 200:
            print("\n✅ Training request successful!")
            
            # Wait a bit for training to complete
            print("Waiting for training to complete...")
            time.sleep(10)  # CNN takes longer
            
            # Check training status
            status_url = f"{BASE_URL}/status/{SESSION_ID}"
            status_response = requests.get(status_url)
            print(f"\nTraining status: {status_response.json()}")
            
        else:
            print(f"\n❌ Training failed: {response.text}")
            
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")

if __name__ == "__main__":
    test_cnn_model()