#!/usr/bin/env python3
"""
Test API integration between frontend and backend
Tests the new endpoints that connect to the verified training pipeline
"""

import requests
import json
import time
import sys

API_BASE = "http://127.0.0.1:8080/api/training"

def test_generate_datasets(session_id):
    """Test the /generate-datasets endpoint"""
    print(f"\n🧪 Testing /generate-datasets/{session_id}")
    
    payload = {
        "model_parameters": {
            "MODE": "Dense",
            "LAY": 2,
            "N": 32,
            "EP": 5,
            "ACTF": "relu"
        },
        "training_split": {
            "train_ratio": 0.7,
            "validation_ratio": 0.2,
            "test_ratio": 0.1,
            "shuffle": True
        }
    }
    
    print(f"📤 Sending request with payload: {json.dumps(payload, indent=2)}")
    
    try:
        response = requests.post(
            f"{API_BASE}/generate-datasets/{session_id}",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        print(f"📥 Response status: {response.status_code}")
        data = response.json()
        print(f"📥 Response data: {json.dumps(data, indent=2)}")
        
        if response.status_code == 200 and data.get('success'):
            print("✅ Dataset generation endpoint works!")
            return True
        else:
            print(f"❌ Dataset generation failed: {data.get('error')}")
            return False
            
    except Exception as e:
        print(f"❌ Error calling API: {str(e)}")
        return False

def test_train_models(session_id):
    """Test the /train-models endpoint"""
    print(f"\n🧪 Testing /train-models/{session_id}")
    
    payload = {
        "model_parameters": {
            "MODE": "CNN",
            "LAY": 2,
            "N": 16,
            "K": 3,
            "EP": 3,
            "ACTF": "ReLU"
        },
        "training_split": {
            "train_ratio": 0.7,
            "validation_ratio": 0.2,
            "test_ratio": 0.1,
            "shuffle": False
        }
    }
    
    print(f"📤 Sending request with payload: {json.dumps(payload, indent=2)}")
    
    try:
        response = requests.post(
            f"{API_BASE}/train-models/{session_id}",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        print(f"📥 Response status: {response.status_code}")
        data = response.json()
        print(f"📥 Response data: {json.dumps(data, indent=2)}")
        
        if response.status_code == 200 and data.get('success'):
            print("✅ Train models endpoint works!")
            print("⏳ Training is running in background...")
            return True
        else:
            print(f"❌ Train models failed: {data.get('error')}")
            return False
            
    except Exception as e:
        print(f"❌ Error calling API: {str(e)}")
        return False

def test_all_models(session_id):
    """Test all 7 model types through the API"""
    models = [
        {
            "name": "Linear",
            "params": {"MODE": "LIN"}
        },
        {
            "name": "Dense Neural Network", 
            "params": {"MODE": "Dense", "LAY": 2, "N": 32, "EP": 2, "ACTF": "relu"}
        },
        {
            "name": "CNN",
            "params": {"MODE": "CNN", "LAY": 2, "N": 16, "K": 3, "EP": 2, "ACTF": "sigmoid"}
        },
        {
            "name": "LSTM",
            "params": {"MODE": "LSTM", "LAY": 1, "N": 16, "EP": 2, "ACTF": "tanh"}
        },
        {
            "name": "AR-LSTM",
            "params": {"MODE": "AR LSTM", "LAY": 1, "N": 16, "EP": 2, "ACTF": "linear"}
        },
        {
            "name": "SVR Direct",
            "params": {"MODE": "SVR_dir", "KERNEL": "rbf", "C": 1.0, "EPSILON": 0.1}
        },
        {
            "name": "SVR MIMO",
            "params": {"MODE": "SVR_MIMO", "KERNEL": "linear", "C": 10.0, "EPSILON": 0.01}
        }
    ]
    
    print("\n🧪 Testing all 7 model types through API")
    
    for model in models:
        print(f"\n📊 Testing {model['name']}...")
        
        payload = {
            "model_parameters": model['params'],
            "training_split": {
                "train_ratio": 0.7,
                "validation_ratio": 0.2, 
                "test_ratio": 0.1,
                "shuffle": True
            }
        }
        
        try:
            response = requests.post(
                f"{API_BASE}/train-models/{session_id}",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                print(f"✅ {model['name']} - API call successful")
            else:
                print(f"❌ {model['name']} - API call failed: {response.status_code}")
                
        except Exception as e:
            print(f"❌ {model['name']} - Error: {str(e)}")
        
        time.sleep(1)

def main():
    """Main test function"""
    print("=" * 60)
    print("🔬 API Integration Test")
    print("=" * 60)
    
    if len(sys.argv) < 2:
        print("\n⚠️  No session ID provided, using test session")
        session_id = "test_session_api_" + str(int(time.time()))
    else:
        session_id = sys.argv[1]
    
    print(f"\n📍 Using session ID: {session_id}")
    
    print("\n" + "=" * 60)
    print("Testing Individual Endpoints")
    print("=" * 60)
    
    datasets_ok = test_generate_datasets(session_id)
    
    training_ok = test_train_models(session_id)
    
    if datasets_ok and training_ok:
        print("\n" + "=" * 60)
        print("Testing All Model Types")
        print("=" * 60)
        test_all_models(session_id)
    
    print("\n" + "=" * 60)
    print("📊 Test Summary")
    print("=" * 60)
    
    if datasets_ok and training_ok:
        print("✅ All API endpoints are working correctly!")
        print("✅ Frontend-backend integration is ready!")
    else:
        print("❌ Some API endpoints failed")
        print("🔧 Please check the error messages above")
    
    print("\n💡 Next steps:")
    print("1. Start the backend: cd my_backend && python app.py")
    print("2. Start the frontend: cd RabensteinerEngineering && npm start")
    print("3. Upload CSV files through the UI")
    print("4. Select model configuration")
    print("5. Click 'Generate Datasets' or 'Train Models'")

if __name__ == "__main__":
    main()
