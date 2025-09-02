# API Integration Documentation

## Overview
This document describes the complete frontend-backend integration for the training system. The integration connects the React frontend with the Flask backend through REST API endpoints.

## Architecture

```
Frontend (React/TypeScript)       Backend (Flask/Python)
├── ModelConfiguration.tsx   →    ├── /api/training/generate-datasets
├── TrainingDataSplit.tsx    →    ├── /api/training/train-models  
├── TrainingApiService.ts    →    ├── ModernMiddlemanRunner
└── Model Parameters         →    └── pipeline_exact.py (verified)
```

## API Endpoints

### 1. Generate Datasets
**Endpoint:** `POST /api/training/generate-datasets/<session_id>`

**Purpose:** Generate training datasets and violin plots with model configuration

**Request Body:**
```json
{
  "model_parameters": {
    "MODE": "Dense",      // Model type: Dense, CNN, LSTM, AR LSTM, SVR_dir, SVR_MIMO, LIN
    "LAY": 2,            // Number of layers (neural networks)
    "N": 64,             // Neurons per layer or filters (neural networks)
    "EP": 10,            // Epochs (neural networks)
    "ACTF": "relu",      // Activation function (neural networks)
    "K": 3,              // Kernel size (CNN only)
    "KERNEL": "rbf",     // Kernel type (SVR only)
    "C": 1.0,            // C parameter (SVR only)
    "EPSILON": 0.1       // Epsilon (SVR only)
  },
  "training_split": {
    "train_ratio": 0.7,
    "validation_ratio": 0.2,
    "test_ratio": 0.1,
    "shuffle": true
  }
}
```

**Response:**
```json
{
  "success": true,
  "message": "Datasets generated successfully",
  "dataset_count": 10,
  "violin_plots": {
    "plot1": "path/to/plot1.png",
    "plot2": "path/to/plot2.png"
  }
}
```

### 2. Train Models
**Endpoint:** `POST /api/training/train-models/<session_id>`

**Purpose:** Train models with user-specified parameters

**Request Body:** Same as generate-datasets

**Response:**
```json
{
  "success": true,
  "message": "Model training started for session xyz",
  "note": "Training is running in background, listen for SocketIO events for progress"
}
```

**SocketIO Events:**
- `training_completed`: Emitted when training finishes successfully
- `training_error`: Emitted if training fails
- `training_progress`: Periodic progress updates

### 3. Run Analysis (Legacy)
**Endpoint:** `POST /api/training/run-analysis/<session_id>`

**Purpose:** Run complete training pipeline (kept for backwards compatibility)

## Model Types and Parameters

### Neural Network Models (Dense, CNN, LSTM, AR LSTM)
- `MODE`: Model type identifier
- `LAY`: Number of layers (1-10)
- `N`: Neurons/filters per layer (1-2048)
- `EP`: Training epochs (1-1000)
- `ACTF`: Activation function (relu, sigmoid, tanh, linear, softmax)
- `K`: Kernel size for CNN only (1-11, odd numbers)

### Support Vector Regression (SVR_dir, SVR_MIMO)
- `MODE`: Model type identifier
- `KERNEL`: Kernel type (linear, poly, rbf, sigmoid)
- `C`: Regularization parameter (0.001-1000)
- `EPSILON`: Epsilon parameter (0.001-1)

### Linear Model (LIN)
- `MODE`: "LIN" (no additional parameters needed)

## Data Flow

1. **Frontend Input**
   - User selects model type in ModelConfiguration component
   - Sets parameters through UI forms
   - Configures training data split percentages

2. **API Call**
   - TrainingApiService converts UI parameters to backend format
   - Validates parameters before sending
   - Makes POST request to appropriate endpoint

3. **Backend Processing**
   - API endpoint receives request
   - ModernMiddlemanRunner prepares model configuration
   - Calls pipeline_exact.py with MDL configuration
   - Runs verified training pipeline

4. **Results**
   - Training results saved to database
   - SocketIO emits completion events
   - Frontend displays results

## Testing

### Run API Integration Test
```bash
cd my_backend/services/Tests
python3 test_api_integration.py [session_id]
```

### Test All Models
```bash
python3 test_all_models.py
```

### Manual Testing
1. Start backend: `cd my_backend && python app.py`
2. Start frontend: `cd RabensteinerEngineering && npm start`
3. Upload CSV files through UI
4. Select model configuration
5. Click "Generate Datasets" or "Train Models"

## Implementation Files

### Backend
- `/my_backend/api/routes/training.py` - API endpoints
- `/my_backend/services/training/middleman_runner.py` - Runner connecting API to pipeline
- `/my_backend/services/training/pipeline_exact.py` - Verified training pipeline
- `/my_backend/services/training/config.py` - MDL configuration class

### Frontend
- `/src/features/training/services/trainingApiService.ts` - API client
- `/src/features/training/components/ModelConfiguration.tsx` - Model selection UI
- `/src/features/training/components/TrainingDataSplit.tsx` - Data split configuration
- `/src/features/training/config/constants.ts` - API endpoints configuration

## Notes

- All 7 model types are fully supported and tested
- Pipeline produces identical results to original training_original.py
- SocketIO provides real-time progress updates
- Background processing prevents request timeouts
- Model parameters are validated on both frontend and backend

## Next Steps

1. ✅ ModernMiddlemanRunner connects to verified pipeline
2. ✅ API endpoints created for dataset generation and training
3. ✅ Model parameters passed correctly from frontend to backend
4. ✅ All 7 model types working through API
5. 🔄 Add progress tracking for dataset generation
6. 🔄 Implement model download endpoint
7. 🔄 Add training results visualization