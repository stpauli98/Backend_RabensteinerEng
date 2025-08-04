# 📋 KOMPLETNO MAPIRANJE TRAINING SISTEMA

**Datum analize**: 2025-01-31  
**Obuhvata**: Frontend (React/TypeScript) → Backend (Flask/Python) → Training System → Database (Supabase)

---

## 🎯 PREGLED SISTEMA

Training sistem omogućava:
- **Upload CSV fajlova** sa chunked upload mehanizmom
- **Konfiguraciju parametara** kroz intuitivni UI
- **Generisanje dataseta** sa automatskim feature engineering
- **Treniranje modela** (Dense NN, CNN, LSTM, AR-LSTM, SVR, Linear)
- **Real-time praćenje** napretka kroz SocketIO
- **Vizualizaciju rezultata** sa interaktivnim grafovima
- **Perzistenciju podataka** kroz Supabase

---

## 🔄 KOMPLETAN WORKFLOW - KORAK PO KORAK

### FAZA 1: FILE UPLOAD (Frontend → Backend)

#### Frontend (Training.tsx)
```typescript
// Korisnik bira fajlove
handleFileSelection() → files.map(file => ({
  name: file.name,
  size: file.size,
  status: 'pending'
}))

// Chunked upload počinje
ChunkedUploader.upload() → {
  chunkSize: 1MB,
  endpoint: '/upload-chunk',
  onProgress: updateFileProgress()
}
```

#### Backend (training.py)
```python
@training_bp.route('/upload-chunk', methods=['POST'])
def upload_chunk():
    # Prima chunk podatke
    chunk_data = request.files['chunk']
    chunk_index = request.form['chunkIndex']
    
    # Čuva u temp folder
    chunk_path = f"chunk_uploads/{upload_id}_{chunk_index}.chunk"
    
    # Kada su svi chunk-ovi primljeni
    if all_chunks_received:
        assemble_file()
        return {'status': 'completed'}
```

### FAZA 2: SESSION CREATION

#### Frontend
```typescript
// Nakon upload-a, kreira se database sesija
handleUploadComplete() → 
  axios.post('/create-database-session', {
    uploadId,
    files: uploadedFiles
  })
```

#### Backend
```python
@training_bp.route('/create-database-session', methods=['POST'])
def create_database_session():
    # Kreira Supabase zapis
    session_data = {
        'session_id': generate_session_id(),
        'files': file_metadata,
        'created_at': datetime.now()
    }
    
    # Upload fajlova na Supabase storage
    supabase.storage.upload(files)
    
    return {'sessionId': session_id}
```

### FAZA 3: CONFIGURATION

#### Frontend
```typescript
// Korisnik konfigurira parametre
const trainingConfig = {
  // Vremenske karakteristike
  timeFeatures: {
    year: true,
    month: true,
    week: true,
    holiday: true
  },
  
  // Vremenski koraci
  timeSteps: {
    input: 24,
    output: 1,
    stepSize: 1,
    offset: 0
  },
  
  // Model parametri (po modelu)
  modelParams: {
    Dense_NN: {
      layers: [64, 32],
      epochs: 50,
      batchSize: 32
    },
    CNN: {
      filters: [32, 64],
      kernelSize: 3,
      epochs: 50
    }
    // ... ostali modeli
  }
}
```

### FAZA 4: DATASET GENERATION

#### Frontend
```typescript
handleGenerateDatasets() → {
  // Šalje konfiguraciju na backend
  axios.post(`/api/training/generate-datasets/${sessionId}`, {
    configuration: trainingConfig
  })
  
  // Pokreće polling za status
  pollDatasetGeneration()
}
```

#### Backend (training_api.py)
```python
@training_api.route('/generate-datasets/<session_id>', methods=['POST'])
def generate_datasets(session_id):
    # Konvertuje frontend parametre u backend format
    mts_config = parameter_converter.convert_to_mts(request.json)
    
    # Pokreće async dataset generaciju
    task = executor.submit(
        pipeline.run_dataset_generation,
        session_id
    )
    
    return {'status': 'processing'}
```

#### Training System (pipeline_integration.py)
```python
def run_dataset_generation_pipeline(session_id, supabase, socketio):
    # 1. Učitava fajlove sa Supabase
    data_loader = DataLoader(supabase)
    input_files, output_files = data_loader.load_session_files(session_id)
    
    # 2. Procesira podatke
    processor = RealDataProcessor()
    datasets = processor.create_ml_datasets(
        input_files, 
        output_files,
        time_features_config
    )
    
    # 3. Generiše violin plotove
    visualizer = create_visualizer()
    violin_plots = visualizer.create_violin_plots(datasets)
    
    # 4. Emituje progress
    socketio.emit('dataset_generation_complete', {
        'session_id': session_id,
        'dataset_count': len(datasets),
        'violin_plots': violin_plots
    })
    
    return {
        'datasets': datasets,
        'visualizations': violin_plots
    }
```

### FAZA 5: MODEL TRAINING

#### Frontend
```typescript
handleTrainModels() → {
  // Šalje model parametre
  axios.post(`/api/training/train-models/${sessionId}`, {
    modelConfigurations: selectedModels
  })
  
  // Real-time praćenje kroz SocketIO
  socket.on('training_progress', (data) => {
    updateTrainingProgress(data)
  })
}
```

#### Backend (training_api.py)
```python
@training_api.route('/train-models/<session_id>', methods=['POST'])
def train_models(session_id):
    # Konvertuje parametre
    mdl_config = parameter_converter.convert_to_mdl(request.json)
    
    # Pokreće training pipeline
    task = executor.submit(
        pipeline.run_model_training,
        session_id,
        mdl_config
    )
    
    return {'status': 'training_started'}
```

#### Training System (model_trainer.py)
```python
class ModelTrainer:
    def train_all_models(self, datasets, config):
        results = {}
        
        for dataset_name, dataset in datasets.items():
            results[dataset_name] = {}
            
            # Trenira svaki omogućen model
            if config.Dense_NN.enabled:
                model = self.train_dense(
                    dataset['X_train'],
                    dataset['y_train'],
                    config.Dense_NN
                )
                results[dataset_name]['Dense_NN'] = {
                    'model': model,
                    'history': training_history,
                    'predictions': model.predict(dataset['X_test'])
                }
            
            # CNN, LSTM, SVR, Linear...
            # Emituje progress nakon svakog modela
            self._emit_progress()
        
        return results
```

### FAZA 6: EVALUATION & RESULTS

#### Training System (results_generator.py)
```python
def generate_results(training_results, session_data):
    evaluation_metrics = {}
    
    for dataset_name, models in training_results.items():
        evaluation_metrics[dataset_name] = {}
        
        for model_name, model_data in models.items():
            # Računa metrije
            metrics = {
                'mae': calculate_mae(actual, predicted),
                'rmse': calculate_rmse(actual, predicted),
                'mape': calculate_mape(actual, predicted),
                'r2': calculate_r2(actual, predicted),
                'smape': calculate_smape(actual, predicted),
                'wape': calculate_wape(actual, predicted),
                'mase': calculate_mase(actual, predicted, naive_forecast)
            }
            
            evaluation_metrics[dataset_name][model_name] = metrics
    
    # Pronalazi najbolji model
    best_model = find_best_model(evaluation_metrics)
    
    return {
        'evaluation_metrics': evaluation_metrics,
        'best_model': best_model,
        'model_comparison': create_comparison_charts()
    }
```

### FAZA 7: VISUALIZATION & DELIVERY

#### Frontend
```typescript
// Prima finalne rezultate
socket.on('training_complete', (results) => {
  setTrainingResults({
    evaluationMetrics: results.evaluation_metrics,
    bestModel: results.best_model,
    visualizations: results.visualizations
  })
  
  // Prikazuje rezultate
  showResultsModal()
})

// Korisnik može download-ovati
handleDownloadResults() → {
  axios.get(`/api/training/results/${sessionId}/download`)
}
```

---

## 📊 DETALJNO MAPIRANJE ORIGINALNOG KODA

### **Mapiranje: training_backend_test_2.py → training_system**

**Originalni fajl**: `training_backend_test_2.py` (3468 linija)  
**Modularizovano u**: 20 specijalizovanih modula

---

## 🎯 **IZVRŠNA SUMARIJA**

Originalni monolitni kod od 3468 linija je sistematski podeljen u 20 funkcionalnih modula koji omogućavaju:
- **Frontend integraciju** kroz RESTful API
- **Real-time praćenje** napretka treniranja
- **Modularno testiranje** i održavanje
- **Scalable arhitekturu** za buduće proširenja

---

## 📊 **DETALJNO MAPIRANJE PO LINIJAMA**

### **1. IMPORTS I MODULE SETUP**

| **Originalne linije** | **Training System Modul** | **Opis** |
|----------------------|---------------------------|-----------|
| **1-36** | Svi moduli (`__init__.py`) | Svi imports (pandas, numpy, sklearn, tensorflow, matplotlib) distribuirani kroz module |

---

### **2. OSNOVNE FUNKCIJE**

#### **`utils.py`** - Osnovne pomoćne funkcije
| **Originalne linije** | **Funkcija** | **Opis** |
|----------------------|-------------|-----------|
| **37-109** | `load()` | Učitavanje i validacija CSV podataka → `load_and_extract_info()` |
| **111-141** | `transf()` | Transformacija vremenskih oznaka → `time_transformation()` |
| **144-154** | `utc_idx_pre()` | Indeksiranje vremenskih serija (prethodni) → `utc_idx_pre()` |
| **157-167** | `utc_idx_post()` | Indeksiranje vremenskih serija (naredni) → `utc_idx_post()` |

---

### **3. MACHINE LEARNING MODELI**

#### **`model_trainer.py`** - Svi ML modeli
| **Originalne linije** | **Funkcija** | **Opis** |
|----------------------|-------------|-----------|
| **170-238** | `train_dense()` | Dense Neural Network treniranje → `train_dense()` |
| **239-320** | `train_cnn()` | Convolutional Neural Network → `train_cnn()` |
| **321-388** | `train_lstm()` | LSTM Network → `train_lstm()` |
| **389-457** | `train_ar_lstm()` | Autoregressive LSTM → `train_ar_lstm()` |
| **458-492** | `train_svr_dir()` | Support Vector Regression (Direct) → `train_svr_dir()` |
| **493-530** | `train_svr_mimo()` | Support Vector Regression (MIMO) → `train_svr_mimo()` |
| **531-553** | `train_linear_model()` | Linear Regression → `train_linear_model()` |

---

### **4. EVALUACIJA I METRIJE**

#### **`results_generator.py`** - Metrijske funkcije
| **Originalne linije** | **Funkcija** | **Opis** |
|----------------------|-------------|-----------|
| **555-566** | `wape()` | Weighted Absolute Percentage Error → `wape()` |
| **570-585** | `smape()` | Symmetric Mean Absolute Percentage Error → `smape()` |
| **588-608** | `mase()` | Mean Absolute Scaled Error → `mase()` |

---

### **5. KONFIGURACIJE I SETUP**

#### **`config.py`** - Sve konfiguracije
| **Originalne linije** | **Klasa/Objekat** | **Opis** |
|----------------------|------------------|-----------|
| **619-632** | `class MTS` | Multivariate Time Series konfiguracija → `class MTS` |
| **634-692** | `HOL` dictionary | Praznici za različite zemlje → `HOL` |
| **798-954** | `class T` | Vremenske karakteristike sa podklasama → `class T` |
| **2046-2141** | `class MDL` | Model konfiguracije → `class MDL` |

#### **`time_features.py`** - Vremenske karakteristike
| **Originalne linije** | **Funkcionalnost** | **Opis** |
|----------------------|-------------------|-----------|
| **798-954** | `T` klasa implementacija | Generisanje vremenskih karakteristika (Y, M, W, D, H) |

#### **`temporal_config.py`** - Dinamička vremenska konfiguracija
| **Originalne linije** | **Funkcionalnost** | **Opis** |
|----------------------|-------------------|-----------|
| **788-790** | Vremenska transformacija | Dinamička konfiguracija parametara |

---

### **6. UČITAVANJE I PROCESIRANJE PODATAKA**

#### **`data_loader.py`** - Učitavanje podataka
| **Originalne linije** | **Funkcionalnost** | **Opis** |
|----------------------|-------------------|-----------|
| **700-751** | Inicijalizacija i učitavanje | Setup i podatak loading → `DataLoader` klasa |
| **986-1015** | Output data loading | Učitavanje izlaznih podataka |

#### **`data_processor.py`** - Procesiranje podataka
| **Originalne linije** | **Funkcionalnost** | **Opis** |
|----------------------|-------------------|-----------|
| **1051-1756** | Dataset kreiranje | Glavna logika za kreiranje dataseta → `DataProcessor` |
| **1079-1756** | Vremenska petlja | Iterativno kreiranje vremenskih serija |

#### **`data_scaling.py`** - Skaliranje podataka
| **Originalne linije** | **Funkcionalnost** | **Opis** |
|----------------------|-------------------|-----------|
| **1812-1873** | Min-Max Scaling | Skaliranje ulaza i izlaza → `DataScaler` klasa |
| **2178-2211** | Dataset skaliranje | Primena skaliranja na datasete |

---

### **7. VISUALIZACIJA**

#### **`visualization.py`** - Grafički prikazi
| **Originalne linije** | **Funkcionalnost** | **Opis** |
|----------------------|-------------------|-----------|
| **1876-2029** | Violina plotovi | Distribucija podataka ulaza/izlaza → `Visualizer` |
| **2336-3244** | Rezultati plotovi | Forecast vs actual vizualizacije |

---

### **8. TRENIRANJE I ORCHESTRACIJA**

#### **`training_pipeline.py`** - Pipeline orchestracija
| **Originalne linije** | **Funkcionalnost** | **Opis** |
|----------------------|-------------------|-----------|
| **2030-2043** | Setup treniranja | Osnovni parametri (train/val/test split) → `TrainingPipeline` |
| **2159-2176** | Randomizacija | Mešanje dataseta |
| **2213-2237** | Finalni dataseti | Kreiranje konačnih train/val/test setova |

#### **`pipeline_integration.py`** - Integracija sistema
| **Originalne linije** | **Funkcionalnost** | **Opis** |
|----------------------|-------------------|-----------|
| **2237-2260** | Model treniranje | Orkestracija treniranja → `run_complete_original_pipeline()` |
| **2260-2307** | Model testiranje | Testiranje modela → `run_model_training_pipeline()` |

---

### **9. RE-SCALING I EVALUACIJA**

| **Originalne linije** | **Training System Modul** | **Funkcionalnost** |
|----------------------|---------------------------|-------------------|
| **2309-2333** | `data_scaling.py` | Re-scaling predviđanja |
| **3245-3468** | `results_generator.py` | Kompletna evaluacija sa metrijama |

---

### **10. NOVI MODULI (DODACI)**

Ovi moduli su **DODANI** za frontend integraciju i ne postoje u originalnom kodu:

#### **`training_api.py`** - RESTful API (2188 linija)
- **NOVO**: Kompletni Flask API sa 15+ endpoints
- **Endpoints**: `/generate-datasets`, `/train-models`, `/results`, `/status`, `/visualizations`
- **SocketIO**: Real-time progress updates

#### **`error_handler.py`** - Error Management (638 linija)
- **NOVO**: Strukturirani error handling sa recovery suggestions
- **Features**: Error categorization, severity levels, user-friendly messages

#### **`progress_manager.py`** - Progress Tracking (615 linija)
- **NOVO**: Real-time praćenje napretka treniranja
- **Features**: Phase tracking, ETA calculation, SocketIO emission

#### **`parameter_converter.py`** - Parameter Conversion (796 linija)
- **NOVO**: Frontend ↔ Backend parameter konverzija
- **Mapiranje**: `2142-2157` (Frontend parameter mapping)

#### **`monitoring_api.py`** - System Monitoring (570 linija)
- **NOVO**: Health checks, performance monitoring

#### **`logging_config.py`** - Structured Logging (556 linija)
- **NOVO**: Konfigurisani logger sa file/console output

---

## 🔄 **WORKFLOW MAPPING**

### **Originalni workflow (training_backend_test_2.py)**
```
Imports → Funkcije → Klase → Input Data → Time Features → Output Data → 
Dataset Creation → Scaling → Visualization → Model Setup → Training → 
Testing → Re-scaling → Plots → Evaluation
```

### **Novi modularni workflow (training_system)**
```
API Request → Parameter Validation → Data Loading → Data Processing → 
Time Features → Dataset Creation → Scaling → Visualization → 
Model Training → Testing → Re-scaling → Results Generation → 
API Response + SocketIO Updates
```

---

## 📈 **STATISTIKE PODELE**

| **Kategorija** | **Originalne linije** | **Training System Modul(i)** | **Broj linija** |
|---------------|----------------------|-------------------------------|-----------------|
| **Imports** | 1-36 (36 linija) | Distribuirano kroz sve module | ~50 linija |
| **Osnovne funkcije** | 37-167 (131 linija) | `utils.py` | 1670 linija |
| **ML Modeli** | 170-553 (384 linija) | `model_trainer.py` | 575 linija |
| **Metrije** | 555-608 (54 linija) | `results_generator.py` | 588 linija |
| **Konfiguracije** | 619-954 (336 linija) | `config.py`, `time_features.py`, `temporal_config.py` | 984 linija |
| **Data Loading** | 700-1015 (316 linija) | `data_loader.py` | 587 linija |
| **Data Processing** | 1051-1756 (706 linija) | `data_processor.py` | 559 linija |
| **Scaling** | 1812-2211 (400 linija) | `data_scaling.py` | 546 linija |
| **Visualization** | 1876-3244 (1368 linija) | `visualization.py` | 625 linija |
| **Training** | 2030-2307 (278 linija) | `training_pipeline.py`, `pipeline_integration.py` | 2538 linija |
| **Evaluation** | 3245-3468 (224 linija) | `results_generator.py` (prošireno) | - |
| **API & Management** | **NOVO** | `training_api.py`, `error_handler.py`, `progress_manager.py`, itd. | 6000+ linija |

**UKUPNO**: 3468 → 15000+ linija (4.3x proširenje sa novim funkcionalnostima)

---

## 🎯 **KLJUČNI UVIDI**

### **1. Funkcionalnost je očuvana**
- **100%** originalnih funkcija je preslikano
- **Identične algoritme** za ML modele
- **Iste metrije** evaluacije

### **2. Dodane su nove funkcionalnosti**
- **RESTful API** za frontend
- **Real-time progress** tracking
- **Error handling** sa recovery
- **Parameter validation**

### **3. Improved Architecture**
- **Modularna struktura** umesto monolita
- **Dependency injection** kroz konstruktore
- **Configuration management** 
- **Proper logging** and monitoring

### **4. Frontend Integration Ready**
- **API endpoints** za sve operacije
- **SocketIO events** za real-time updates
- **JSON serialization** svih rezultata
- **User-friendly error** messages

---

## 🚀 **SLEDEĆI KORACI**

### **Za nastavak rada**:

1. **API Testiranje**
   ```bash
   # Test dataset generation
   POST /api/training/generate-datasets/<session_id>
   
   # Test model training  
   POST /api/training/train-models/<session_id>
   
   # Check status
   GET /api/training/status/<session_id>
   ```

2. **Frontend Integracija**
   - Koristiti `/api/training/*` endpoints
   - Implementirati SocketIO listeners
   - Handling error responses

3. **Debugging**
   - Svaki modul može biti testiran nezavisno
   - Structured logging u `logs/` direktoriju
   - Error details u API responses

---

**Ovaj mapping omogućava potpuno razumevanje kako je originalni kod transformisan u modularni sistem spreman za production deployment sa frontend integracijom.**

---

## 🛡️ ERROR HANDLING & RECOVERY

### Frontend Error Handling
```typescript
// Training.tsx
const handleError = (error: TrainingError) => {
  // Kategorizuje grešku
  const errorCategory = categorizeError(error);
  
  // Prikazuje user-friendly poruku
  showErrorModal({
    title: errorCategory.title,
    message: errorCategory.userMessage,
    suggestions: errorCategory.recoverySuggestions,
    technicalDetails: error.details
  });
  
  // Log za debugging
  console.error('Training error:', error);
}
```

### Backend Error Categories (error_handler.py)
```python
ERROR_CATEGORIES = {
    'DATA_VALIDATION': {
        'severity': 'high',
        'user_message': 'Invalid data format',
        'recovery': ['Check CSV format', 'Verify column names']
    },
    'INSUFFICIENT_DATA': {
        'severity': 'high',
        'user_message': 'Not enough data points',
        'recovery': ['Add more data', 'Reduce time steps']
    },
    'MODEL_CONVERGENCE': {
        'severity': 'medium',
        'user_message': 'Model failed to converge',
        'recovery': ['Adjust hyperparameters', 'Try different model']
    },
    'RESOURCE_EXHAUSTION': {
        'severity': 'critical',
        'user_message': 'System resources exhausted',
        'recovery': ['Reduce batch size', 'Use fewer models']
    }
}
```

---

## 📡 REAL-TIME KOMUNIKACIJA (SocketIO)

### Event Flow
```
Frontend                  Backend                   Training System
    |                        |                             |
    |------ connect -------->|                             |
    |<----- connected -------|                             |
    |                        |                             |
    |--- start_training ---->|                             |
    |                        |---- execute_training ------>|
    |                        |                             |
    |                        |<--- progress_update --------|
    |<-- training_progress --|                             |
    |                        |                             |
    |                        |<--- phase_complete ---------|
    |<--- phase_update ------|                             |
    |                        |                             |
    |                        |<--- training_complete ------|
    |<-- training_complete --|                             |
```

### SocketIO Events

#### Frontend → Backend
- `join_session`: Pridružuje se session room-u
- `start_training`: Pokreće training
- `cancel_training`: Prekida training
- `get_status`: Traži trenutni status

#### Backend → Frontend
- `training_progress`: Progress update (0-100%)
- `phase_update`: Prelazak između faza
- `model_complete`: Završen pojedinačni model
- `training_complete`: Svi modeli završeni
- `error`: Greška tokom treniranja

---

## 💾 DATABASE SCHEMA (Supabase)

### Tables

#### training_sessions
```sql
CREATE TABLE training_sessions (
    id UUID PRIMARY KEY,
    session_id TEXT UNIQUE,
    status TEXT, -- 'uploading', 'configuring', 'processing', 'completed', 'failed'
    created_at TIMESTAMP,
    updated_at TIMESTAMP,
    user_id UUID REFERENCES users(id),
    configuration JSONB,
    metadata JSONB
);
```

#### training_files
```sql
CREATE TABLE training_files (
    id UUID PRIMARY KEY,
    session_id TEXT REFERENCES training_sessions(session_id),
    file_name TEXT,
    file_type TEXT, -- 'input' or 'output'
    file_size BIGINT,
    storage_path TEXT,
    uploaded_at TIMESTAMP
);
```

#### training_results
```sql
CREATE TABLE training_results (
    id UUID PRIMARY KEY,
    session_id TEXT REFERENCES training_sessions(session_id),
    evaluation_metrics JSONB,
    model_performance JSONB,
    best_model JSONB,
    summary JSONB,
    status TEXT,
    completed_at TIMESTAMP
);
```

#### training_visualizations
```sql
CREATE TABLE training_visualizations (
    id UUID PRIMARY KEY,
    session_id TEXT REFERENCES training_sessions(session_id),
    plot_name TEXT,
    plot_type TEXT,
    plot_data_base64 TEXT,
    metadata JSONB,
    created_at TIMESTAMP
);
```

#### training_logs
```sql
CREATE TABLE training_logs (
    id UUID PRIMARY KEY,
    session_id TEXT,
    message TEXT,
    level TEXT, -- 'INFO', 'WARNING', 'ERROR'
    step_number INTEGER,
    step_name TEXT,
    progress_percentage INTEGER,
    created_at TIMESTAMP
);
```

---

## 🔐 SECURITY CONSIDERATIONS

### File Upload Security
- **Chunk validation**: Svaki chunk se validira pre čuvanja
- **File type checking**: Samo CSV fajlovi su dozvoljeni
- **Size limits**: Max 500MB po fajlu
- **Virus scanning**: Integracija sa ClamAV (opciono)

### API Security
- **Authentication**: Supabase JWT tokens
- **Rate limiting**: 100 requests/minute per user
- **CORS**: Strict origin checking
- **Input validation**: Svi parametri se validiraju

### Data Security
- **Encryption at rest**: Supabase enkriptuje sve podatke
- **Encryption in transit**: HTTPS za sve komunikacije
- **Access control**: Row-level security u Supabase
- **Data isolation**: Svaki user vidi samo svoje podatke

---

## 🚀 PERFORMANCE OPTIMIZATIONS

### Frontend
- **Lazy loading**: Komponente se učitavaju po potrebi
- **Memoization**: React.memo za expensive komponente
- **Virtual scrolling**: Za velike rezultate
- **WebWorkers**: Za heavy computations

### Backend
- **Async processing**: ThreadPoolExecutor za training
- **Caching**: Redis za često korišćene podatke
- **Database pooling**: Connection pool za Supabase
- **Chunked responses**: Streaming za velike rezultate

### Training System
- **Batch processing**: Optimizovano za GPU
- **Early stopping**: Prekida training ako nema napretka
- **Model checkpointing**: Čuva najbolje modele
- **Memory management**: Garbage collection nakon svake faze

---

## 📈 MONITORING & LOGGING

### Metrics
- **Training duration**: Po modelu i ukupno
- **Resource usage**: CPU, RAM, GPU
- **Model performance**: Praćenje metrika tokom treniranja
- **Error rates**: Po tipu greške

### Logging Levels
```python
# logging_config.py
LOGGING_CONFIG = {
    'version': 1,
    'handlers': {
        'file': {
            'class': 'logging.FileHandler',
            'filename': 'logs/training_system.log',
            'formatter': 'detailed'
        },
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'simple'
        }
    },
    'root': {
        'level': 'INFO',
        'handlers': ['file', 'console']
    }
}
```

---

## 🔄 DEPLOYMENT CONSIDERATIONS

### Docker Configuration
```dockerfile
FROM python:3.9-slim

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application
COPY . /app
WORKDIR /app

# Run with gunicorn
CMD ["gunicorn", "--workers=1", "--threads=8", "--timeout=0", 
     "--bind=0.0.0.0:8080", "app:app"]
```

### Environment Variables
```bash
# .env
SUPABASE_URL=https://xxx.supabase.co
SUPABASE_KEY=xxx
FLASK_ENV=production
SECRET_KEY=xxx
REDIS_URL=redis://localhost:6379
```

### Scaling Strategies
- **Horizontal scaling**: Multiple worker instances
- **Queue system**: Celery za long-running tasks
- **Load balancing**: Nginx reverse proxy
- **CDN**: Za static assets i visualizations

---

## 🎯 ZAKLJUČAK

Ovaj dokument pruža kompletno mapiranje training sistema od korisničkog interfejsa do finalnih rezultata. Sistem je dizajniran za:

1. **Skalabilnost**: Može handle multiple concurrent trainings
2. **Pouzdanost**: Comprehensive error handling i recovery
3. **Performance**: Optimizovan za brzinu i efikasnost
4. **User Experience**: Real-time updates i intuitivni UI
5. **Maintainability**: Modularna arhitektura olakšava održavanje

Svaki korak je dokumentovan sa code examples i može se lakše debug-ovati kroz structured logging i monitoring.