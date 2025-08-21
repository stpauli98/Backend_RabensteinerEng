# PLAN ZA MODULARIZACIJU TRAINING SISTEMA

## CILJ
Podjela training_backend_test_2.py na modularne komponente koje omogućavaju:
1. Prijem podataka iz baze preko training.py
2. Obradu podataka kroz ML pipeline
3. Vraćanje rezultata na frontend preko API endpoints

## TRENUTNO STANJE
- training_backend_test_2.py je monolitni fajl od 3,468 linija
- Hardcoded file paths i konfiguracija
- Nema povratni tok podataka na frontend
- middleman_runner.py koristi subprocess umjesto modularnih poziva

## NOVA ARHITEKTURA

### training_system/
```
training_system/
├── __init__.py
├── config.py              # Konfiguracija i konstante
├── data_loader.py          # Čitanje podataka iz baze
├── data_processor.py       # Obrada i čišćenje podataka
├── model_trainer.py        # ML modeli i treniranje
├── results_generator.py    # Formatiranje rezultata
├── visualization.py        # Kreiranje plotova i grafikona
├── training_api.py         # API endpoints za rezultate
├── training_pipeline.py    # Glavni orchestrator
└── utils.py               # Pomoćne funkcije
```

## DETALJNI PLAN IMPLEMENTACIJE

### FAZA 1: PRIPREMA INFRASTRUKTURE (Dani 1-2)

#### Korak 1.1: Kreiranje API endpoints za rezultate
**Fajl:** `training_api.py`
**Šta treba:**
- GET `/api/training/results/<session_id>` - Vraća rezultate treniranja
- GET `/api/training/status/<session_id>` - Vraća status treniranja
- GET `/api/training/progress/<session_id>` - Vraća progress treniranja
- GET `/api/training/visualizations/<session_id>` - Vraća plotove i grafikone

#### Korak 1.2: Kreiranje database schema za rezultate
**Potrebno u Supabase:**
- Tabela `training_results` za čuvanje rezultata
- Tabela `training_progress` za praćenje statusa
- Tabela `training_visualizations` za slike/plotove

#### Korak 1.3: Modifikacija middleman_runner.py
**Šta treba:**
- Dodaj funkciju za čuvanje rezultata u bazu
- Dodaj SocketIO emit za progress updates
- Pripremi za poziv modularnih funkcija umjesto subprocess

### FAZA 2: IZVLAČENJE OSNOVNIH KOMPONENTI (Dani 3-5)

#### Korak 2.1: Kreiranje config.py
**Šta se izvlači iz training_backend_test_2.py:**
- MTS klasa (oko linija 619-692)
- MDL klasa (oko linija 2046-2141)
- HOL dictionary (praznici)
- Sve globalne konstante i konfiguracije

#### Korak 2.2: Kreiranje data_loader.py
**Šta treba:**
- Funkcija `load_session_data(session_id)` - čita iz baze
- Funkcija `download_session_files(session_id)` - skida CSV fajlove
- Funkcija `prepare_file_paths(session_id)` - priprema putanje za obradu

#### Korak 2.3: Kreiranje utils.py
**Šta se izvlači:**
- load() funkcija (linija 37-168)
- Pomoćne funkcije za datum/vrijeme
- Utility funkcije za file management

### FAZA 3: OBRADA PODATAKA (Dani 6-8)

#### Korak 3.1: Kreiranje data_processor.py
**Šta se izvlači iz training_backend_test_2.py:**
- transf() funkcija
- utc_idx_pre() i utc_idx_post() funkcije
- T klasa za time features (oko linija 798-955)
- Dataset preparation logika (oko linija 1055-1873)

#### Korak 3.2: Testiranje data processing
**Šta treba:**
- Kreirati test sa jednim session_id
- Provjeriti da li se podaci pravilno učitavaju i obrađuju
- Validirati da output odgovara originalnom

### FAZA 4: MACHINE LEARNING MODELI (Dani 9-12)

#### Korak 4.1: Kreiranje model_trainer.py
**Šta se izvlači iz training_backend_test_2.py:**
- train_dense() funkcija (oko linija 170-220)
- train_cnn() funkcija (oko linija 221-280)
- train_lstm() funkcija (oko linija 281-340)
- train_ar_lstm() funkcija (oko linija 341-400)
- train_svr_dir() funkcija (oko linija 401-460)
- train_svr_mimo() funkcija (oko linija 461-520)
- train_linear_model() funkcija (oko linija 521-553)

#### Korak 4.2: Kreiranje model factory pattern
**Šta treba:**
- Funkcija `create_model(model_type, config)`
- Funkcija `train_model(model, data, config)`
- Standardizovani interface za sve modele

### FAZA 5: REZULTATI I VIZUALIZACIJA (Dani 13-15)

#### Korak 5.1: Kreiranje results_generator.py
**Šta se izvlači:**
- wape(), smape(), mase() funkcije (oko linija 3245-3467)
- Evaluation logika
- Kreiranje df_eval i df_eval_ts DataFrames

#### Korak 5.2: Kreiranje visualization.py
**Šta se izvlači:**
- Matplotlib/seaborn kodovi (oko linija 1874-2032, 2340-2885)
- Violin plots funkcije
- Forecast visualization funkcije

### FAZA 6: ORCHESTRACIJA (Dani 16-18)

#### Korak 6.1: Kreiranje training_pipeline.py
**Šta treba:**
- Glavna funkcija `run_training_pipeline(session_id)`
- Koordinacija svih modula
- Error handling i logging
- Progress reporting

#### Korak 6.2: Integracija sa postojećim sistemom
**Šta treba:**
- Modifikacija middleman_runner.py da koristi novi pipeline
- Testiranje kompletnog toka
- Validacija rezultata

### FAZA 7: FINALNO TESTIRANJE (Dani 19-20)

#### Korak 7.1: End-to-end testiranje
- Test kompletnog toka: Frontend → API → Database → Processing → Results → Frontend
- Performance testiranje
- Error handling testiranje

#### Korak 7.2: Cleanup i optimizacija
- Uklanjanje training_backend_test_2.py (kada sve radi)
- Code review i refactoring
- Dokumentacija

## DETALJNI SADRŽAJ FAJLOVA

### config.py
```python
# MTS klasa - multivariate time series konfiguracija
# MDL klasa - model konfiguracija
# HOL dictionary - praznici
# Konstante za procesiranje
```

### data_loader.py
```python
# load_session_data() - čita session podatke iz baze
# download_session_files() - skida CSV fajlove iz storage
# prepare_file_paths() - priprema putanje za processing
```

### data_processor.py
```python
# process_session_data() - glavni processing workflow
# transf() - data transformation
# T klasa - time features
# Dataset preparation funkcije
```

### model_trainer.py
```python
# train_all_models() - trenira sve modele
# train_dense(), train_cnn(), train_lstm() - specifični modeli
# model_factory() - kreiranje modela
# evaluate_models() - evaluacija
```

### results_generator.py
```python
# generate_results() - formatira rezultate za frontend
# calculate_metrics() - računa sve metrije
# create_evaluation_dataframes() - df_eval, df_eval_ts
```

### visualization.py
```python
# create_visualizations() - generiše sve plotove
# create_violin_plots() - violin plotovi
# create_forecast_plots() - forecast vizualizacije
# save_plots_as_base64() - konvertuje u base64 za frontend
```

### training_api.py
```python
# API endpoints za rezultate
# @app.route('/api/training/results/<session_id>')
# @app.route('/api/training/status/<session_id>')
# @app.route('/api/training/visualizations/<session_id>')
```

### training_pipeline.py
```python
# run_training_pipeline() - glavna orchestracija
# Poziva sve module redom
# Čuva rezultate u bazu
# Emituje progress preko SocketIO
```

## KLJUČNE FUNKCIJE IZ training_backend_test_2.py

### Data Loading (linija 37-168)
- `load()` - učitava podatke iz CSV fajlova
- Parsira UTC timestamps
- Izvlači metadata (start/end time, broj redova)

### Data Transformation (linija 700-1049)
- `transf()` - transformiše podatke
- Interpolacija, outlier removal
- Data scaling i normalizacija

### Time Features (linija 798-955)
- `T` klasa - generiše time-based features
- Praznike, dan u sedmici, mjesec, itd.

### Model Training (linija 170-553)
- 7 različitih modela
- Dense NN, CNN, LSTM, AR-LSTM, SVR, Linear
- Svaki model ima svoju train_*() funkciju

### Evaluation (linija 3245-3467)
- Custom metrije: wape(), smape(), mase()
- Standardne metrije: MAE, MAPE, MSE, RMSE
- Generiranje evaluation DataFrames

### Visualization (linija 1874-2032, 2340-2885)
- Violin plots za distribuciju grešaka
- Forecast plots za predviđanja
- Matplotlib/seaborn integracija

## AŽURIRANI PLAN BAZIRAN NA POSTOJEĆOJ ARHITEKTURI

**VAŽNO OTKRIĆE:** Postojeći supabase_client.py već ima kompletnu database integraciju!

### POSTOJEĆA INFRASTRUKTURA:
- ✅ Supabase client s get_supabase_client()
- ✅ Session UUID mapping sistem
- ✅ Database schema: sessions, session_mappings, time_info, zeitschritte, files, csv_file_refs
- ✅ Complex JSONB time_info struktura s category_data
- ✅ File upload i storage sistem
- ✅ API endpoints u training.py

### TREBAŠ DODATI:
1. **Database Tables za rezultate:**
   - `training_results` - glavni rezultati
   - `training_progress` - praćenje statusa
   - `training_logs` - logovi
   - `training_visualizations` - plotovi i grafovi

2. **API endpoints za rezultate:**
   - `/api/training/results/<session_id>`
   - `/api/training/status/<session_id>` 
   - `/api/training/visualizations/<session_id>`

3. **Modifikacija data_loader.py:**
   - Koristiti postojeći supabase_client umjesto kreiranja novog
   - Koristiti postojeće funkcije za dohvaćanje session podataka

4. **Integration s middleman_runner.py:**
   - Zamijeniti subprocess s modularnim pozivima
   - Dodati progress tracking i results persistence

## REVIDIRANE FAZE:

### FAZA 1: Database Extensions (1 dan)
- Kreiraj nove tabele za training rezultate
- Testiraj da rade s postojećim session sistemom

### FAZA 2: Modifikacija postojećih modula (2 dana)
- Ažuriraj data_loader.py da koristi postojeći supabase_client
- Ažuriraj training_api.py s realnim database pozivima
- Integriraj s postojećim session UUID sistemom

### FAZA 3: Middleman Integration (1 dan)  
- Modifikuj middleman_runner.py da koristi training_pipeline
- Dodaj SocketIO progress tracking
- Testiraj kompletni flow

### FAZA 4: Frontend Integration (1 dan)
- Dodaj frontend pozive za dohvaćanje rezultata
- Testiraj end-to-end flow
- Provjeri da sve radi s postojećim Training.tsx

## RIZICI I MITIGATION

### Rizik 1: Hardcoded Paths
**Problem:** training_backend_test_2.py ima hardcoded file paths
**Rješenje:** data_loader.py će dinamički kreirati putanje

### Rizik 2: Circular Dependencies
**Problem:** Moduli mogu imati circular imports
**Rješenje:** Koristiti dependency injection i factory pattern

### Rizik 3: Performance
**Problem:** Cjepkanje može usporavati sistem
**Rješenje:** Lazy loading i optimizovane imports

### Rizik 4: Data Integrity
**Problem:** Podaci se mogu pokvariti tokom cjepkanja
**Rješenje:** Postupno testiranje i validacija na svakom koraku

## ✅ ZAVRŠENO (Status: 2025-01-10)

### FAZA 2: IZVLAČENJE OSNOVNIH KOMPONENTI ✅ KOMPLETNO
- ✅ **config.py** - MTS, T, MDL klase i HOL dictionary (linije 619-692, 798-954, 2046-2141)
- ✅ **data_loader.py** - load() funkcija i data processing (linije 37-168)
- ✅ **data_processor.py** - transf() funkcija i time features (linije 113-141)
- ✅ **model_trainer.py** - Svih 7 ML modela (linije 170-551)
  - train_dense(), train_cnn(), train_lstm(), train_ar_lstm()
  - train_svr_dir(), train_svr_mimo(), train_linear_model()
- ✅ **results_generator.py** - Evaluation funkcije wape(), smape(), mase() (linije 555-608)
- ✅ **visualization.py** - Violin plots i distribution plots (linije 1876-2026)

### TESTIRANJE ✅ KOMPLETNO
Svi ekstraktovani moduli testirani sa 100% pass rate:
- ✅ test_data_loader_real.py (4/4 testova)
- ✅ test_data_processor_real.py (4/4 testova)  
- ✅ test_config_real.py (6/6 testova)
- ✅ test_model_trainer_real.py (6/6 testova)
- ✅ test_results_generator_real.py (6/6 testova)
- ✅ test_visualization_real.py (7/7 testova)

### DEPENDENCY MANAGEMENT ✅ KOMPLETNO
- ✅ TensorFlow instaliran i funkcionalan
- ✅ Seaborn instaliran i funkcionalan
- ✅ Sve dependencies rade sa extracted funkcijama

## 🔄 TRENUTNI FOKUS: BACKEND INTEGRATION ZA FRONTEND READINESS

### FAZA 1: BACKEND INFRASTRUCTURE (KRITICNO ZA FRONTEND)
**Status:** IN PROGRESS
**Cilj:** Pripremiti backend da frontend može da se integriše seamlessly

#### PRIORITET 1A: TrainingPipeline Integration ⚡ COMPLETED ✅
**Fajl:** `training_pipeline.py`
**Status:** COMPLETED 🎉 
**Šta je urađeno:**
- ✅ Extracted moduli testirani (100%)
- ✅ Zameniti placeholder pozive sa realnim funkcijama 
- ✅ Integrisati DataLoader, ModelTrainer, ResultsGenerator, Visualization
- ✅ End-to-end test sa realnim session podacima (test_complete_integration.py)
- ✅ Error handling i progress reporting
- ✅ Pipeline integration module kreiran (pipeline_integration.py)
- ✅ Real functions uspešno zamenjene u TrainingPipeline.run_training_pipeline()

#### PRIORITET 1B: Database Results Tables ⚡ COMPLETED ✅
**Status:** COMPLETED 🎉
**Šta je urađeno:**
- ✅ Kreirane sve potrebne tabele u database_results_schema.sql
- ✅ training_results - session results sa JSONB evaluation_metrics
- ✅ training_visualizations - base64 plots za frontend display  
- ✅ training_logs - detaljni progress logs
- ✅ Proper indexing i RLS policies dodane
- ✅ TrainingPipeline._save_results_to_database() integrisano

#### PRIORITET 1C: Results API Endpoints ⚡ URGENT  
**Fajl:** `training_api.py`
**Status:** EXISTS but NOT TESTED
**Blokira:** Frontend data fetching
**Potrebni endpoints:**
- `GET /api/training/results/{sessionId}` - evaluation metrics
- `GET /api/training/visualizations/{sessionId}` - base64 plots  
- `GET /api/training/status/{sessionId}` - training status
- `GET /api/training/progress/{sessionId}` - real-time progress

### FAZA 2: MIDDLEMAN RUNNER REFACTOR ⚡ URGENT
**Status:** PENDING
**Blokira:** `/api/training/run-analysis` endpoint (frontend koristi ovo!)

#### Korak 2A: Zameniti subprocess sa TrainingPipeline
**Fajl:** `middleman_runner.py`  
**Trenutno:** Poziva subprocess sa training_backend_test_2.py
**Treba:** Poziva TrainingPipeline.run() sa extracted modulima
**Frontend dependency:** Training.tsx klika "Run Analysis" dugme

#### Korak 2B: Real-time SocketIO Integration
**Blokira:** Frontend progress tracking
**Šta treba:**
- Progress events tokom training-a
- Status updates (running → completed → error)
- Result completion notifications

### FAZA 3: E2E TESTING & VALIDATION
**Status:** PENDING
**Cilj:** Garantovati da backend prima frontend podatke i vraća rezultate

#### Korak 3A: Session Data Flow Testing
- Frontend → `/api/training/run-analysis` → TrainingPipeline → Results
- Test sa realnim session podacima iz Training.tsx
- Validirati da results API endpoints vraćaju pravilne podatke

#### Korak 3B: Database Integration Testing  
- Test da se rezultati pravilno čuvaju u nove tabele
- Test da API endpoints čitaju iz baze
- Test session isolation (multiple users)

## 📊 PROGRESS TRACKING (AŽURIRANO)

**Ukupan napredak za Frontend Readiness:** 75% 🚀

### Backend Infrastructure:
- **Core Extraction:** 100% ✅ (Svi moduli izvučeni i testirani)
- **Pipeline Integration:** 100% ✅ (Real functions integrisani u TrainingPipeline)
- **Database Results:** 100% ✅ (Tabele kreiran, save methods implementirani)  
- **Results API:** 20% ⏳ (Exists but untested - NEXT PRIORITY)
- **Middleman Refactor:** 0% ⏳ (NEXT PRIORITY)
- **SocketIO Integration:** 50% ✅ (Progress tracking implementiran, treba testing)

### Frontend Readiness Assessment:
- **Data Input:** 100% ✅ (Training.tsx kompletno)
- **Session Management:** 100% ✅ (UUID sessions, persistence)
- **API Communication:** 100% ✅ (Patterns postoje)
- **Results Display:** 0% ❌ (Čeka backend endpoints)
- **Progress Tracking:** 0% ❌ (Čeka SocketIO)

## 🎯 IMMEDIATE ACTIONS (SLEDEĆIH 24h) - AŽURIRANO

### ✅ ZAVRŠENO DANAS:
1. **✅ TrainingPipeline Integration** - Real extracted functions integrisani
2. **✅ Database Schema** - Sve tabele kreacije i save methods
3. **✅ Complete Testing** - End-to-end integration test passes
4. **✅ Pipeline Integration Module** - Real implementations povezani

### SLEDEĆI KORACI:
1. **Jutro:** Testirati/fiksovati results API endpoints  
2. **Podne:** Refaktor middleman_runner.py da koristi TrainingPipeline
3. **Veče:** E2E test: Frontend → Backend → Results → Display

## 🎯 FRONTEND INTEGRATION TRIGGER

**KADA POČETI FRONTEND:** Kad backend prođe ovaj test:
```
Training.tsx → "Run Analysis" → middleman_runner → TrainingPipeline → 
Real extracted functions → Database results → API endpoints → JSON response
```

**ETA za trigger:** 2-3 dana (ako nema blokera)
**Risk Level:** SREDNJI (Kompleksna integracija, ali komponente testirane)

## 🚨 WORKFLOW RESTRUCTURING PLAN (URGENT)

**PROBLEM IDENTIFIED:** Current backend immediately trains models after "Run Analysis", but should follow proper sequence:

### ORIGINAL INTENDED WORKFLOW:
1. **Upload & Process Data** → Generate datasets and show count
2. **Generate & Display Violin Plots** → Show data distribution visualizations  
3. **User Model Parameter Selection** → Wait for user to choose model types, layers, neurons, epochs
4. **Train Models** → Only after user provides parameters

### CURRENT INCORRECT WORKFLOW:
1. Upload & Process Data
2. ❌ **IMMEDIATELY TRAIN MODELS** (wrong!)

### REQUIRED CHANGES:

#### BACKEND RESTRUCTURING:
1. **Split `/api/training/run-analysis` endpoint:**
   - Current: Processes data + trains models
   - New: Only processes data + generates violin plots
   
2. **Create new `/api/training/generate-datasets/` endpoint:**
   - Generate datasets from processed data
   - Create violin plots using Seaborn
   - Return dataset count and base64 violin plots
   
3. **Modify existing `/api/training/train-models/` endpoint:**
   - Accept user model parameters (Dense/CNN layers, neurons, epochs, activation functions)
   - Train models only after user selection
   
#### FRONTEND CHANGES:
4. **Update Training.tsx workflow:**
   - "Run Analysis" → call `/api/training/generate-datasets/`
   - Display violin plots and dataset count
   - Show model parameter selection form
   - "Train Models" button → call `/api/training/train-models/` with parameters

#### FILES TO MODIFY:
- `middleman_runner.py` - Split analysis and training logic
- `training_pipeline.py` - Separate dataset generation from training
- `training.py` - Add new API endpoints
- `Training.tsx` - Add model parameter form and restructure workflow

### IMPLEMENTATION PRIORITY:
1. **Backend endpoint restructuring** (CRITICAL)
2. **Frontend model parameter form** (HIGH)
3. **Testing complete workflow** (HIGH)

---

Ovaj plan je živ dokument - ažuriram ga kako radiš!