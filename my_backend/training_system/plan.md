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

## NEXT STEPS

1. **Odgovori na pitanja** gore
2. **Kreiraj database schema** ako treba
3. **Počni sa Fazom 1** - API endpoints
4. **Testiraj svaki korak** prije prelaska na sljedeći
5. **Dokumentuj sve promjene** u ovom fajlu

Ovaj plan je živ dokument - mjenjaj ga kako radiš!