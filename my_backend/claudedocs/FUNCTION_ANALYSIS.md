# 🔍 Detaljna Analiza Funkcija - training.py

**Total Functions**: 54 funkcije
**Total Lines**: 4,338
**HTTP Endpoints**: 37
**Helper Functions**: 17

---

## 📊 Kategorizacija po Tipu

### 🔧 Helper Functions (17) - Prebacuju se u Service Layer

| # | Funkcija | Linija | Destinacija | Kompleksnost |
|---|----------|--------|-------------|--------------|
| 1 | `validate_session_id()` | 36 | `utils/validation.py` | 🟢 Low |
| 2 | `create_error_response()` | 52 | `utils/validation.py` | 🟢 Low |
| 3 | `create_success_response()` | 60 | `utils/validation.py` | 🟢 Low |
| 4 | `extract_file_metadata_fields()` | 70 | `utils/metadata_utils.py` | 🟢 Low |
| 5 | `calculate_n_dat_from_session()` | 104 | `services/training/session_manager.py` | 🟡 Medium |
| 6 | `extract_file_metadata()` | 153 | `utils/metadata_utils.py` | 🟡 Medium |
| 7 | `verify_file_hash()` | 194 | `services/training/upload_manager.py` | 🟢 Low |
| 8 | `assemble_file_locally()` | 203 | `services/training/upload_manager.py` | 🟡 Medium |
| 9 | `save_session_metadata_locally()` | 313 | `services/training/upload_manager.py` | 🟡 Medium |
| 10 | `get_session_metadata_locally()` | 372 | `services/training/upload_manager.py` | 🟡 Medium |
| 11 | `print_session_files()` | 424 | `services/training/upload_manager.py` | 🟢 Low |
| 12 | `update_session_metadata()` | 714 | `services/training/session_manager.py` | 🟡 Medium |
| 13 | `verify_session_files()` | 738 | `services/training/upload_manager.py` | 🟡 Medium |
| 14 | `save_session_to_database()` | 780 | `services/training/session_manager.py` | 🔴 High |
| 15 | `cleanup_incomplete_uploads()` | 3889 | `services/training/upload_manager.py` | 🟡 Medium |

### 🌐 HTTP Endpoints (37) - Ostaju u training.py, pozivaju service

---

## 📦 Kategorizacija po Domenu

---

## 1️⃣ **UPLOAD & FILE MANAGEMENT** (8 funkcija)

### Helper Functions → `services/training/upload_manager.py`

| Funkcija | Linija | Šta Radi | Dependencies | LOC |
|----------|--------|----------|--------------|-----|
| `verify_file_hash()` | 194 | Verifikuje SHA256 hash fajla | hashlib | ~10 |
| `assemble_file_locally()` | 203 | Spaja chunkove u finalni fajl | os, BytesIO | ~110 |
| `save_session_metadata_locally()` | 313 | Sprema metadata u JSON | os, json | ~60 |
| `get_session_metadata_locally()` | 372 | Čita metadata iz JSON | os, json | ~50 |
| `print_session_files()` | 424 | Debug - ispisuje fajlove | logger | ~60 |
| `verify_session_files()` | 738 | Verifikuje integritet fajlova | os, pandas | ~40 |
| `cleanup_incomplete_uploads()` | 3889 | Briše stare incomplete uploade | Path, shutil, time | ~40 |

**Total LOC**: ~370 linija

### HTTP Endpoints → Ostaju u `training.py`

| Endpoint | Linija | Method | Decorators | Poziva Service |
|----------|--------|--------|------------|----------------|
| `/upload-chunk` | 482 | POST | @require_auth, @require_subscription, @check_processing_limit | `save_chunk()` |
| `/finalize-session` | 804 | POST | - | `finalize_upload_session()` |

**Service Functions to Create**:
```python
# services/training/upload_manager.py
- save_chunk(session_id, chunk_data, metadata) -> dict
- finalize_upload_session(session_id) -> dict
```

---

## 2️⃣ **SESSION MANAGEMENT** (17 funkcija)

### Helper Functions → `services/training/session_manager.py`

| Funkcija | Linija | Šta Radi | Dependencies | LOC |
|----------|--------|----------|--------------|-----|
| `calculate_n_dat_from_session()` | 104 | Broji ukupan broj data samplea | pandas, os | ~50 |
| `update_session_metadata()` | 714 | Ažurira session metadata | json, os | ~25 |
| `save_session_to_database()` | 780 | Sprema session u Supabase | supabase_client | ~25 |

**Total LOC**: ~100 linija

### HTTP Endpoints → Ostaju u `training.py`

| Endpoint | Linija | Method | Purpose | Service Call |
|----------|--------|--------|---------|--------------|
| `/list-sessions` | 850 | GET | Lista svih sesija | `get_all_sessions()` |
| `/session/<session_id>` | 1015 | GET | Detalji sesije (local) | `get_session_details()` |
| `/session/<session_id>/database` | 1074 | GET | Detalji sesije (DB) | `get_session_from_db()` |
| `/session-status/<session_id>` | 1165 | GET | Status sesije | `get_session_status()` |
| `/init-session` | 1308 | POST | Inicijalizacija nove sesije | `initialize_new_session()` |
| `/save-time-info` | 1370 | POST | Sprema time info | `save_time_info()` |
| `/create-database-session` | 1424 | POST | Kreira DB session | `create_db_session()` |
| `/get-session-uuid/<session_id>` | 1448 | GET | Dohvata UUID sesije | `get_or_create_uuid()` |
| `/save-zeitschritte` | 1486 | POST | Sprema zeitschritte | `save_zeitschritte()` |
| `/session/<session_id>/delete` | 1540 | POST | Briše sesiju kompletno | `delete_session_completely()` |
| `/get-zeitschritte/<session_id>` | 1702 | GET | Dohvata zeitschritte | `get_zeitschritte()` |
| `/get-time-info/<session_id>` | 1743 | GET | Dohvata time info | `get_time_info()` |
| `/delete-all-sessions` | 3146 | POST | Briše sve sesije | `delete_all_sessions()` |
| `/session-name-change` | 4266 | POST | Mijenja ime sesije | `change_session_name()` |

**Service Functions to Create**:
```python
# services/training/session_manager.py
- get_all_sessions(user_id, limit) -> list
- get_session_details(session_id) -> dict
- get_session_from_db(session_id) -> dict
- get_session_status(session_id) -> dict
- initialize_new_session(session_data) -> dict
- save_time_info(session_id, time_info) -> dict
- create_db_session(session_id) -> dict
- get_or_create_uuid(session_id) -> str
- save_zeitschritte(session_id, zeitschritte) -> dict
- delete_session_completely(session_id) -> dict
- get_zeitschritte(session_id) -> dict
- get_time_info(session_id) -> dict
- delete_all_sessions() -> dict
- change_session_name(session_id, new_name) -> dict
```

---

## 3️⃣ **CSV FILE MANAGEMENT** (4 funkcije)

### HTTP Endpoints → Ostaju u `training.py`

| Endpoint | Linija | Method | Purpose | Service Call |
|----------|--------|--------|---------|--------------|
| `/csv-files/<session_id>` | 1783 | GET | Lista CSV fajlova | `get_csv_files()` |
| `/csv-files` | 1828 | POST | Kreira novi CSV | `create_csv_file()` |
| `/csv-files/<file_id>` | 1908 | PUT | Ažurira CSV | `update_csv_file()` |
| `/csv-files/<file_id>` | 1969 | DELETE | Briše CSV | `delete_csv_file()` |

**Service Functions to Create**:
```python
# services/training/csv_manager.py (NOVI)
- get_csv_files(session_id, file_type) -> list
- create_csv_file(file_data) -> dict
- update_csv_file(file_id, updates) -> dict
- delete_csv_file(file_id) -> dict
```

**Decision**: Kreirati novi `csv_manager.py` ili ostaviti u `training.py`?
**Preporuka**: Ostaviti u `training.py` (mala logika, CRUD operacije)

---

## 4️⃣ **TRAINING RESULTS** (3 funkcije)

### HTTP Endpoints → Ostaju u `training.py`

| Endpoint | Linija | Method | Purpose | Current Logic | LOC |
|----------|--------|--------|---------|---------------|-----|
| `/results/<session_id>` | 2025 | GET | Osnovni rezultati | Direct DB fetch | ~85 |
| `/get-training-results/<session_id>` | 2110 | GET | Detaljni rezultati | DB + Storage fetch | ~10 |
| `/status/<session_id>` | 2560 | GET | Training status | DB fetch sa fallback | ~80 |

**Analysis**: Ove funkcije su relativno jednostavne i samo dohvaćaju podatke iz baze.

**Decision**: **OSTAJU u training.py** - jednostavni GET endpointi

---

## 5️⃣ **VISUALIZATION** (3 funkcije) 🎨

### HTTP Endpoints → Pozivaju `visualization.py`

| Endpoint | Linija | Method | Purpose | Current LOC | Service Function |
|----------|--------|--------|---------|-------------|------------------|
| `/plot-variables/<session_id>` | 2119 | GET | Dohvata dostupne varijable | ~70 | `get_available_variables()` |
| `/visualizations/<session_id>` | 2187 | GET | Lista svih vizualizacija | ~50 | `get_session_visualizations()` |
| `/generate-plot` | 2236 | POST | Generira custom plot | ~320 | `generate_custom_plot()` |

**Total Complex Logic**: ~440 linija → Prebacuje se u service

**Service Functions** (dodati u postojeći `visualization.py`):
```python
# services/training/visualization.py
class Visualizer:
    # Existing methods...

    # NEW:
    def get_available_variables(session_id: str) -> dict
    def get_session_visualizations(session_id: str) -> list
    def generate_custom_plot(session_id: str, plot_config: dict) -> dict
```

---

## 6️⃣ **DATASET GENERATION & TRAINING** (2 funkcije) 🤖

### HTTP Endpoints → Pozivaju postojeće services

| Endpoint | Linija | Method | Purpose | Current LOC | Existing Service |
|----------|--------|--------|---------|-------------|------------------|
| `/generate-datasets/<session_id>` | 2639 | POST | Generira train/val/test datasete | ~170 | `middleman_runner.py` |
| `/train-models/<session_id>` | 2810 | POST | Trenira ML modele | ~335 | `middleman_runner.py` |

**Analysis**:
- `generate_datasets`: Poziva `ModernMiddlemanRunner.run_training_script()` - VEĆ POSTOJI
- `train_models`: Kompleksna logika, pokreće training u background thread

**Service Functions** (kreirati u `data_processor.py` ili `middleman_runner.py`):
```python
# services/training/data_processor.py ili middleman_runner.py
- generate_training_datasets(session_id, config) -> dict  # NOVO ako treba
- start_training_job(session_id, model_config) -> dict    # Wrapper oko existinga
```

**Decision**: Većina logike VEĆ POSTOJI u `middleman_runner.py`.
Samo treba refaktorirati endpoint da poziva service umjesto direktno izvršavati logiku.

---

## 7️⃣ **EVALUATION TABLES** (2 funkcije) 📊

### HTTP Endpoints → Ostaju u `training.py`

| Endpoint | Linija | Method | Purpose | Current LOC | Service? |
|----------|--------|--------|---------|-------------|----------|
| `/evaluation-tables/<session_id>` | 3369 | GET | Dohvata evaluation tablice | ~165 | Možda |
| `/save-evaluation-tables/<session_id>` | 3536 | POST | Sprema evaluation tablice | ~70 | Možda |

**Analysis**:
- GET: Kompleksna logika - rekonstrukcija podataka iz training results
- POST: Srednje kompleksna - transformacija i spremanje

**Service Functions to Consider**:
```python
# services/training/results_generator.py (već postoji!)
- get_evaluation_tables(session_id) -> dict
- save_evaluation_tables(session_id, tables) -> dict
```

**Decision**: Prebaciti biznis logiku u postojeći `results_generator.py`

---

## 8️⃣ **SCALER MANAGEMENT** (3 funkcije) ⚖️

### HTTP Endpoints → Pozivaju `scaler_manager.py`

| Endpoint | Linija | Method | Purpose | Current LOC | Service Function |
|----------|--------|--------|---------|-------------|------------------|
| `/scalers/<session_id>` | 3606 | GET | Dohvata scaler info | ~55 | `get_session_scalers()` |
| `/scalers/<session_id>/download` | 3662 | GET | Download scalers kao .pkl | ~75 | `create_scaler_download_package()` |
| `/scale-data/<session_id>` | 3736 | POST | Skalira nove podatke | ~190 | `scale_new_data()` |

**Total Complex Logic**: ~320 linija → Prebacuje se u service

**Service Functions** (dodati u postojeći `scaler_manager.py`):
```python
# services/training/scaler_manager.py
# Existing: create_scaling_lists, fit_scalers, apply_scaling, process_and_scale_data

# NEW:
def get_session_scalers(session_id: str) -> dict
def create_scaler_download_package(session_id: str) -> bytes
def scale_new_data(session_id: str, input_data: dict) -> dict
```

---

## 9️⃣ **MODEL MANAGEMENT** (3 funkcije) 🏗️

### HTTP Endpoints → Kreirati `model_manager.py`

| Endpoint | Linija | Method | Purpose | Current LOC | Service Function |
|----------|--------|--------|---------|-------------|------------------|
| `/save-model/<session_id>` | 3929 | POST | Sprema modele u Storage | ~190 | `save_models_to_storage()` |
| `/list-models-database/<session_id>` | 4120 | GET | Lista svih modela | ~50 | `list_available_models()` |
| `/download-model-h5/<session_id>` | 4170 | GET | Download model kao H5/PKL | ~95 | `download_model_file()` |

**Total Complex Logic**: ~335 linija → Kreirati novi service

**Service Functions** (kreirati NOVI `model_manager.py`):
```python
# services/training/model_manager.py (NOVI FAJL)
def save_models_to_storage(session_id: str) -> dict
def list_available_models(session_id: str) -> list
def download_model_file(session_id: str, model_type: str) -> BytesIO
def _extract_models_from_results(results: dict) -> list  # Helper
```

---

## 📊 Summary Statistics

### Funkcije po Destinaciji:

| Destinacija | Broj Funkcija | Ukupno LOC | Status |
|-------------|---------------|------------|--------|
| **utils/validation.py** | 3 | ~50 | ✅ Kreirati |
| **utils/metadata_utils.py** | 2 | ~100 | ✅ Kreirati |
| **services/training/upload_manager.py** | 7 | ~370 | ✅ Kreirati |
| **services/training/session_manager.py** | 17 | ~600 | ✅ Kreirati |
| **services/training/visualization.py** | 3 | ~440 | ⚡ Proširiti postojeći |
| **services/training/scaler_manager.py** | 3 | ~320 | ⚡ Proširiti postojeći |
| **services/training/model_manager.py** | 3 | ~335 | ✅ Kreirati |
| **services/training/results_generator.py** | 2 | ~235 | ⚡ Proširiti postojeći |
| **services/training/csv_manager.py** | 4 | ~150 | ❓ Optional |
| **Ostaje u training.py** | 10 | ~175 | 🟢 Jednostavni GET |

**TOTAL**: 54 funkcije | ~2,775 LOC prebačeno u services

**training.py nakon refaktoriranja**: ~1,563 LOC (samo HTTP routing)

---

## 🎯 Prioritizacija za Refaktoring

### **Tier 1: Brzo i Jednostavno** (Faza 0)
1. ✅ `utils/validation.py` - 3 funkcije, 50 LOC
2. ✅ `utils/metadata_utils.py` - 2 funkcije, 100 LOC

**Benefit**: Odmah smanjuje training.py za 150 linija
**Rizik**: Vrlo nizak
**Trajanje**: 30 min

---

### **Tier 2: Proof of Concept** (Faza 1-2)
3. ⚡ Proširi `visualization.py` - 3 funkcije, 440 LOC
4. ⚡ Proširi `scaler_manager.py` - 3 funkcije, 320 LOC

**Benefit**: Smanjenje za dodatnih 760 linija, proof-of-concept service pattern
**Rizik**: Nizak (proširivanje postojećih modula)
**Trajanje**: 6-8 sati

---

### **Tier 3: Novi Service Moduli** (Faza 3-5)
5. ✅ Kreiraj `model_manager.py` - 3 funkcije, 335 LOC
6. ✅ Kreiraj `session_manager.py` - 17 funkcija, 600 LOC
7. ✅ Kreiraj `upload_manager.py` - 7 funkcije, 370 LOC

**Benefit**: Kompletna separacija odgovornosti
**Rizik**: Srednji (nove dependencies, testiranje)
**Trajanje**: 12-15 sati

---

### **Tier 4: Optional** (Faza 6)
8. ⚡ Proširi `results_generator.py` - 2 funkcije, 235 LOC
9. ❓ Kreiraj `csv_manager.py` - 4 funkcije, 150 LOC (OPTIONAL)

**Benefit**: Kompletna organizacija
**Rizik**: Nizak
**Trajanje**: 4-6 sati

---

## 🔍 Dependency Graph

```
training.py (HTTP Layer)
    │
    ├── utils/validation.py
    │   ├── validate_session_id()
    │   ├── create_error_response()
    │   └── create_success_response()
    │
    ├── utils/metadata_utils.py
    │   ├── extract_file_metadata_fields()
    │   └── extract_file_metadata()
    │
    ├── services/training/upload_manager.py
    │   ├── verify_file_hash()
    │   ├── save_chunk()
    │   ├── assemble_file_locally()
    │   ├── finalize_upload_session()
    │   └── cleanup_incomplete_uploads()
    │
    ├── services/training/session_manager.py
    │   ├── initialize_new_session()
    │   ├── get_all_sessions()
    │   ├── get_session_details()
    │   ├── delete_session_completely()
    │   └── ... (14 more)
    │
    ├── services/training/visualization.py
    │   ├── get_available_variables()
    │   ├── get_session_visualizations()
    │   └── generate_custom_plot()
    │
    ├── services/training/scaler_manager.py
    │   ├── get_session_scalers()
    │   ├── create_scaler_download_package()
    │   └── scale_new_data()
    │
    ├── services/training/model_manager.py
    │   ├── save_models_to_storage()
    │   ├── list_available_models()
    │   └── download_model_file()
    │
    └── services/training/results_generator.py
        ├── get_evaluation_tables()
        └── save_evaluation_tables()
```

---

## 📝 Zaključci i Preporuke

### ✅ Što Raditi:

1. **Počni sa Tier 1** (utils moduli) - brzo, sigurno, odmah pokazuje napredak
2. **Nastavi sa Tier 2** (proširivanje postojećih) - proof-of-concept, vidiš da pattern radi
3. **Kreiraj Tier 3** (novi moduli) - kompletna transformacija
4. **Optional Tier 4** - ako imaš vremena

### ⚠️ Što NE Raditi:

1. **NE miksati faze** - završi jednu pa kreni drugu
2. **NE skip testove** - testiraj nakon svake faze
3. **NE brisati stari kod odmah** - komentiraj pa briši tek nakon što sve radi
4. **NE raditi sve u jednom commit** - jedan checkpoint po fazi

### 🎯 Expected Outcome:

**PRIJE**:
```
training.py: 4,338 linija - KAOS
```

**POSLIJE**:
```
training.py: ~1,560 linija (samo HTTP routing)
+ 8 dobro organiziranih service modula
+ Jasna separacija odgovornosti
+ Lako za testiranje
+ Lako za održavanje
```

---

**ANALIZA KOMPLETNA! Spreman za Fazu 0! 🚀**
