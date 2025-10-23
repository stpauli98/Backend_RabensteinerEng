# 📋 Refactoring Checklist - Training Module

Koristi ovaj checklist da pratiš napredak tokom refaktoriranja.

---

## ✅ PRE-REFACTORING (Obavezno!)

- [ ] **Git branch kreiran**: `refactor/training-module-split`
- [ ] **Backup kreiran**: `training.py.BACKUP_YYYYMMDD_HHMMSS`
- [ ] **Initial commit**: Pre-refactoring checkpoint
- [ ] **Backend radi**: Docker containers up and running
- [ ] **Health check pass**: `curl http://localhost:8080/health`
- [ ] **Baseline test**: Svi ključni endpointi rade

---

## FAZA 0: Priprema (30 min)

### Korak 0.1: Utils - Validation
- [ ] Kreiran `utils/validation.py`
- [ ] Funkcija `validate_session_id()` prebačena
- [ ] Funkcija `create_error_response()` prebačena
- [ ] Funkcija `create_success_response()` prebačena
- [ ] Test import: `from utils.validation import ...` radi

### Korak 0.2: Utils - Metadata
- [ ] Kreiran `utils/metadata_utils.py`
- [ ] Funkcija `extract_file_metadata_fields()` prebačena
- [ ] Funkcija `extract_file_metadata()` prebačena
- [ ] Test import radi

### Korak 0.3: Update training.py
- [ ] Import statements dodani
- [ ] Stare funkcije zakomentarisane (NE izbrisane!)
- [ ] Backend se pokreće bez grešaka

### ✅ Checkpoint 0
- [ ] Backend restart: `docker-compose restart backend`
- [ ] Test `/health` endpoint
- [ ] Test `/list-sessions` endpoint
- [ ] **Git commit**: "Phase 0: Extract validation and metadata utils"

---

## FAZA 1: Visualizations (3-4h)

### Korak 1.1: Analiza
- [ ] Pronađene sve visualization funkcije
- [ ] Identificirana biznis logika za ekstrakciju

### Korak 1.2: Service Layer
- [ ] `services/training/visualization.py` otvoren
- [ ] Metoda `get_available_variables()` dodana
- [ ] Metoda `get_session_visualizations()` dodana
- [ ] Metoda `generate_custom_plot()` dodana
- [ ] Helper metode `_extract_variable_data()` i `_save_visualization_to_db()` dodane
- [ ] Import test pass

### Korak 1.3: HTTP Layer
- [ ] Endpoint `get_plot_variables()` refaktoriran
- [ ] Endpoint `get_training_visualizations()` refaktoriran
- [ ] Endpoint `generate_plot()` refaktoriran
- [ ] Stari kod zakomentarisan

### ✅ Checkpoint 1
- [ ] Backend restart
- [ ] Test `GET /plot-variables/<session_id>`
- [ ] Test `GET /visualizations/<session_id>`
- [ ] Test `POST /generate-plot`
- [ ] **Git commit**: "Phase 1: Refactor visualization functions to service layer"

---

## FAZA 2: Scalers (3-4h)

### Korak 2.1: Service Layer
- [ ] `services/training/scaler_manager.py` otvoren
- [ ] Funkcija `get_session_scalers()` dodana
- [ ] Funkcija `create_scaler_download_package()` dodana
- [ ] Funkcija `scale_new_data()` dodana

### Korak 2.2: HTTP Layer
- [ ] Endpoint `get_scalers()` refaktoriran
- [ ] Endpoint `download_scalers_as_save_files()` refaktoriran
- [ ] Endpoint `scale_input_data()` refaktoriran

### ✅ Checkpoint 2
- [ ] Backend restart
- [ ] Test `GET /scalers/<session_id>`
- [ ] Test `GET /scalers/<session_id>/download`
- [ ] Test `POST /scale-data/<session_id>`
- [ ] **Git commit**: "Phase 2: Refactor scaler functions to service layer"

---

## FAZA 3: Model Management (4-5h)

### Korak 3.1: Novi Service Module
- [ ] **Kreiran** `services/training/model_manager.py`
- [ ] Funkcija `save_models_to_storage()` implementirana
- [ ] Funkcija `list_available_models()` implementirana
- [ ] Funkcija `download_model_file()` implementirana
- [ ] Helper `_extract_models_from_results()` implementiran

### Korak 3.2: HTTP Layer
- [ ] Endpoint `save_model()` refaktoriran
- [ ] Endpoint `list_models_database()` refaktoriran
- [ ] Endpoint `download_model_h5()` refaktoriran

### ✅ Checkpoint 3
- [ ] Backend restart
- [ ] Test `POST /save-model/<session_id>`
- [ ] Test `GET /list-models-database/<session_id>`
- [ ] Test `GET /download-model-h5/<session_id>`
- [ ] **Git commit**: "Phase 3: Add model management service layer"

---

## FAZA 4: Session Management (5-6h)

### Korak 4.1: Novi Service Module
- [ ] **Kreiran** `services/training/session_manager.py`
- [ ] Funkcija `initialize_new_session()` implementirana
- [ ] Funkcija `get_all_sessions()` implementirana
- [ ] Funkcija `get_session_details()` implementirana
- [ ] Funkcija `delete_session_completely()` implementirana
- [ ] Funkcija `calculate_session_n_dat()` implementirana

### Korak 4.2: HTTP Layer
- [ ] Endpoint `init_session()` refaktoriran
- [ ] Endpoint `list_sessions()` refaktoriran
- [ ] Endpoint `get_session()` refaktoriran
- [ ] Endpoint `delete_session()` refaktoriran

### ✅ Checkpoint 4
- [ ] Backend restart
- [ ] Test `POST /init-session`
- [ ] Test `GET /list-sessions`
- [ ] Test `GET /session/<session_id>`
- [ ] Test `POST /session/<session_id>/delete`
- [ ] **Git commit**: "Phase 4: Add session management service layer"

---

## FAZA 5: Upload Management (4-5h)

### Korak 5.1: Novi Service Module
- [ ] **Kreiran** `services/training/upload_manager.py`
- [ ] Funkcija `verify_file_hash()` implementirana
- [ ] Funkcija `save_chunk()` implementirana
- [ ] Funkcija `assemble_chunks()` implementirana
- [ ] Funkcija `finalize_upload_session()` implementirana
- [ ] Funkcija `cleanup_old_uploads()` implementirana

### Korak 5.2: HTTP Layer
- [ ] Endpoint `upload_chunk()` refaktoriran
- [ ] Endpoint `finalize_session()` refaktoriran

### ✅ Checkpoint 5
- [ ] Backend restart
- [ ] Test `POST /upload-chunk`
- [ ] Test `POST /finalize-session`
- [ ] Test cleanup funkcionalnost
- [ ] **Git commit**: "Phase 5: Add upload management service layer"

---

## FAZA 6: Final Testing & Validation (1 dan)

### Korak 6.1: Kompletno Testiranje
- [ ] Test script `test_all_endpoints.sh` kreiran
- [ ] Svi endpointi testirani
- [ ] Edge cases testirani
- [ ] Error handling validiran

### Korak 6.2: Code Quality Review
- [ ] **Code Quality Checklist:**
  - [ ] No commented-out code
  - [ ] Consistent naming conventions
  - [ ] Proper error handling
  - [ ] Appropriate logging
  - [ ] No hardcoded values

- [ ] **Architecture Checklist:**
  - [ ] HTTP layer samo poziva service
  - [ ] Service layer bez Flask dependencies
  - [ ] Clear separation of concerns
  - [ ] No circular dependencies

- [ ] **Security Checklist:**
  - [ ] Input validation na svim endpointima
  - [ ] Proper error messages
  - [ ] Authentication decorators
  - [ ] No security vulnerabilities

### Korak 6.3: Final Cleanup
- [ ] Sve zakomentarisane funkcije izbrisane
- [ ] Imports organizovani
- [ ] Docstrings dodani gdje fale
- [ ] Line length check (<100 chars)

### Korak 6.4: Dokumentacija
- [ ] `REFACTORING_COMPLETE.md` kreiran
- [ ] Architecture diagram update
- [ ] API documentation update

### ✅ Final Checkpoint
- [ ] `pytest tests/ -v` pass
- [ ] `flake8` pass
- [ ] Test coverage report generated
- [ ] **Git commit**: "Refactoring complete: training.py modularized"
- [ ] Pull Request kreiran
- [ ] Code review completed
- [ ] Merged to main

---

## 📊 Metrički Napredak

Popuni nakon svake faze:

| Faza | Lines Extracted | training.py Size | Status |
|------|----------------|------------------|--------|
| 0 - Utils | ~150 | ~4,200 | ⬜ |
| 1 - Visualizations | ~300 | ~3,900 | ⬜ |
| 2 - Scalers | ~350 | ~3,550 | ⬜ |
| 3 - Models | ~400 | ~3,150 | ⬜ |
| 4 - Sessions | ~500 | ~2,650 | ⬜ |
| 5 - Uploads | ~450 | ~2,200 | ⬜ |
| 6 - Final Cleanup | ~1,400 | ~800 | ⬜ |

**Ciljni rezultat**: training.py < 1,000 linija

---

## ⚠️ Troubleshooting

### Ako backend ne startuje:
```bash
# Check logs
docker-compose logs backend

# Check imports
python3 -c "from api.routes.training import bp"

# Rollback to last checkpoint
git log --oneline -5
git checkout <last-good-commit>
```

### Ako endpoint ne radi:
```bash
# Check endpoint registration
grep -r "register_blueprint" api/routes/

# Check route definition
grep -A 5 "@bp.route('/endpoint')" api/routes/training.py

# Check service import
grep "from services.training" api/routes/training.py
```

### Ako test fails:
```bash
# Detailed error
pytest tests/test_training.py::test_name -vv

# Check database connection
python3 -c "from utils.database import get_supabase_client; print(get_supabase_client())"

# Check file permissions
ls -la api/routes/uploads/file_uploads/
```

---

## 🎯 Definicija "Done"

Faza je kompletna kada:
- ✅ Kod je refaktoriran prema planu
- ✅ Svi testovi prolaze
- ✅ Backend se pokreće bez grešaka
- ✅ Endpointi vraćaju očekivane odgovore
- ✅ Commit kreiran sa jasnom porukom
- ✅ Checkpoint dokumentiran

---

**Uspješan refactoring! 🎉**
