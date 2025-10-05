# Cloud.py Test Suite

Kompletan test suite za `api/routes/cloud.py` endpoint-e.

## 📁 Struktura

```
tests/
├── conftest.py                      # Pytest konfiguracija i fixtures
├── test_cloud.py                    # Glavni test fajl
├── fixtures/                        # Test data
│   ├── sample_temperature.csv
│   ├── sample_load.csv
│   └── sample_load_with_gaps.csv
└── README.md                        # Ova dokumentacija
```

## 🧪 Vrste testova

### 1. **Unit testovi** (funkcije)
- ✅ `calculate_bounds()` - Testiranje tolerancija
- ✅ `interpolate_data()` - Testiranje interpolacije
- ✅ Constant vs Dependent tolerance
- ✅ Negativne vrednosti i edge cases

### 2. **Integration testovi** (endpointi)
- ✅ `POST /upload-chunk` - Chunk upload flow
- ✅ `POST /complete` - Kompletiranje chunked upload-a
- ✅ `POST /clouddata` - Direktan upload i procesiranje
- ✅ `POST /interpolate-chunked` - Interpolacija sa streaming-om
- ✅ `POST /prepare-save` - Priprema fajla za download
- ✅ `GET /download/<file_id>` - Download fajla

### 3. **Edge case testovi**
- ✅ Duplicate timestamps
- ✅ No matching timestamps
- ✅ NaN vrednosti
- ✅ Tolerancije van opsega
- ✅ Empty data after cleaning
- ✅ Invalid chunk indices

### 4. **Performance testovi**
- ✅ Veliki fajlovi (10000+ rows)
- ✅ Streaming response (20000+ rows)
- ✅ Memory usage during chunk processing

## 🚀 Pokretanje testova

### Svi testovi
```bash
pytest
```

### Samo unit testovi
```bash
pytest -m unit
```

### Samo integration testovi
```bash
pytest -m integration
```

### Bez slow testova
```bash
pytest -m "not slow"
```

### Sa coverage report-om
```bash
pytest --cov=api.routes.cloud --cov-report=html
```

### Verbose output
```bash
pytest -v
```

## 🐳 Docker testiranje

### Build Docker image
```bash
docker build -t my_backend -f Dockerfile.optimized .
```

### Run testovi u Docker-u
```bash
docker run --rm my_backend pytest
```

### Run sa coverage u Docker-u
```bash
docker run --rm my_backend pytest --cov=api.routes.cloud --cov-report=term-missing
```

## 📊 Coverage cilj

**Cilj: >85% code coverage**

Trenutno pokriveni endpointi:
- ✅ `/upload-chunk` 
- ✅ `/complete`
- ✅ `/clouddata`
- ✅ `/interpolate-chunked`
- ✅ `/prepare-save`
- ✅ `/download/<file_id>`

Pokrivene funkcije:
- ✅ `calculate_bounds()`
- ✅ `interpolate_data()`
- ✅ `_process_data_frames()`
- ✅ `get_chunk_dir()`

## 🔧 Instalacija dependencies

```bash
pip install -r requirements.txt
```

Potrebni paketi za testiranje:
- `pytest==8.3.4`
- `pytest-cov==6.0.0`
- `pytest-flask==1.3.0`
- `pytest-mock==3.14.0`

## 📝 Pisanje novih testova

### Test pattern
```python
class TestNewFeature:
    """Test description"""
    
    def test_success_case(self, client):
        """Test successful operation"""
        response = client.post('/endpoint', json={...})
        assert response.status_code == 200
        assert response.get_json()['success'] is True
    
    def test_error_case(self, client):
        """Test error handling"""
        response = client.post('/endpoint', json={...})
        assert response.status_code == 400
        assert 'error' in response.get_json()['data']
```

### Korišćenje fixtures
```python
def test_with_fixtures(client, sample_temp_df, upload_id):
    """Test koristi postojeće fixtures"""
    # sample_temp_df i upload_id su automatski dostupni
    pass
```

## 🐛 Debug testova

### Run sa pdb debugger-om
```bash
pytest --pdb
```

### Show print statements
```bash
pytest -s
```

### Run single test
```bash
pytest tests/test_cloud.py::TestUploadChunkEndpoint::test_valid_chunk_upload
```

## ⚠️ Poznati problemi

1. **Streaming responses** - Potreban custom parser za NDJSON
2. **Chunk cleanup** - Autocleanup može failovati na Windows-u
3. **Large files** - Performance testovi su označeni sa `@pytest.mark.slow`

## 📈 CI/CD Integration

### GitHub Actions example
```yaml
- name: Run tests
  run: |
    pip install -r requirements.txt
    pytest --cov=api.routes.cloud --cov-report=xml
    
- name: Upload coverage
  uses: codecov/codecov-action@v3
  with:
    file: ./coverage.xml
```

## 🎯 Sledeci koraci

- [ ] Dodati parametrizovane testove za različite REG/TR kombinacije
- [ ] E2E testovi sa pravim frontend pozivima
- [ ] Load testing sa locust/k6
- [ ] Security testovi (SQL injection, XSS)
- [ ] Mutation testing sa mutpy
