# Backend Tests

Comprehensive test suite za load_data.py refaktoring.

## Setup

### 1. Instalacija Dependencies

```bash
# Koristi virtual environment (preporučeno)
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# ili
venv\Scripts\activate  # Windows

# Instaliraj test dependencies
pip install -r requirements-test.txt

# Instaliraj main dependencies
pip install -r requirements.txt
```

### 2. Pokretanje Testova

```bash
# Svi testovi
pytest

# Sa verbose output
pytest -v

# Specifičan fajl
pytest my_backend/tests/test_setup.py

# Sa coverage reportom
pytest --cov=my_backend --cov-report=html

# Samo unit testovi
pytest -m unit

# Samo integration testovi
pytest -m integration
```

### 3. Coverage Report

```bash
# Generiši HTML coverage report
pytest --cov=my_backend --cov-report=html

# Otvori report
open htmlcov/index.html  # Mac
xdg-open htmlcov/index.html  # Linux
start htmlcov/index.html  # Windows
```

## Test Struktura

```
my_backend/tests/
├── conftest.py              # Shared fixtures
├── test_setup.py            # Framework validation tests
├── fixtures/                # Test data files
│   ├── sample_data.csv
│   ├── sample_data_german.csv
│   ├── sample_data_semicolon.csv
│   └── sample_data_separate_datetime.csv
├── api/
│   └── routes/
│       └── test_load_data.py     # (TODO: Route tests)
├── core/
│   └── state/
│       └── test_upload_manager.py # (TODO: State manager tests)
└── utils/
    └── test_datetime_parser.py   # (TODO: Parser tests)
```

## Fixtures

### Flask Fixtures
- `app` - Flask app instance
- `client` - Flask test client
- `mock_socketio` - Mocked SocketIO
- `mock_auth` - Bypasses auth
- `mock_subscription` - Bypasses subscription checks

### Data Fixtures
- `sample_csv_content` - Basic CSV string
- `sample_csv_semicolon` - Semicolon delimited
- `sample_csv_german_format` - German date format
- `sample_csv_separate_datetime` - Date/time in separate columns
- `sample_dataframe` - Pandas DataFrame
- `supported_date_formats` - List of 19 supported formats

### Upload Fixtures
- `upload_chunk_data` - Sample chunk upload data
- `temp_csv_file` - Temporary CSV file

## Progress

### ✅ FAZA 1 DAN 1: Test Framework Setup
- [x] Test directories created
- [x] pytest.ini configured
- [x] conftest.py with fixtures
- [x] Sample test data files
- [x] Setup validation tests

### 🔄 FAZA 1 DAN 2-3: Helper Function Tests
- [ ] Date/time utility tests
- [ ] CSV parsing tests
- [ ] Validation tests

### ⏸️ FAZA 1 DAN 4-5: Route Handler Tests
- [ ] /upload-chunk tests
- [ ] /finalize-upload tests
- [ ] /cancel-upload tests
- [ ] Integration tests

## Tips

### Debugging Failed Tests
```bash
# Stop at first failure
pytest -x

# Show local variables on failure
pytest -l

# Enter debugger on failure
pytest --pdb
```

### Watch Mode
```bash
# Install pytest-watch
pip install pytest-watch

# Auto-run tests on file changes
ptw
```

### Parallel Execution
```bash
# Install pytest-xdist
pip install pytest-xdist

# Run tests in parallel
pytest -n auto
```

## Next Steps

1. **Dan 2:** Implementirati test_helpers.py
   - test_clean_time()
   - test_detect_delimiter()
   - test_clean_file_content()
   - test_check_date_format()
   
2. **Dan 3:** Implementirati test_parsing.py
   - test_parse_datetime_column()
   - test_parse_datetime()
   - test_convert_to_utc()

3. **Dan 4-5:** Implementirati test_routes.py
   - Route handler tests
   - Integration tests

## Troubleshooting

### Import Errors
```python
# Ako imaš import errors, dodaj path u conftest.py:
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
```

### Coverage Not Working
```bash
# Provjeri da li je .coveragerc konfiguran
# Ili koristi --cov-config=pytest.ini
```

### Slow Tests
```bash
# Mark slow tests
@pytest.mark.slow
def test_expensive_operation():
    pass

# Skip slow tests
pytest -m "not slow"
```
