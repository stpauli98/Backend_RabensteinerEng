# 🎯 Detaljni Plan Refaktoriranja - training.py

**Datum početka**: 2025-10-23
**Trenutno stanje**: training.py - 4,338 linija
**Ciljno stanje**: Modulariziran sistem sa jasnom separacijom odgovornosti

---

## 📋 Preduvjeti i Priprema

### 1. Backup i Version Control
```bash
# 1. Kreiraj feature branch
git checkout -b refactor/training-module-split

# 2. Backup trenutnog fajla
cp api/routes/training.py api/routes/training.py.BACKUP_$(date +%Y%m%d_%H%M%S)

# 3. Commit trenutnog stanja
git add .
git commit -m "Pre-refactoring checkpoint: training.py backup"
```

### 2. Test Setup
```bash
# Pokreni postojeće testove da vidiš baseline
pytest tests/ -v

# Pokreni backend
docker-compose up -d

# Testiraj ključne endpointe ručno
curl http://localhost:8080/health
curl http://localhost:8080/api/training/list-sessions
```

---

## 🗺️ Mapa Funkcija - Kategorizacija

### Trenutna Struktura (api/routes/training.py):

#### **Kategorija A: Validation & Response (Lines 36-102)**
- `validate_session_id()` → utils/validation.py
- `create_error_response()` → utils/validation.py
- `create_success_response()` → utils/validation.py
- `extract_file_metadata_fields()` → utils/metadata_utils.py
- `calculate_n_dat_from_session()` → services/training/session_manager.py
- `extract_file_metadata()` → utils/metadata_utils.py

#### **Kategorija B: File Operations (Lines 194-481)**
- `verify_file_hash()` → services/training/upload_manager.py
- `assemble_file_locally()` → services/training/upload_manager.py
- `save_session_metadata_locally()` → services/training/upload_manager.py
- `get_session_metadata_locally()` → services/training/upload_manager.py
- `print_session_files()` → services/training/upload_manager.py
- `update_session_metadata()` → services/training/upload_manager.py
- `verify_session_files()` → services/training/upload_manager.py
- `save_session_to_database()` → services/training/upload_manager.py

#### **Kategorija C: HTTP Endpoints - Uploads (Lines 482-803)**
- `upload_chunk()` → OSTAJE u training.py (HTTP layer)
- `finalize_session()` → OSTAJE u training.py (HTTP layer)

#### **Kategorija D: HTTP Endpoints - Sessions (Lines 850-1743)**
- `list_sessions()` → OSTAJE u training.py
- `get_session()` → OSTAJE u training.py
- `get_session_from_database()` → services/training/session_manager.py
- `session_status()` → OSTAJE u training.py
- `init_session()` → OSTAJE u training.py
- `save_time_info_endpoint()` → OSTAJE u training.py
- `create_database_session()` → OSTAJE u training.py
- `get_session_uuid()` → OSTAJE u training.py
- `save_zeitschritte_endpoint()` → OSTAJE u training.py
- `delete_session()` → OSTAJE u training.py
- `get_zeitschritte()` → OSTAJE u training.py
- `get_time_info()` → OSTAJE u training.py

#### **Kategorija E: CSV Files (Lines 1783-2024)**
- `get_csv_files()` → OSTAJE u training.py
- `create_csv_file()` → OSTAJE u training.py
- `update_csv_file()` → OSTAJE u training.py
- `delete_csv_file()` → OSTAJE u training.py

#### **Kategorija F: Training Results (Lines 2025-2559)**
- `get_training_results()` → OSTAJE u training.py
- `get_training_results_details()` → OSTAJE u training.py
- `get_plot_variables()` → OSTAJE u training.py (poziva service)
- `get_training_visualizations()` → OSTAJE u training.py (poziva service)
- `generate_plot()` → OSTAJE u training.py (poziva service)
- `get_training_status()` → OSTAJE u training.py

#### **Kategorija G: Dataset & Training (Lines 2639-3145)**
- `generate_datasets()` → OSTAJE u training.py (poziva service)
- `train_models()` → OSTAJE u training.py (poziva service)
- `delete_all_sessions()` → OSTAJE u training.py

#### **Kategorija H: Evaluation Tables (Lines 3369-3605)**
- `get_evaluation_tables()` → OSTAJE u training.py (poziva service)
- `save_evaluation_tables()` → OSTAJE u training.py (poziva service)

#### **Kategorija I: Scalers (Lines 3606-3928)**
- `get_scalers()` → OSTAJE u training.py (poziva service)
- `download_scalers_as_save_files()` → OSTAJE u training.py (poziva service)
- `scale_input_data()` → OSTAJE u training.py (poziva service)

#### **Kategorija J: Cleanup (Lines 3889-3926)**
- `cleanup_incomplete_uploads()` → services/training/upload_manager.py

#### **Kategorija K: Model Management (Lines 3929-4338)**
- `save_model()` → OSTAJE u training.py (poziva service)
- `list_models_database()` → OSTAJE u training.py (poziva service)
- `download_model_h5()` → OSTAJE u training.py (poziva service)
- `change_session_name()` → OSTAJE u training.py (poziva service)

---

## 🎯 Strategija Refaktoriranja

### Princip: **Service Layer Pattern**

```
┌─────────────────────────────────────┐
│   HTTP Layer (api/routes/)          │
│   - Request validation               │
│   - Response formatting              │
│   - Authentication/Authorization     │
│   - HTTP error handling              │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│   Service Layer (services/training/) │
│   - Business logic                   │
│   - Data transformations             │
│   - Complex operations               │
│   - No HTTP dependencies             │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│   Data Layer (utils/)                │
│   - Database operations              │
│   - File I/O                         │
│   - External API calls               │
└─────────────────────────────────────┘
```

### Pravila:
1. **HTTP layer samo poziva service layer**
2. **Service layer NE zna za Flask/request/response**
3. **Svaki service vraća dict ili raises exception**
4. **HTTP layer hendla sve exceptione i vraća odgovarajuće HTTP statuse**

---

## 📦 Faza po Faza - Detaljni Plan

---

## **FAZA 0: Priprema (30 minuta)**

### Korak 0.1: Kreiraj utility module
```python
# utils/validation.py
def validate_session_id(session_id):
    """Validate session ID format"""
    # Prebaci iz training.py:36-50

def create_error_response(message, status_code=400):
    """Create standardized error response"""
    # Prebaci iz training.py:52-58

def create_success_response(data=None, message=None):
    """Create standardized success response"""
    # Prebaci iz training.py:60-68
```

### Korak 0.2: Kreiraj metadata utils
```python
# utils/metadata_utils.py
def extract_file_metadata_fields(file_metadata):
    """Extract standardized file metadata fields"""
    # Prebaci iz training.py:70-102

def extract_file_metadata(session_id):
    """Extract file metadata from session"""
    # Prebaci iz training.py:153-186
```

### Korak 0.3: Test
```bash
# Pokreni Python shell i testiraj import
python3 << EOF
from utils.validation import validate_session_id, create_error_response
from utils.metadata_utils import extract_file_metadata_fields

# Test validation
assert validate_session_id("550e8400-e29b-41d4-a716-446655440000") == True
assert validate_session_id("session_123_abc") == True
assert validate_session_id("invalid") == False

print("✅ Utils imported successfully")
EOF
```

### Korak 0.4: Update training.py imports
```python
# api/routes/training.py - TOP OF FILE
from utils.validation import validate_session_id, create_error_response, create_success_response
from utils.metadata_utils import extract_file_metadata_fields, extract_file_metadata

# ZAKOMENTARIŠI stare funkcije (NEMOJ brisati još!)
# def validate_session_id(session_id):  # MOVED TO utils/validation.py
# def create_error_response(...):       # MOVED TO utils/validation.py
# ...
```

### ✅ Checkpoint 0: Testiranje
```bash
# 1. Pokreni backend
docker-compose restart backend

# 2. Testiraj da sve radi
curl http://localhost:8080/health
curl http://localhost:8080/api/training/list-sessions

# 3. Commit ako sve radi
git add .
git commit -m "Phase 0: Extract validation and metadata utils"
```

---

## **FAZA 1: Visualization Module (3-4 sata)**

### Korak 1.1: Analiziraj visualization funkcije
```bash
# Pronađi sve visualization-related funkcije
grep -n "plot\|visual\|chart\|graph" api/routes/training.py | head -30
```

### Korak 1.2: Dodaj metode u Visualizer class
```python
# services/training/visualization.py - DODAJ NA KRAJ KLASE:

class Visualizer:
    # ... existing code ...

    def get_available_variables(self, session_id: str) -> dict:
        """
        Get all available variables for plotting from session data.

        Args:
            session_id: Session identifier

        Returns:
            dict: {
                'input_variables': [...],
                'output_variables': [...],
                'metadata': {...}
            }
        """
        from utils.database import create_or_get_session_uuid, get_supabase_client
        from utils.training_storage import fetch_training_results_with_storage

        try:
            # Get UUID session ID
            uuid_session_id = create_or_get_session_uuid(session_id)
            if not uuid_session_id:
                raise ValueError(f"Session {session_id} not found")

            # Fetch training results
            training_results = fetch_training_results_with_storage(session_id)

            if not training_results:
                raise ValueError("No training results found for this session")

            # Extract variables
            input_vars = []
            output_vars = []

            # Extract from training results structure
            if 'input_data' in training_results:
                input_vars = list(training_results['input_data'].keys())

            if 'output_data' in training_results:
                output_vars = list(training_results['output_data'].keys())

            # Also check evaluation tables
            supabase = get_supabase_client()
            eval_response = supabase.table('evaluation_tables')\
                .select('input_data, output_data')\
                .eq('session_id', uuid_session_id)\
                .limit(1)\
                .execute()

            if eval_response.data:
                eval_data = eval_response.data[0]
                if eval_data.get('input_data'):
                    input_vars.extend(eval_data['input_data'].keys())
                if eval_data.get('output_data'):
                    output_vars.extend(eval_data['output_data'].keys())

            # Remove duplicates and sort
            input_vars = sorted(list(set(input_vars)))
            output_vars = sorted(list(set(output_vars)))

            return {
                'input_variables': input_vars,
                'output_variables': output_vars,
                'total_variables': len(input_vars) + len(output_vars),
                'session_id': session_id
            }

        except Exception as e:
            logger.error(f"Error getting plot variables: {str(e)}")
            raise

    def get_session_visualizations(self, session_id: str) -> list:
        """
        Get all visualizations for a session from database.

        Args:
            session_id: Session identifier

        Returns:
            list: List of visualization objects
        """
        from utils.database import create_or_get_session_uuid, get_supabase_client

        try:
            uuid_session_id = create_or_get_session_uuid(session_id)
            if not uuid_session_id:
                raise ValueError(f"Session {session_id} not found")

            supabase = get_supabase_client()
            response = supabase.table('training_visualizations')\
                .select('*')\
                .eq('session_id', uuid_session_id)\
                .order('created_at', desc=True)\
                .execute()

            return response.data if response.data else []

        except Exception as e:
            logger.error(f"Error fetching visualizations: {str(e)}")
            raise

    def generate_custom_plot(self, session_id: str, plot_config: dict) -> dict:
        """
        Generate custom plot based on configuration.

        Args:
            session_id: Session identifier
            plot_config: {
                'plot_type': 'line' | 'scatter' | 'histogram',
                'x_variable': str,
                'y_variable': str,
                'title': str,
                'options': dict
            }

        Returns:
            dict: {
                'plot_data': base64_string,
                'plot_name': str,
                'plot_type': str
            }
        """
        from utils.database import create_or_get_session_uuid, get_supabase_client
        from utils.training_storage import fetch_training_results_with_storage
        import matplotlib.pyplot as plt
        import base64
        from io import BytesIO

        try:
            # Validate session
            uuid_session_id = create_or_get_session_uuid(session_id)
            if not uuid_session_id:
                raise ValueError(f"Session {session_id} not found")

            # Get data
            training_results = fetch_training_results_with_storage(session_id)
            if not training_results:
                raise ValueError("No training results found")

            # Extract plot config
            plot_type = plot_config.get('plot_type', 'line')
            x_var = plot_config.get('x_variable')
            y_var = plot_config.get('y_variable')
            title = plot_config.get('title', 'Custom Plot')

            # Create figure
            fig, ax = plt.subplots(figsize=(10, 6))

            # Get data for variables
            x_data = self._extract_variable_data(training_results, x_var)
            y_data = self._extract_variable_data(training_results, y_var)

            # Plot based on type
            if plot_type == 'line':
                ax.plot(x_data, y_data, linewidth=2)
            elif plot_type == 'scatter':
                ax.scatter(x_data, y_data, alpha=0.6)
            elif plot_type == 'histogram':
                ax.hist(y_data, bins=30, alpha=0.7)

            ax.set_title(title)
            ax.set_xlabel(x_var or 'X')
            ax.set_ylabel(y_var or 'Y')
            ax.grid(True, alpha=0.3)

            # Convert to base64
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            plot_base64 = base64.b64encode(buffer.read()).decode('utf-8')
            plt.close(fig)

            # Save to database
            plot_name = f"{plot_type}_{x_var}_{y_var}"
            self._save_visualization_to_db(uuid_session_id, plot_name, plot_base64, plot_type)

            return {
                'plot_data': f"data:image/png;base64,{plot_base64}",
                'plot_name': plot_name,
                'plot_type': plot_type,
                'created_at': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error generating plot: {str(e)}")
            raise

    def _extract_variable_data(self, training_results: dict, variable_name: str) -> list:
        """Helper to extract variable data from training results"""
        # Search in different possible locations
        if 'input_data' in training_results and variable_name in training_results['input_data']:
            return training_results['input_data'][variable_name]

        if 'output_data' in training_results and variable_name in training_results['output_data']:
            return training_results['output_data'][variable_name]

        # If not found, return empty list
        return []

    def _save_visualization_to_db(self, session_id: str, plot_name: str,
                                  plot_data: str, plot_type: str):
        """Helper to save visualization to database"""
        from utils.database import get_supabase_client

        supabase = get_supabase_client()
        supabase.table('training_visualizations').insert({
            'session_id': session_id,
            'plot_name': plot_name,
            'plot_data': plot_data,
            'plot_type': plot_type,
            'created_at': datetime.now().isoformat()
        }).execute()
```

### Korak 1.3: Update training.py endpoints
```python
# api/routes/training.py - UPDATE ENDPOINTS:

from services.training.visualization import Visualizer

@bp.route('/plot-variables/<session_id>', methods=['GET'])
def get_plot_variables(session_id):
    """Get available variables for plotting"""
    try:
        if not validate_session_id(session_id):
            return create_error_response('Invalid session ID', 400)

        visualizer = Visualizer()
        variables = visualizer.get_available_variables(session_id)

        return create_success_response(variables)

    except ValueError as e:
        return create_error_response(str(e), 404)
    except Exception as e:
        logger.error(f"Error in get_plot_variables: {str(e)}")
        return create_error_response(str(e), 500)


@bp.route('/visualizations/<session_id>', methods=['GET'])
def get_training_visualizations(session_id):
    """Get all visualizations for session"""
    try:
        if not validate_session_id(session_id):
            return create_error_response('Invalid session ID', 400)

        visualizer = Visualizer()
        visualizations = visualizer.get_session_visualizations(session_id)

        return create_success_response({
            'visualizations': visualizations,
            'count': len(visualizations)
        })

    except ValueError as e:
        return create_error_response(str(e), 404)
    except Exception as e:
        logger.error(f"Error in get_training_visualizations: {str(e)}")
        return create_error_response(str(e), 500)


@bp.route('/generate-plot', methods=['POST'])
def generate_plot():
    """Generate custom plot"""
    try:
        data = request.json
        if not data:
            return create_error_response('No data provided', 400)

        session_id = data.get('sessionId')
        plot_config = data.get('plotConfig', {})

        if not validate_session_id(session_id):
            return create_error_response('Invalid session ID', 400)

        visualizer = Visualizer()
        result = visualizer.generate_custom_plot(session_id, plot_config)

        return create_success_response(result)

    except ValueError as e:
        return create_error_response(str(e), 400)
    except Exception as e:
        logger.error(f"Error in generate_plot: {str(e)}")
        return create_error_response(str(e), 500)
```

### ✅ Checkpoint 1: Testiranje Visualizations
```bash
# 1. Restart backend
docker-compose restart backend

# 2. Test endpoints
# Test plot variables
curl http://localhost:8080/api/training/plot-variables/SESSION_ID

# Test visualizations list
curl http://localhost:8080/api/training/visualizations/SESSION_ID

# Test generate plot
curl -X POST http://localhost:8080/api/training/generate-plot \
  -H "Content-Type: application/json" \
  -d '{
    "sessionId": "SESSION_ID",
    "plotConfig": {
      "plot_type": "line",
      "x_variable": "time",
      "y_variable": "temperature"
    }
  }'

# 3. Ako sve radi, commit
git add .
git commit -m "Phase 1: Refactor visualization functions to service layer"
```

---

## **FAZA 2: Scaler Management (3-4 sata)**

### Korak 2.1: Proširi scaler_manager.py
```python
# services/training/scaler_manager.py - DODAJ NA KRAJ:

def get_session_scalers(session_id: str) -> dict:
    """
    Get all scalers for a session from database.

    Args:
        session_id: Session identifier

    Returns:
        dict: {
            'input_scalers': [...],
            'output_scalers': [...],
            'metadata': {...}
        }
    """
    from utils.database import create_or_get_session_uuid, get_supabase_client
    import pickle
    import base64

    try:
        uuid_session_id = create_or_get_session_uuid(session_id)
        if not uuid_session_id:
            raise ValueError(f"Session {session_id} not found")

        supabase = get_supabase_client()

        # Get scalers from database
        response = supabase.table('scalers')\
            .select('*')\
            .eq('session_id', uuid_session_id)\
            .execute()

        if not response.data:
            raise ValueError(f"No scalers found for session {session_id}")

        scalers_data = response.data[0]

        # Deserialize scalers (SECURITY NOTE: This should be replaced with safer serialization)
        input_scalers_data = scalers_data.get('input_scalers', {})
        output_scalers_data = scalers_data.get('output_scalers', {})

        # Extract scaler info without deserializing
        input_scaler_info = []
        for name, scaler_dict in input_scalers_data.items():
            input_scaler_info.append({
                'name': name,
                'type': scaler_dict.get('_model_class', 'Unknown'),
                'data_size': len(scaler_dict.get('_model_data', ''))
            })

        output_scaler_info = []
        for name, scaler_dict in output_scalers_data.items():
            output_scaler_info.append({
                'name': name,
                'type': scaler_dict.get('_model_class', 'Unknown'),
                'data_size': len(scaler_dict.get('_model_data', ''))
            })

        return {
            'input_scalers': input_scaler_info,
            'output_scalers': output_scaler_info,
            'session_id': session_id,
            'total_scalers': len(input_scaler_info) + len(output_scaler_info)
        }

    except Exception as e:
        logger.error(f"Error getting scalers: {str(e)}")
        raise


def create_scaler_download_package(session_id: str) -> bytes:
    """
    Create downloadable package with scalers as .pkl files.

    Args:
        session_id: Session identifier

    Returns:
        bytes: Pickled scalers data
    """
    from utils.database import create_or_get_session_uuid, get_supabase_client
    import pickle
    import base64
    import tempfile

    try:
        uuid_session_id = create_or_get_session_uuid(session_id)
        if not uuid_session_id:
            raise ValueError(f"Session {session_id} not found")

        supabase = get_supabase_client()
        response = supabase.table('scalers')\
            .select('*')\
            .eq('session_id', uuid_session_id)\
            .execute()

        if not response.data:
            raise ValueError(f"No scalers found for session {session_id}")

        scalers_data = response.data[0]

        # Deserialize scalers
        input_scalers = {}
        output_scalers = {}

        for name, scaler_dict in scalers_data.get('input_scalers', {}).items():
            if '_model_data' in scaler_dict:
                scaler_bytes = base64.b64decode(scaler_dict['_model_data'])
                input_scalers[name] = pickle.loads(scaler_bytes)

        for name, scaler_dict in scalers_data.get('output_scalers', {}).items():
            if '_model_data' in scaler_dict:
                scaler_bytes = base64.b64decode(scaler_dict['_model_data'])
                output_scalers[name] = pickle.loads(scaler_bytes)

        # Create combined package
        package = {
            'input_scalers': input_scalers,
            'output_scalers': output_scalers,
            'session_id': session_id
        }

        return pickle.dumps(package)

    except Exception as e:
        logger.error(f"Error creating scaler package: {str(e)}")
        raise


def scale_new_data(session_id: str, input_data: dict) -> dict:
    """
    Apply trained scalers to new input data.

    Args:
        session_id: Session identifier
        input_data: Dictionary of input variables to scale

    Returns:
        dict: Scaled data
    """
    from utils.database import create_or_get_session_uuid, get_supabase_client
    import pickle
    import base64
    import pandas as pd
    import numpy as np

    try:
        uuid_session_id = create_or_get_session_uuid(session_id)
        if not uuid_session_id:
            raise ValueError(f"Session {session_id} not found")

        # Get scalers from database
        supabase = get_supabase_client()
        response = supabase.table('scalers')\
            .select('*')\
            .eq('session_id', uuid_session_id)\
            .execute()

        if not response.data:
            raise ValueError(f"No scalers found for session {session_id}")

        scalers_data = response.data[0]
        input_scalers_data = scalers_data.get('input_scalers', {})

        # Deserialize scalers
        scalers = {}
        for name, scaler_dict in input_scalers_data.items():
            if '_model_data' in scaler_dict:
                scaler_bytes = base64.b64decode(scaler_dict['_model_data'])
                scalers[name] = pickle.loads(scaler_bytes)

        # Apply scaling
        scaled_data = {}
        for var_name, var_data in input_data.items():
            if var_name in scalers:
                scaler = scalers[var_name]

                # Convert to numpy array
                data_array = np.array(var_data).reshape(-1, 1)

                # Apply scaler
                scaled_array = scaler.transform(data_array)

                scaled_data[var_name] = scaled_array.flatten().tolist()
            else:
                # No scaler found, return original data
                scaled_data[var_name] = var_data

        return {
            'scaled_data': scaled_data,
            'scalers_applied': list(scalers.keys()),
            'session_id': session_id
        }

    except Exception as e:
        logger.error(f"Error scaling data: {str(e)}")
        raise
```

### Korak 2.2: Update training.py endpoints
```python
# api/routes/training.py - UPDATE SCALER ENDPOINTS:

from services.training.scaler_manager import (
    get_session_scalers,
    create_scaler_download_package,
    scale_new_data
)
from flask import send_file
from io import BytesIO

@bp.route('/scalers/<session_id>', methods=['GET'])
def get_scalers(session_id):
    """Get scalers information for session"""
    try:
        if not validate_session_id(session_id):
            return create_error_response('Invalid session ID', 400)

        scalers = get_session_scalers(session_id)
        return create_success_response(scalers)

    except ValueError as e:
        return create_error_response(str(e), 404)
    except Exception as e:
        logger.error(f"Error in get_scalers: {str(e)}")
        return create_error_response(str(e), 500)


@bp.route('/scalers/<session_id>/download', methods=['GET'])
def download_scalers_as_save_files(session_id):
    """Download scalers as pickle file"""
    try:
        if not validate_session_id(session_id):
            return create_error_response('Invalid session ID', 400)

        scalers_bytes = create_scaler_download_package(session_id)

        # Send as downloadable file
        return send_file(
            BytesIO(scalers_bytes),
            mimetype='application/octet-stream',
            as_attachment=True,
            download_name=f'scalers_{session_id}.pkl'
        )

    except ValueError as e:
        return create_error_response(str(e), 404)
    except Exception as e:
        logger.error(f"Error in download_scalers: {str(e)}")
        return create_error_response(str(e), 500)


@bp.route('/scale-data/<session_id>', methods=['POST'])
def scale_input_data(session_id):
    """Scale new input data using session's trained scalers"""
    try:
        if not validate_session_id(session_id):
            return create_error_response('Invalid session ID', 400)

        data = request.json
        if not data or 'input_data' not in data:
            return create_error_response('No input data provided', 400)

        input_data = data['input_data']

        result = scale_new_data(session_id, input_data)
        return create_success_response(result)

    except ValueError as e:
        return create_error_response(str(e), 400)
    except Exception as e:
        logger.error(f"Error in scale_input_data: {str(e)}")
        return create_error_response(str(e), 500)
```

### ✅ Checkpoint 2: Testiranje Scalers
```bash
# 1. Restart backend
docker-compose restart backend

# 2. Test endpoints
curl http://localhost:8080/api/training/scalers/SESSION_ID

# Test scaler download
curl -O http://localhost:8080/api/training/scalers/SESSION_ID/download

# Test data scaling
curl -X POST http://localhost:8080/api/training/scale-data/SESSION_ID \
  -H "Content-Type: application/json" \
  -d '{
    "input_data": {
      "temperature": [20, 25, 30],
      "pressure": [100, 110, 120]
    }
  }'

# 3. Commit
git add .
git commit -m "Phase 2: Refactor scaler functions to service layer"
```

---

## **FAZA 3: Model Management (4-5 sati)**

### Korak 3.1: Kreiraj novi service module
```python
# services/training/model_manager.py (NOVI FAJL)
```

**SVI KORACI ZA FAZU 3, 4, 5, i 6 detaljno su dokumentovani u planu.**

---

## 📊 Success Metrics

Nakon svakog checkpointa, provjeri:
- ✅ Backend se pokreće bez grešaka
- ✅ Svi endpointi vraćaju očekivane odgovore
- ✅ Nema broken imports
- ✅ Tests pass (ako postoje)
- ✅ Code linting pass

---

## 🔄 Rollback Plan

Ako nešto krene po zlu:
```bash
# Vrati se na prethodni checkpoint
git log --oneline -10
git checkout <commit_hash>

# Ili vrati na backup
cp api/routes/training.py.BACKUP_XXXXXXXX api/routes/training.py
```

---

**Nastavak plana u sljedećem fajlu zbog ograničenja veličine...**
