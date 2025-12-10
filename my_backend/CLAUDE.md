# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Docker (Recommended)
```bash
# Build with environment file
docker build --build-arg ENV_FILE=.env -t my_backend .

# Run container
docker run -p 8080:8080 --env-file .env my_backend

# Build and run combined
docker build --build-arg ENV_FILE=.env -t my_backend . && docker run -p 8080:8080 --env-file .env my_backend
```

### Local Development
```bash
python app.py  # Runs on port 8080
```

### Health Check
```bash
curl http://localhost:8080/health
```

## Architecture Overview

This is a Flask backend using **Domain-Driven Design** with the following structure:

### Core (`core/`)
- `app_factory.py` - Flask app factory with SocketIO, CORS, error handlers
- `blueprints.py` - Central blueprint registration for all domains
- `socketio_handlers.py` - WebSocket event handlers

### Domains (`domains/`)
Each domain follows the pattern: `api/` (endpoints) + `services/` (business logic) + `data/` (data processing)

| Domain | Prefix | Purpose |
|--------|--------|---------|
| `training/` | `/api/training` | ML model training, 36 endpoints |
| `processing/` | `/api/firstProcessing`, `/api/dataProcessingMain` | CSV data processing |
| `upload/` | `/api/loadRowData` | Chunked file upload |
| `adjustments/` | `/api/adjustmentsOfData` | Data adjustments |
| `cloud/` | `/api/cloud` | Cloud analysis, interpolation |
| `payments/` | `/api/stripe` | Stripe subscriptions |

### Shared Infrastructure (`shared/`)
- `auth/` - JWT authentication (`@require_auth`), subscription checks (`@require_subscription`)
- `database/` - Supabase client and operations (`get_supabase_client`, `save_session_to_supabase`)
- `payments/` - Stripe helpers
- `tracking/` - Usage tracking (`increment_processing_count`)

### Key Patterns

**Authentication Decorators** (order matters):
```python
@bp.route('/endpoint', methods=['POST'])
@require_auth
@require_subscription
@check_processing_limit
def endpoint():
    user_id = g.user_id  # Available after @require_auth
```

**Supabase Operations**:
```python
from shared.database.operations import get_supabase_client
supabase = get_supabase_client(use_service_role=True)  # Bypasses RLS
```

**WebSocket Progress Updates**:
```python
from flask import current_app
socketio = current_app.extensions.get('socketio')
socketio.emit('training_status_update', data, room=session_id)
```

### Data Flow
1. **Upload**: Chunked CSV upload → `uploads/file_uploads/session_*`
2. **Process**: Data cleaning, interpolation, adjustments
3. **Train**: Generate datasets → Train models (Dense, CNN, LSTM, SVR, Linear)
4. **Results**: Stored in Supabase, scalers saved for inference

### Environment Variables
Required in `.env`:
- `SUPABASE_URL`, `SUPABASE_KEY`, `SUPABASE_SERVICE_ROLE_KEY`
- `STRIPE_SECRET_KEY`, `STRIPE_PUBLISHABLE_KEY`, `STRIPE_WEBHOOK_SECRET`
- `FRONTEND_URL`

### File Storage
- `chunk_uploads/` - Temporary chunks during upload
- `temp_uploads/` - Processed files (auto-cleanup every 30 min)
- `uploads/file_uploads/` - Session files
