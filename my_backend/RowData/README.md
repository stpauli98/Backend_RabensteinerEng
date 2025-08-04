# RowData Module

Refaktorisan modul za procesiranje CSV/TXT fajlova sa vremenskim serijama.

## Struktura

```
RowData/
├── __init__.py              # Main module exports
├── load_row_data.py         # Flask Blueprint sa endpoint-ima
├── services/                # Business logic servisi
│   ├── file_upload_service.py    # Chunked upload management
│   ├── date_parsing_service.py   # Date/time parsing
│   └── data_processing_service.py # CSV processing
├── repositories/            # Data persistence
│   └── upload_repository.py      # Redis storage
├── utils/                   # Utility moduli
│   ├── validators.py            # Input validation
│   ├── exceptions.py            # Custom exceptions
│   └── auth.py                  # Authentication/authorization
├── config/                  # Configuration
│   └── settings.py              # Module settings
└── tasks/                   # Async tasks (future)
```

## Funkcionalnosti

### 1. Sigurnost
- JWT autentifikacija (može se isključiti)
- Rate limiting
- File size limits (100MB)
- Input validacija
- Path traversal zaštita

### 2. Performanse
- Streaming upload/processing
- Chunk-by-chunk obrada
- Redis za metadata storage
- Memorijska efikasnost

### 3. Date/Time podrška
- 18+ date formata
- Custom format podrška
- Timezone konverzija
- Separated date/time columns

## API Endpoints

### Upload Chunk
```
POST /api/loadRowData/upload-chunk
Auth: Required
Rate limit: 100/min
```

### Finalize Upload
```
POST /api/loadRowData/finalize-upload
Auth: Required
Rate limit: 10/min
```

### Cancel Upload
```
POST /api/loadRowData/cancel-upload
Auth: Required
```

### Check Status
```
GET /api/loadRowData/check-status/<upload_id>
Auth: Required
```

### Prepare Save
```
POST /api/loadRowData/prepare-save
Auth: Required
```

### Download
```
GET /api/loadRowData/download/<file_id>
Auth: Required
Rate limit: 50/min
```

## Konfiguracija

### Environment Variables
```bash
# Redis
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_ROWDATA_DB=1

# Security
ROWDATA_REQUIRE_AUTH=true
JWT_SECRET=your-secret-key

# Storage
ROWDATA_STORAGE_PATH=/tmp/row_data_uploads

# Logging
LOG_LEVEL=INFO
```

### Settings Override
Edituj `config/settings.py` za custom konfiguraciju.

## Dependency Requirements

```python
flask
flask-socketio
pandas
redis
pytz
flask-limiter
pyjwt
```

## Error Handling

Modul koristi custom exception klase:
- `ValidationError` - Invalid input
- `UploadError` - Upload problemi
- `ProcessingError` - Processing greške
- `AuthenticationError` - Auth problemi
- `RateLimitError` - Rate limit exceeded

## Socket.IO Events

- `upload_progress` - Progress updates
- `join_upload_room` - Join room za real-time updates

## Primeri korišćenja

### Python Client
```python
import requests

# Upload chunk
with open('chunk.dat', 'rb') as f:
    response = requests.post(
        'http://localhost:5000/api/loadRowData/upload-chunk',
        files={'fileChunk': f},
        data={
            'uploadId': 'unique-id',
            'chunkIndex': 0,
            'totalChunks': 10,
            'delimiter': ',',
            # ... other params
        },
        headers={'Authorization': 'Bearer <token>'}
    )
```

## Migracija sa starog sistema

1. Promeni import u `app.py`:
   ```python
   from RowData import rowdata_blueprint
   ```

2. Registruj blueprint:
   ```python
   app.register_blueprint(rowdata_blueprint, url_prefix='/api/loadRowData')
   ```

3. API ostaje kompatibilan - nema potrebe za frontend promenama

## Budući razvoj

- [ ] Celery podrška za async processing
- [ ] S3 storage za velike fajlove
- [ ] Batch processing API
- [ ] WebSocket streaming upload
- [ ] Prometheus metrics