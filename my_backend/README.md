# Backend - Rabensteiner Engineering

Backend aplikacija za Rabensteiner Engineering sistem za obradu podataka i treniranje ML modela.

## 📋 Pregled

Backend omogućava:
- Upload i obradu CSV podataka u chunk-ovima
- Procesiranje i transformaciju vremenskih serija
- Treniranje različitih ML modela (Dense, CNN, LSTM, SVR, Linear)
- Real-time praćenje progresa preko WebSocket-a
- Cloud integraciju sa Supabase
- Automatsko čišćenje starih fajlova

## 🚀 Instalacija

### Preduslovi
- Python 3.8+
- pip
- Virtual environment (preporučeno)

### Koraci instalacije

1. **Kloniraj repozitorijum**
```bash
git clone https://github.com/your-repo/Backend_RabensteinerEng.git
cd Backend_RabensteinerEng/my_backend
```

2. **Kreiraj virtuelno okruženje**
```bash
python -m venv venv
source venv/bin/activate  # Na Windows: venv\Scripts\activate
```

3. **Instaliraj zavisnosti**
```bash
pip install -r requirements.txt
```

4. **Konfiguriši environment varijable**
```bash
cp .env.example .env
# Edituj .env fajl sa tvojim podešavanjima
```

## 🏗️ Arhitektura

```
my_backend/
├── api/routes/           # API endpoints (Flask blueprints)
│   ├── adjustments.py    # Data adjustments endpoints
│   ├── cloud.py          # Cloud operations
│   ├── data_processing.py# Main data processing
│   ├── first_processing.py# Initial processing
│   ├── load_data.py      # Data upload endpoints
│   └── training.py       # ML training endpoints
├── services/             # Business logic
│   ├── adjustments/      # Data adjustment services
│   ├── cloud/            # Cloud integration
│   ├── data_processing/  # Processing logic
│   ├── training/         # ML training pipeline
│   └── upload/           # File upload handlers
├── core/                 # Application core
│   ├── app_factory.py    # Flask app creation
│   ├── extensions.py     # Flask extensions
│   └── socketio_handlers.py # WebSocket handlers
├── utils/                # Utilities
│   └── database.py       # Supabase client
├── models/               # Data models
├── config/               # Configuration
├── storage/              # File storage
└── app.py               # Entry point
```

## 🔧 Konfiguracija

### Environment varijable (.env)

```env
# Flask
FLASK_ENV=development
PORT=8080

# Supabase
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key

# Upload settings
MAX_CONTENT_LENGTH=104857600  # 100MB
UPLOAD_FOLDER=uploads
TEMP_FOLDER=temp_uploads
```

## 🚦 Pokretanje

### Development mode
```bash
python app.py
```

Server će biti dostupan na `http://localhost:8080`

### Production mode
```bash
gunicorn -w 4 -b 0.0.0.0:8080 --timeout 300 app:app
```

### Docker
```bash
docker build -t rabensteiner-backend .
docker run -p 8080:8080 --env-file .env rabensteiner-backend
```

## 📡 API Endpoints

### Health Check
- `GET /` - Status servera
- `GET /health` - Health check

### Data Upload
- `POST /api/loadRowData/upload-chunk` - Upload podataka u chunk-ovima
- `POST /api/loadRowData/finalize-upload` - Finalizacija upload-a
- `POST /api/loadRowData/cancel-upload` - Prekini upload

### Data Processing
- `POST /api/firstProcessing/upload_chunk` - Inicijalna obrada
- `POST /api/dataProcessingMain/upload-chunk` - Glavna obrada
- `POST /api/adjustmentsOfData/process` - Prilagođavanje podataka

### Training
- `POST /api/training/generate-dataset` - Generisanje dataset-a
- `POST /api/training/train` - Treniranje modela
- `GET /api/training/status/<session_id>` - Status treniranja

### Cloud Operations
- `POST /api/cloud/upload-chunk` - Upload na cloud
- `POST /api/cloud/clouddata` - Preuzmi cloud podatke
- `POST /api/cloud/interpolate-chunked` - Interpolacija podataka

## 🔌 WebSocket Events

### Client → Server
- `connect` - Konekcija na server
- `join` - Pridruži se room-u
- `join_training_session` - Pridruži se training sesiji
- `request_training_status` - Zatraži status treniranja

### Server → Client
- `upload_progress` - Progres upload-a
- `training_status_update` - Update statusa treniranja
- `dataset_status_update` - Status generisanja dataset-a
- `processing_error` - Greška u procesiranju

## 🧪 Testiranje

```bash
# Pokreni unit testove
python -m pytest tests/

# Sa coverage
python -m pytest --cov=. tests/
```

## 📦 Deployment

### Google Cloud Run
```bash
gcloud builds submit --tag gcr.io/PROJECT_ID/rabensteiner-backend
gcloud run deploy --image gcr.io/PROJECT_ID/rabensteiner-backend --platform managed
```

### Render.com
1. Poveži GitHub repozitorijum
2. Podesi environment varijable
3. Deploy automatski na push

## 🐛 Debugging

### Logovi
```python
import logging
logger = logging.getLogger(__name__)
logger.debug("Debug message")
logger.info("Info message")
logger.error("Error message")
```

### Common Issues

**Port već u upotrebi:**
```bash
lsof -i :8080
kill -9 <PID>
```

**Import greške:**
Proveri da si u pravom direktorijumu i da je virtuelno okruženje aktivno.

**Supabase konekcija:**
Proveri da su SUPABASE_URL i SUPABASE_KEY ispravno podešeni u .env fajlu.

## 📄 Licenca

Proprietary - Rabensteiner Engineering

## 👥 Tim

- Backend Development Team
- ML Engineering Team

## 📞 Kontakt

Za pitanja i podršku, kontaktiraj development tim.

---

**Verzija:** 1.0.0  
**Poslednje ažuriranje:** August 2024