# Backend - Rabensteiner Engineering

Backend application for Rabensteiner Engineering data processing and ML model training system.

## 📋 Overview

The backend provides:
- Chunked CSV data upload and processing
- Time series processing and transformation
- Training of various ML models (Dense, CNN, LSTM, SVR, Linear)
- Real-time progress tracking via WebSocket
- Cloud integration with Supabase
- Automatic cleanup of old files

## 🚀 Installation

### Prerequisites
- Python 3.8+
- pip
- Virtual environment (recommended)

### Installation Steps

1. **Clone the repository**
```bash
git clone https://github.com/your-repo/Backend_RabensteinerEng.git
cd Backend_RabensteinerEng/my_backend
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure environment variables**
```bash
cp .env.example .env
# Edit .env file with your settings
```

## 🏗️ Architecture

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

## 🔧 Configuration

### Environment Variables (.env)

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

## 🚦 Running the Application

### Development Mode
```bash
python app.py
```

Server will be available at `http://localhost:8080`

### Production Mode
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
- `GET /` - Server status
- `GET /health` - Health check

### Data Upload
- `POST /api/loadRowData/upload-chunk` - Upload data in chunks
- `POST /api/loadRowData/finalize-upload` - Finalize upload
- `POST /api/loadRowData/cancel-upload` - Cancel upload

### Data Processing
- `POST /api/firstProcessing/upload_chunk` - Initial processing
- `POST /api/dataProcessingMain/upload-chunk` - Main processing
- `POST /api/adjustmentsOfData/process` - Data adjustments

### Training
- `POST /api/training/generate-dataset` - Generate dataset
- `POST /api/training/train` - Train models
- `GET /api/training/status/<session_id>` - Training status

### Cloud Operations
- `POST /api/cloud/upload-chunk` - Cloud upload
- `POST /api/cloud/clouddata` - Retrieve cloud data
- `POST /api/cloud/interpolate-chunked` - Data interpolation

## 🔌 WebSocket Events

### Client → Server
- `connect` - Connect to server
- `join` - Join room
- `join_training_session` - Join training session
- `request_training_status` - Request training status

### Server → Client
- `upload_progress` - Upload progress
- `training_status_update` - Training status update
- `dataset_status_update` - Dataset generation status
- `processing_error` - Processing error

## 🧪 Testing

```bash
# Run unit tests
python -m pytest tests/

# With coverage
python -m pytest --cov=. tests/
```

## 📦 Deployment

### Google Cloud Run
```bash
gcloud builds submit --tag gcr.io/PROJECT_ID/rabensteiner-backend
gcloud run deploy --image gcr.io/PROJECT_ID/rabensteiner-backend --platform managed
```

### Render.com
1. Connect GitHub repository
2. Set environment variables
3. Auto-deploy on push

## 🐛 Debugging

### Logging
```python
import logging
logger = logging.getLogger(__name__)
logger.debug("Debug message")
logger.info("Info message")
logger.error("Error message")
```

### Common Issues

**Port already in use:**
```bash
lsof -i :8080
kill -9 <PID>
```

**Import errors:**
Check that you're in the correct directory and virtual environment is activated.

**Supabase connection:**
Verify that SUPABASE_URL and SUPABASE_KEY are correctly set in .env file.

## 📄 License

Proprietary - Rabensteiner Engineering

## 👥 Team

- Backend Development Team
- ML Engineering Team

## 📞 Contact

For questions and support, contact the development team.

---

**Version:** 1.0.0  
**Last Updated:** August 2024