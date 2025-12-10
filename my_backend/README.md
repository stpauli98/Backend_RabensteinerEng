# Backend - Rabensteiner Engineering

Backend application for Rabensteiner Engineering data processing and ML model training system.

## Overview

The backend provides:
- Chunked CSV data upload and processing
- Time series processing and transformation
- Training of various ML models (Dense, CNN, LSTM, SVR, Linear)
- Real-time progress tracking via WebSocket
- Cloud integration with Supabase
- Stripe payment processing
- Automatic cleanup of old files

## Installation

### Prerequisites
- Python 3.9+
- Docker (recommended)

### Docker (Recommended)

```bash
# Build
docker build --build-arg ENV_FILE=.env -t my_backend .

# Run
docker run -p 8080:8080 --env-file .env my_backend
```

### Local Development

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run
python app.py
```

## Environment Variables

Create a `.env` file with:

```env
# Supabase
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_anon_key
SUPABASE_SERVICE_ROLE_KEY=your_service_role_key

# Stripe
STRIPE_SECRET_KEY=your_stripe_secret_key
STRIPE_PUBLISHABLE_KEY=your_stripe_publishable_key
STRIPE_WEBHOOK_SECRET=your_webhook_secret

# Frontend
FRONTEND_URL=http://localhost:3000
```

## Architecture

```
my_backend/
├── app.py                      # Entry point
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
│
├── core/                       # Application core
│   ├── app.py                  # Flask app (alternative)
│   ├── app_factory.py          # Flask app factory
│   ├── blueprints.py           # Blueprint registration
│   └── socketio_handlers.py    # WebSocket handlers
│
├── domains/                    # Domain-driven architecture
│   ├── training/               # ML training domain
│   │   ├── api/                # Training endpoints (36 routes)
│   │   ├── data/               # Data processing, scaling
│   │   ├── ml/                 # Models, scalers, trainers
│   │   └── services/           # Session, upload, visualization
│   │
│   ├── processing/             # Data processing domain
│   │   ├── api/                # Processing endpoints
│   │   └── services/           # CSV processing, cleaning
│   │
│   ├── upload/                 # File upload domain
│   │   ├── api/                # Upload endpoints
│   │   └── services/           # State management, parsing
│   │
│   ├── adjustments/            # Data adjustments domain
│   │   ├── api/                # Adjustment endpoints
│   │   └── services/           # Adjustment logic
│   │
│   ├── cloud/                  # Cloud analysis domain
│   │   ├── api/                # Cloud endpoints
│   │   └── services/           # Regression, interpolation
│   │
│   └── payments/               # Stripe payments domain
│       └── api/                # Payment endpoints
│
├── shared/                     # Shared infrastructure
│   ├── auth/                   # JWT authentication, subscription checks
│   ├── database/               # Supabase client and operations
│   ├── payments/               # Stripe helpers
│   ├── storage/                # Storage service
│   ├── tracking/               # Usage tracking
│   └── exceptions/             # Error handling
│
├── middleware/                 # Authentication middleware
├── services/                   # Scheduled cleanup service
├── utils/                      # Helper utilities
└── sql/                        # SQL schema references
```

## API Endpoints

### Health Check
- `GET /` - Server status
- `GET /health` - Health check

### Training (`/api/training`)
- `POST /upload-chunk` - Upload CSV chunk
- `POST /finalize-session` - Finalize upload session
- `GET /list-sessions` - List training sessions
- `POST /generate-datasets/<session_id>` - Generate datasets
- `POST /train-models/<session_id>` - Train ML models
- `GET /results/<session_id>` - Get training results
- `GET /scalers/<session_id>` - Get scalers info
- `POST /scale-data/<session_id>` - Scale new data

### Data Upload (`/api/loadRowData`)
- `POST /upload-chunk` - Upload data chunk
- `POST /finalize-upload` - Finalize upload
- `POST /cancel-upload` - Cancel upload

### Processing (`/api/firstProcessing`, `/api/dataProcessingMain`)
- `POST /upload_chunk` - Process data chunks
- `POST /process` - Main processing

### Adjustments (`/api/adjustmentsOfData`)
- `POST /process` - Apply data adjustments

### Cloud (`/api/cloud`)
- `POST /upload-chunk` - Cloud upload
- `POST /clouddata` - Retrieve cloud data
- `POST /interpolate-chunked` - Data interpolation

### Payments (`/api/stripe`)
- `POST /create-checkout-session` - Create Stripe checkout
- `POST /webhook` - Stripe webhook handler
- `GET /subscription-status` - Get subscription status

## WebSocket Events

### Client → Server
- `connect` - Connect to server
- `join` - Join room
- `join_training_session` - Join training session

### Server → Client
- `upload_progress` - Upload progress
- `training_status_update` - Training status
- `dataset_status_update` - Dataset generation status
- `violin_plot_progress` - Violin plot generation progress

## Running

### Development
```bash
python app.py
# Server at http://localhost:8080
```

### Production (Docker)
```bash
docker build --build-arg ENV_FILE=.env -t my_backend .
docker run -p 8080:8080 --env-file .env my_backend
```

### Docker Compose
```bash
docker-compose up --build
```

## Common Issues

**Port already in use:**
```bash
lsof -i :8080
kill -9 <PID>
```

**Supabase connection:**
Verify SUPABASE_URL and SUPABASE_KEY in .env file.

**Stripe webhooks:**
Use Stripe CLI for local testing:
```bash
stripe listen --forward-to localhost:8080/api/stripe/webhook
```

---

**Version:** 2.0.0
**Last Updated:** December 2024
