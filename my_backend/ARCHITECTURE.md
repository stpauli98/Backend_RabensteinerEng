# Backend Architecture

## Overview
This Flask backend has been restructuredd following clean architecture principles with proper separation of concerns.

## Directory Structure

```
my_backend/
├── api/                        # API Layer - HTTP request/response handling
│   ├── routes/                 # All API route definitions (blueprints)
│   │   ├── adjustments.py     # Data adjustments endpoints
│   │   ├── cloud.py            # Cloud storage endpoints
│   │   ├── data_processing.py # Data processing endpoints
│   │   ├── first_processing.py# Initial processing endpoints
│   │   ├── load_data.py       # Data loading endpoints
│   │   └── training.py        # Training endpoints
│   └── middleware/             # Middleware components
│
├── services/                   # Business Logic Layer
│   ├── adjustments/           # Data adjustment services
│   │   └── cleanup.py         # File cleanup service
│   ├── cloud/                 # Cloud services
│   ├── data_processing/       # Data processing logic
│   ├── training/              # ML training services
│   │   ├── config.py          # Training configuration
│   │   ├── data_loader.py     # Data loading utilities
│   │   ├── data_processor.py  # Data processing utilities
│   │   ├── data_scaling.py    # Data scaling utilities
│   │   ├── model_trainer.py   # Model training logic
│   │   ├── pipeline_integration.py # Pipeline integration
│   │   ├── progress_manager.py # Progress tracking
│   │   ├── results_generator.py # Results generation
│   │   ├── time_features.py   # Time feature engineering
│   │   ├── training_api.py    # Training API blueprint
│   │   ├── training_pipeline.py # Training pipeline
│   │   ├── utils.py           # Training utilities
│   │   └── visualization.py   # Visualization utilities
│   └── upload/                # Upload services
│
├── models/                     # Data Models Layer
│   └── (database models)
│
├── utils/                      # Utility Functions
│   └── database.py            # Database client (Supabase)
│
├── config/                     # Configuration
│   └── (configuration files)
│
├── core/                       # Core Application Setup
│   ├── app_factory.py         # Flask application factory
│   ├── extensions.py          # Flask extensions (SocketIO, CORS)
│   └── socketio_handlers.py   # SocketIO event handlers
│
├── storage/                    # File Storage
│   ├── cache/
│   ├── logs/
│   ├── processed/
│   ├── sessions/
│   ├── temp/
│   └── uploads/
│
├── app.py                      # Main entry point (simplified)
└── requirements.txt            # Python dependencies
```

## Architecture Principles

### 1. Separation of Concerns
- **API Layer** (`api/routes/`): Handles HTTP requests/responses only
- **Business Logic** (`services/`): Contains all business logic and data processing
- **Data Access** (`utils/database.py`): Centralized database access
- **Core** (`core/`): Application initialization and configuration

### 2. Dependency Flow
```
app.py → core/app_factory.py → api/routes/* → services/* → utils/*
```

### 3. Blueprint Organization
Each route file is a Flask Blueprint with its own namespace:
- `/api/loadRowData` - Data loading endpoints
- `/api/firstProcessing` - Initial processing endpoints
- `/api/cloud` - Cloud storage endpoints
- `/api/adjustmentsOfData` - Data adjustment endpoints
- `/api/training` - Training endpoints

### 4. Service Layer Pattern
Business logic is separated from routes:
- Routes handle request/response
- Services contain business logic
- Utils provide shared functionality

## Key Components

### Core Application (`core/`)
- **app_factory.py**: Creates and configures Flask app
- **extensions.py**: Initializes Flask extensions (SocketIO, CORS)
- **socketio_handlers.py**: WebSocket event handlers

### API Routes (`api/routes/`)
- Each file contains Flask blueprints
- Routes delegate to services for business logic
- Handles request validation and response formatting

### Services (`services/`)
- **training/**: Complete ML training pipeline
- **adjustments/**: Data adjustment and cleanup services
- **cloud/**: Cloud storage operations
- **data_processing/**: Data processing logic

### Utilities (`utils/`)
- **database.py**: Supabase client and database operations

## Migration Notes

### Import Changes
All imports have been updated:
- `from supabase_client import` → `from utils.database import`
- `from training_system.` → `from services.training.`
- `from adjustmentsOfData import cleanup_old_files` → `from services.adjustments.cleanup import cleanup_old_files`

### File Relocations
- `supabase_client.py` → `utils/database.py`
- `training_system/*` → `services/training/*`
- All route files → `api/routes/*`

### Preserved Functionality
- All endpoints remain at the same URLs
- All business logic preserved
- WebSocket functionality intact
- Background jobs continue working

## Running the Application

```bash
python app.py
```

The application will start on port 8080 (or the PORT environment variable).

## Future Improvements

1. **Complete Service Extraction**: Move remaining business logic from routes to services
2. **Model Definitions**: Create proper SQLAlchemy or Pydantic models
3. **Configuration Management**: Centralize configuration in `config/`
4. **Testing**: Add unit and integration tests in `tests/`
5. **API Documentation**: Add OpenAPI/Swagger documentation
6. **Error Handling**: Centralized error handling middleware
7. **Logging**: Structured logging configuration
8. **Dependency Injection**: Consider using Flask-Injector for DI