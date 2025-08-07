# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Running the Application
- **Development**: `python app.py` - Runs Flask app with SocketIO on port 8080
- **Production**: `gunicorn -k eventlet -w 1 -b 0.0.0.0:8080 --timeout 300 app:app`
- **Docker Build**: `docker build -t my_backend .`
- **Docker Run**: `docker run -p 8080:8080 my_backend`

### Dependencies
- **Install**: `pip install -r requirements.txt`
- **Virtual Environment**: Use `venv_tf` as the project virtual environment
- **Update**: Manually update `requirements.txt` when adding new packages

### Testing
- No automated test framework is currently configured
- Manual testing via API endpoints
- Health check endpoint: `GET /health`
- Training system tests: `python training_system/test_parameter_converter.py`

## Architecture Overview

### Core Application Structure
The application is a Flask-based data processing backend with real-time WebSocket support via SocketIO. It follows a modular blueprint architecture where each major feature is isolated in its own module.

### Key Modules

#### Main Application Layer
- **app.py**: Flask application initialization, blueprint registration, CORS configuration, SocketIO setup, and scheduled cleanup tasks
- **supabase_client.py**: Database operations and connection management for Supabase

#### Data Processing Pipeline
1. **firstProcessing.py**: Initial CSV processing with chunked file upload support
   - Handles file uploads in chunks (10MB per chunk, 500MB max total)
   - Supports various resampling methods (mean, interpolation, nearest)
   - Real-time progress tracking via SocketIO

2. **load_row_data.py**: Raw data loading module
   - Multiple datetime format support
   - Session-based data management
   - Format detection and validation

3. **data_processing_main.py**: Advanced data cleaning and filtering
   - Outlier detection and removal
   - Data interpolation
   - Measurement failure elimination
   - Statistical filtering

4. **adjustmentsOfData.py**: Data adjustment operations
   - Temporary file management
   - Automatic cleanup every 30 minutes
   - Session persistence

#### Machine Learning & Analysis
- **training.py**: ML model training using scikit-learn
  - Feature extraction from file metadata
  - Model persistence and management
  
- **training_system/**: Complete training pipeline module
  - Modular architecture with separate components for data loading, processing, scaling
  - Progress management and monitoring
  - Parameter conversion and configuration

- **cloud.py**: Cloud-based data analysis
  - Matplotlib visualization generation
  - Statistical analysis
  - Chart export functionality

- **plotting.py**: Data visualization endpoints
  - Dynamic plot generation
  - Multiple chart types support

### Data Flow Architecture

1. **Upload Phase**: Files uploaded in chunks to `chunk_uploads/`
2. **Processing Pipeline**: 
   - firstProcessing → load_row_data → data_processing_main → adjustmentsOfData
   - Each module emits progress via SocketIO
3. **Storage**: 
   - Temporary files in `temp_uploads/` (auto-cleanup after 30 minutes)
   - Session files in `uploads/file_uploads/session_*`
4. **Persistence**: Metadata and results saved to Supabase

### API Structure

All endpoints follow RESTful conventions with blueprints:
- `/api/firstProcessing/*` - Initial data processing
- `/api/loadRowData/*` - Raw data operations
- `/api/adjustmentsOfData/*` - Data adjustments
- `/api/training/*` - ML training endpoints
- `/api/cloud/*` - Cloud analysis features

### WebSocket Integration

SocketIO provides real-time updates for long-running operations:
- Room-based architecture (clients join rooms by `uploadId`)
- Progress events emitted during processing
- Connection management with 60s timeout, 25s ping interval

### Environment Configuration

Required environment variables:
- `SUPABASE_URL`: Database URL
- `SUPABASE_KEY`: Database access key
- `PORT`: Server port (default: 8080)

### File Management

Directory structure:
```
chunk_uploads/       # Temporary chunk storage during upload
temp_uploads/        # Processed files with auto-cleanup
uploads/
  └── file_uploads/
      └── session_*  # Session-based file organization
```

### Security & CORS

- CORS enabled for `localhost:3000` and all origins in development
- Max file size: 100MB configured in Flask
- Chunk size limit: 10MB per chunk
- Total upload limit: 500MB

### Logging

All modules use Python's logging module with INFO level:
- Timestamp format: `%(asctime)s - %(name)s - %(levelname)s - %(message)s`
- SocketIO logging disabled to reduce noise
- Request/response logging for debugging

### Production Deployment

Docker deployment with:
- Python 3.9 slim base image
- Gunicorn with eventlet worker
- Single worker process for SocketIO compatibility
- 300-second timeout for long-running operations
- Health check every 30 seconds