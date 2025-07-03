# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Running the Application
- **Development**: `python app.py` - Runs Flask app with SocketIO on port 8080
- **Production**: `gunicorn app:app` - Uses gunicorn with configuration from Dockerfile
- **Docker**: `docker build -t my_backend .` then `docker run -p 8080:8080 my_backend`

### Dependencies
- **Install**: `pip install -r requirements.txt`
- **Update**: Update `requirements.txt` manually when adding new dependencies

### Testing
- No automated test framework is configured
- Manual testing via API endpoints and health check at `/health`

## Architecture Overview

### Core Application Structure
- **app.py**: Main Flask application with SocketIO integration, blueprint registration, and scheduled cleanup
- **Blueprint-based modular design**: Each major feature is a separate blueprint module

### Key Modules
- **firstProcessing.py**: Initial CSV data processing, chunked file upload handling
- **load_row_data.py**: Raw data loading with support for multiple datetime formats
- **data_processing_main.py**: Advanced data cleaning and filtering operations
- **adjustmentsOfData.py**: Data adjustment operations with temporary file management
- **training.py**: Machine learning model training and file metadata extraction
- **cloud.py**: Cloud-based data analysis with matplotlib visualization
- **supabase_client.py**: Database operations using Supabase client

### Data Flow Architecture
1. **File Upload**: Chunked upload system stores temporary files in `chunk_uploads/`
2. **Processing Pipeline**: Data flows through firstProcessing → load_row_data → data_processing_main → adjustmentsOfData
3. **Real-time Updates**: SocketIO provides progress tracking during long-running operations
4. **Temporary Storage**: Files stored in `temp_uploads/` with automatic cleanup every 30 minutes
5. **Persistence**: Session data and metadata saved to Supabase database

### WebSocket Integration
- **SocketIO**: Real-time communication for file processing progress
- **Room-based**: Clients join rooms based on uploadId for targeted updates
- **Progress Events**: Emit progress updates during data processing operations

### File Management
- **Chunk Storage**: `chunk_uploads/` for temporary file chunks during upload
- **Temp Storage**: `temp_uploads/` for processed files with automatic cleanup
- **Session Storage**: `uploads/file_uploads/session_*` for organized file sessions

### Environment Configuration
- **Production**: Uses gunicorn with 1 worker, 8 threads, no timeout
- **CORS**: Configured for `localhost:3000` with full method support
- **Logging**: INFO level logging across all modules
- **Database**: Supabase integration via environment variables (`SUPABASE_URL`, `SUPABASE_KEY`)

### Data Processing Features
- **CSV Processing**: Support for multiple datetime formats and data cleaning
- **Interpolation**: Time series data interpolation with configurable parameters
- **Outlier Detection**: Statistical outlier removal with customizable thresholds
- **Visualization**: Matplotlib-based chart generation for data analysis
- **Machine Learning**: Scikit-learn integration for model training and predictions