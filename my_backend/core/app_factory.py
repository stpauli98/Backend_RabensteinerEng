"""Application factory for Flask app"""
import os
import logging
from datetime import datetime as dat
from flask import Flask, jsonify, request
from flask_socketio import SocketIO
from flask_cors import CORS
from apscheduler.schedulers.background import BackgroundScheduler

from core.socketio_handlers import register_socketio_handlers

socketio = SocketIO()
cors = CORS()

# Configure logging level from environment (default: INFO for production)
# Set LOG_LEVEL=DEBUG in .env for verbose logging during development
_log_level = getattr(logging, os.environ.get('LOG_LEVEL', 'INFO').upper(), logging.INFO)
logging.basicConfig(
    level=_log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress noisy third-party loggers (they flood logs with INFO level messages)
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('apscheduler').setLevel(logging.WARNING)

def create_app():
    """Application factory function"""
    app = Flask(__name__)
    
    app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024

    # CORS origins from environment (default: "*" for backward compatibility)
    # Set CORS_ORIGINS=https://entropia-seven.vercel.app,http://localhost:3000 in production
    _cors_origins = os.environ.get('CORS_ORIGINS', '*')
    _socketio_cors = _cors_origins.split(',') if _cors_origins != '*' else '*'

    # Initialize SocketIO
    socketio.init_app(app,
                     cors_allowed_origins=_socketio_cors,
                     async_mode='threading',
                     logger=False,
                     engineio_logger=False,
                     ping_timeout=60,
                     ping_interval=25,
                     transports=['polling', 'websocket'],
                     always_connect=True)
    app.extensions['socketio'] = socketio

    # Initialize CORS with configurable origins
    _flask_cors_origins = _cors_origins.split(',') if _cors_origins != '*' else ["http://localhost:3000", "http://127.0.0.1:3000", "https://entropia-seven.vercel.app", "*"]
    cors.init_app(app, resources={
        r"/*": {
            "origins": _flask_cors_origins,
            "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
            "allow_headers": ["Content-Type", "Authorization", "Accept", "Origin", "X-Requested-With"],
            "expose_headers": ["Content-Disposition", "Content-Length"],
            "supports_credentials": True,
            "max_age": 3600
        }
    })

    register_socketio_handlers(socketio)
    
    @app.before_request
    def handle_preflight():
        if request.method == "OPTIONS":
            response = app.make_default_options_response()
            headers = response.headers
            headers['Access-Control-Allow-Origin'] = request.headers.get('Origin', '*')
            headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS, PATCH'
            headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization, Accept, Origin, X-Requested-With'
            headers['Access-Control-Allow-Credentials'] = 'true'
            headers['Access-Control-Max-Age'] = '3600'
            return response
    
    from core.blueprints import register_blueprints
    register_blueprints(app)
    
    @app.errorhandler(400)
    def bad_request(error):
        logger.error(f"Bad request (400): {error}")
        if hasattr(error, 'response') and error.response:
            return error.response
        return jsonify({'error': 'Bad Request', 'message': str(error)}), 400

    @app.errorhandler(413)
    def payload_too_large(error):
        logger.error(f"Payload too large (413): {error}")
        if hasattr(error, 'response') and error.response:
            return error.response
        return jsonify({'error': 'Payload Too Large', 'message': 'Request entity is too large'}), 413

    @app.errorhandler(500)
    def internal_error(error):
        logger.error(f"Internal server error (500): {error}")
        if hasattr(error, 'response') and error.response:
            return error.response
        return jsonify({'error': 'Internal Server Error', 'message': str(error)}), 500
    
    @app.route('/health')
    def health():
        return jsonify(status="ok"), 200
    
    @app.route('/')
    def index():
        try:
            logger.info("Handling request to index endpoint")
            return jsonify({
                'status': 'online',
                'message': 'Backend service is running',
                'version': '1.0.0',
                'timestamp': str(dat.now()),
                'port': os.environ.get('PORT', '8080'),
                'env': os.environ.get('FLASK_ENV', 'production')
            })
        except Exception as e:
            logger.error(f"Error in index route: {e}")
            return jsonify({'error': str(e)}), 500
    
    scheduler = BackgroundScheduler(daemon=True)

    from services.adjustments.cleanup import cleanup_old_files

    def run_cleanup_with_app_context():
        with app.app_context():
            try:
                result = cleanup_old_files()
                logger.info(f"Scheduled cleanup completed: {result.get('message', 'No message')}")
            except Exception as e:
                logger.error(f"Error in scheduled cleanup: {str(e)}")

    def run_chunk_cleanup():
        """Clean up expired chunk uploads from Supabase Storage"""
        try:
            from shared.storage.chunk_service import chunk_storage_service
            deleted = chunk_storage_service.cleanup_expired_uploads(max_age_hours=1.0)
            if deleted > 0:
                logger.info(f"Cleaned up {deleted} expired chunk uploads from Supabase Storage")
        except Exception as e:
            logger.error(f"Error in chunk cleanup: {str(e)}")

    def run_processed_files_cleanup():
        """Clean up old processed files from Supabase Storage (24h expiry)"""
        try:
            from shared.storage.service import storage_service
            deleted = storage_service.cleanup_all_old_files(max_age_hours=24)
            if deleted > 0:
                logger.info(f"Cleaned up {deleted} old processed files from Supabase Storage")
        except Exception as e:
            logger.error(f"Error in processed files cleanup: {str(e)}")

    scheduler.add_job(run_cleanup_with_app_context, 'interval', minutes=30, id='cleanup_job')
    scheduler.add_job(run_chunk_cleanup, 'interval', minutes=30, id='chunk_cleanup_job')
    scheduler.add_job(run_processed_files_cleanup, 'interval', hours=6, id='processed_files_cleanup_job')
    scheduler.start()
    
    return app, socketio
