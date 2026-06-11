"""Application factory for Flask app"""
import os
import logging

# Enable structured debug logging for anomaly detection when DEBUG_ANOMALY=true.
# Must run at import time so the logger level is set before any request handler runs.
if os.getenv("DEBUG_ANOMALY", "false").lower() == "true":
    logging.getLogger("domains.adjustments.debug").setLevel(logging.DEBUG)
    # Make sure root handlers also pass DEBUG through
    for _h in logging.getLogger().handlers:
        _h.setLevel(logging.DEBUG)
from datetime import datetime as dat
from flask import Flask, jsonify, request
from flask_socketio import SocketIO
from flask_cors import CORS
from apscheduler.schedulers.background import BackgroundScheduler

from core.socketio_handlers import register_socketio_handlers
from core.rate_limits import limiter

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

# Suppress noisy third-party loggers (they flood logs with DEBUG/INFO messages)
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('httpcore').setLevel(logging.WARNING)
logging.getLogger('hpack').setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('apscheduler').setLevel(logging.WARNING)
logging.getLogger('tensorflow').setLevel(logging.WARNING)
logging.getLogger('h5py').setLevel(logging.WARNING)
logging.getLogger('tzlocal').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)
# Stripe SDK logs an INFO line for every API call ("Request to Stripe api ...")
# and DEBUG includes full response bodies (customer email, sub IDs, etc.).
# Suppress to WARNING so production logs only surface real problems.
logging.getLogger('stripe').setLevel(logging.WARNING)

def _resolve_cors_origins() -> list[str]:
    """
    Resolve allowed CORS origins.

    In production (FLASK_ENV=production), CORS_ORIGINS must be explicitly
    set to a comma-separated list. If absent, raise — better to fail at
    boot than silently allow all origins with credentials.

    In development, default to local Vite/CRA dev servers.
    """
    raw = os.environ.get('CORS_ORIGINS')
    if not raw:
        if os.environ.get('FLASK_ENV') == 'production':
            raise RuntimeError(
                "CORS_ORIGINS is required in production. "
                "Set it to a comma-separated list of allowed origins."
            )
        return ['http://localhost:3000', 'http://localhost:5173']
    return [o.strip() for o in raw.split(',') if o.strip()]


def create_app():
    """Application factory function"""
    app = Flask(__name__)

    app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024

    # CORS origins resolved via fail-closed helper.
    # Production requires CORS_ORIGINS env var; dev falls back to localhost.
    allowed_origins = _resolve_cors_origins()

    # Initialize SocketIO
    socketio.init_app(app,
                     cors_allowed_origins=allowed_origins,
                     async_mode='threading',
                     logger=False,
                     engineio_logger=False,
                     ping_timeout=60,
                     ping_interval=25,
                     transports=['polling', 'websocket'],
                     always_connect=True)
    app.extensions['socketio'] = socketio

    # Initialize CORS - Flask-Cors 6.0 uses top-level kwargs instead of resources dict
    # supports_credentials=True requires explicit origins (no wildcard "*")
    cors.init_app(app,
        origins=allowed_origins,
        methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
        allow_headers=["Content-Type", "Authorization", "Accept", "Origin", "X-Requested-With"],
        expose_headers=[
            "Content-Disposition",
            "Content-Length",
            "X-Model-Env-Python",
            "X-Model-Env-TensorFlow",
            "X-Model-Env-Keras",
            "X-Model-Env-Numpy",
        ],
        supports_credentials=True,
        max_age=3600
    )

    limiter.init_app(app)

    register_socketio_handlers(socketio)
    
    from core.blueprints import register_blueprints
    register_blueprints(app)
    
    @app.errorhandler(400)
    def bad_request(error):
        logger.error(f"Bad request (400): {error}")
        if hasattr(error, 'response') and error.response:
            return error.response
        return jsonify({'success': False, 'code': 'BAD_REQUEST', 'error': str(error)}), 400

    @app.errorhandler(413)
    def payload_too_large(error):
        logger.error(f"Payload too large (413): {error}")
        if hasattr(error, 'response') and error.response:
            return error.response
        return jsonify({'success': False, 'code': 'PAYLOAD_TOO_LARGE', 'error': 'Request entity is too large'}), 413

    # W11-ADV-4: global JSON handlers for 404/405/500 to keep the error
    # contract consistent. Flask's defaults return HTML for these, which
    # breaks FE error mappers that key off `code`.
    @app.errorhandler(404)
    def _not_found(error):
        # Routes that set their own custom response (e.g., abort(404, response=...))
        # should still win.
        if hasattr(error, 'response') and error.response:
            return error.response
        return jsonify({
            'success': False,
            'code': 'NOT_FOUND',
            'error': 'Resource not found',
        }), 404

    @app.errorhandler(405)
    def _method_not_allowed(error):
        if hasattr(error, 'response') and error.response:
            return error.response
        return jsonify({
            'success': False,
            'code': 'METHOD_NOT_ALLOWED',
            'error': f'Method {request.method} not allowed for this endpoint',
        }), 405

    @app.errorhandler(500)
    def internal_error(error):
        # logger.exception captures stack trace server-side. Client only
        # sees the standardized {success, code, error} contract — no leak.
        logger.exception("Unhandled server error (500)")
        if hasattr(error, 'response') and error.response:
            return error.response
        return jsonify({
            'success': False,
            'code': 'INTERNAL_ERROR',
            'error': 'An unexpected error occurred',
        }), 500
    
    @app.route('/health')
    def health():
        return jsonify(status="ok"), 200
    
    @app.route('/')
    def index():
        try:
            logger.debug("Handling request to index endpoint")
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
                logger.debug(f"Scheduled cleanup completed: {result.get('message', 'No message')}")
            except Exception as e:
                logger.error(f"Error in scheduled cleanup: {str(e)}")

    def run_chunk_cleanup():
        """Clean up expired chunk uploads from local filesystem"""
        try:
            from domains.processing.services.local_chunk_service import local_chunk_service
            deleted = local_chunk_service.cleanup_all_expired()
            if deleted > 0:
                logger.debug(f"Cleaned up {deleted} expired chunk uploads from local filesystem")
        except Exception as e:
            logger.error(f"Error in chunk cleanup: {str(e)}")

    def run_processed_files_cleanup():
        """Clean up old processed files from Supabase Storage (24h expiry) - for legacy files"""
        try:
            from shared.storage.service import storage_service
            deleted = storage_service.cleanup_all_old_files(max_age_hours=24)
            if deleted > 0:
                logger.debug(f"Cleaned up {deleted} old processed files from Supabase Storage")
        except Exception as e:
            logger.error(f"Error in processed files cleanup: {str(e)}")

    def run_local_processed_results_cleanup():
        """Clean up old processed results from local filesystem (24h expiry)"""
        try:
            from domains.processing.services.local_chunk_service import local_chunk_service
            deleted = local_chunk_service.cleanup_old_processed_results(max_age_hours=24)
            if deleted > 0:
                logger.debug(f"Cleaned up {deleted} old processed results from local filesystem")
        except Exception as e:
            logger.error(f"Error in local processed results cleanup: {str(e)}")

    scheduler.add_job(run_cleanup_with_app_context, 'interval', minutes=30, id='cleanup_job')
    scheduler.add_job(run_chunk_cleanup, 'interval', minutes=30, id='chunk_cleanup_job')
    scheduler.add_job(run_processed_files_cleanup, 'interval', hours=6, id='processed_files_cleanup_job')
    scheduler.add_job(run_local_processed_results_cleanup, 'interval', hours=6, id='local_processed_cleanup_job')
    scheduler.start()

    # Clean up orphaned training progress entries on startup
    # This handles cases where backend crashed/restarted during training
    def run_training_progress_cleanup():
        """Clean up stale training_progress entries (orphaned from crashed trainings)"""
        try:
            from domains.training.services.training_tracker import cleanup_stale_training_progress
            cleanup_stale_training_progress()
        except Exception as e:
            logger.error(f"Error in training progress cleanup: {str(e)}")

    # Run immediately on startup
    run_training_progress_cleanup()

    # Also run periodically (every 5 minutes) to catch any missed cleanups
    scheduler.add_job(run_training_progress_cleanup, 'interval', minutes=5, id='training_progress_cleanup_job')

    return app, socketio
