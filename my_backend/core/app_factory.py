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

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_app():
    """Application factory function"""
    app = Flask(__name__)
    
    app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024

    # Initialize SocketIO
    socketio.init_app(app,
                     cors_allowed_origins="*",
                     async_mode='threading',
                     logger=False,
                     engineio_logger=False,
                     ping_timeout=60,
                     ping_interval=25,
                     transports=['polling', 'websocket'],
                     always_connect=True)
    app.extensions['socketio'] = socketio

    # Initialize CORS
    cors.init_app(app, resources={
        r"/*": {
            "origins": ["http://localhost:3000", "http://127.0.0.1:3000", "https://entropia-seven.vercel.app", "*"],
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
    
    scheduler.add_job(run_cleanup_with_app_context, 'interval', minutes=30, id='cleanup_job')
    scheduler.start()
    
    return app, socketio
