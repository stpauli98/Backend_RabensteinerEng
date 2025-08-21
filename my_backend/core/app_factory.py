"""Application factory for Flask app"""
import os
import logging
from datetime import datetime as dat
from flask import Flask, jsonify, request
from apscheduler.schedulers.background import BackgroundScheduler

from core.extensions import init_extensions
from core.socketio_handlers import register_socketio_handlers

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_app():
    """Application factory function"""
    app = Flask(__name__)
    
    # Configure request size limits
    app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100 MB limit
    
    # Initialize extensions
    socketio = init_extensions(app)
    
    # Register Socket.IO handlers
    register_socketio_handlers(socketio)
    
    # Add explicit OPTIONS handler for all routes
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
    
    # Register blueprints
    from api.routes import register_blueprints
    register_blueprints(app)
    
    # Error handlers
    @app.errorhandler(400)
    def bad_request(error):
        logger.error(f"Bad request (400): {error}")
        return jsonify({'error': 'Bad Request', 'message': str(error)}), 400
    
    @app.errorhandler(413)
    def payload_too_large(error):
        logger.error(f"Payload too large (413): {error}")
        return jsonify({'error': 'Payload Too Large', 'message': 'Request entity is too large'}), 413
    
    @app.errorhandler(500)
    def internal_error(error):
        logger.error(f"Internal server error (500): {error}")
        return jsonify({'error': 'Internal Server Error', 'message': str(error)}), 500
    
    # Health check endpoint
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
    
    # Initialize the scheduler
    scheduler = BackgroundScheduler(daemon=True)
    
    # Import cleanup function
    from services.adjustments.cleanup import cleanup_old_files
    
    # Create a wrapper function that runs cleanup_old_files within the app context
    def run_cleanup_with_app_context():
        with app.app_context():
            try:
                result = cleanup_old_files()
                logger.info(f"Scheduled cleanup completed: {result.get('message', 'No message')}")
            except Exception as e:
                logger.error(f"Error in scheduled cleanup: {str(e)}")
    
    # Schedule the wrapper function to run every 30 minutes
    scheduler.add_job(run_cleanup_with_app_context, 'interval', minutes=30, id='cleanup_job')
    # Start the scheduler
    scheduler.start()
    
    return app, socketio