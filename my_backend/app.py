import os
import logging

from flask_socketio import SocketIO
from datetime import datetime as dat
from flask import Flask, jsonify
from flask_cors import CORS
from apscheduler.schedulers.background import BackgroundScheduler

from firstProcessing import bp as first_processing_bp
from load_row_data import bp as load_row_data_bp
from data_processing_main import bp as data_processing_bp
from training import bp as training_bp
from adjustmentsOfData import bp as adjustmentsOfData_bp
from adjustmentsOfData import cleanup_old_files
from cloud import bp as cloud_bp

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Nakon inicijalizacije Flask aplikacije i CORS-a, dodajte:
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

# Configure CORS with more permissive settings
CORS(app, resources={
    r"/*": {
        "origins": ["http://localhost:3000", "*"],
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
        "allow_headers": ["Content-Type", "Authorization", "Accept", "Origin", "X-Requested-With"],
        "expose_headers": ["Content-Disposition", "Content-Length"],
        "supports_credentials": True,
        "max_age": 3600
    }
})

# Register blueprints with correct prefixes
app.register_blueprint(data_processing_bp)
app.register_blueprint(load_row_data_bp, url_prefix='/api/loadRowData')
app.register_blueprint(first_processing_bp, url_prefix='/api/firstProcessing')
app.register_blueprint(cloud_bp, url_prefix='/api/cloud')
app.register_blueprint(adjustmentsOfData_bp, url_prefix='/api/adjustmentsOfData')
app.register_blueprint(training_bp, url_prefix='/api/training')

# Health check endpoint
@app.route('/health')
def health():
    return jsonify(status="ok"), 200

@socketio.on('connect')
def handle_connect():
    logger.info("Client connected")

@socketio.on('disconnect')
def handle_disconnect():
    logger.info("Client disconnected")

@socketio.on('join')
def handle_join(data):
    """Client joins a room based on uploadId"""
    if 'uploadId' in data:
        upload_id = data['uploadId']
        from flask_socketio import join_room
        join_room(upload_id)
        socketio.emit('status', {'message': f'Joined room: {upload_id}'}, room=upload_id)

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

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=8080, debug=False)
