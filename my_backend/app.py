import os
import logging

from flask_socketio import SocketIO
from datetime import datetime as dat
from flask import Flask, jsonify, request
from flask_cors import CORS
from apscheduler.schedulers.background import BackgroundScheduler

from firstProcessing import bp as first_processing_bp
from load_row_data import bp as load_row_data_bp
from data_processing_main import bp as data_processing_bp
from training import bp as training_bp
from adjustmentsOfData import bp as adjustmentsOfData_bp
from adjustmentsOfData import cleanup_old_files
from cloud import bp as cloud_bp
from training_system.training_api import training_api_bp

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configure request size limits
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100 MB limit

# Nakon inicijalizacije Flask aplikacije i CORS-a, dodajte:
socketio = SocketIO(app, 
                   cors_allowed_origins="*", 
                   async_mode='threading',
                   logger=False,
                   engineio_logger=False,
                   ping_timeout=60,
                   ping_interval=25)

# Register socketio in app extensions for current_app access
app.extensions['socketio'] = socketio

# Configure CORS with more permissive settings
CORS(app, resources={
    r"/*": {
        "origins": ["http://localhost:3000", "http://127.0.0.1:3000", "*"],
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
        "allow_headers": ["Content-Type", "Authorization", "Accept", "Origin", "X-Requested-With"],
        "expose_headers": ["Content-Disposition", "Content-Length"],
        "supports_credentials": True,
        "max_age": 3600
    }
})

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

# Register blueprints with correct prefixes
app.register_blueprint(data_processing_bp)
app.register_blueprint(load_row_data_bp, url_prefix='/api/loadRowData')
app.register_blueprint(first_processing_bp, url_prefix='/api/firstProcessing')
app.register_blueprint(cloud_bp, url_prefix='/api/cloud')
app.register_blueprint(adjustmentsOfData_bp, url_prefix='/api/adjustmentsOfData')
app.register_blueprint(training_bp, url_prefix='/api/training')
app.register_blueprint(training_api_bp)  # Already has /api/training prefix

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

@socketio.on('connect')
def handle_connect():
    try:
        logger.info("Client connected")
    except Exception as e:
        logger.error(f"Error in connect handler: {str(e)}")

@socketio.on('disconnect')
def handle_disconnect():
    try:
        logger.info("Client disconnected")
    except Exception as e:
        logger.error(f"Error in disconnect handler: {str(e)}")

# RowData Socket.IO event handlers
@socketio.on('join_upload_room')
def handle_join_upload_room(data):
    """Client joins a room for upload progress tracking"""
    try:
        upload_id = data.get('uploadId')
        if upload_id:
            from flask_socketio import join_room, emit
            join_room(upload_id)
            logger.info(f"Client joined upload room: {upload_id}")
            emit('joined_room', {'uploadId': upload_id}, room=upload_id)
    except Exception as e:
        logger.error(f"Error in join_upload_room: {str(e)}")

# Enhanced SocketIO event handlers for training system
@socketio.on('join_training_session')
def handle_join_training_session(data):
    """
    Allow clients to join training session rooms for real-time updates
    """
    try:
        session_id = data.get('session_id')
        if session_id:
            room = f"training_{session_id}"
            from flask_socketio import join_room, emit
            join_room(room)
            logger.info(f"Client joined training room: {room}")
            
            # Send confirmation
            emit('training_session_joined', {
                'status': 'success',
                'session_id': session_id,
                'room': room,
                'message': 'Successfully joined training session'
            })
        else:
            from flask_socketio import emit
            emit('training_session_error', {
                'status': 'error',
                'message': 'Session ID is required'
            })
    except Exception as e:
        logger.error(f"Error joining training session: {str(e)}")
        from flask_socketio import emit
        emit('training_session_error', {
            'status': 'error', 
            'message': f'Failed to join session: {str(e)}'
        })

@socketio.on('leave_training_session')
def handle_leave_training_session(data):
    """
    Allow clients to leave training session rooms
    """
    try:
        session_id = data.get('session_id')
        if session_id:
            room = f"training_{session_id}"
            from flask_socketio import leave_room, emit
            leave_room(room)
            logger.info(f"Client left training room: {room}")
            
            emit('training_session_left', {
                'status': 'success',
                'session_id': session_id,
                'message': 'Successfully left training session'
            })
    except Exception as e:
        logger.error(f"Error leaving training session: {str(e)}")

@socketio.on('request_training_status')
def handle_request_training_status(data):
    """
    Handle requests for current training status
    """
    try:
        session_id = data.get('session_id')
        if session_id:
            # Get current training status from database
            from supabase_client import get_supabase_client
            from training_system.training_api import create_or_get_session_uuid
            
            supabase = get_supabase_client()
            uuid_session_id = create_or_get_session_uuid(session_id)
            
            if uuid_session_id:
                # Get latest training result
                results = supabase.table('training_results').select('*').eq('session_id', uuid_session_id).order('created_at', desc=True).limit(1).execute()
                
                if results.data:
                    training_status = results.data[0]
                    from flask_socketio import emit
                    emit('training_status_update', {
                        'session_id': session_id,
                        'status': training_status.get('status', 'unknown'),
                        'message': f"Current status: {training_status.get('status', 'unknown')}",
                        'last_updated': training_status.get('updated_at', training_status.get('created_at')),
                        'models_trained': training_status.get('summary', {}).get('models_trained', 0) if training_status.get('summary') else 0
                    })
                else:
                    from flask_socketio import emit
                    emit('training_status_update', {
                        'session_id': session_id,
                        'status': 'not_found',
                        'message': 'No training data found for this session'
                    })
            else:
                from flask_socketio import emit
                emit('training_status_error', {
                    'session_id': session_id,
                    'message': 'Session not found'
                })
    except Exception as e:
        logger.error(f"Error getting training status: {str(e)}")
        from flask_socketio import emit
        emit('training_status_error', {
            'message': f'Failed to get status: {str(e)}'
        })

@socketio.on('join')
def handle_join(data):
    """Client joins a room based on uploadId"""
    try:
        if 'uploadId' in data:
            upload_id = data['uploadId']
            from flask_socketio import join_room
            join_room(upload_id)
            logger.info(f"Client joined Socket.IO room: {upload_id}")
            
            # Send status confirmation
            socketio.emit('status', {'message': f'Joined room: {upload_id}'}, room=upload_id)
    except Exception as e:
        logger.error(f"Error in join handler: {str(e)}")

@socketio.on('request_dataset_status')
def handle_request_dataset_status(data):
    """
    Handle requests for dataset generation status updates
    """
    try:
        session_id = data.get('session_id')
        if session_id:
            logger.info(f"Frontend requesting dataset status for session: {session_id}")
            
            # Get dataset status from database
            from supabase_client import get_supabase_client
            from training_system.training_api import create_or_get_session_uuid
            
            supabase = get_supabase_client()
            uuid_session_id = create_or_get_session_uuid(session_id)
            
            if uuid_session_id:
                # Get latest training result
                results = supabase.table('training_results').select('*').eq('session_id', uuid_session_id).order('created_at', desc=True).limit(1).execute()
                
                if results.data:
                    dataset_status = results.data[0]
                    from flask_socketio import emit
                    emit('dataset_status_update', {
                        'session_id': session_id,
                        'status': dataset_status.get('status', 'unknown'),
                        'message': dataset_status.get('error_message') if dataset_status.get('status') == 'data_validation_error' else f"Current status: {dataset_status.get('status', 'unknown')}",
                        'error_details': dataset_status.get('error_details') if dataset_status.get('status') == 'data_validation_error' else None,
                        'last_updated': dataset_status.get('updated_at', dataset_status.get('created_at')),
                        'processing_stopped': dataset_status.get('summary', {}).get('processing_stopped', False) if dataset_status.get('summary') else False
                    })
                else:
                    from flask_socketio import emit
                    emit('dataset_status_update', {
                        'session_id': session_id,
                        'status': 'not_found',
                        'message': 'No dataset generation data found for this session'
                    })
            else:
                from flask_socketio import emit
                emit('dataset_status_error', {
                    'session_id': session_id,
                    'message': 'Session not found'
                })
    except Exception as e:
        logger.error(f"Error getting dataset status: {str(e)}")
        from flask_socketio import emit
        emit('dataset_status_error', {
            'message': f'Failed to get dataset status: {str(e)}'
        })

# Add global SocketIO error handler
@socketio.on_error_default
def default_error_handler(e):
    logger.error(f"SocketIO error: {str(e)}")
    return False

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
    socketio.run(app, host='0.0.0.0', port=8080, debug=False, allow_unsafe_werkzeug=True)
