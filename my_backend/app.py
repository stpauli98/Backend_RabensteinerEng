import os
import logging

from flask_socketio import SocketIO
from datetime import datetime as dat
from flask import Flask, jsonify, request
from flask_cors import CORS

# Import consolidated blueprints
from api.data_pipeline import bp as data_pipeline_bp
from api.analytics import bp as analytics_bp
from api.machine_learning import bp as machine_learning_bp
from api.system import bp as system_bp
from api.backward_compatibility import bp as compatibility_bp, COMPATIBILITY_ROUTES

# Import centralized scheduler service
from services.scheduler import init_scheduler, start_scheduler

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

# ============================================================================
# REGISTER CONSOLIDATED BLUEPRINTS
# ============================================================================

# Register new consolidated blueprints
app.register_blueprint(data_pipeline_bp, url_prefix='/api/data')
app.register_blueprint(analytics_bp, url_prefix='/api/analytics')  
app.register_blueprint(machine_learning_bp, url_prefix='/api/ml')
app.register_blueprint(system_bp, url_prefix='/api/system')

# Register backward compatibility blueprint
app.register_blueprint(compatibility_bp)

# Register legacy compatibility routes dynamically
for route_config in COMPATIBILITY_ROUTES:
    app.add_url_rule(
        route_config['rule'],
        endpoint=route_config['endpoint'],
        view_func=route_config['handler'],
        methods=route_config['methods']
    )

logger.info("Registered consolidated API blueprints with backward compatibility")

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

# Legacy health check endpoint - redirects to system health
@app.route('/health')
def health():
    from flask import redirect
    return redirect('/api/system/health', code=301)

# Debug endpoint removed - already exists below

@socketio.on('connect')
def handle_connect(sid=None):
    try:
        if sid:
            logger.info(f"Client {sid} connected")
        else:
            logger.info("Client connected")
    except Exception as e:
        logger.error(f"Error in connect handler: {str(e)}")

@socketio.on('disconnect')
def handle_disconnect(sid=None):
    try:
        if sid:
            logger.info(f"Client {sid} disconnected")
        else:
            logger.info("Client disconnected")
    except Exception as e:
        logger.error(f"SocketIO error: {str(e)}")

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
            from flask_socketio import join_room
            join_room(room)
            logger.info(f"Client joined training room: {room}")
            
            # Send confirmation
            from flask_socketio import emit
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
            from services.supabase_client import get_supabase_client
            from models.training_system.training_api import create_or_get_session_uuid
            
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
                    emit('training_status_update', {
                        'session_id': session_id,
                        'status': 'not_found',
                        'message': 'No training data found for this session'
                    })
            else:
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
            from services.supabase_client import get_supabase_client
            from models.training_system.training_api import create_or_get_session_uuid
            
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
                    emit('dataset_status_update', {
                        'session_id': session_id,
                        'status': 'not_found',
                        'message': 'No dataset generation data found for this session'
                    })
            else:
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

# Initialize centralized scheduler
scheduler = init_scheduler(app)
start_scheduler()

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 8080))
    socketio.run(app, host='0.0.0.0', port=port, debug=False, allow_unsafe_werkzeug=True)
