"""SocketIO event handlers"""
import logging
from flask_socketio import join_room, leave_room, emit
from utils.database import get_supabase_client
from services.training.training_api import create_or_get_session_uuid

logger = logging.getLogger(__name__)

def register_socketio_handlers(socketio):
    """Register all SocketIO event handlers"""
    
    @socketio.on('connect')
    def handle_connect():
        try:
            logger.info("Client connected")
        except Exception as e:
            logger.error(f"Error in connect handler: {str(e)}")

    @socketio.on('disconnect')
    def handle_disconnect(*args, **kwargs):
        try:
            logger.info("Client disconnected")
            return True
        except Exception as e:
            logger.error(f"Error in disconnect handler: {str(e)}")
            return False

    @socketio.on('join_upload_room')
    def handle_join_upload_room(data):
        """Client joins a room for upload progress tracking"""
        try:
            upload_id = data.get('uploadId')
            if upload_id:
                join_room(upload_id)
                logger.info(f"Client joined upload room: {upload_id}")
                emit('joined_room', {'uploadId': upload_id}, room=upload_id)
        except Exception as e:
            logger.error(f"Error in join_upload_room: {str(e)}")

    @socketio.on('join_training_session')
    def handle_join_training_session(data):
        """
        Allow clients to join training session rooms for real-time updates
        """
        try:
            session_id = data.get('session_id')
            if session_id:
                room = f"training_{session_id}"
                join_room(room)
                logger.info(f"Client joined training room: {room}")
                
                emit('training_session_joined', {
                    'status': 'success',
                    'session_id': session_id,
                    'room': room,
                    'message': 'Successfully joined training session'
                })
            else:
                emit('training_session_error', {
                    'status': 'error',
                    'message': 'Session ID is required'
                })
        except Exception as e:
            logger.error(f"Error joining training session: {str(e)}")
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
            session_id = data.get('session_id') if data else None
            if session_id:
                room = f"training_{session_id}"
                leave_room(room)
                logger.info(f"Client left training room: {room}")
                
                emit('training_session_left', {
                    'status': 'success',
                    'session_id': session_id,
                    'message': 'Successfully left training session'
                })
            return True
        except Exception as e:
            logger.error(f"Error leaving training session: {str(e)}")
            return False

    @socketio.on('request_training_status')
    def handle_request_training_status(data):
        """
        Handle requests for current training status
        """
        try:
            session_id = data.get('session_id')
            if session_id:
                supabase = get_supabase_client()
                uuid_session_id = create_or_get_session_uuid(session_id)
                
                if uuid_session_id:
                    results = supabase.table('training_results').select('*').eq('session_id', uuid_session_id).order('created_at', desc=True).limit(1).execute()
                    
                    if results.data:
                        training_status = results.data[0]
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
            emit('training_status_error', {
                'message': f'Failed to get status: {str(e)}'
            })

    @socketio.on('join')
    def handle_join(data):
        """Client joins a room based on uploadId"""
        try:
            if 'uploadId' in data:
                upload_id = data['uploadId']
                join_room(upload_id)
                logger.info(f"Client joined Socket.IO room: {upload_id}")
                
                socketio.emit('status', {'message': f'Joined room: {upload_id}'}, room=upload_id)
        except Exception as e:
            logger.error(f"Error in join handler: {str(e)}")


    @socketio.on('leave')
    def handle_leave(data):
        """Client leaves a room based on uploadId"""
        try:
            if 'uploadId' in data:
                upload_id = data['uploadId']
                leave_room(upload_id)
                logger.info(f"Client left Socket.IO room: {upload_id}")
                
                emit('left_room', {'uploadId': upload_id})
        except Exception as e:
            logger.error(f"Error in leave handler: {str(e)}")
    @socketio.on('request_dataset_status')
    def handle_request_dataset_status(data):
        """
        Handle requests for dataset generation status updates
        """
        try:
            session_id = data.get('session_id')
            if session_id:
                logger.info(f"Frontend requesting dataset status for session: {session_id}")
                
                supabase = get_supabase_client()
                uuid_session_id = create_or_get_session_uuid(session_id)
                
                if uuid_session_id:
                    results = supabase.table('training_results').select('*').eq('session_id', uuid_session_id).order('created_at', desc=True).limit(1).execute()
                    
                    if results.data:
                        dataset_status = results.data[0]
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
            emit('dataset_status_error', {
                'message': f'Failed to get dataset status: {str(e)}'
            })

    @socketio.on_error_default
    def default_error_handler(e):
        logger.error(f"SocketIO error: {str(e)}")
        return False
