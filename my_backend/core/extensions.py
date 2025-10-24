"""Flask extensions initialization"""
from flask_socketio import SocketIO
from flask_cors import CORS

socketio = SocketIO()
cors = CORS()

def init_extensions(app):
    """Initialize Flask extensions with app"""
    
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
    
    return socketio
