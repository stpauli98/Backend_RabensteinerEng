"""Main entry point for the Flask application"""
import os
from core.app_factory import create_app

# Create the application
app, socketio = create_app()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    socketio.run(app, host='0.0.0.0', port=port, debug=False, allow_unsafe_werkzeug=True)