#!/usr/bin/env python3
"""
Test Flask app sa samo RowData modulom
"""
import os
from flask import Flask
from flask_socketio import SocketIO
from flask_cors import CORS

# Postavi da koristi file storage
os.environ['ROWDATA_STORAGE_BACKEND'] = 'file'
os.environ['ROWDATA_FILE_STORAGE_PATH'] = '/tmp/rowdata_test'

# Kreiraj Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'test-secret-key'

# CORS
CORS(app, resources={r"/*": {"origins": "*"}})

# Socket.IO
socketio = SocketIO(app, cors_allowed_origins="*")

# Importuj i registruj RowData blueprint
from RowData import rowdata_blueprint
app.register_blueprint(rowdata_blueprint, url_prefix='/api/loadRowData')

# Registruj u app context za RowData
app.extensions['socketio'] = socketio

@app.route('/')
def home():
    return {"message": "Test RowData server is running!"}

@app.route('/health')
def health():
    return {"status": "healthy", "module": "RowData"}

if __name__ == '__main__':
    print("=" * 50)
    print("Starting test server with RowData module...")
    print("Using file-based storage (no Redis required)")
    print("=" * 50)
    
    socketio.run(app, 
                 host='0.0.0.0', 
                 port=5001,  # Koristimo port 5001 da ne interferira sa glavnom app
                 debug=True,
                 allow_unsafe_werkzeug=True)  # Za development