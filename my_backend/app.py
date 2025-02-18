from flask import Flask, request, jsonify
from flask_cors import CORS
import firstProcessing
import load_row_data

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return jsonify({
        'status': 'online',
        'message': 'Backend service is running',
        'version': '1.0.0'
    })

CORS(app, resources={
    r"/*": {
        "origins": [
            "https://rabensteinerengineering.onrender.com",
            "https://localhost:3000",
            "http://localhost:3000"  # Za lokalni development
        ],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": [
            "Content-Type",
            "Accept",
            "Authorization",
            "X-Requested-With",
            "Content-Length",
            "Content-Range",
            "X-Content-Type-Options"
        ],
        "expose_headers": [
            "Content-Length",
            "Content-Range",
            "Content-Encoding"
        ],
        "supports_credentials": True,
        "max_age": 1728000  # 20 days
    }
})

# Define API prefix for firstProcessing routes
API_PREFIX = '/api/firstProcessing'

@app.route(f'{API_PREFIX}/upload_chunk', methods=['POST'])
def upload_chunk_endpoint():
    return firstProcessing.upload_chunk(request)

@app.route(f'{API_PREFIX}/prepare-save', methods=['POST'])
def prepare_save_endpoint():
    return firstProcessing.prepare_save(request)

@app.route(f'{API_PREFIX}/download/<file_id>', methods=['GET'])
def download_file_endpoint(file_id):
    return firstProcessing.download_file(file_id, request)

@app.route('/load-data', methods=['POST'])
def load_data_endpoint():
    try:
        data = request.get_json()
        # Add your data loading logic here using load_row_data.py functions
        result = load_row_data.load_data()
        return jsonify({"status": "success", "data": result})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
