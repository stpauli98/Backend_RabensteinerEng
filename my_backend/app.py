from flask import Flask, request, jsonify
from flask_cors import CORS
import firstProcessing
import load_row_data

app = Flask(__name__)
CORS(app)

# API configuration
API_PREFIX_LOAD_ROW_DATA = '/api/loadRowData'
API_PREFIX_FIRST_PROCESSING = '/api/firstProcessing'

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
            "https://backend-rabensteinereng.onrender.com",
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

#LoadRowData

@app.route(f'{API_PREFIX_LOAD_ROW_DATA}/upload', methods=['POST'])
def load_row_data_upload_endpoint():
    return load_row_data.upload_files()

@app.route(f'{API_PREFIX_LOAD_ROW_DATA}/download/<file_id>', methods=['GET'])
def load_row_data_download_endpoint(file_id):
    return load_row_data.download_file(file_id)

@app.route(f'{API_PREFIX_LOAD_ROW_DATA}/prepare-save', methods=['POST'])
def load_row_data_prepare_save_endpoint():
    return load_row_data.prepare_save(request)


#FirstProcessing

@app.route(f'{API_PREFIX_FIRST_PROCESSING}/upload_chunk', methods=['POST'])
def first_processing_upload_chunk_endpoint():
    return firstProcessing.upload_chunk(request)

@app.route(f'{API_PREFIX_FIRST_PROCESSING}/prepare-save', methods=['POST'])
def first_processing_prepare_save_endpoint():
    return firstProcessing.prepare_save(request)

@app.route(f'{API_PREFIX_FIRST_PROCESSING}/download/<file_id>', methods=['GET'])
def first_processing_download_file_endpoint(file_id):
    return firstProcessing.download_file(file_id, request)


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
