from flask import Flask, request, jsonify
from flask_cors import CORS
import firstProcessing
import load_row_data
import data_processing_main
from data_processing_main import zweite_bearbeitung, prepare_save, download_file

app = Flask(__name__)
CORS(app)

# API configuration
API_PREFIX_LOAD_ROW_DATA = '/api/loadRowData'
API_PREFIX_FIRST_PROCESSING = '/api/firstProcessing'
API_PREFIX_DATA_PROCESSING_MAIN = '/api/dataProcessingMain'

@app.route('/')
def index():
    return jsonify({
        'status': 'online',
        'message': 'Backend service is running',
        'version': '1.0.0'
    })

# Configure CORS with more permissive settings
CORS(app, resources={
    r"/*": {
        "origins": [
            "https://rabensteinerengineering.onrender.com",
            "https://backend-rabensteinereng.onrender.com",
            "http://localhost:3000",
            "https://localhost:3000"
        ],
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
        "allow_headers": "*",
        "expose_headers": "*",
        "supports_credentials": True,
        "max_age": 600  # 10 minutes
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

#DataProcessingMain

@app.route(f'{API_PREFIX_DATA_PROCESSING_MAIN}/zweite-bearbeitung', methods=['POST'])
def data_processing_main_zweite_bearbeitung_endpoint():
    return zweite_bearbeitung()

@app.route(f'{API_PREFIX_DATA_PROCESSING_MAIN}/prepare-save', methods=['POST'])
def data_processing_main_prepare_save_endpoint():
    return prepare_save(request)

@app.route(f'{API_PREFIX_DATA_PROCESSING_MAIN}/download/<file_id>', methods=['GET'])
def data_processing_main_download_file_endpoint(file_id):
    return download_file(file_id, request)



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
