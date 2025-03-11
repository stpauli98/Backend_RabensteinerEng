import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import firstProcessing
import load_row_data
import data_processing_main
import adjustmentsOfData
import cloud

# API configuration
API_PREFIX_LOAD_ROW_DATA = '/api/loadRowData'
API_PREFIX_FIRST_PROCESSING = '/api/firstProcessing'
API_PREFIX_DATA_PROCESSING_MAIN = '/api/dataProcessingMain'
API_PREFIX_ADJUSTMENTS_OF_DATA = '/api/adjustmentsOfDataMain'
API_PREFIX_CLOUD = '/api/cloud'

app = Flask(__name__)

# Definiši CORS iz ENV varijable
ALLOWED_ORIGIN = os.getenv("ACCESS_CONTROL_ALLOW_ORIGIN", "https://rabensteinerengineering.onrender.com")

# Globalni CORS
CORS(app, resources={r"/*": {"origins": ALLOWED_ORIGIN}}, supports_credentials=True)

@app.after_request
def apply_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = ALLOWED_ORIGIN
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    response.headers["Access-Control-Allow-Credentials"] = "true"
    return response

# Eksplicitno podrži OPTIONS zahteve
@app.route("/api/dataProcessingMain/upload-chunk", methods=["OPTIONS"])
def options_handler():
    response = jsonify({"message": "CORS preflight OK"})
    response.headers["Access-Control-Allow-Origin"] = ALLOWED_ORIGIN
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    response.headers["Access-Control-Allow-Credentials"] = "true"
    return response

@app.route('/')
def index():
    return jsonify({"message": "Backend service is running", "CORS": ALLOWED_ORIGIN})

#LoadRowData

@app.route(f'{API_PREFIX_LOAD_ROW_DATA}/upload', methods=['POST'])
def load_row_data_upload_endpoint():
    return load_row_data.upload_files()

@app.route(f'{API_PREFIX_LOAD_ROW_DATA}/prepare-save', methods=['POST'])
def load_row_data_prepare_save_endpoint():
    return load_row_data.prepare_save(request)

@app.route(f'{API_PREFIX_LOAD_ROW_DATA}/download/<file_id>', methods=['GET'])
def load_row_data_download_endpoint(file_id):
    return load_row_data.download_file(file_id)

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

@app.route(f'{API_PREFIX_DATA_PROCESSING_MAIN}/upload-chunk', methods=['POST'])
def data_processing_main_upload_chunk_endpoint():
    return data_processing_main.upload_chunk(request)

@app.route(f'{API_PREFIX_DATA_PROCESSING_MAIN}/prepare-save', methods=['POST'])
def data_processing_main_prepare_save_endpoint():
    return data_processing_main.prepare_save(request)

@app.route(f'{API_PREFIX_DATA_PROCESSING_MAIN}/download/<file_id>', methods=['GET'])
def data_processing_main_download_file_endpoint(file_id):
    return data_processing_main.download_file(file_id, request)

#AdjustmentsOfData

@app.route(f'{API_PREFIX_ADJUSTMENTS_OF_DATA}/analysedata', methods=['POST'])
def adjustments_of_data_analysedata_endpoint():
    return adjustmentsOfData.analyse_data(request)

@app.route(f'{API_PREFIX_ADJUSTMENTS_OF_DATA}/adjustdata', methods=['POST'])
def adjustments_of_data_adjustdata_endpoint():
    return adjustmentsOfData.adjust_data(request)

@app.route(f'{API_PREFIX_ADJUSTMENTS_OF_DATA}/prepare-save', methods=['POST'])
def adjustments_of_data_prepare_save_endpoint():
    return adjustmentsOfData.prepare_save(request)

@app.route(f'{API_PREFIX_ADJUSTMENTS_OF_DATA}/download/<file_id>', methods=['GET'])
def adjustments_of_data_download_file_endpoint(file_id):
    return adjustmentsOfData.download_file(file_id, request)

#Cloud

@app.route(f'{API_PREFIX_CLOUD}/clouddata', methods=['POST'])
def cloud_clouddata_endpoint():
    return cloud.clouddata(request)

@app.route(f'{API_PREFIX_CLOUD}/interpolate', methods=['POST'])
def cloud_interpolate_endpoint():
    return cloud.interpolate(request) 

@app.route(f'{API_PREFIX_CLOUD}/prepare-save', methods=['POST'])
def cloud_prepare_save_endpoint():
    return cloud.prepare_save(request)

@app.route(f'{API_PREFIX_CLOUD}/download/<file_id>', methods=['GET'])
def cloud_download_file_endpoint(file_id):
    return cloud.download_file(file_id, request)



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
