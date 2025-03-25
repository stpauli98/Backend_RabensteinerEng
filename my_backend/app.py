import os
import logging

# Import adjustmentsOfData module
import adjustmentsOfData

from datetime import datetime as dat
from flask import Flask, request, jsonify
from flask_cors import CORS

from firstProcessing import bp as first_processing_bp
from load_row_data import bp as load_row_data_bp
from data_processing_main import bp as data_processing_main_bp
from cloud import bp as cloud_bp

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


app = Flask(__name__)

# Configure CORS with more permissive settings
CORS(app, resources={
    r"/*": {
        "origins": ["*"],
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
        "allow_headers": ["Content-Type", "Authorization", "Accept", "Origin"],
        "expose_headers": ["Content-Disposition", "Content-Length"],
        "supports_credentials": True,
        "max_age": 3600
    }
})

# Register blueprints with correct prefixes
app.register_blueprint(data_processing_main_bp, url_prefix='/api/dataProcessingMain')
app.register_blueprint(load_row_data_bp, url_prefix='/api/loadRowData')
app.register_blueprint(first_processing_bp, url_prefix='/api/firstProcessing')
app.register_blueprint(cloud_bp, url_prefix='/api/cloud')

#API prefix
API_PREFIX_ADJUSTMENTS_OF_DATA = '/api/adjustmentsOfData'

# AdjustmentsOfData routes
@app.route(f'{API_PREFIX_ADJUSTMENTS_OF_DATA}/upload-chunk', methods=['POST'])
def adjustments_of_data_upload_chunk():
    return adjustmentsOfData.upload_chunk()

@app.route(f'{API_PREFIX_ADJUSTMENTS_OF_DATA}/analyse-data', methods=['POST'])
def adjustments_of_data_analyse_data():
    return adjustmentsOfData.analyse_data()

@app.route(f'{API_PREFIX_ADJUSTMENTS_OF_DATA}/adjust-data-chunk', methods=['POST'])
def adjustments_of_data_adjust_data():
    return adjustmentsOfData.adjust_data()

@app.route(f'{API_PREFIX_ADJUSTMENTS_OF_DATA}/adjustdata/complete', methods=['POST'])
def adjustments_of_data_complete():
    return adjustmentsOfData.complete_adjustment()

@app.route(f'{API_PREFIX_ADJUSTMENTS_OF_DATA}/prepare-save', methods=['POST'])
def adjustments_of_data_prepare_save():
    return adjustmentsOfData.prepare_save()

@app.route(f'{API_PREFIX_ADJUSTMENTS_OF_DATA}/download/<file_id>', methods=['GET'])
def adjustments_of_data_download(file_id):
    return adjustmentsOfData.download_file(file_id)


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


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
