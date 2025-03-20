import os
from datetime import datetime as dat
from flask import Flask, request, jsonify
from flask_cors import CORS
from firstProcessing import bp as first_processing_bp
from load_row_data import bp as load_row_data_bp
from data_processing_main import bp as data_processing_main_bp
from adjustmentsOfData import bp as adjustments_of_data_bp
from cloud import bp as cloud_bp

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
app.register_blueprint(adjustments_of_data_bp, url_prefix='/api/adjustmentsOfData')
app.register_blueprint(cloud_bp, url_prefix='/api/cloud')



# API configuration
#API_PREFIX_LOAD_ROW_DATA = '/api/loadRowData'
#API_PREFIX_FIRST_PROCESSING = '/api/firstProcessing'
#API_PREFIX_DATA_PROCESSING_MAIN = '/api/dataProcessingMain'
#API_PREFIX_ADJUSTMENTS_OF_DATA = '/api/adjustmentsOfDataMain'
#API_PREFIX_CLOUD = '/api/cloud'

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


#LoadRowData

#@app.route(f'{API_PREFIX_LOAD_ROW_DATA}/upload-chunk', methods=['POST'])
#def load_row_data_upload_chunk_endpoint():
#    return load_row_data.upload_chunk(request)

#@app.route(f'{API_PREFIX_LOAD_ROW_DATA}/prepare-save', methods=['POST'])
#def load_row_data_prepare_save_endpoint():
 #   return load_row_data.prepare_save(request)

#@app.route(f'{API_PREFIX_LOAD_ROW_DATA}/download/<file_id>', methods=['GET'])
#def load_row_data_download_endpoint(file_id):
 #   return load_row_data.download_file(file_id)

#FirstProcessing

#@app.route(f'{API_PREFIX_FIRST_PROCESSING}/upload_chunk', methods=['POST'])
#def first_processing_upload_chunk_endpoint():
 #   return firstProcessing.upload_chunk(request)

#@app.route(f'{API_PREFIX_FIRST_PROCESSING}/prepare-save', methods=['POST'])
#def first_processing_prepare_save_endpoint():
 #   return firstProcessing.prepare_save(request)

#@app.route(f'{API_PREFIX_FIRST_PROCESSING}/download/<file_id>', methods=['GET'])
#def first_processing_download_file_endpoint(file_id):
 #   return firstProcessing.download_file(file_id, request)


@app.route('/health', methods=['GET'])
def health_check():
    try:
        logger.info("Handling request to health check endpoint")
        response = {
            'status': 'healthy',
            'timestamp': str(dat.now()),
            'port': os.environ.get('PORT', '8080'),
            'env': os.environ.get('FLASK_ENV', 'production'),
            'blueprints': list(app.blueprints.keys())
        }
        logger.info(f"Health check response: {response}")
        return jsonify(response), 200
    except Exception as e:
        logger.error(f"Error in health check: {e}")
        return jsonify({'error': str(e), 'status': 'unhealthy'}), 500

#DataProcessingMain

#@app.route(f'{API_PREFIX_DATA_PROCESSING_MAIN}/zweite-bearbeitung', methods=['POST'])
#def data_processing_main_zweite_bearbeitung_endpoint():
 #   return data_processing_main.zweite_bearbeitung(request)



#AdjustmentsOfData

#@app.route(f'{API_PREFIX_ADJUSTMENTS_OF_DATA}/analysedata', methods=['POST'])
#def adjustments_of_data_analysedata_endpoint():
 #   return adjustmentsOfData.analyse_data(request)

#@app.route(f'{API_PREFIX_ADJUSTMENTS_OF_DATA}/adjustdata', methods=['POST'])
#def adjustments_of_data_adjustdata_endpoint():
 #   return adjustmentsOfData.adjust_data(request)

#@app.route(f'{API_PREFIX_ADJUSTMENTS_OF_DATA}/prepare-save', methods=['POST'])
#def adjustments_of_data_prepare_save_endpoint():
 #   return adjustmentsOfData.prepare_save(request)

#@app.route(f'{API_PREFIX_ADJUSTMENTS_OF_DATA}/download/<file_id>', methods=['GET'])
#def adjustments_of_data_download_file_endpoint(file_id):
 #   return adjustmentsOfData.download_file(file_id, request)

#Cloud

#@app.route(f'{API_PREFIX_CLOUD}/clouddata', methods=['POST'])
#def cloud_clouddata_endpoint():
 #   return cloud.clouddata(request)

#@app.route(f'{API_PREFIX_CLOUD}/interpolate', methods=['POST'])
#def cloud_interpolate_endpoint():
 #   return cloud.interpolate(request) 

#@app.route(f'{API_PREFIX_CLOUD}/prepare-save', methods=['POST'])
#def cloud_prepare_save_endpoint():
 #   return cloud.prepare_save(request)

#@app.route(f'{API_PREFIX_CLOUD}/download/<file_id>', methods=['GET'])
#def cloud_download_file_endpoint(file_id):
 #   return cloud.download_file(file_id, request)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
