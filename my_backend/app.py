from flask import Flask, request, jsonify
from flask_cors import CORS
import firstProcessing
import load_row_data

app = Flask(__name__)
CORS(app)

@app.route('/process-csv', methods=['POST'])
def process_csv_endpoint():
    try:
        data = request.get_json()
        # Add your processing logic here using firstProcessing.py functions
        result = firstProcessing.process_csv()
        return jsonify({"status": "success", "data": result})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

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
