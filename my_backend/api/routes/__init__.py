"""API routes registration"""

def register_blueprints(app):
    """Register all blueprints with the Flask app"""
    
    # Import all route blueprints
    from api.routes.data_processing import bp as data_processing_bp
    from api.routes.load_data import bp as load_row_data_bp
    from api.routes.first_processing import bp as first_processing_bp
    from api.routes.cloud import bp as cloud_bp
    from api.routes.adjustments import bp as adjustmentsOfData_bp
    from api.routes.training import bp as training_bp
    # NOTE: training_api_bp has been deprecated - all endpoints moved to training_bp
    # Backup available at: services/training/training_api.py.backup_20251017_130931

    # Register blueprints with correct prefixes
    app.register_blueprint(data_processing_bp)
    app.register_blueprint(load_row_data_bp, url_prefix='/api/loadRowData')
    app.register_blueprint(first_processing_bp, url_prefix='/api/firstProcessing')
    app.register_blueprint(cloud_bp, url_prefix='/api/cloud')
    app.register_blueprint(adjustmentsOfData_bp, url_prefix='/api/adjustmentsOfData')
    app.register_blueprint(training_bp, url_prefix='/api/training')