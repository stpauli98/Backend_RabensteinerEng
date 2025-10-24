"""API routes registration"""

def register_blueprints(app):
    """Register all blueprints with the Flask app"""
    
    from api.routes.data_processing import bp as data_processing_bp
    from api.routes.load_data import bp as load_row_data_bp
    from api.routes.first_processing import bp as first_processing_bp
    from api.routes.cloud import bp as cloud_bp
    from api.routes.adjustments import bp as adjustmentsOfData_bp
    from api.routes.training import bp as training_bp
    from api.routes.auth_example import auth_example_bp

    app.register_blueprint(data_processing_bp)
    app.register_blueprint(load_row_data_bp, url_prefix='/api/loadRowData')
    app.register_blueprint(first_processing_bp, url_prefix='/api/firstProcessing')
    app.register_blueprint(cloud_bp, url_prefix='/api/cloud')
    app.register_blueprint(adjustmentsOfData_bp, url_prefix='/api/adjustmentsOfData')
    app.register_blueprint(training_bp, url_prefix='/api/training')
    app.register_blueprint(auth_example_bp)
