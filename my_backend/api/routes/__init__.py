"""API routes registration"""

def register_blueprints(app):
    """Register all blueprints with the Flask app"""

    # Domain blueprints (new architecture)
    from domains.processing import first_processing_bp, data_processing_bp
    from domains.training.api import training_bp
    from domains.upload import load_data_bp
    from domains.adjustments import adjustments_bp
    from domains.cloud import cloud_bp

    # Legacy blueprints (pending migration)
    from api.routes.auth_example import auth_example_bp
    from api.routes.stripe_routes import stripe_bp

    # Register domain blueprints
    app.register_blueprint(data_processing_bp)
    app.register_blueprint(first_processing_bp, url_prefix='/api/firstProcessing')
    app.register_blueprint(training_bp, url_prefix='/api/training')
    app.register_blueprint(load_data_bp, url_prefix='/api/loadRowData')
    app.register_blueprint(adjustments_bp, url_prefix='/api/adjustmentsOfData')
    app.register_blueprint(cloud_bp, url_prefix='/api/cloud')

    # Register legacy blueprints
    app.register_blueprint(auth_example_bp)
    app.register_blueprint(stripe_bp)
