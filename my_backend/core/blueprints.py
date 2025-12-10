"""Blueprint registration for Flask app"""


def register_blueprints(app):
    """Register all blueprints with the Flask app"""
    from domains.processing import first_processing_bp, data_processing_bp
    from domains.training.api import training_bp
    from domains.upload import load_data_bp
    from domains.adjustments import adjustments_bp
    from domains.cloud import cloud_bp
    from domains.payments import stripe_bp

    app.register_blueprint(data_processing_bp)
    app.register_blueprint(first_processing_bp, url_prefix='/api/firstProcessing')
    app.register_blueprint(training_bp, url_prefix='/api/training')
    app.register_blueprint(load_data_bp, url_prefix='/api/loadRowData')
    app.register_blueprint(adjustments_bp, url_prefix='/api/adjustmentsOfData')
    app.register_blueprint(cloud_bp, url_prefix='/api/cloud')
    app.register_blueprint(stripe_bp, url_prefix='/api/stripe')
