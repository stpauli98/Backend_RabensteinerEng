"""
Training API Blueprint Aggregator

This module aggregates all training sub-blueprints into a single parent blueprint.
The sub-blueprints are:
- session_routes: 15 session management endpoints
- upload_routes: 5 file upload endpoints
- training_routes: 5 ML training endpoints
- visualization_routes: 5 visualization endpoints
- model_routes: 6 model storage endpoints
"""

from flask import Blueprint

# Import sub-blueprints
from .session_routes import bp as session_bp
from .upload_routes import bp as upload_bp
from .training_routes import bp as training_bp
from .visualization_routes import bp as visualization_bp
from .model_routes import bp as model_bp

# Create parent blueprint
bp = Blueprint('training', __name__)

# Register all sub-blueprints (without additional prefix - routes stay at /api/training/*)
bp.register_blueprint(session_bp)
bp.register_blueprint(upload_bp)
bp.register_blueprint(training_bp)
bp.register_blueprint(visualization_bp)
bp.register_blueprint(model_bp)

# Keep backward compatibility export name
training_bp = bp

__all__ = ['bp', 'training_bp']
