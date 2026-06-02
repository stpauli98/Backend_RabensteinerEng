"""
Environment info endpoint.

Endpoint:
- GET /environment-info — return runtime library versions for model compatibility

W11-A T7: hardened with @limiter.limit and the standardized error contract
(shared.responses.errors.error_response). No session_id on this endpoint,
so there is no UUID guard.
"""

import sys
import tensorflow as tf
import keras
import numpy as np

from flask import Blueprint, jsonify

from .common import require_auth, get_logger
from core.rate_limits import limiter, training_limit_string
from shared.responses.errors import error_response as _err

bp = Blueprint('training_environment', __name__)
logger = get_logger(__name__)


@bp.route('/environment-info', methods=['GET'])
@limiter.limit(training_limit_string)
@require_auth
def environment_info():
    """Return backend runtime library versions.

    Clients can use these versions to align their local environment before
    loading downloaded .keras model files, avoiding deserialization errors
    caused by version drift.
    """
    try:
        tf_version = tf.__version__
        keras_version = keras.__version__
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        numpy_version = np.__version__

        return jsonify({
            "python": python_version,
            "tensorflow": tf_version,
            "keras": keras_version,
            "numpy": numpy_version,
            "pip_install": f"pip install tensorflow=={tf_version} keras=={keras_version}",
        })
    except Exception:
        logger.exception("Failed to read environment info")
        return _err(
            'INTERNAL_ERROR',
            'Failed to read environment info',
            500,
            suggestion='Please try again. If the problem persists, contact support.',
        )
