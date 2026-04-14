"""
Environment info endpoint.

Endpoint:
- GET /environment-info — return runtime library versions for model compatibility
"""

import sys
import tensorflow as tf
import keras
import numpy as np

from flask import Blueprint, jsonify

from .common import require_auth

bp = Blueprint('training_environment', __name__)


@bp.route('/environment-info', methods=['GET'])
@require_auth
def environment_info():
    """Return backend runtime library versions.

    Clients can use these versions to align their local environment before
    loading downloaded .keras model files, avoiding deserialization errors
    caused by version drift.
    """
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
