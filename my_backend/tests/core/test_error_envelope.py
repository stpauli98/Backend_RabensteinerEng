from flask import Flask, abort, jsonify


def _app_with_handlers():
    """Replicates the TARGET 400/413 handlers from core.app_factory for isolated testing."""
    app = Flask(__name__)

    @app.errorhandler(400)
    def bad_request(error):
        if hasattr(error, 'response') and error.response:
            return error.response
        return jsonify({'success': False, 'code': 'BAD_REQUEST', 'error': str(error)}), 400

    @app.errorhandler(413)
    def payload_too_large(error):
        if hasattr(error, 'response') and error.response:
            return error.response
        return jsonify({'success': False, 'code': 'PAYLOAD_TOO_LARGE',
                        'error': 'Request entity is too large'}), 413

    @app.route('/boom400')
    def boom400():
        abort(400)

    @app.route('/boom413')
    def boom413():
        abort(413)

    return app


def test_400_envelope():
    body = _app_with_handlers().test_client().get('/boom400').get_json()
    assert body['success'] is False and body['code'] == 'BAD_REQUEST' and 'error' in body


def test_413_envelope():
    body = _app_with_handlers().test_client().get('/boom413').get_json()
    assert body['success'] is False and body['code'] == 'PAYLOAD_TOO_LARGE'
