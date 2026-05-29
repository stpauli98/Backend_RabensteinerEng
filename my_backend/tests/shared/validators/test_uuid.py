"""Unit tests for shared.validators.uuid."""
import json
import pytest
from flask import Flask

from shared.validators.uuid import validate_uuid_format


@pytest.fixture
def app():
    app = Flask(__name__)
    return app


class TestValidateUuidFormat:
    def test_returns_none_for_valid_uuid_v4(self, app):
        with app.test_request_context():
            result = validate_uuid_format("4633c88e-36fb-446d-a17e-90374359875c")
            assert result is None

    def test_returns_400_for_malformed_uuid(self, app):
        with app.test_request_context():
            result = validate_uuid_format("not-a-uuid")
            assert result is not None
            response, status = result
            assert status == 400
            body = json.loads(response.get_data(as_text=True))
            assert body['success'] is False
            assert body['code'] == 'BAD_UUID'
            assert 'session_id' in body['error']

    def test_returns_400_for_empty_string(self, app):
        with app.test_request_context():
            result = validate_uuid_format("")
            assert result is not None
            _, status = result
            assert status == 400

    def test_returns_400_for_none(self, app):
        with app.test_request_context():
            result = validate_uuid_format(None)
            assert result is not None
            _, status = result
            assert status == 400

    def test_returns_400_for_path_traversal_attempt(self, app):
        with app.test_request_context():
            result = validate_uuid_format("../../../etc/passwd")
            assert result is not None
            _, status = result
            assert status == 400
