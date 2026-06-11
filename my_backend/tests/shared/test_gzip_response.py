import json
from flask import Flask, Response
from shared.responses.gzip import gzip_json_response, _GZIP_MIN_BYTES


def _run(body: bytes, accept="gzip"):
    app = Flask(__name__)
    with app.test_request_context(headers={"Accept-Encoding": accept}):
        return gzip_json_response(Response(body, mimetype="application/json"))


def test_large_json_is_gzipped():
    big = json.dumps({"x": "y" * (_GZIP_MIN_BYTES + 100)}).encode()
    assert _run(big).headers.get("Content-Encoding") == "gzip"


def test_small_json_not_gzipped():
    assert _run(b'{"x":1}').headers.get("Content-Encoding") is None


def test_no_gzip_when_not_accepted():
    big = json.dumps({"x": "y" * (_GZIP_MIN_BYTES + 100)}).encode()
    assert _run(big, accept="identity").headers.get("Content-Encoding") is None
