"""Tests for the adjustments blueprint gzip after_request hook (`_gzip_json`).

The hook compresses large JSON responses so full-resolution anomaly plot
payloads stay under Cloud Run's 32 MiB cap without dropping datapoints.
"""
import gzip
import json

from flask import Flask, jsonify

from domains.adjustments.api.adjustments import _gzip_json


def _ctx(accept="gzip"):
    app = Flask(__name__)
    return app.test_request_context(headers={"Accept-Encoding": accept})


def _large_payload():
    # > 1 KiB JSON so it crosses the compression threshold
    return jsonify({"data": ["x" * 100] * 200})


def test_compresses_large_json_when_accepted():
    app = Flask(__name__)
    with _ctx("gzip"):
        resp = _gzip_json(_large_payload())
    assert resp.headers.get("Content-Encoding") == "gzip"
    assert "Accept-Encoding" in resp.headers.get("Vary", "")
    decoded = gzip.decompress(resp.get_data())
    assert "data" in json.loads(decoded)
    assert int(resp.headers["Content-Length"]) == len(resp.get_data())


def test_skips_when_client_does_not_accept_gzip():
    with _ctx("identity"):
        resp = _gzip_json(_large_payload())
    assert resp.headers.get("Content-Encoding") is None
    # body is still valid, uncompressed JSON
    assert "data" in json.loads(resp.get_data())


def test_skips_small_bodies():
    with _ctx("gzip"):
        resp = _gzip_json(jsonify({"ok": True}))
    assert resp.headers.get("Content-Encoding") is None


def test_skips_non_json_responses():
    from flask import Response
    with _ctx("gzip"):
        resp = _gzip_json(Response("x" * 5000, mimetype="text/plain"))
    assert resp.headers.get("Content-Encoding") is None
