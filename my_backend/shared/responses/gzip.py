"""Reusable gzip-after-request helper for large JSON responses (Cloud Run 32 MiB guard)."""
import gzip
import logging
from flask import request, Response

logger = logging.getLogger(__name__)

_GZIP_MIN_BYTES = 1024


def gzip_json_response(response: Response) -> Response:
    """Gzip a large JSON response when the client accepts it; safe no-op otherwise."""
    try:
        accept = request.headers.get('Accept-Encoding', '')
        if (
            'gzip' not in accept.lower()
            or response.direct_passthrough
            or response.status_code < 200
            or response.status_code in (204, 304)
            or response.headers.get('Content-Encoding')
            or response.mimetype != 'application/json'
        ):
            return response
        data = response.get_data()
        if len(data) < _GZIP_MIN_BYTES:
            return response
        compressed = gzip.compress(data, compresslevel=6)
        response.set_data(compressed)
        response.headers['Content-Encoding'] = 'gzip'
        response.headers['Content-Length'] = str(len(compressed))
        response.headers.add('Vary', 'Accept-Encoding')
    except Exception:
        logger.exception("gzip after_request failed; returning uncompressed")
    return response
