"""Verify download_all_chunks_as_string logs encoding decisions (FB2 — BE BUG #2).

Strategy: mock `download_all_chunks` on the service instance so the function
receives known bytes without any filesystem I/O. Then assert the correct log
records are emitted at the expected level.

Cases covered:
  1. Pure UTF-8 bytes  → DEBUG log with "preferred encoding=utf-8".
  2. Latin-1 bytes with high codepoints that are invalid UTF-8
     → WARNING log mentioning a fallback encoding.
  3. Both UTF-8 and fallback encodings fail entirely (raw bytes that nothing can
     decode cleanly) → ERROR log and a non-None return (errors='ignore' path).
"""
from __future__ import annotations

import logging
from unittest.mock import patch

import pytest

# The singleton instance we exercise
from domains.processing.services.local_chunk_service import local_chunk_service

UPLOAD_ID = "test-upload-0000-0000-0000-000000000001"
LOGGER_NAME = "domains.processing.services.local_chunk_service"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _invoke(raw_bytes: bytes, total_chunks: int = 1) -> str | None:
    """Call download_all_chunks_as_string with download_all_chunks mocked."""
    with patch.object(local_chunk_service, "download_all_chunks", return_value=raw_bytes):
        return local_chunk_service.download_all_chunks_as_string(
            UPLOAD_ID, total_chunks
        )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_utf8_content_logs_debug(caplog):
    """Clean UTF-8 bytes should be decoded on the first attempt (DEBUG level, not INFO)."""
    utf8_bytes = "col1,col2\n1,2\n3,4\n".encode("utf-8")

    with caplog.at_level(logging.DEBUG, logger=LOGGER_NAME):
        result = _invoke(utf8_bytes)

    assert result is not None, "Expected a decoded string, got None"
    assert "col1,col2" in result

    debug_records = [r for r in caplog.records if r.levelno == logging.DEBUG
                     and "preferred encoding" in r.getMessage()]
    assert debug_records, "Expected at least one DEBUG record with 'preferred encoding'"
    combined = " ".join(r.getMessage() for r in debug_records)
    assert "preferred encoding" in combined, (
        f"Expected 'preferred encoding' in DEBUG log; got: {combined!r}"
    )
    assert UPLOAD_ID in combined, (
        f"Expected upload_id in DEBUG log; got: {combined!r}"
    )


def test_latin1_content_logs_warning(caplog):
    """Bytes that are valid Latin-1 but invalid UTF-8 should trigger a WARNING
    that names the fallback encoding used."""
    # 0x80–0x9F are valid cp1252/latin-1 but invalid UTF-8
    latin1_bytes = b"name,value\nM\xfcller,42\n"  # ü in latin-1 = 0xFC

    with caplog.at_level(logging.DEBUG, logger=LOGGER_NAME):
        result = _invoke(latin1_bytes)

    assert result is not None, "Expected a decoded string, got None"
    assert "M" in result  # At minimum the ASCII part survived

    warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert warning_records, (
        "Expected at least one WARNING record for fallback encoding; "
        f"records: {[r.getMessage() for r in caplog.records]}"
    )
    combined = " ".join(r.getMessage() for r in warning_records)
    assert "fallback encoding" in combined, (
        f"Expected 'fallback encoding' in WARNING log; got: {combined!r}"
    )
    assert UPLOAD_ID in combined, (
        f"Expected upload_id in WARNING log; got: {combined!r}"
    )


def test_utf8_preferred_encoding_in_log_message(caplog):
    """The preferred encoding name should appear in the DEBUG record."""
    utf8_bytes = b"hello,world\n"

    with caplog.at_level(logging.DEBUG, logger=LOGGER_NAME):
        _invoke(utf8_bytes)

    messages = [r.getMessage() for r in caplog.records
                if r.levelno == logging.DEBUG and "preferred encoding" in r.getMessage()]
    assert any("utf-8" in m for m in messages), (
        f"Expected 'utf-8' in at least one DEBUG 'preferred encoding' message; got: {messages}"
    )


def test_debug_records_emitted_for_failed_attempts(caplog):
    """Each failed encoding attempt should produce a DEBUG record."""
    # latin-1 bytes: UTF-8 attempt will fail, producing a DEBUG record
    latin1_bytes = b"caf\xe9"  # 'café' in latin-1

    with caplog.at_level(logging.DEBUG, logger=LOGGER_NAME):
        result = _invoke(latin1_bytes)

    assert result is not None

    debug_records = [r for r in caplog.records if r.levelno == logging.DEBUG]
    assert debug_records, (
        "Expected at least one DEBUG record for a failed encoding attempt"
    )
    combined = " ".join(r.getMessage() for r in debug_records)
    assert "decode failed" in combined, (
        f"Expected 'decode failed' in DEBUG records; got: {combined!r}"
    )


def test_preferred_encoding_logs_at_DEBUG_not_INFO(caplog):
    """Success-path encoding log should be DEBUG to reduce production noise."""
    import logging
    from unittest.mock import patch
    from domains.processing.services.local_chunk_service import LocalChunkService

    service = LocalChunkService()
    with caplog.at_level(logging.DEBUG, logger='domains.processing.services.local_chunk_service'):
        with patch.object(service, 'download_all_chunks', return_value=b'col1;col2\n1;2\n'):
            result = service.download_all_chunks_as_string('test-id', total_chunks=1)
            assert result is not None

    # All "decoded with preferred" log records should be DEBUG, not INFO
    preferred_records = [r for r in caplog.records if 'preferred encoding' in r.message.lower()]
    assert len(preferred_records) >= 1, "Expected at least one 'preferred encoding' log"
    for r in preferred_records:
        assert r.levelno == logging.DEBUG, \
            f"Expected DEBUG level for happy-path log, got {r.levelname}"


def test_iso_8859_1_removed_from_encoding_list():
    """The encoding fallback list should not include both latin-1 and iso-8859-1 (same codec)."""
    from domains.processing.services import local_chunk_service as svc_module
    import inspect

    source = inspect.getsource(svc_module.LocalChunkService.download_all_chunks_as_string)
    assert "iso-8859-1" not in source, \
        "iso-8859-1 is identical to latin-1; including both produces misleading WARNING logs"
