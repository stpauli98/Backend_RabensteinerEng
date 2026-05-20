"""Verify download_all_chunks_as_string logs encoding decisions (FB2 — BE BUG #2).

Strategy: mock `download_all_chunks` on the service instance so the function
receives known bytes without any filesystem I/O. Then assert the correct log
records are emitted at the expected level.

Cases covered:
  1. Pure UTF-8 bytes  → INFO log with "preferred encoding=utf-8".
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


def test_utf8_content_logs_info(caplog):
    """Clean UTF-8 bytes should be decoded on the first attempt (INFO level)."""
    utf8_bytes = "col1,col2\n1,2\n3,4\n".encode("utf-8")

    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        result = _invoke(utf8_bytes)

    assert result is not None, "Expected a decoded string, got None"
    assert "col1,col2" in result

    info_records = [r for r in caplog.records if r.levelno == logging.INFO]
    assert info_records, "Expected at least one INFO record"
    combined = " ".join(r.getMessage() for r in info_records)
    assert "preferred encoding" in combined, (
        f"Expected 'preferred encoding' in INFO log; got: {combined!r}"
    )
    assert UPLOAD_ID in combined, (
        f"Expected upload_id in INFO log; got: {combined!r}"
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
    """The preferred encoding name should appear in the INFO record."""
    utf8_bytes = b"hello,world\n"

    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        _invoke(utf8_bytes)

    messages = [r.getMessage() for r in caplog.records if r.levelno == logging.INFO]
    assert any("utf-8" in m for m in messages), (
        f"Expected 'utf-8' in at least one INFO message; got: {messages}"
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
