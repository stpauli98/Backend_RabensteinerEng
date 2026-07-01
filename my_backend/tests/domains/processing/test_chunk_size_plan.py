import os
import pytest
from domains.processing.services.chunk_handler import combine_chunks_efficiently


def _mk(tmp_path, n, mb_each):
    for i in range(n):
        (tmp_path / f"chunk_{i:04d}.part").write_bytes(b"x" * (mb_each * 1024 * 1024))
    return str(tmp_path)


def test_over_plan_limit_raises(tmp_path):
    d = _mk(tmp_path, 2, 1)  # 2 MB total
    with pytest.raises(ValueError):
        combine_chunks_efficiently(d, 2, max_size_bytes=1 * 1024 * 1024)


def test_unlimited_none_ok(tmp_path):
    d = _mk(tmp_path, 2, 1)
    name, size = combine_chunks_efficiently(d, 2, max_size_bytes=None)
    assert size == 2 * 1024 * 1024
    os.unlink(name)  # clean up the returned temp file
