import os
import time

import services.adjustments.cleanup as cleanup


def test_processed_csv_survives_longer_than_raw(tmp_path, monkeypatch):
    monkeypatch.setattr(cleanup, "UPLOAD_FOLDER", str(tmp_path))
    raw = tmp_path / "data.csv"
    processed = tmp_path / "data_1.csv"
    raw.write_text("x")
    processed.write_text("y")
    old = time.time() - 2 * 60 * 60  # 2 hours: > 60 min, < 24 h
    os.utime(raw, (old, old))
    os.utime(processed, (old, old))
    cleanup.cleanup_old_files()
    assert not raw.exists(), "raw upload should be cleaned after 60 min"
    assert processed.exists(), "processed CSV should survive (24 h TTL)"


def test_processed_csv_cleaned_after_24h(tmp_path, monkeypatch):
    monkeypatch.setattr(cleanup, "UPLOAD_FOLDER", str(tmp_path))
    processed = tmp_path / "data_1.csv"
    processed.write_text("y")
    old = time.time() - 25 * 60 * 60
    os.utime(processed, (old, old))
    cleanup.cleanup_old_files()
    assert not processed.exists(), "processed CSV should be cleaned after 24 h"
