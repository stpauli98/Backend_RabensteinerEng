from shared.auth.subscription import plan_file_size_bytes

def test_unlimited_sentinel_returns_none():
    assert plan_file_size_bytes({'max_file_size_mb': -1}) is None

def test_zero_blocks_uploads():
    assert plan_file_size_bytes({'max_file_size_mb': 0}) == 0

def test_positive_mb_to_bytes():
    assert plan_file_size_bytes({'max_file_size_mb': 100}) == 100 * 1024 * 1024

def test_missing_key_is_unlimited():
    assert plan_file_size_bytes({}) is None
