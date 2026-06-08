import pytest
import domains.processing.services.local_chunk_service as lcs


@pytest.mark.parametrize("bad", ["../x", "a/b", "..", "x\\y", "", {"a": 1}, 5, "a b"])
def test_validate_file_id_rejects(bad):
    with pytest.raises((ValueError, TypeError)):
        lcs._validate_file_id(bad)


def test_validate_file_id_accepts_normal():
    assert lcs._validate_file_id("userid_abcDEF123-_") == "userid_abcDEF123-_"
