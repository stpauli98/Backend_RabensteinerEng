import pytest
import shared.database.persistence as p


@pytest.mark.parametrize("bad", ["999999999", "0", "-3", "abc", 10**9])
def test_rejects_out_of_range(bad):
    with pytest.raises(ValueError):
        p.validate_zeitschritte_window(bad)


def test_valid_passes():
    assert p.validate_zeitschritte_window("96") == 96
