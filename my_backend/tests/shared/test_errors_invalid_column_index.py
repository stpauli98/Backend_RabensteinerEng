"""Tests for InvalidColumnIndexError."""
import pytest
from shared.exceptions import InvalidColumnIndexError, ValidationError, LoadDataException


class TestInvalidColumnIndexError:
    def test_inherits_from_validation_error(self):
        assert issubclass(InvalidColumnIndexError, ValidationError)
        assert issubclass(InvalidColumnIndexError, LoadDataException)

    def test_message_contains_index_and_bound(self):
        err = InvalidColumnIndexError(index=5, max_index=2)
        assert "5" in err.message
        assert "2" in err.message

    def test_error_code(self):
        err = InvalidColumnIndexError(index=5, max_index=2)
        assert err.error_code == "INVALID_COLUMN_INDEX"

    def test_details_contain_index_info(self):
        err = InvalidColumnIndexError(index=5, max_index=2)
        assert err.details["index"] == 5
        assert err.details["max_index"] == 2

    def test_to_dict_is_json_serializable(self):
        err = InvalidColumnIndexError(index=5, max_index=2)
        d = err.to_dict()
        assert d["error_code"] == "INVALID_COLUMN_INDEX"
        assert "details" in d
