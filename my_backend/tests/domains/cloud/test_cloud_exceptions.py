"""Test the CloudException hierarchy + to_dict() shape (mirror W9 test_anomaly_exceptions.py)."""
import pytest

from shared.exceptions.errors import (
    CloudException,
    TimestampMismatchError,
    InsufficientMatchingPointsError,
    ColumnDetectionError,
    ToleranceBoundsEmptyError,
    UnknownToleranceTypeError,
)


def test_cloud_exception_base_has_error_code():
    e = CloudException("test message")
    assert e.error_code == "CLOUD_ERROR"
    assert e.message == "test message"
    d = e.to_dict()
    assert d["error_code"] == "CLOUD_ERROR"
    assert d["error"] == "test message"


def test_timestamp_mismatch_error():
    e = TimestampMismatchError(file1="predictor.csv", file2="target.csv")
    assert e.error_code == "TIMESTAMP_MISMATCH"
    assert e.details["file1"] == "predictor.csv"
    assert e.details["file2"] == "target.csv"


def test_insufficient_matching_points_error():
    e = InsufficientMatchingPointsError(matched=4, minimum=100)
    assert e.error_code == "INSUFFICIENT_MATCHING_POINTS"
    assert e.details["matched"] == 4
    assert e.details["minimum"] == 100


def test_column_detection_error():
    e = ColumnDetectionError(column_type="temperature", available=["UTC", "value"])
    assert e.error_code == "COLUMN_NOT_FOUND"
    assert e.details["column_type"] == "temperature"
    assert e.details["available"] == ["UTC", "value"]


def test_tolerance_bounds_empty_error():
    e = ToleranceBoundsEmptyError(tolerance_type="cnt")
    assert e.error_code == "TOLERANCE_BOUNDS_EMPTY"
    assert e.details["tolerance_type"] == "cnt"


def test_unknown_tolerance_type_error():
    e = UnknownToleranceTypeError(provided="xyz")
    assert e.error_code == "UNKNOWN_TOLERANCE_TYPE"
    assert e.details["provided"] == "xyz"


def test_cloud_exception_to_dict_shape():
    e = TimestampMismatchError(file1="a.csv", file2="b.csv")
    d = e.to_dict()
    assert d["error_code"] == "TIMESTAMP_MISMATCH"
    assert "error" in d
    assert d["details"]["file1"] == "a.csv"
