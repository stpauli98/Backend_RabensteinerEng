"""Unit tests for csv_safe_cell formula injection sanitizer."""
import pytest
from domains.processing.utils.csv_sanitizer import csv_safe_cell


@pytest.mark.parametrize("dangerous", [
    "=1+1",
    "+SUM(A1:A2)",
    "-CMD",
    "@FOO",
])
def test_sanitizes_leading_formula_char(dangerous):
    """Cells starting with =, +, -, @ get prefixed with a single quote."""
    result = csv_safe_cell(dangerous)
    assert result == "'" + dangerous


@pytest.mark.parametrize("dangerous", [
    "\t=1+1",
    "\r+SUM",
    " =evil",
])
def test_sanitizes_whitespace_then_formula_char(dangerous):
    """Cells with leading whitespace followed by a formula char are also dangerous."""
    result = csv_safe_cell(dangerous)
    assert result == "'" + dangerous


@pytest.mark.parametrize("safe", [
    "hello",
    "123.45",
    "1=2",     # equals not at start
    "",        # empty
    "  hello", # leading whitespace but not followed by formula char
])
def test_leaves_safe_strings_alone(safe):
    """Cells that don't start with formula chars are returned unchanged."""
    assert csv_safe_cell(safe) == safe


def test_leaves_non_strings_alone():
    """Numbers, None, booleans pass through untouched."""
    assert csv_safe_cell(42) == 42
    assert csv_safe_cell(3.14) == 3.14
    assert csv_safe_cell(None) is None
    assert csv_safe_cell(True) is True
