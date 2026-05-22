"""Tests for domains.adjustments.services.utils.detect_delimiter"""
import pytest

from domains.adjustments.services.utils import detect_delimiter


def test_detects_semicolon_german_default():
    assert detect_delimiter("UTC;value;flag\n1;2;3\n") == ";"


def test_detects_comma():
    assert detect_delimiter("UTC,value,flag\n1,2,3\n") == ","


def test_detects_tab():
    assert detect_delimiter("UTC\tvalue\tflag\n1\t2\t3\n") == "\t"


def test_detects_pipe():
    assert detect_delimiter("UTC|value|flag\n1|2|3\n") == "|"


def test_picks_highest_count_when_multiple_present():
    # Pipe appears 2x; semicolon appears 1x → pipe wins
    assert detect_delimiter("UTC|value|flag;extra\n") == "|"


def test_defaults_to_semicolon_when_no_delimiter():
    assert detect_delimiter("UTCvalueflag\n") == ";"


def test_empty_first_line_defaults_to_semicolon():
    assert detect_delimiter("\n") == ";"
