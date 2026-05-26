"""Test the AnomalyException hierarchy + to_dict() shape."""
import pytest

from shared.exceptions.errors import (
    AnomalyException,
    STLPeriodAlignmentError,
    LSTMHyperparameterCapError,
    SBADSlopeRuleError,
    ThresholdOutOfRangeError,
    ParameterOutOfRangeError,
)


def test_anomaly_exception_base_has_error_code():
    e = AnomalyException("test message")
    assert e.error_code == "ANOMALY_ERROR"
    assert e.message == "test message"
    d = e.to_dict()
    assert d["error_code"] == "ANOMALY_ERROR"
    assert d["error"] == "test message"


def test_stl_period_alignment_error_subclass():
    e = STLPeriodAlignmentError(period_h=24, dt_avg_h=0.083)
    assert e.error_code == "STL_PERIOD_NOT_ALIGNED"
    assert e.details["period_h"] == 24
    assert e.details["dt_avg_h"] == 0.083


def test_lstm_hyperparameter_cap_error():
    e = LSTMHyperparameterCapError(param="neurons", value=2048, max_value=1024)
    assert e.error_code == "LSTM_HYPERPARAM_OUT_OF_RANGE"
    assert e.details["param"] == "neurons"
    assert e.details["value"] == 2048
    assert e.details["max"] == 1024


def test_sbad_slope_rule_error():
    e = SBADSlopeRuleError(provided="chg_max")
    assert e.error_code == "SBAD_SLOPE_REQUIRES_BOTH"
    assert e.details["provided"] == "chg_max"


def test_threshold_out_of_range_error():
    e = ThresholdOutOfRangeError(value=-5, min_value=0)
    assert e.error_code == "THRESHOLD_OUT_OF_RANGE"
    assert e.details["value"] == -5
    assert e.details["min"] == 0


def test_parameter_out_of_range_error():
    e = ParameterOutOfRangeError(param="eq_max", value=-10, min_value=0)
    assert e.error_code == "PARAM_OUT_OF_RANGE"
    assert e.details["param"] == "eq_max"
    assert e.details["value"] == -10
    assert e.details["min"] == 0


def test_anomaly_exception_to_dict_shape():
    e = STLPeriodAlignmentError(period_h=24, dt_avg_h=0.5)
    d = e.to_dict()
    assert d['error_code'] == 'STL_PERIOD_NOT_ALIGNED'
    assert d['error'].startswith('STL period')
    assert d['details']['period_h'] == 24
    assert d['details']['dt_avg_h'] == 0.5
