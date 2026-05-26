"""Unit tests for anomaly detection validators — exact error message comparison."""
import pandas as pd
import pytest

from domains.adjustments.services.anomaly_pipeline import build_par_dict
from domains.adjustments.services.anomaly_validators import (
    check_comp,
    check_float,
    check_ge_zero,
    check_gt_zero,
    check_integer,
    validate_par_dict,
    validate_param_single,
)
from shared.exceptions.errors import AnomalyException


def descriptor(value, name_en="Test", name_de="Test"):
    return {"value": value, "unit": None, "name": {"en": name_en, "de": name_de}}


def test_check_float_passes_numeric():
    assert check_float(descriptor(3.5)) == 3.5
    assert check_float(descriptor("2.7")) == 2.7


def test_check_float_raises_de_message():
    with pytest.raises(AnomalyException) as exc:
        check_float(descriptor("abc", "X", "X"), lang="de")
    assert "kann nicht in eine Fließkommazahl konvertiert werden" in str(exc.value)


def test_check_integer_raises_for_decimal():
    with pytest.raises(AnomalyException) as exc:
        check_integer(descriptor(1.5, "Y", "Y"), lang="en")
    assert "is no integer" in str(exc.value)


def test_check_gt_zero_de_message():
    with pytest.raises(AnomalyException) as exc:
        check_gt_zero(descriptor(-5, "EQ", "EQ"), lang="de")
    # Localized message preserved in suggestions[0]
    assert "muss größer als 0 sein" in exc.value.suggestions[0]


def test_check_ge_zero_passes_zero():
    check_ge_zero(descriptor(0, "X", "X"))


def test_check_ge_zero_fails_negative():
    with pytest.raises(AnomalyException):
        check_ge_zero(descriptor(-0.1, "X", "X"))


def test_check_comp_violation_en():
    v_max = descriptor(10, "MAX", "MAX")
    v_min = descriptor(20, "MIN", "MIN")
    with pytest.raises(AnomalyException) as exc:
        check_comp(v_max, v_min, lang="en")
    assert "must be greater than the input value" in str(exc.value)


def test_validate_par_dict_eq_max_negative():
    par = build_par_dict({"eqMax": -5})
    with pytest.raises(AnomalyException) as exc:
        validate_par_dict(par, dt_avg=pd.Timedelta(minutes=3), lang="de")
    # Localized message preserved in suggestions[0]
    msg = exc.value.suggestions[0]
    assert "Maximal zulässige Dauer konstanter Werte" in msg
    assert "muss größer als 0 sein" in msg


def test_validate_par_dict_period_misaligned():
    par = build_par_dict({"stl": {"run": True, "periodH": 1.5}})
    with pytest.raises(AnomalyException) as exc:
        validate_par_dict(par, dt_avg=pd.Timedelta(minutes=4), lang="en")
    # Localized message preserved in suggestions[0]
    assert "must be an integer multiple of the time step" in exc.value.suggestions[0]


def test_validate_par_dict_full_defaults_ok():
    par = build_par_dict({
        "eqMax": 15, "gapMax": 60, "dec": 1, "lgMin": 720,
        "vMax": 180, "vMin": 0, "el0": True,
        "sbad": {"chgMax": 20, "lgMax": 120},
        "stl": {"run": True, "periodH": 24},
        "lstm": {"run": False},
    })
    # Should not raise
    validate_par_dict(par, dt_avg=pd.Timedelta(minutes=3), lang="en")


def test_validate_param_single_eq_max_de():
    par = build_par_dict({})
    with pytest.raises(AnomalyException) as exc:
        validate_param_single("EQ_MAX", -5, par, dt_avg=None, lang="de")
    # Localized message preserved in suggestions[0]
    assert "muss größer als 0 sein" in exc.value.suggestions[0]


def test_validate_param_single_dec_must_be_int():
    par = build_par_dict({})
    with pytest.raises(AnomalyException) as exc:
        validate_param_single("DEC", 1.5, par, dt_avg=None, lang="en")
    assert "is no integer" in str(exc.value)


def test_validate_param_single_v_max_lt_v_min():
    par = build_par_dict({"vMin": 100})
    with pytest.raises(AnomalyException) as exc:
        validate_param_single("V_MAX", 50, par, dt_avg=None, lang="en")
    assert "must be greater than the input value" in str(exc.value)


def test_validate_param_single_sbad_chg_max_negative():
    par = build_par_dict({})
    with pytest.raises(AnomalyException) as exc:
        validate_param_single("SBAD.CHG_MAX", -1, par, dt_avg=None, lang="de")
    # Localized message preserved in suggestions[0]
    assert "muss größer als 0" in exc.value.suggestions[0]


def test_validate_par_dict_stl_run_without_period_h_raises():
    par = build_par_dict({"stl": {"run": True}})
    par["STL"]["var"]["PERIOD_H"]["value"] = None  # explicitly clear
    with pytest.raises(AnomalyException) as exc:
        validate_par_dict(par, dt_avg=pd.Timedelta(minutes=3), lang="de")
    assert "muss eingegeben werden" in str(exc.value)


def test_validate_par_dict_lstm_run_without_period_h_raises():
    par = build_par_dict({"lstm": {"run": True}})
    par["LSTM"]["var"]["PERIOD_H"]["value"] = None
    with pytest.raises(AnomalyException) as exc:
        validate_par_dict(par, dt_avg=pd.Timedelta(minutes=3), lang="en")
    assert "must be set when LSTM is enabled" in str(exc.value)


def test_build_par_dict_lstm_period_h_default_24():
    par = build_par_dict({})
    assert par["LSTM"]["var"]["PERIOD_H"]["value"] == 24


def test_build_par_dict_lstm_threshold_default_100():
    par = build_par_dict({})
    assert par["LSTM"]["var"]["THRESHOLD"]["value"] == 100


def test_validate_par_dict_lstm_neurons_over_cap():
    par = build_par_dict({"lstm": {"run": True, "neurons": 5000}})
    with pytest.raises(AnomalyException) as exc:
        validate_par_dict(par, dt_avg=pd.Timedelta(minutes=3), lang="en")
    assert "exceeds the maximum" in str(exc.value)


def test_validate_par_dict_lstm_epochs_over_cap_de():
    par = build_par_dict({"lstm": {"run": True, "epochs": 9999}})
    with pytest.raises(AnomalyException) as exc:
        validate_par_dict(par, dt_avg=pd.Timedelta(minutes=3), lang="de")
    # Localized message preserved in suggestions[0]
    assert "überschreitet das Maximum" in exc.value.suggestions[0]


@pytest.mark.parametrize(
    "name",
    [
        "EQ_MAX",
        "GAP_MAX",
        "DEC",
        "LG_MIN",
        "V_MAX",
        "V_MIN",
        "SBAD.CHG_MAX",
        "SBAD.LG_MAX",
        "STL.PERIOD_H",
        "LSTM.PERIOD_H",
        "LSTM.NEURONS",
        "LSTM.EPOCHS",
        "LSTM.BATCH_SIZE",
    ],
)
def test_validate_param_single_none_value_is_noop(name):
    """None means 'not set' — must mirror validate_par_dict policy.

    Bug repro: clearing a field in the UI sent value=null to /validate-param,
    which crashed with 'Der Eingabewert None kann nicht in eine
    Fließkommazahl konvertiert werden'. Empty must be a legal in-progress
    state for live on-blur validation.
    """
    par = build_par_dict({})
    validate_param_single(name, None, par, dt_avg=None, lang="de")
    validate_param_single(name, None, par, dt_avg=None, lang="en")


def test_validate_param_single_none_does_not_mutate_par():
    """The short-circuit must not write None back into the par dict."""
    par = build_par_dict({"eqMax": 15})
    original = par["EQ_MAX"]["value"]
    validate_param_single("EQ_MAX", None, par, dt_avg=None, lang="de")
    assert par["EQ_MAX"]["value"] == original
