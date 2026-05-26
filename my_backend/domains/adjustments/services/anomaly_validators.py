"""
Anomaly detection parameter validators.

Localized validation functions ported from `anomaly_detection_1.py` (L110-181).
Each function takes a parameter descriptor `v = {"value": x, "name": {"en": str, "de": str}}`
plus an explicit `lang` argument. Raises ValueError with the localized message
matching the original Python script byte-for-byte.

Also exposes `validate_par_dict(par, dt_avg, lang)` which mirrors the preflight
checks at L790-921 (dependency checks + period-step alignment for STL).
"""
from __future__ import annotations

import math
from typing import Dict, Any, Optional

import pandas as pd

from domains.adjustments.services.anomaly_helpers import hms, tr
from shared.exceptions.errors import (
    AnomalyException,
    STLPeriodAlignmentError,
    LSTMHyperparameterCapError,
    SBADSlopeRuleError,
    ParameterOutOfRangeError,
)


def _name(v: Dict[str, Any], lang: str) -> str:
    return v["name"].get(lang, v["name"].get("en"))


def check_float(v: Dict[str, Any], lang: str = "en") -> float:
    """Convert v['value'] to float; raise AnomalyException with localized message on failure."""
    try:
        return float(v["value"])
    except (ValueError, TypeError):
        name = _name(v, lang)
        value = v["value"]
        raise AnomalyException(
            tr(
                f"The input value '{value}' for '{name}' cannot be converted to a float.",
                f"Der Eingabewert '{value}' für '{name}' kann nicht in eine Fließkommazahl konvertiert werden.",
                lang,
            ),
            error_code='PARAM_NOT_FLOAT',
            details={'param': name, 'value': str(value)},
        )


def check_integer(v: Dict[str, Any], lang: str = "en") -> int:
    """Verify v['value'] is an integer (whole number); raise AnomalyException otherwise."""
    value = v["value"]
    try:
        if int(value) == float(value):
            return int(value)
    except (ValueError, TypeError):
        pass
    name = _name(v, lang)
    raise AnomalyException(
        tr(
            f"The input value '{value}' for '{name}' is no integer.",
            f"Der Eingabewert '{value}' für '{name}' ist kein Integer.",
            lang,
        ),
        error_code='PARAM_NOT_INTEGER',
        details={'param': name, 'value': str(value)},
    )


def check_gt_zero(v: Dict[str, Any], lang: str = "en") -> None:
    """Raise AnomalyException (ParameterOutOfRangeError) if v['value'] is <= 0."""
    if v["value"] is None or v["value"] <= 0:
        name = _name(v, lang)
        value = v["value"]
        raise ParameterOutOfRangeError(
            param=name,
            value=value if value is not None else 0,
            min_value=0,
            suggestions=[
                tr(
                    f"The input value '{value}' for '{name}' must be greater than 0.",
                    f"Der Eingabewert '{value}' für '{name}' muss größer als 0 sein.",
                    lang,
                ),
            ],
        )


def check_ge_zero(v: Dict[str, Any], lang: str = "en") -> None:
    """Raise AnomalyException (ParameterOutOfRangeError) if v['value'] is < 0."""
    if v["value"] is None or v["value"] < 0:
        name = _name(v, lang)
        value = v["value"]
        raise ParameterOutOfRangeError(
            param=name,
            value=value if value is not None else -1,
            min_value=0,
            suggestions=[
                tr(
                    f"The input value '{value}' for '{name}' must be greater than or equal to 0.",
                    f"Der Eingabewert '{value}' für '{name}' muss größer oder gleich 0 sein.",
                    lang,
                ),
            ],
        )


def check_comp(v1: Dict[str, Any], v2: Dict[str, Any], lang: str = "en") -> None:
    """Ensure v1['value'] >= v2['value']; raise AnomalyException otherwise."""
    if v1["value"] is None or v2["value"] is None:
        return
    if v1["value"] < v2["value"]:
        name1 = _name(v1, lang)
        name2 = _name(v2, lang)
        value1 = v1["value"]
        value2 = v2["value"]
        raise AnomalyException(
            tr(
                f"The input value '{value1}' for '{name1}' must be greater than the input value '{value2}' for '{name2}'.",
                f"Der Eingabewert '{value1}' von '{name1}' muss größer als der Eingabewert '{value2}' von '{name2}' sein.",
                lang,
            ),
            error_code='PARAM_COMPARISON_FAILED',
            details={
                'param1': name1, 'value1': value1,
                'param2': name2, 'value2': value2,
            },
        )


def validate_par_dict(par: Dict[str, Any], dt_avg: Optional[pd.Timedelta], lang: str = "en") -> Dict[str, Any]:
    """
    Preflight validation matching Python script L790-921.

    Validates parameter coherence + dependencies. Mutates `par` to coerce
    numeric strings → numeric (matching the original behavior). Returns the
    mutated par for convenience.

    Raises ValueError with localized message on first failure.
    """
    # TIME-BASED THRESHOLDS
    if par["EQ_MAX"]["value"] is not None:
        par["EQ_MAX"]["value"] = check_float(par["EQ_MAX"], lang)
        check_gt_zero(par["EQ_MAX"], lang)

    if par["GAP_MAX"]["value"] is not None:
        par["GAP_MAX"]["value"] = check_float(par["GAP_MAX"], lang)
        check_ge_zero(par["GAP_MAX"], lang)

    if par["DEC"]["value"] is not None:
        par["DEC"]["value"] = check_integer(par["DEC"], lang)
        check_ge_zero(par["DEC"], lang)

    if par["LG_MIN"]["value"] is not None:
        par["LG_MIN"]["value"] = check_float(par["LG_MIN"], lang)
        check_ge_zero(par["LG_MIN"], lang)

    # VALUE LIMITS
    v_max = par["V_MAX"]
    v_min = par["V_MIN"]
    if v_max["value"] is not None:
        v_max["value"] = check_float(v_max, lang)
    if v_min["value"] is not None:
        v_min["value"] = check_float(v_min, lang)
    if v_max["value"] is not None and v_min["value"] is not None:
        check_comp(v_max, v_min, lang)

    # SLOPE-BASED ANOMALY DETECTION
    sbad = par["SBAD"]
    sbad_values = [sbad["var"][k]["value"] for k in sbad["var"]]
    if any(val is not None for val in sbad_values):
        if sbad["var"]["CHG_MAX"]["value"] is not None:
            sbad["var"]["CHG_MAX"]["value"] = check_float(sbad["var"]["CHG_MAX"], lang)
            check_gt_zero(sbad["var"]["CHG_MAX"], lang)
        if sbad["var"]["LG_MAX"]["value"] is not None:
            sbad["var"]["LG_MAX"]["value"] = check_float(sbad["var"]["LG_MAX"], lang)
            check_ge_zero(sbad["var"]["LG_MAX"], lang)
        if sbad_values.count(None) == 1:
            # Determine which of CHG_MAX / LG_MAX was provided
            provided = 'chg_max' if sbad["var"]["CHG_MAX"]["value"] is not None else 'lg_max'
            raise SBADSlopeRuleError(
                provided=provided,
                suggestions=[
                    tr(
                        f"All parameters for '{sbad['name']['en']}' must be set.",
                        f"Alle Parameter für '{sbad['name']['de']}' müssen eingegeben werden.",
                        lang,
                    ),
                ],
            )

    # STL — period-step alignment requires dt_avg
    if par["STL"]["run"]:
        period_h = par["STL"]["var"]["PERIOD_H"]
        if period_h["value"] is None:
            raise AnomalyException(
                tr(
                    f"Parameter '{period_h['name']['en']}' must be set when STL is enabled.",
                    f"Parameter '{period_h['name']['de']}' muss eingegeben werden, wenn STL aktiviert ist.",
                    lang,
                ),
                error_code='STL_PERIOD_REQUIRED',
                details={'param': period_h['name']['en']},
            )
        period_h["value"] = check_float(period_h, lang)
        check_gt_zero(period_h, lang)
        if dt_avg is None:
            raise AnomalyException(
                tr(
                    "STL validation requires a known time step; load CSV first.",
                    "STL-Prüfung benötigt eine bekannte Zeitschrittweite; CSV zuerst laden.",
                    lang,
                ),
                error_code='STL_REQUIRES_DT_AVG',
            )
        period = par["STL"]["var"]["PERIOD"]
        period["value"] = period_h["value"] * 3600 / dt_avg.total_seconds()
        if not math.isclose(period["value"], round(period["value"]), rel_tol=1e-9):
            dt_avg_h = dt_avg.total_seconds() / 3600.0
            raise STLPeriodAlignmentError(
                period_h=period_h["value"],
                dt_avg_h=dt_avg_h,
                suggestions=[
                    tr(
                        f"The input value '{period_h['value']}' (→ '{hms(period_h['value'], 'hours')}') for "
                        f"'{period_h['name']['en']}' must be an integer multiple of the time step "
                        f"'{hms(dt_avg, 'timedelta')}' of the data set.",
                        f"Der Eingabewert von '{period_h['value']}' (→ '{hms(period_h['value'], 'hours')}') für "
                        f"die '{period_h['name']['de']}' muss ein ganzzahliges Vielfaches der Zeitschrittweite "
                        f"'{hms(dt_avg, 'timedelta')}' des Datensatzes sein.",
                        lang,
                    ),
                ],
            )

    # LSTM
    if par["LSTM"]["run"]:
        period_h = par["LSTM"]["var"]["PERIOD_H"]
        if period_h["value"] is None:
            raise AnomalyException(
                tr(
                    f"Parameter '{period_h['name']['en']}' must be set when LSTM is enabled.",
                    f"Parameter '{period_h['name']['de']}' muss eingegeben werden, wenn LSTM aktiviert ist.",
                    lang,
                ),
                error_code='LSTM_PERIOD_REQUIRED',
                details={'param': period_h['name']['en']},
            )
        period_h["value"] = check_float(period_h, lang)
        check_gt_zero(period_h, lang)
        if dt_avg is not None:
            par["LSTM"]["var"]["PERIOD"]["value"] = int(
                period_h["value"] * 3600 / dt_avg.total_seconds()
            )

        # Upper bound on LSTM hyperparams — guards against resource exhaustion
        # (a single authenticated request with epochs=10000, neurons=10000
        # would block the Cloud Run worker for hours).
        upper_bounds = {"NEURONS": 1024, "EPOCHS": 500, "BATCH_SIZE": 4096}
        param_keys = {"NEURONS": "neurons", "EPOCHS": "epochs", "BATCH_SIZE": "batch_size"}
        for key in ("NEURONS", "EPOCHS", "BATCH_SIZE"):
            param = par["LSTM"]["var"][key]
            if param["value"] is not None:
                param["value"] = check_integer(param, lang)
                check_gt_zero(param, lang)
                cap = upper_bounds[key]
                if param["value"] > cap:
                    name = _name(param, lang)
                    raise LSTMHyperparameterCapError(
                        param=param_keys[key],
                        value=param["value"],
                        max_value=cap,
                        suggestions=[
                            tr(
                                f"The input value '{param['value']}' for '{name}' "
                                f"exceeds the maximum of {cap}.",
                                f"Der Eingabewert '{param['value']}' für '{name}' "
                                f"überschreitet das Maximum von {cap}.",
                                lang,
                            ),
                        ],
                    )

    return par


def validate_param_single(name: str, value, par: Dict[str, Any], dt_avg: Optional[pd.Timedelta], lang: str = "en") -> None:
    """
    Validate a single parameter value in isolation (used by `/validate-param` endpoint).

    `name` is the param key path (e.g. "EQ_MAX", "SBAD.CHG_MAX", "STL.PERIOD_H").
    Raises ValueError with localized message on failure.

    `None` is treated as "not set" — a legal in-progress state for on-blur
    validation, matching `validate_par_dict` and the original
    `anomaly_detection_1.py` script (L794: `if par[X]["value"] is not None`).
    Cross-field constraints and required-when-enabled checks (SBAD pair,
    STL/LSTM run-without-period) are enforced at submit time by
    `validate_par_dict`.
    """
    descriptor = _resolve_descriptor(par, name)
    if descriptor is None:
        raise AnomalyException(
            tr(
                f"Unknown parameter '{name}'.",
                f"Unbekannter Parameter '{name}'.",
                lang,
            ),
            error_code='UNKNOWN_PARAMETER',
            details={'param': name},
        )

    # None = "not set" — skip per-field validation. Booleans (EL0, STL.run,
    # LSTM.run) never come through as None from the frontend; they fall
    # through to the boolean isinstance check below.
    if value is None and name not in ("EL0", "STL.run", "LSTM.run"):
        return

    # Build a temp descriptor with the candidate value to validate
    candidate = dict(descriptor)
    candidate["value"] = value

    # Per-parameter validation rules (mirror L790-921)
    if name in ("EQ_MAX", "SBAD.CHG_MAX", "STL.PERIOD_H", "LSTM.PERIOD_H"):
        candidate["value"] = check_float(candidate, lang)
        check_gt_zero(candidate, lang)
    elif name in ("GAP_MAX", "LG_MIN", "SBAD.LG_MAX"):
        candidate["value"] = check_float(candidate, lang)
        check_ge_zero(candidate, lang)
    elif name == "DEC":
        candidate["value"] = check_integer(candidate, lang)
        check_ge_zero(candidate, lang)
    elif name in ("V_MAX", "V_MIN"):
        candidate["value"] = check_float(candidate, lang)
        # V_MAX vs V_MIN cross-check uses values currently in `par`
        if name == "V_MAX" and par["V_MIN"]["value"] is not None:
            check_comp(candidate, par["V_MIN"], lang)
        elif name == "V_MIN" and par["V_MAX"]["value"] is not None:
            check_comp(par["V_MAX"], candidate, lang)
    elif name in ("LSTM.NEURONS", "LSTM.EPOCHS", "LSTM.BATCH_SIZE"):
        candidate["value"] = check_integer(candidate, lang)
        check_gt_zero(candidate, lang)
    elif name == "EL0":
        # Boolean: nothing to validate beyond presence
        if not isinstance(value, bool):
            raise AnomalyException(
                tr(
                    f"The input value '{value}' for '{_name(candidate, lang)}' must be true or false.",
                    f"Der Eingabewert '{value}' für '{_name(candidate, lang)}' muss true oder false sein.",
                    lang,
                ),
                error_code='PARAM_NOT_BOOLEAN',
                details={'param': name, 'value': str(value)},
            )
    elif name in ("STL.run", "LSTM.run"):
        if not isinstance(value, bool):
            raise AnomalyException(
                tr(
                    f"The input value '{value}' for '{name}' must be true or false.",
                    f"Der Eingabewert '{value}' für '{name}' muss true oder false sein.",
                    lang,
                ),
                error_code='PARAM_NOT_BOOLEAN',
                details={'param': name, 'value': str(value)},
            )
    else:
        raise AnomalyException(
            tr(
                f"No validation rule for parameter '{name}'.",
                f"Keine Prüfregel für Parameter '{name}'.",
                lang,
            ),
            error_code='NO_VALIDATION_RULE',
            details={'param': name},
        )


def _resolve_descriptor(par: Dict[str, Any], name: str) -> Optional[Dict[str, Any]]:
    """Resolve dotted parameter path (e.g. 'SBAD.CHG_MAX') to its descriptor dict."""
    parts = name.split(".")
    if len(parts) == 1:
        return par.get(parts[0])
    if len(parts) == 2:
        section = par.get(parts[0])
        if section is None:
            return None
        if "var" in section:
            return section["var"].get(parts[1])
        return section.get(parts[1])
    return None
