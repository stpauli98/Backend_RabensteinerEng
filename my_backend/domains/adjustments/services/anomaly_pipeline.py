"""
Anomaly detection pipeline phases.

Atomic, side-effect-free (modulo `df` mutation) functions ported from
`anomaly_detection_1.py` (L926-1529). Each phase accepts a DataFrame plus
the parameter values it needs, performs detection, NaN-s anomalies, and
optionally re-interpolates.

Bugs corrected vs. original Python:
  - L913: `epochs = meth["var"]["NEURONS"]` → use EPOCHS
  - L1568: filename uses Path.stem (handled in API layer, not here)
  - L1241/L1380: matplotlib Qt-only window calls — not ported
  - L1389: `plt.legend()` without label — plot building lives elsewhere
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from domains.adjustments.services.anomaly_helpers import (
    ProgressCallback,
    intrpl,
    rd,
    slope_calc,
    tr,
    create_sequences,
)
from domains.adjustments.debug_log import log_phase, dlog

logger = logging.getLogger(__name__)


# Deterministic seed for `prepare_lstm` RNG sources (Python `random`, NumPy,
# TensorFlow). Re-seeded per call so that recovery via
# `_ensure_lstm_intermediate` reproduces the exact model that the user
# previewed at /start (otherwise Keras weight init + optimizer momentum
# would silently shift residuals for the same threshold). The integer value
# itself is arbitrary; stability across calls is what matters.
_LSTM_SEED = 42


# ---------------------------------------------------------------------------
# Parameter dictionary factory
# ---------------------------------------------------------------------------

def build_par_dict(values: Dict[str, Any]) -> Dict[str, Any]:
    """
    Construct the structured `par` dictionary expected by validators and
    pipeline phases, populating localized `name` metadata.

    `values` is a flat dict from the API layer using camelCase or UPPER keys:
        eqMax, gapMax, dec, lgMin, vMax, vMin, el0,
        sbad: {chgMax, lgMax},
        stl: {run, periodH, threshold},
        lstm: {run, periodH, neurons, epochs, batchSize, threshold}

    Missing keys default to None / False.
    """
    def get(d, *keys, default=None):
        for k in keys:
            if k in d and d[k] is not None:
                return d[k]
        return default

    sbad_in = values.get("sbad") or {}
    stl_in = values.get("stl") or {}
    lstm_in = values.get("lstm") or {}

    par: Dict[str, Any] = {
        "EQ_MAX": _scalar(
            get(values, "eqMax", "EQ_MAX"),
            unit="min",
            name_en="Maximum allowable duration of constant values",
            name_de="Maximal zulässige Dauer konstanter Werte",
        ),
        "GAP_MAX": _scalar(
            get(values, "gapMax", "GAP_MAX"),
            unit="min",
            name_en="Maximum allowable gap for linear interpolation",
            name_de="Maximal zulässige Lücke für lineare Interpolation",
        ),
        "DEC": _scalar(
            get(values, "dec", "DEC"),
            unit=None,
            name_en="Number of decimal places",
            name_de="Anzahl der Dezimalstellen",
        ),
        "LG_MIN": _scalar(
            get(values, "lgMin", "LG_MIN"),
            unit="min",
            name_en="Minimum length of valid data segments",
            name_de="Mindestlänge gültiger Datenabschnitte",
        ),
        "V_MAX": _scalar(
            get(values, "vMax", "V_MAX"),
            unit=None,
            name_en="Maximum allowable value",
            name_de="Maximal zulässiger Wert",
        ),
        "V_MIN": _scalar(
            get(values, "vMin", "V_MIN"),
            unit=None,
            name_en="Minimum allowable value",
            name_de="Minimal zulässiger Wert",
        ),
        "EL0": _scalar(
            bool(get(values, "el0", "EL0", default=False)),
            unit="bool",
            name_en="Removing zeros",
            name_de="Nullwerte entfernen",
        ),
        "SBAD": {
            "name": {
                "en": "Slope-based anomaly detection",
                "de": "Steigungsbasierte Anomalieerkennung",
            },
            "var": {
                "CHG_MAX": _scalar(
                    get(sbad_in, "chgMax", "CHG_MAX"),
                    unit="#/min",
                    name_en="Slope threshold",
                    name_de="Schwellwert für die Steigung",
                ),
                "LG_MAX": _scalar(
                    get(sbad_in, "lgMax", "LG_MAX"),
                    unit="min",
                    name_en="Duration threshold",
                    name_de="Schwellwert für die Anomaliedauer",
                ),
            },
        },
        "STL": {
            "name": {
                "en": "Anomaly detection by seasonal-trend decomposition using LOESS (STL)",
                "de": "Anomalieerkennung durch saisonale Trendzerlegung mit LOESS (STL)",
            },
            "run": bool(get(stl_in, "run", default=False)),
            "var": {
                "PERIOD_H": _scalar(
                    get(stl_in, "periodH", "PERIOD_H", default=24),
                    unit="h",
                    name_en="Period duration",
                    name_de="Periodendauer",
                ),
                "PERIOD": _scalar(None, unit="timesteps",
                                  name_en="Period duration",
                                  name_de="Periodendauer"),
                "THRESHOLD": _scalar(
                    get(stl_in, "threshold", "THRESHOLD"),
                    unit=None,
                    name_en="Threshold for anomaly detection",
                    name_de="Schwellwert für die Anomalieerkennung",
                ),
            },
        },
        "LSTM": {
            "name": {
                "en": "Anomaly detection by Long Short-Term Memory (LSTM)",
                "de": "Anomalieerkennung mit Long Short-Term Memory (LSTM)",
            },
            "run": bool(get(lstm_in, "run", default=False)),
            "var": {
                "PERIOD_H": _scalar(
                    get(lstm_in, "periodH", "PERIOD_H", default=24),
                    unit="h",
                    name_en="Period duration",
                    name_de="Periodendauer",
                ),
                "PERIOD": _scalar(None, unit="timesteps",
                                  name_en="Period duration",
                                  name_de="Periodendauer"),
                "NEURONS": _scalar(
                    get(lstm_in, "neurons", "NEURONS", default=64),
                    unit=None,
                    name_en="Number of neurons",
                    name_de="Anzahl der Neuronen",
                ),
                "EPOCHS": _scalar(
                    get(lstm_in, "epochs", "EPOCHS", default=20),
                    unit=None,
                    name_en="Maximum number of training runs",
                    name_de="Maximale Anzahl an Trainingsdurchläufen",
                ),
                "BATCH_SIZE": _scalar(
                    get(lstm_in, "batchSize", "BATCH_SIZE", default=16),
                    unit=None,
                    name_en="Number of data points processed at once before the weights are updated",
                    name_de="Anzahl an Datenpunkten, welche auf einmal verarbeitet werden, bevor die Gewichte aktualisiert werden",
                ),
                "THRESHOLD": _scalar(
                    get(lstm_in, "threshold", "THRESHOLD", default=100),
                    unit=None,
                    name_en="Threshold for anomaly detection",
                    name_de="Schwellwert für die Anomalieerkennung",
                ),
            },
        },
    }

    return _attach_units(par)


def _scalar(value, unit, name_en, name_de):
    return {
        "value": value,
        "unit": unit,
        "name": {"en": name_en, "de": name_de},
    }


def _attach_units(par):
    """Append `[unit]` suffix to localized name strings (port of unit() helper)."""
    def walk(node):
        if isinstance(node, dict):
            if "value" in node and "name" in node and "unit" in node:
                u = node.get("unit")
                if u and u != "bool":
                    for lang in ("en", "de"):
                        n = node["name"].get(lang)
                        if n and f"[{u}]" not in n:
                            node["name"][lang] = f"{n} [{u}]"
            for v in node.values():
                walk(v)
    walk(par)
    return par


# ---------------------------------------------------------------------------
# Pipeline phases
# ---------------------------------------------------------------------------

@log_phase("preprocess_constants")
def process_constants(
    df: pd.DataFrame,
    eq_max: float,
    gap_max: Optional[float],
    dec,
    lang: str = "en",
    progress_callback: ProgressCallback = None,
) -> pd.DataFrame:
    """Port of L926-971 — NaN segments where value is constant for ≥ eq_max minutes."""
    if eq_max is None:
        return df

    n = len(df)
    if n < 2:
        return df

    label = tr("Removing constant values", "Entfernung konstanter Werte", lang)
    frm = 0
    idx_strt = None
    last_progress = 0.0

    times = df.iloc[:, 0].values
    values = df.iloc[:, 1].values

    for i in range(1, n):
        idx_end = None
        v_prev = values[i - 1]
        v_curr = values[i]

        # Opening analysis time window
        if v_prev == v_curr and frm == 0 and not pd.isna(v_curr):
            idx_strt = i - 1
            frm = 1
        # Closing analysis time window
        elif v_prev != v_curr and frm == 1:
            idx_end = i - 1

        if i == n - 1 and frm == 1:
            idx_end = i

        if idx_end is not None and idx_strt is not None:
            t_end = pd.Timestamp(times[idx_end])
            t_strt = pd.Timestamp(times[idx_strt])
            frm_width = (t_end - t_strt).total_seconds() / 60.0

            if frm_width >= eq_max:
                values[idx_strt : idx_end + 1] = np.nan

            frm = 0

        if progress_callback is not None:
            progress = i / max(n - 1, 1)
            if progress - last_progress >= 0.05 or i == n - 1:
                progress_callback(label, progress)
                last_progress = progress

    df.iloc[:, 1] = values

    if gap_max is not None:
        df = intrpl(df, gap_max, dec, lang, progress_callback=progress_callback)

    return df


@log_phase("preprocess_zeros")
def process_zeros(
    df: pd.DataFrame,
    el0: bool,
    gap_max: Optional[float],
    dec,
    lang: str = "en",
    progress_callback: ProgressCallback = None,
) -> pd.DataFrame:
    """Port of L973-986 — NaN all zero values, then interpolate."""
    if not el0:
        return df

    label = tr("Removing zeros", "Nullwerte entfernen", lang)
    if progress_callback is not None:
        progress_callback(label, 0.0)

    df.iloc[:, 1] = df.iloc[:, 1].mask(df.iloc[:, 1] == 0, np.nan)

    if progress_callback is not None:
        progress_callback(label, 1.0)

    if gap_max is not None:
        df = intrpl(df, gap_max, dec, lang, progress_callback=progress_callback)

    return df


@log_phase("preprocess_range")
def process_range(
    df: pd.DataFrame,
    v_max: Optional[float],
    v_min: Optional[float],
    gap_max: Optional[float],
    dec,
    lang: str = "en",
    progress_callback: ProgressCallback = None,
) -> pd.DataFrame:
    """Port of L988-1011 — NaN values outside [v_min, v_max], then interpolate."""
    if v_max is None and v_min is None:
        return df

    label = tr("Range filtering", "Range-Begrenzung", lang)
    if progress_callback is not None:
        progress_callback(label, 0.0)

    col = df.columns[1]
    mask = pd.Series(False, index=df.index)
    if v_max is not None:
        mask |= df[col] > v_max
    if v_min is not None:
        mask |= df[col] < v_min

    df.loc[mask, col] = np.nan

    if progress_callback is not None:
        progress_callback(label, 1.0)

    if gap_max is not None:
        df = intrpl(df, gap_max, dec, lang, progress_callback=progress_callback)

    return df


@log_phase("sbad")
def process_sbad(
    df: pd.DataFrame,
    chg_max: float,
    lg_max: float,
    gap_max: Optional[float],
    dec,
    lang: str = "en",
    progress_callback: ProgressCallback = None,
) -> Tuple[pd.DataFrame, int]:
    """
    Slope-based anomaly detection — port of L1013-1205.

    Iteratively detects anomalies via two strategies (immediate-after-NaN and
    analyzing-time-window) until the anomaly count stabilizes (3 equal counts
    in a row). Returns (df_modified, total_anomalies_detected).
    """
    if chg_max is None or lg_max is None:
        return df, 0

    label = tr(
        "Slope-based anomaly detection",
        "Steigungsbasierte Anomalieerkennung",
        lang,
    )

    df_slope = slope_calc(df)
    count_nan_strt = int(df.iloc[:, 1].isna().sum())
    count_an = [0]
    iteration = 0
    max_iterations = 50  # safety bound — guards against pathological non-convergence

    while iteration < max_iterations:
        iteration += 1
        if progress_callback is not None:
            progress_callback(f"{label} (iter {iteration})", 0.0)

        # ---- Pass 1: detect anomalies immediately after NaN ----
        n = len(df)
        frm = 0
        slope_values = df_slope.iloc[:, 1].values
        values = df.iloc[:, 1].values

        for i in range(n):
            if i == 0 or pd.isna(values[i]):
                frm = 1
            elif not pd.isna(slope_values[i]) and frm == 1:
                chg = slope_values[i]
                if abs(chg) > chg_max:
                    values[i - 1] = np.nan
                else:
                    frm = 0

        df.iloc[:, 1] = values
        count_nan_int = int(df.iloc[:, 1].isna().sum())
        count_an.append(count_nan_int - count_nan_strt)

        if len(count_an) >= 3 and len(set(count_an[-3:])) == 1:
            break

        # ---- Pass 2: ATW-based detection ----
        df_slope = slope_calc(df)
        slope_values = df_slope.iloc[:, 1].values
        values = df.iloc[:, 1].values
        times = df.iloc[:, 0].values

        frm = 0
        idx_strt_1 = None
        idx_strt_2 = None
        sign_strt = None

        for i in range(n):
            if not pd.isna(slope_values[i]):
                chg = slope_values[i]
                sign = (1 if chg > 0 else 0) - (1 if chg < 0 else 0)

                if frm == 0:
                    if abs(chg) > chg_max:
                        frm = 1
                        idx_strt_1 = i
                        idx_strt_2 = i
                        sign_strt = sign
                elif frm == 1:
                    if abs(chg) > chg_max and sign_strt == sign:
                        idx_strt_2 = i
                    elif abs(chg) > chg_max and sign_strt != sign:
                        idx_end = i
                        for i_frm in range(idx_strt_1, idx_end):
                            values[i_frm] = np.nan
                        frm = 0
                        idx_strt_1 = idx_strt_2 = sign_strt = None
                    else:
                        t_prev = pd.Timestamp(times[i - 1])
                        t_strt2 = pd.Timestamp(times[idx_strt_2])
                        gap = (t_prev - t_strt2).total_seconds() / 60.0
                        if gap > lg_max:
                            frm = 0
                            idx_strt_1 = idx_strt_2 = sign_strt = None
            else:
                if frm == 1:
                    idx_end = i
                    t_endm1 = pd.Timestamp(times[idx_end - 1]) if idx_end - 1 >= 0 else pd.Timestamp(times[0])
                    t_strt2 = pd.Timestamp(times[idx_strt_2])
                    gap = (t_endm1 - t_strt2).total_seconds() / 60.0
                    if gap <= lg_max:
                        for i_frm in range(idx_strt_1, idx_end):
                            values[i_frm] = np.nan
                    frm = 0
                    idx_strt_1 = idx_strt_2 = sign_strt = None

            # End of array with open ATW
            if i + 1 == n and frm == 1:
                for i_frm in range(idx_strt_1, n):
                    values[i_frm] = np.nan

        df.iloc[:, 1] = values
        count_nan_int = int(df.iloc[:, 1].isna().sum())
        count_an.append(count_nan_int - count_nan_strt)

        if len(count_an) >= 3 and len(set(count_an[-3:])) == 1:
            break

        df_slope = slope_calc(df)

    if iteration >= max_iterations:
        logger.warning(
            "process_sbad reached max_iterations=%d without count stabilization; "
            "last counts: %s",
            max_iterations,
            count_an[-3:] if len(count_an) >= 3 else count_an,
        )

    if progress_callback is not None:
        progress_callback(label, 1.0)

    if gap_max is not None:
        df = intrpl(df, gap_max, dec, lang, progress_callback=progress_callback)

    return df, count_an[-1]


@log_phase("stl_prep")
def prepare_stl(
    df: pd.DataFrame,
    period: int,
    lang: str = "en",
    progress_callback: ProgressCallback = None,
) -> Tuple[Any, pd.Series]:
    """
    Run STL decomposition (no threshold yet). Port of L1207-1253 (data part only,
    excluding the matplotlib plot which is built later by plot_builder).

    Raises ValueError if dataset still contains NaNs.
    Returns (stl_result, time_values).
    """
    from statsmodels.tsa.seasonal import STL

    if int(df.iloc[:, 1].isna().sum()) > 0:
        raise ValueError(
            tr(
                "'Anomaly detection by seasonal-trend decomposition using LOESS (STL)' "
                "cannot be performed because the dataset contains NaN.",
                "'Anomalieerkennung durch saisonale Trendzerlegung mit LOESS (STL)' "
                "kann nicht durchgeführt werden, weil NaN im Datensatz vorhanden sind.",
                lang,
            )
        )

    stl_label = tr("STL decomposition", "STL-Zerlegung", lang)
    if progress_callback is not None:
        progress_callback(stl_label, 0.0)

    stl = STL(df.iloc[:, 1], period=int(period), robust=True)
    result = stl.fit()

    if progress_callback is not None:
        progress_callback(stl_label, 1.0)

    time_values = pd.to_datetime(df.iloc[:, 0])
    return result, time_values


@log_phase("stl_apply")
def apply_stl_threshold(
    df: pd.DataFrame,
    stl_result,
    threshold: float,
    gap_max: Optional[float],
    dec,
    lang: str = "en",
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Port of L1273-1315 — mark residuals exceeding threshold as anomalies, NaN them, interpolate.
    Returns (df_modified, anomaly_mask).
    """
    if threshold is None:
        raise ValueError(
            tr(
                "Parameter 'Threshold for anomaly detection' must be set.",
                "Parameter 'Schwellwert für die Anomalieerkennung' muss eingegeben werden.",
                lang,
            )
        )

    stl_filter = stl_result.resid.abs() > float(threshold)
    df.loc[stl_filter, df.columns[1]] = np.nan

    if gap_max is not None:
        df = intrpl(df, gap_max, dec, lang)

    return df, stl_filter.values


@log_phase("lstm_prep")
def prepare_lstm(
    df: pd.DataFrame,
    period: int,
    neurons: int,
    epochs: int,
    batch_size: int,
    lang: str = "en",
    progress_callback: ProgressCallback = None,
) -> Tuple[pd.DataFrame, Any]:
    """
    Port of L1331-1394 (data part only) — train LSTM, compute residuals.
    Returns (results_df, model). results_df has columns:
      timestamp, value, forecast, residual, absolute_error

    Raises ValueError if dataset still contains NaNs (mirrors prepare_stl guard,
    since MinMaxScaler + LSTM both fail silently on NaN inputs).
    """
    if int(df.iloc[:, 1].isna().sum()) > 0:
        raise ValueError(
            tr(
                "'Anomaly detection by Long Short-Term Memory (LSTM)' "
                "cannot be performed because the dataset contains NaN.",
                "'Anomalieerkennung mit Long Short-Term Memory (LSTM)' "
                "kann nicht durchgeführt werden, weil NaN im Datensatz vorhanden sind.",
                lang,
            )
        )

    from keras.models import Sequential
    from keras.layers import Dense, Input, LSTM
    from sklearn.preprocessing import MinMaxScaler

    label = tr(
        "Anomaly detection by Long Short-Term Memory (LSTM)",
        "Anomalieerkennung mit Long Short-Term Memory (LSTM)",
        lang,
    )
    if progress_callback is not None:
        progress_callback(label, 0.0)

    scaler = MinMaxScaler()
    col_name = df.columns[1]
    scaled_col = f"{col_name} (scal)"
    df_local = df.copy()
    df_local[scaled_col] = scaler.fit_transform(df_local.iloc[:, [1]])

    X, y = create_sequences(df_local[scaled_col].values, int(period))
    X = X.reshape((X.shape[0], X.shape[1], 1))

    # Seed RNG sources so weight init / optimizer momentum are deterministic.
    # Required for `_ensure_lstm_intermediate` recovery to reproduce the same
    # model the user previewed at /start. Keras 3 exposes a one-shot helper
    # that covers Python `random`, NumPy and TensorFlow; older Keras versions
    # need the three calls explicitly.
    try:
        import keras.utils as _keras_utils  # local import — keeps top-of-file imports lean
        _keras_utils.set_random_seed(_LSTM_SEED)
    except (ImportError, AttributeError):
        import random as _random
        import tensorflow as _tf
        _random.seed(_LSTM_SEED)
        np.random.seed(_LSTM_SEED)
        _tf.random.set_seed(_LSTM_SEED)

    model = Sequential()
    model.add(Input(shape=(int(period), 1)))
    model.add(LSTM(int(neurons), activation="tanh"))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse")

    if progress_callback is not None:
        from keras.callbacks import LambdaCallback
        total_epochs = int(epochs)
        epoch_cb = LambdaCallback(
            on_epoch_end=lambda epoch, logs: progress_callback(
                label, (epoch + 1) / total_epochs
            )
        )
        fit_callbacks = [epoch_cb]
    else:
        fit_callbacks = None

    model.fit(X, y, epochs=int(epochs), batch_size=int(batch_size), verbose=0, callbacks=fit_callbacks)

    predicted_scaled = model.predict(X, verbose=0)
    predicted = scaler.inverse_transform(predicted_scaled)
    actual = scaler.inverse_transform(y.reshape(-1, 1))

    result_time = df_local.iloc[int(period):, 0].reset_index(drop=True)
    results = pd.DataFrame({
        "timestamp": result_time,
        "value": actual.flatten(),
        "forecast": predicted.flatten(),
    })
    results["residual"] = results["value"] - results["forecast"]
    results["absolute_error"] = np.abs(results["residual"])

    if progress_callback is not None:
        progress_callback(label, 1.0)

    return results, model


@log_phase("lstm_apply")
def apply_lstm_threshold(
    df: pd.DataFrame,
    results_df: pd.DataFrame,
    threshold: float,
    gap_max: Optional[float],
    dec,
    lang: str = "en",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Port of L1417-1467 — mark rows whose absolute_error exceeds threshold as
    anomalies, NaN those rows in df, interpolate.
    Returns (df_modified, anomalies_df).
    """
    if threshold is None:
        raise ValueError(
            tr(
                "Parameter 'Threshold for anomaly detection' must be set.",
                "Parameter 'Schwellwert für die Anomalieerkennung' muss eingegeben werden.",
                lang,
            )
        )

    results_df = results_df.copy()
    results_df["anomaly"] = results_df["absolute_error"] > float(threshold)
    anomalies = results_df[results_df["anomaly"]]

    df.loc[
        df.iloc[:, 0].isin(anomalies["timestamp"]),
        df.columns[1],
    ] = np.nan

    if gap_max is not None:
        df = intrpl(df, gap_max, dec, lang)

    return df, anomalies


@log_phase("finalize")
def process_short_ranges(
    df: pd.DataFrame,
    lg_min: Optional[float],
    lang: str = "en",
    progress_callback: ProgressCallback = None,
) -> pd.DataFrame:
    """Port of L1470-1529 — NaN valid-value segments shorter than lg_min minutes."""
    if lg_min is None:
        return df

    label = tr(
        "Removing excessively short value ranges",
        "Eliminierung kurzer gültiger Zeitfenster",
        lang,
    )

    n = len(df)
    if n == 0:
        return df

    times = df.iloc[:, 0].values
    values = df.iloc[:, 1].values

    frm = 0
    idx_strt = None
    last_progress = 0.0

    for i in range(n):
        v_curr = values[i]

        if not pd.isna(v_curr) and frm == 0:
            idx_strt = i
            frm = 1
        elif pd.isna(v_curr) and frm == 1:
            idx_end = i
            t_endm1 = pd.Timestamp(times[idx_end - 1])
            t_strt = pd.Timestamp(times[idx_strt])
            gap = (t_endm1 - t_strt).total_seconds() / 60.0
            if gap < lg_min:
                values[idx_strt:idx_end] = np.nan
            frm = 0
            idx_strt = None

        if progress_callback is not None:
            progress = i / max(n - 1, 1)
            if progress - last_progress >= 0.05 or i == n - 1:
                progress_callback(label, progress)
                last_progress = progress

    # Trailing open window: keep values (matches Python's drop-through behaviour
    # at L1494 where end of loop without NaN does not close the ATW).
    df.iloc[:, 1] = values
    return df
