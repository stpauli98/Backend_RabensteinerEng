"""S1: MASE must degrade to NaN (like WAPE), never crash the metrics dict.

A constant/flat target series makes the MASE naive baseline 0. Previously this
raised ZeroDivisionError, which the pipeline's broad except turned into an empty
metrics dict (all of MAE/RMSE/... lost). MASE should return NaN and let the other
metrics through.

S2: original-scale metrics must be computed from the unscaled ground truth via
inverse_scale_predictions(), not duplicated from the scaled metrics.
"""
import math

import numpy as np

from domains.training.ml.exact import (
    calculate_evaluation_metrics,
    _calculate_single_timestep_metrics,
)


# --------------------------------------------------------------------------- S1
def test_constant_series_does_not_raise_and_keeps_other_metrics():
    # Constant ground truth -> naive MAE == 0. Shape (samples, timesteps, features).
    y_true = np.full((4, 3, 1), 5.0)
    y_pred = y_true + 0.1
    m = calculate_evaluation_metrics(y_true, y_pred)
    # Other metrics computed normally
    assert m["MAE"] > 0
    assert m["RMSE"] > 0
    # MASE degrades to NaN instead of raising
    assert math.isnan(m["MASE"])


def test_constant_single_timestep_yields_nan_mase():
    v_true = np.full(5, 2.0)
    v_pred = np.array([2.0, 2.1, 1.9, 2.0, 2.2])
    ts = _calculate_single_timestep_metrics(v_true, v_pred)
    assert math.isnan(ts["mase"])
    assert ts["mae"] >= 0


def test_too_few_points_yields_nan_mase():
    v_true = np.array([3.0])
    v_pred = np.array([3.5])
    ts = _calculate_single_timestep_metrics(v_true, v_pred)
    assert math.isnan(ts["mase"])


# --------------------------------------------------------------------------- S2
def test_inverse_scale_predictions_roundtrips_via_scaler():
    from sklearn.preprocessing import MinMaxScaler

    from domains.training.ml.exact import inverse_scale_predictions

    # One output feature scaled from real units [0, 100] into [0, 1].
    scaler = MinMaxScaler().fit(np.array([[0.0], [100.0]]))
    o_scalers = {0: scaler}

    # Scaled predictions (n=2, timesteps=3, features=1) representing real [10,20,30]/[40,50,60].
    real = np.array([[[10.0], [20.0], [30.0]], [[40.0], [50.0], [60.0]]])
    scaled = np.empty_like(real)
    for i in range(real.shape[0]):
        scaled[i, :, 0] = scaler.transform(real[i, :, 0].reshape(-1, 1)).ravel()

    restored = inverse_scale_predictions(scaled, o_scalers)
    assert np.allclose(restored, real, atol=1e-6)
    # Original input untouched (helper must not mutate its argument)
    assert not np.allclose(scaled, real)


def test_original_metrics_differ_from_scaled_when_data_is_scaled():
    """When a scaler maps real units into [0,1], original-scale MAE must be larger
    than scaled MAE (it is in real units), proving the two are computed distinctly."""
    from sklearn.preprocessing import MinMaxScaler

    from domains.training.ml.exact import inverse_scale_predictions

    scaler = MinMaxScaler().fit(np.array([[0.0], [100.0]]))
    o_scalers = {0: scaler}

    real_true = np.array([[[10.0], [20.0], [30.0]], [[40.0], [50.0], [60.0]]])
    real_pred = real_true + 5.0  # 5-unit error in real space

    scaled_true = np.empty_like(real_true)
    scaled_pred = np.empty_like(real_pred)
    for i in range(real_true.shape[0]):
        scaled_true[i, :, 0] = scaler.transform(real_true[i, :, 0].reshape(-1, 1)).ravel()
        scaled_pred[i, :, 0] = scaler.transform(real_pred[i, :, 0].reshape(-1, 1)).ravel()

    scaled_metrics = calculate_evaluation_metrics(scaled_true, scaled_pred)
    pred_orig = inverse_scale_predictions(scaled_pred, o_scalers)
    orig_metrics = calculate_evaluation_metrics(real_true, pred_orig)

    # Scaled MAE ~0.05 (5/100), original MAE ~5.0 — distinctly different magnitudes.
    assert orig_metrics["MAE"] > scaled_metrics["MAE"] * 10
    assert abs(orig_metrics["MAE"] - 5.0) < 1e-6
