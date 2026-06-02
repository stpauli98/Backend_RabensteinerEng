# W11 Evaluation-Metrics Correctness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix three correctness bugs in the W11 ML-training evaluation path: MASE crashing wipes all metrics (S1), "original-scale" metrics are silently fake (S2), and LGBMR crashes with ≥2 output features (S3).

**Architecture:** All three are pre-existing logic bugs in merged `main`, in `domains/training/ml/exact.py` (metrics computation + the `run_exact_training_pipeline` evaluation block) and `domains/training/ml/trainer.py` (`train_lgbmr`). Fixes are surgical: make MASE return `np.nan` like WAPE already does; compute the original-scale metrics from `tst_y_orig` + inverse-scaled predictions (a small extracted helper, reused by the existing 12-level block); and reshape LGBMR I/O by the real `n_features_out` instead of a hard-coded `1`.

**Tech Stack:** Python 3.9, numpy, scikit-learn, lightgbm, pytest. Backend tests run in Docker (`docker compose run --rm -T backend pytest …`).

---

## File Structure

- `domains/training/ml/exact.py`
  - `calculate_evaluation_metrics` (`:92`) and `_calculate_single_timestep_metrics` (`:237`) — MASE no longer raises (S1).
  - New module-level helper `inverse_scale_predictions(predictions, o_scalers)` — lifts the inverse-transform loop currently inlined at `:665-688` so it can be reused (S2).
  - `run_exact_training_pipeline` evaluation block (`:586-597`) — wire real original-scale metrics (S2); LGBMR test/val predict reshape (`:582`, `:640`) — use `n_features_out` (S3).
- `domains/training/ml/trainer.py`
  - `train_lgbmr` (`:416`) — flatten `train_y` by `n_timesteps * n_features_out` (S3).
- `tests/domains/training/ml/test_evaluation_metrics.py` (new) — S1 + S2 unit tests.
- `tests/domains/training/ml/test_lgbmr_multioutput.py` (new) — S3 unit test.

---

## Task 1 (S1): MASE returns NaN instead of crashing the whole metrics dict

**Why:** MASE (`exact.py:152-166` total, `:291-303` per-timestep) raises `ZeroDivisionError`/`ValueError` when the naive baseline is 0 (any output timestep where all test samples share one value — a flat/offline/saturated series) or when there are ≤1 points. The raise aborts `calculate_evaluation_metrics` entirely; it is caught by the broad `except` at `:722`, which replaces **all** metrics with empty dicts. WAPE already returns `np.nan` in the same situation — make MASE consistent so one degenerate timestep yields `MASE=NaN` while the other 7 metrics survive.

**Files:**
- Test: `tests/domains/training/ml/test_evaluation_metrics.py`
- Modify: `domains/training/ml/exact.py:152-166` and `domains/training/ml/exact.py:291-305`

- [ ] **Step 1: Write the failing tests**

Create `tests/domains/training/ml/test_evaluation_metrics.py`:

```python
"""S1: MASE must degrade to NaN (like WAPE), never crash the metrics dict.

A constant/flat target series makes the MASE naive baseline 0. Previously this
raised ZeroDivisionError, which the pipeline's broad except turned into an empty
metrics dict (all of MAE/RMSE/... lost). MASE should return NaN and let the other
metrics through.
"""
import numpy as np
import math
from domains.training.ml.exact import (
    calculate_evaluation_metrics,
    _calculate_single_timestep_metrics,
)


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `docker compose run --rm -T backend pytest tests/domains/training/ml/test_evaluation_metrics.py -q -p no:cacheprovider`
Expected: FAIL — `ZeroDivisionError: Naive MAE ist 0 – MASE nicht definiert.` (and `ValueError` for the too-few-points case).

- [ ] **Step 3: Fix the total-MASE branch (`exact.py:152-166`)**

Replace:

```python
    # MASE - EXACT MATCH: raises exceptions (original lines 597-617)
    m = 1  # Saisonalität
    n_mase = len(y_true_flat)
    mae_forecast = sum(abs(yt - yp) for yt, yp in zip(y_true_flat, y_pred_flat)) / n_mase

    if n_mase <= m:
        raise ValueError("Zu wenig Daten für gewählte Saisonalität m.")

    naive_errors = [abs(y_true_flat[t] - y_true_flat[t - m]) for t in range(m, n_mase)]
    mae_naive = sum(naive_errors) / len(naive_errors)

    if mae_naive == 0:
        raise ZeroDivisionError("Naive MAE ist 0 – MASE nicht definiert.")

    mase = mae_forecast / mae_naive
```

with:

```python
    # MASE - degrades to NaN (like WAPE) when undefined, instead of raising and
    # taking down the entire metrics dict via the pipeline's broad except.
    m = 1  # Saisonalität
    n_mase = len(y_true_flat)
    mae_forecast = sum(abs(yt - yp) for yt, yp in zip(y_true_flat, y_pred_flat)) / n_mase

    if n_mase <= m:
        # Too few points for the chosen seasonality -> MASE undefined.
        mase = np.nan
    else:
        naive_errors = [abs(y_true_flat[t] - y_true_flat[t - m]) for t in range(m, n_mase)]
        mae_naive = sum(naive_errors) / len(naive_errors)
        if mae_naive == 0:
            # Constant series -> naive baseline 0 -> MASE undefined.
            mase = np.nan
        else:
            mase = mae_forecast / mae_naive
```

- [ ] **Step 4: Fix the per-timestep MASE branch (`exact.py:291-305`)**

Replace:

```python
    # MASE - EXACT MATCH: raises exceptions (original lines 597-617)
    m = 1  # Saisonalität
    n_mase = len(v_true)
    mae_forecast = sum(abs(yt - yp) for yt, yp in zip(v_true, v_pred)) / n_mase

    if n_mase <= m:
        raise ValueError("Zu wenig Daten für gewählte Saisonalität m.")

    naive_errors = [abs(v_true[t] - v_true[t - m]) for t in range(m, n_mase)]
    mae_naive_val = sum(naive_errors) / len(naive_errors)

    if mae_naive_val == 0:
        raise ZeroDivisionError("Naive MAE ist 0 – MASE nicht definiert.")

    ts_mase = float(mae_forecast / mae_naive_val)
```

with:

```python
    # MASE - degrades to NaN (like WAPE) when undefined, instead of raising.
    m = 1  # Saisonalität
    n_mase = len(v_true)
    mae_forecast = sum(abs(yt - yp) for yt, yp in zip(v_true, v_pred)) / n_mase

    if n_mase <= m:
        ts_mase = float("nan")
    else:
        naive_errors = [abs(v_true[t] - v_true[t - m]) for t in range(m, n_mase)]
        mae_naive_val = sum(naive_errors) / len(naive_errors)
        if mae_naive_val == 0:
            ts_mase = float("nan")
        else:
            ts_mase = float(mae_forecast / mae_naive_val)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `docker compose run --rm -T backend pytest tests/domains/training/ml/test_evaluation_metrics.py -q -p no:cacheprovider`
Expected: PASS (3 passed).

- [ ] **Step 6: Commit**

```bash
git add tests/domains/training/ml/test_evaluation_metrics.py domains/training/ml/exact.py
git commit -m "fix(training): MASE degrades to NaN instead of wiping all eval metrics

A constant/flat output timestep makes the MASE naive baseline 0; the
ZeroDivisionError was caught by the pipeline's broad except and replaced the
entire metrics dict with empty dicts. MASE now returns NaN like WAPE, so a
single degenerate timestep no longer destroys MAE/RMSE/MAPE/etc.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 2 (S3): LGBMR supports multiple output features

**Why:** `train_lgbmr` (`trainer.py:433`) does `y_flat = train_y.reshape(n_samples, n_timesteps)`, and the predict reshapes (`exact.py:582`, `:640`) hard-code `(n_samples, n_timesteps, 1)`. `train_y` is `(n_samples, n_timesteps, n_features_out)`, so with ≥2 output features the reshape throws `ValueError: cannot reshape array of size … into shape (n, ts)`. SVR_dir/MIMO/LIN avoid this by looping one model per output feature. Fix LGBMR to flatten/reshape by the real `n_features_out`. (Done before Task 3 so the S2 wiring reasons about correctly-shaped predictions.)

**Files:**
- Test: `tests/domains/training/ml/test_lgbmr_multioutput.py`
- Modify: `domains/training/ml/trainer.py:427-448`, `domains/training/ml/exact.py:581-582`, `domains/training/ml/exact.py:639-640`

- [ ] **Step 1: Write the failing test**

Create `tests/domains/training/ml/test_lgbmr_multioutput.py`:

```python
"""S3: LGBMR must handle >1 output feature.

train_lgbmr flattened train_y to (n_samples, n_timesteps), which only works for a
single output feature; >=2 output features raised a numpy reshape ValueError.
"""
import numpy as np
import pytest
from types import SimpleNamespace
from domains.training.ml.trainer import train_lgbmr


def _mdl():
    return SimpleNamespace(N_ESTIMATORS=10, LEARNING_RATE=0.1, MAX_DEPTH=3)


def test_lgbmr_two_output_features_trains_and_predicts_correct_width():
    n_samples, n_timesteps, n_feat_in, n_feat_out = 20, 4, 3, 2
    rng = np.random.RandomState(0)
    train_x = rng.rand(n_samples, n_timesteps, n_feat_in)
    train_y = rng.rand(n_samples, n_timesteps, n_feat_out)

    model = train_lgbmr(train_x, train_y, _mdl())

    import pandas as pd
    x_flat = train_x.reshape(n_samples, -1)
    cols = [f"x_{k}" for k in range(x_flat.shape[1])]
    pred = model.predict(pd.DataFrame(x_flat, columns=cols))
    # Flattened width must be timesteps * output features so it reshapes to (n, ts, feat).
    assert pred.shape == (n_samples, n_timesteps * n_feat_out)
    pred.reshape(n_samples, n_timesteps, n_feat_out)  # must not raise


def test_lgbmr_single_output_feature_still_works():
    n_samples, n_timesteps, n_feat_in = 20, 4, 3
    rng = np.random.RandomState(1)
    train_x = rng.rand(n_samples, n_timesteps, n_feat_in)
    train_y = rng.rand(n_samples, n_timesteps, 1)
    model = train_lgbmr(train_x, train_y, _mdl())

    import pandas as pd
    x_flat = train_x.reshape(n_samples, -1)
    cols = [f"x_{k}" for k in range(x_flat.shape[1])]
    pred = model.predict(pd.DataFrame(x_flat, columns=cols))
    assert pred.shape == (n_samples, n_timesteps * 1)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `docker compose run --rm -T backend pytest tests/domains/training/ml/test_lgbmr_multioutput.py -q -p no:cacheprovider`
Expected: FAIL — `test_lgbmr_two_output_features...` raises `ValueError: cannot reshape array of size 160 into shape (20,4)` inside `train_lgbmr`.

- [ ] **Step 3: Fix `train_lgbmr` (`trainer.py:427-433`)**

Replace:

```python
    n_samples, n_timesteps, n_features_in = train_x.shape

    # Flatten 3D input to 2D: (n_samples, n_timesteps * n_features_in)
    x_flat = train_x.reshape(n_samples, -1)

    # Flatten 3D output to 2D: (n_samples, n_timesteps)
    y_flat = train_y.reshape(n_samples, n_timesteps)
```

with:

```python
    n_samples, n_timesteps, n_features_in = train_x.shape
    n_features_out = train_y.shape[2]

    # Flatten 3D input to 2D: (n_samples, n_timesteps * n_features_in)
    x_flat = train_x.reshape(n_samples, -1)

    # Flatten 3D output to 2D: (n_samples, n_timesteps * n_features_out).
    # Reshaping to (n_samples, n_timesteps) only works for a single output feature;
    # MultiOutputRegressor handles the full flattened width and predict() returns it
    # in the same C-order, so exact.py reshapes back to (n, n_timesteps, n_features_out).
    y_flat = train_y.reshape(n_samples, n_timesteps * n_features_out)
```

- [ ] **Step 4: Fix the LGBMR test-prediction reshape (`exact.py:581-582`)**

Replace:

```python
                test_predictions = mdl.predict(x_flat_df)
                test_predictions = test_predictions.reshape(n_samples, n_timesteps, 1)
```

with:

```python
                test_predictions = mdl.predict(x_flat_df)
                n_features_out = tst_y.shape[2]
                test_predictions = test_predictions.reshape(n_samples, n_timesteps, n_features_out)
```

- [ ] **Step 5: Fix the LGBMR val-prediction reshape (`exact.py:639-640`)**

Replace:

```python
                val_predictions = mdl.predict(x_flat_df)
                val_predictions = val_predictions.reshape(n_samples, n_timesteps, 1)
```

with:

```python
                val_predictions = mdl.predict(x_flat_df)
                n_features_out = val_y.shape[2]
                val_predictions = val_predictions.reshape(n_samples, n_timesteps, n_features_out)
```

- [ ] **Step 6: Run test to verify it passes**

Run: `docker compose run --rm -T backend pytest tests/domains/training/ml/test_lgbmr_multioutput.py -q -p no:cacheprovider`
Expected: PASS (2 passed).

- [ ] **Step 7: Commit**

```bash
git add tests/domains/training/ml/test_lgbmr_multioutput.py domains/training/ml/trainer.py domains/training/ml/exact.py
git commit -m "fix(training): LGBMR supports multiple output features

train_lgbmr flattened train_y to (n_samples, n_timesteps) and predict reshaped to
(n, ts, 1), hard-coding a single output feature; >=2 output features raised a numpy
reshape ValueError. Flatten/reshape by the real n_features_out instead.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 3 (S2): "Original-scale" metrics are computed from original data, not duplicated

**Why:** `exact.py:588-591` assigns `original_metrics = test_metrics` in **both** branches, so `evaluation_metrics['test_metrics_original']` always equals the scaled-space metrics. The original-scale ground truth (`tst_y_orig`, `:459`) and the inverse-transform of predictions (`:665-688`) already exist — they're just not wired into the headline metrics. Result: the UI shows "original" MAE/RMSE in normalized space (e.g. 0.03 instead of ~120 kW). Extract the inverse-scale loop into a helper and compute genuine original-scale metrics; reuse the helper in the 12-level block (DRY).

**Files:**
- Test: `tests/domains/training/ml/test_evaluation_metrics.py` (append)
- Modify: `domains/training/ml/exact.py` — add `inverse_scale_predictions` helper; rewrite `:586-597`; reuse helper at `:665-688`.

- [ ] **Step 1: Write the failing test (append to `test_evaluation_metrics.py`)**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `docker compose run --rm -T backend pytest "tests/domains/training/ml/test_evaluation_metrics.py::test_inverse_scale_predictions_roundtrips_via_scaler" -q -p no:cacheprovider`
Expected: FAIL — `ImportError: cannot import name 'inverse_scale_predictions'`.

- [ ] **Step 3: Add the `inverse_scale_predictions` helper to `exact.py`**

Insert immediately after `_calculate_single_timestep_metrics` (after its `return {...}` block, before `def train_non_keras_model`):

```python
def inverse_scale_predictions(predictions, o_scalers):
    """Inverse-transform scaled predictions back to original units.

    predictions: (n_samples, n_timesteps, n_features_out) or (n_samples, n_timesteps).
    o_scalers: dict {feature_index: fitted scaler}. Returns a new array (input is not
    mutated); features without a scaler are left as-is.
    """
    if o_scalers is None or len(o_scalers) == 0:
        return predictions

    restored = np.copy(predictions)
    n_samples = predictions.shape[0]
    n_features_out = predictions.shape[-1] if predictions.ndim > 2 else 1

    for i in range(n_samples):
        for i1 in range(n_features_out):
            if i1 in o_scalers and o_scalers[i1] is not None:
                if predictions.ndim == 3:
                    restored[i, :, i1] = o_scalers[i1].inverse_transform(
                        predictions[i, :, i1].reshape(-1, 1)
                    ).ravel()
                elif predictions.ndim == 2:
                    restored[i, :] = o_scalers[0].inverse_transform(
                        predictions[i, :].reshape(-1, 1)
                    ).ravel()
    return restored
```

- [ ] **Step 4: Run helper test to verify it passes**

Run: `docker compose run --rm -T backend pytest "tests/domains/training/ml/test_evaluation_metrics.py::test_inverse_scale_predictions_roundtrips_via_scaler" -q -p no:cacheprovider`
Expected: PASS.

- [ ] **Step 5: Wire genuine original-scale metrics (`exact.py:586-597`)**

Replace:

```python
            test_metrics = calculate_evaluation_metrics(tst_y, test_predictions)
            
            if tst_y_orig is not None:
                original_metrics = test_metrics
            else:
                original_metrics = test_metrics
            
            evaluation_metrics = {
                'test_metrics_scaled': test_metrics,
                'test_metrics_original': original_metrics,
                'model_type': mdl_config.MODE
            }
```

with:

```python
            test_metrics = calculate_evaluation_metrics(tst_y, test_predictions)

            # Original-scale metrics: inverse-transform predictions and compare against
            # the unscaled ground truth so reported MAE/RMSE are in real units (e.g. kW),
            # not normalized space. Fall back to scaled metrics if originals are missing.
            if tst_y_orig is not None and o_scalers:
                try:
                    test_predictions_orig = inverse_scale_predictions(test_predictions, o_scalers)
                    original_metrics = calculate_evaluation_metrics(tst_y_orig, test_predictions_orig)
                except Exception as orig_err:
                    logger.warning(f"Original-scale metrics failed, using scaled: {orig_err}")
                    original_metrics = test_metrics
            else:
                original_metrics = test_metrics

            evaluation_metrics = {
                'test_metrics_scaled': test_metrics,
                'test_metrics_original': original_metrics,
                'model_type': mdl_config.MODE
            }
```

- [ ] **Step 6: Reuse the helper in the 12-level block (`exact.py:665-688`) — DRY**

Replace:

```python
                if o_scalers is not None and len(o_scalers) > 0:
                    test_predictions_orig = np.copy(test_predictions)
                    n_tst_samples = test_predictions.shape[0]
                    n_ft_o = test_predictions.shape[-1] if len(test_predictions.shape) > 2 else 1

                    logger.info(f"Inverse scaling predictions: {n_tst_samples} samples, {n_ft_o} features")

                    for i in range(n_tst_samples):
                        for i1 in range(n_ft_o):
                            if i1 in o_scalers and o_scalers[i1] is not None:
                                if len(test_predictions.shape) == 3:
                                    test_predictions_orig[i, :, i1] = o_scalers[i1].inverse_transform(
                                        test_predictions[i, :, i1].reshape(-1, 1)
                                    ).ravel()
                                elif len(test_predictions.shape) == 2:
                                    test_predictions_orig[i, :] = o_scalers[0].inverse_transform(
                                        test_predictions[i, :].reshape(-1, 1)
                                    ).ravel()

                    eval_fcst = test_predictions_orig
                    logger.info(f"Predictions inverse scaled. Original range: [{np.min(test_predictions):.4f}, {np.max(test_predictions):.4f}] -> [{np.min(eval_fcst):.2f}, {np.max(eval_fcst):.2f}]")
                else:
                    eval_fcst = test_predictions
                    logger.warning("No output scalers available - using scaled predictions for evaluation")
```

with:

```python
                if o_scalers is not None and len(o_scalers) > 0:
                    eval_fcst = inverse_scale_predictions(test_predictions, o_scalers)
                    logger.info(f"Predictions inverse scaled. Range: [{np.min(test_predictions):.4f}, {np.max(test_predictions):.4f}] -> [{np.min(eval_fcst):.2f}, {np.max(eval_fcst):.2f}]")
                else:
                    eval_fcst = test_predictions
                    logger.warning("No output scalers available - using scaled predictions for evaluation")
```

- [ ] **Step 7: Add an end-to-end original-vs-scaled metrics test (append to `test_evaluation_metrics.py`)**

```python
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
    orig_metrics = calculate_evaluation_metrics(scaled_true.copy(), pred_orig) \
        if False else calculate_evaluation_metrics(real_true, pred_orig)

    # Scaled MAE ~0.05 (5/100), original MAE ~5.0 — distinctly different magnitudes.
    assert orig_metrics["MAE"] > scaled_metrics["MAE"] * 10
    assert abs(orig_metrics["MAE"] - 5.0) < 1e-6
```

- [ ] **Step 8: Run the full new test file to verify all pass**

Run: `docker compose run --rm -T backend pytest tests/domains/training/ml/test_evaluation_metrics.py -q -p no:cacheprovider`
Expected: PASS (5 passed).

- [ ] **Step 9: Commit**

```bash
git add tests/domains/training/ml/test_evaluation_metrics.py domains/training/ml/exact.py
git commit -m "fix(training): compute real original-scale metrics instead of duplicating scaled

test_metrics_original was assigned test_metrics in both branches, so the reported
'original-scale' MAE/RMSE were actually normalized-space numbers. Extract the
inverse-transform loop into inverse_scale_predictions() and compute genuine
original-scale metrics from tst_y_orig; reuse the helper in the 12-level block.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: Regression — full suite stays green

**Files:** none (verification only).

- [ ] **Step 1: Run the whole training-domain + shared suite in Docker**

Run: `docker compose run --rm -T backend pytest tests/ -q -p no:cacheprovider`
Expected: all previously-passing tests still pass (772 baseline) + the 5 new tests; 0 failures. Note any pre-existing skips unchanged.

- [ ] **Step 2: If green, the branch is ready for PR**

```bash
git log --oneline -4
git push -u origin fix/w11-eval-metrics-correctness
```

---

## Self-Review

- **Spec coverage:** S1 → Task 1; S3 → Task 2; S2 → Task 3; regression → Task 4. S4 (dead `else: test_predictions = tst_y` at `:584`) is intentionally **not** fixed — it's unreachable (dispatch raises at `:529`) and cosmetic; left as-is to keep this PR focused on the three real bugs.
- **Type consistency:** `inverse_scale_predictions(predictions, o_scalers)` defined in Task 3 Step 3, used identically in Step 5/6 and in the helper test. `n_features_out` derived from `*.shape[2]` consistently in trainer + exact. Metrics dict keys (`MAE`, `MASE`, `mase`, `test_metrics_original`) match existing code.
- **Placeholder scan:** every code step shows full replacement text; no TBD/TODO.
- **Ordering note:** Task 2 (S3) precedes Task 3 (S2) so that by the time S2 reasons about `test_predictions`, LGBMR already yields correctly-shaped `(n, ts, feat)` arrays; the `inverse_scale_predictions` helper is shape-agnostic regardless.
