"""Regression tests for predict scaling math.

Verifies that preprocess_input scales each FEATURE (not timestep), and
postprocess_output inverse-scales each OUTPUT (not timestep). The pre-fix
code silently produced incorrect predictions for batch>1 and partially-
incorrect for batch=1.

Pre-fix bug summary:
- preprocess_input iterated `range(input_array.shape[1])` for 3-D input
  (N, T, F), treating timesteps as features. scaler[i] (fit on a 1-D feature
  column) was then applied to a (N, F) slice — wrong axis, broadcast error
  swallowed by a try/except.
- postprocess_output iterated `range(predictions.shape[1])` for 3-D output
  (N, T, O), KeyError'd past output_scalers[0] (only 1 output → only key 0),
  silently caught the exception and returned raw scaled values.
"""

import numpy as np
import pytest
from sklearn.preprocessing import MinMaxScaler

from domains.training.services.prediction_service import PredictionService


@pytest.fixture
def service():
    """Bypass __init__ — we only test the pure (input, scalers) → array methods."""
    svc = PredictionService.__new__(PredictionService)
    svc.session_id = "test"
    svc.user_id = "u1"
    svc._scalers = None
    svc._model = None
    svc._model_filename = None
    svc._training_results = None
    return svc


def _make_scalers_input(n, low=0, high=100):
    """Build {'input': {idx: MinMaxScaler}} fit on [low, high] per feature."""
    inner = {}
    for i in range(n):
        s = MinMaxScaler()
        s.fit(np.array([[low], [high]], dtype=float))
        inner[i] = s
    return {'input': inner, 'output': {}}


def _make_scalers_output(specs):
    """Build {'output': {idx: MinMaxScaler}} fit on the (low, high) tuples."""
    inner = {}
    for i, (low, high) in enumerate(specs):
        s = MinMaxScaler()
        s.fit(np.array([[low], [high]], dtype=float))
        inner[i] = s
    return {'input': {}, 'output': inner}


# ─── preprocess_input ─────────────────────────────────────────────────────────

def test_preprocess_2d_scales_each_feature(service):
    """(N, F) shape: each column min-max scaled within its own feature range."""
    arr = np.array([[0.0, 0.0, 0.0], [50.0, 50.0, 50.0], [100.0, 100.0, 100.0]])
    scalers = _make_scalers_input(3, 0, 100)
    result = service.preprocess_input(arr, scalers)
    np.testing.assert_allclose(
        result, np.array([[0, 0, 0], [0.5, 0.5, 0.5], [1, 1, 1]]), atol=1e-6
    )


def test_preprocess_3d_scales_each_feature_across_timesteps(service):
    """(N, T, F) shape: feature 0 across all timesteps scales uniformly by its scaler."""
    # 2 batches, 3 timesteps, 2 features. Feature 0 ranges 0-100, feature 1 ranges 0-200.
    arr = np.array([
        [[0, 0], [50, 100], [100, 200]],
        [[25, 50], [75, 150], [50, 100]],
    ], dtype=float)
    scalers = {
        'input': {
            0: MinMaxScaler().fit(np.array([[0], [100]], dtype=float)),
            1: MinMaxScaler().fit(np.array([[0], [200]], dtype=float)),
        },
        'output': {},
    }
    result = service.preprocess_input(arr, scalers)
    # Feature 0: divide by 100; feature 1: divide by 200
    expected = np.array([
        [[0, 0], [0.5, 0.5], [1, 1]],
        [[0.25, 0.25], [0.75, 0.75], [0.5, 0.5]],
    ])
    np.testing.assert_allclose(result, expected, atol=1e-6)


def test_preprocess_skips_missing_scaler(service):
    """If scalers dict missing feature_idx 1, that column is left unscaled."""
    arr = np.array([[0.0, 50.0], [100.0, 50.0]])
    scalers = {
        'input': {0: MinMaxScaler().fit(np.array([[0], [100]], dtype=float))},
        'output': {},
    }  # no key 1
    result = service.preprocess_input(arr, scalers)
    np.testing.assert_allclose(result, np.array([[0, 50], [1, 50]]), atol=1e-6)


def test_preprocess_1d_single_sample(service):
    """1-D input (n_features,) — single sample, non-sequence model."""
    arr = np.array([0.0, 50.0, 100.0])
    scalers = _make_scalers_input(3, 0, 100)
    result = service.preprocess_input(arr, scalers)
    np.testing.assert_allclose(result, np.array([0.0, 0.5, 1.0]), atol=1e-6)


def test_preprocess_raises_on_invalid_ndim(service):
    """4-D+ unsupported — must raise, not silently return wrong values."""
    arr = np.zeros((2, 3, 4, 5))
    scalers = _make_scalers_input(5)
    with pytest.raises(ValueError, match="ndim"):
        service.preprocess_input(arr, scalers)


# ─── postprocess_output ──────────────────────────────────────────────────────

def test_postprocess_2d_unscales_each_output(service):
    """(N, O) output: each column inverse-transformed via its own scaler."""
    pred = np.array([[0.0, 0.5], [1.0, 0.0]])
    scalers = _make_scalers_output([(0, 10), (0, 100)])
    result = service.postprocess_output(pred, scalers)
    # 2-D multi-output returns nested list
    np.testing.assert_allclose(
        np.array(result), np.array([[0, 50], [10, 0]]), atol=1e-6
    )


def test_postprocess_3d_unscales_across_timesteps(service):
    """(N, T, O) output: each timestep inverse-scaled per output feature.

    Pre-fix: this would KeyError at output_scalers[1] (only key 0 exists),
    fall into the try/except, and return raw scaled values.
    """
    pred = np.array([
        [[0.0], [0.5], [1.0]],
        [[0.25], [0.75], [0.5]],
    ])
    scalers = _make_scalers_output([(0, 1000)])  # scale to range 0-1000
    result = service.postprocess_output(pred, scalers)
    expected = np.array([
        [[0], [500], [1000]],
        [[250], [750], [500]],
    ])
    np.testing.assert_allclose(np.array(result), expected, atol=1e-6)


def test_postprocess_1d_reshapes_to_flat_list(service):
    """(N,) input gets reshaped to (N, 1), unscaled, returned as flat list."""
    pred = np.array([0.0, 0.5, 1.0])
    scalers = _make_scalers_output([(0, 100)])
    result = service.postprocess_output(pred, scalers)
    np.testing.assert_allclose(np.array(result).flatten(), np.array([0, 50, 100]), atol=1e-6)


def test_postprocess_raises_on_invalid_ndim(service):
    """4-D+ unsupported; must raise explicitly (not silently return wrong values)."""
    pred = np.zeros((2, 3, 4, 5))
    scalers = _make_scalers_output([(0, 1)])
    with pytest.raises(ValueError, match="ndim"):
        service.postprocess_output(pred, scalers)


# ─── regression: batch>1 was silently wrong before fix ──────────────────────

def test_batch_predictions_are_actually_scaled(service):
    """Pre-fix: batch=100, (N, T, F) silently returned unscaled values w/ success:true.

    This is the canonical regression: every feature value is 50, every
    scaler maps [0, 100] → [0, 1], so the post-fix result must be all 0.5.
    The buggy code returned all 50.0 (raw) because the broadcast error in the
    timestep loop got swallowed.
    """
    arr = np.ones((100, 49, 7)) * 50.0  # batch=100, 49 timesteps, 7 features, all values = 50
    scalers = _make_scalers_input(7, 0, 100)  # each scaler fits [0,100]
    result = service.preprocess_input(arr, scalers)
    # Every value should now be 0.5 (50/100)
    assert np.all(np.isclose(result, 0.5)), (
        f"Expected all values to be 0.5 after scaling, got min={result.min()}, max={result.max()}"
    )
    # And definitely NOT 50 (which is what the bug returned)
    assert not np.any(np.isclose(result, 50.0)), (
        "Found unscaled values (50.0) — scaling was not applied correctly"
    )


def test_postprocess_batch_3d_all_timesteps_unscaled(service):
    """Pre-fix: only timestep[0] was inverse-scaled (key 0 worked); rest stayed raw.

    Confirms every timestep gets the same inverse-transform applied.
    """
    # (N=10, T=49, O=1) — all values 0.5 in scaled space; expect 5000 after unscale
    pred = np.ones((10, 49, 1)) * 0.5
    scalers = _make_scalers_output([(0, 10000)])
    result = np.array(service.postprocess_output(pred, scalers))
    assert result.shape == (10, 49, 1)
    assert np.all(np.isclose(result, 5000.0)), (
        f"Expected all unscaled values to be 5000.0, got min={result.min()}, max={result.max()}"
    )


def test_preprocess_does_not_swallow_scaler_errors(service):
    """If a scaler raises (e.g. wrong feature shape), the error must propagate.

    The pre-fix code wrapped scaler.transform() in try/except logger.warning,
    masking real bugs. This regression confirms we no longer swallow.
    """
    # Scaler fit on a single feature; we pass a 2-D array but plug in a
    # broken scaler that raises on transform.
    class _BrokenScaler:
        def transform(self, x):
            raise RuntimeError("intentional scaler failure")

    arr = np.array([[1.0, 2.0]])
    scalers = {'input': {0: _BrokenScaler()}, 'output': {}}
    with pytest.raises(RuntimeError, match="intentional scaler failure"):
        service.preprocess_input(arr, scalers)
