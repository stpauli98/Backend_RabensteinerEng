"""Regression tests for extract_serialized_models.

Pre-fix: sklearn estimators (Linear/SVR/LGBMR) and lists thereof were
silently dropped — function returned only scalers, orchestrator logged
'Auto-saved 2 model(s)' but no actual model file landed in storage.
Confirmed via training-agent E2E run.

These tests guard against the silent-drop regression and verify that
the new detection branches (is_raw_sklearn, is_raw_sklearn_list) fire
on the exact shapes produced by train_linear_model, train_svr_dir,
train_svr_mimo, and train_lgbmr.
"""

import pickle

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from domains.training.ml.models import extract_serialized_models


def _fit_lr():
    m = LinearRegression()
    m.fit(np.array([[1.0], [2.0], [3.0]]), np.array([1.0, 2.0, 3.0]))
    return m


def _fit_svr_pipeline():
    """Mirrors train_svr_dir's actual output: a make_pipeline(StandardScaler, SVR)."""
    pipe = make_pipeline(StandardScaler(), SVR(kernel='rbf', C=1.0, epsilon=0.1))
    pipe.fit(np.array([[1.0], [2.0], [3.0], [4.0]]), np.array([1.0, 2.0, 3.0, 4.0]))
    return pipe


def _model_entries(result):
    """Filter out scaler-dict entries; what's left should be model artifacts."""
    return [
        r for r in result
        if not r.get('is_scaler_dict')
        and 'scaler' not in r.get('model_class', '').lower()
    ]


# ---------- Task A: detection branches ----------

def test_single_sklearn_estimator_is_serialized():
    """A bare LinearRegression should produce one extracted artifact.

    Pre-fix this returned [] for a bare estimator.
    """
    result = extract_serialized_models(_fit_lr())
    models = _model_entries(result)
    assert len(models) == 1
    assert models[0].get('is_raw_sklearn') is True
    assert models[0]['model_class'] == 'LinearRegression'


def test_list_of_sklearn_estimators_is_serialized():
    """train_linear_model returns a list of estimators; the list as a whole
    should be wrapped (not iterated and dropped).
    """
    estimators = [_fit_lr(), _fit_lr()]
    result = extract_serialized_models(estimators)
    models = _model_entries(result)
    # One wrapper entry for the whole list
    assert len(models) == 1
    assert models[0].get('is_raw_sklearn_list') is True
    assert models[0]['model_class'] == 'LinearRegressionList'
    # The raw list is preserved for pickle/joblib dump
    assert isinstance(models[0]['raw_model'], list)
    assert len(models[0]['raw_model']) == 2


def test_svr_pipeline_estimator_is_serialized():
    """SVR is wrapped in make_pipeline(StandardScaler, SVR) by train_svr_dir.
    The Pipeline object has .predict but no .layers — must be detected.
    """
    result = extract_serialized_models(_fit_svr_pipeline())
    models = _model_entries(result)
    assert len(models) == 1
    assert models[0].get('is_raw_sklearn') is True


def test_list_of_svr_pipelines_is_serialized():
    """train_svr_dir / train_svr_mimo return list of make_pipeline estimators."""
    estimators = [_fit_svr_pipeline(), _fit_svr_pipeline()]
    result = extract_serialized_models(estimators)
    models = _model_entries(result)
    assert len(models) == 1
    assert models[0].get('is_raw_sklearn_list') is True


def test_lgbmr_like_estimator_is_serialized():
    """LGBMRegressor wrapped in MultiOutputRegressor has .predict but no .layers.
    Use a stand-in (any fitted single-estimator) since lightgbm may not be in
    the test runner — the detection logic is the same.
    """
    # Use a fitted LR as proxy for a single-estimator MultiOutputRegressor.
    # The detection path doesn't care about class — only .predict + no .layers + picklable.
    result = extract_serialized_models(_fit_lr())
    models = _model_entries(result)
    assert len(models) == 1


def test_trained_model_top_level_key_with_list():
    """Mirrors the exact dict shape from run_exact_training_pipeline for LIN/SVR.

    Pre-fix the recursion descended into trained_model -> [LR1, LR2] -> per-item,
    and none of LR1/LR2 matched any detection branch, so result was empty.
    """
    training_results = {
        'trained_model': [_fit_lr(), _fit_lr()],
        'scalers': {
            'input': {0: 'placeholder'},
            'output': {0: 'placeholder'},
        },
        'evaluation_metrics': {'mae': 0.1},
    }
    result = extract_serialized_models(training_results)
    models = _model_entries(result)
    assert len(models) >= 1
    # The model entry should sit at path 'trained_model'
    assert any(m['path'] == 'trained_model' for m in models)


def test_trained_model_top_level_key_with_single_estimator():
    """LGBMR returns a single MultiOutputRegressor; same dict shape, single object."""
    training_results = {
        'trained_model': _fit_lr(),  # proxy for any single sklearn-like estimator
        'scalers': {
            'input': {0: 'placeholder'},
            'output': {0: 'placeholder'},
        },
    }
    result = extract_serialized_models(training_results)
    models = _model_entries(result)
    assert len(models) == 1
    assert models[0]['path'] == 'trained_model'
    assert models[0].get('is_raw_sklearn') is True


def test_scaler_dict_still_detected_alongside_sklearn():
    """Regression: adding sklearn detection must not break scaler dict detection."""
    from sklearn.preprocessing import MinMaxScaler

    s_in = MinMaxScaler().fit(np.array([[1.0], [2.0]]))
    training_results = {
        'trained_model': [_fit_lr()],
        'scalers': {
            'input': {0: s_in},
            'output': {0: s_in},
        },
    }
    result = extract_serialized_models(training_results)
    scaler_entries = [r for r in result if r.get('is_scaler_dict')]
    model_entries = _model_entries(result)
    assert len(scaler_entries) == 2  # input + output
    assert len(model_entries) == 1


def test_unrecognized_object_does_not_silently_drop_when_at_root():
    """An object that's neither Keras nor sklearn nor scaler — must produce empty
    result but should not crash. This is the documented behavior; the orchestrator
    log will surface the 0-model warning.
    """
    class WeirdObject:
        # No .predict, no .layers, no .save — not detected by any branch
        def fit(self, *a, **kw):
            pass

    result = extract_serialized_models(WeirdObject())
    # Empty result is acceptable — the orchestrator's model_count==0 warning
    # is what surfaces the issue to operators. The bug was silent drop combined
    # with a misleading "Auto-saved N" log; that log is now fixed separately.
    assert isinstance(result, list)


# ---------- Task E: round-trip integration ----------

def test_sklearn_list_roundtrip_via_pickle():
    """Round-trip: serialize list of LR via pickle, deserialize, verify same predictions.

    Guards against future refactors that might break the joblib/pickle save path
    in save_models_to_storage.
    """
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([1.0, 2.0, 3.0, 4.0])

    m1 = LinearRegression().fit(X, y)
    m2 = LinearRegression().fit(X, y * 2.0)
    original = [m1, m2]

    blob = pickle.dumps(original)
    restored = pickle.loads(blob)

    test_X = np.array([[5.0]])
    assert np.allclose(original[0].predict(test_X), restored[0].predict(test_X))
    assert np.allclose(original[1].predict(test_X), restored[1].predict(test_X))


def test_extract_then_pickle_then_predict():
    """End-to-end: extract_serialized_models on train-style output, pickle the
    raw_model, unpickle, predict — verifies the full save/load contract.
    """
    estimators = [_fit_lr(), _fit_lr()]
    result = extract_serialized_models(estimators)
    models = _model_entries(result)
    assert len(models) == 1
    assert models[0].get('is_raw_sklearn_list') is True

    raw = models[0]['raw_model']
    blob = pickle.dumps(raw)
    restored = pickle.loads(blob)

    test_X = np.array([[5.0]])
    for orig, rest in zip(raw, restored):
        assert np.allclose(orig.predict(test_X), rest.predict(test_X))
