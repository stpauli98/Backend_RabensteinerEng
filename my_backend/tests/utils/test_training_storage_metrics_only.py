"""Tests for metrics-only training-results download (#112).

The evaluation-tables endpoint must recover df_eval even when the trained Keras
model that sits in the same pickle cannot be deserialized (version skew across
deploys). download_training_results_metrics_only skips Keras reconstruction
entirely, so the metrics survive regardless of the model's Keras version.
"""
import io
import gzip
import pickle
from unittest.mock import Mock, patch

import pytest

from utils.training_storage import (
    download_training_results_metrics_only,
    _SkipModelsUnpickler,
    _SkippedKerasObject,
    _metrics_cache,
    _results_cache,
)


def _make_results_with_model():
    """Build a results dict shaped like a real training_results pickle: model + metrics."""
    from tensorflow import keras
    model = keras.Sequential([
        keras.layers.Input((4,)),
        keras.layers.Dense(512, activation="relu"),
        keras.layers.Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    return {
        "model": model,
        "model_type": "Dense",
        "evaluation_metrics": {
            "df_eval": {"load": {"delta [min]": [15.0, 30.0], "MAE": [1.1, 2.2], "RMSE": [3.3, 4.4]}},
            "df_eval_ts": {"load": {"15.0": {"MAE": 1.1}}},
            "model_type": "Dense",
        },
    }


def _gzip_pickle(obj):
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", compresslevel=6) as gz:
        pickle.dump(obj, gz, protocol=pickle.HIGHEST_PROTOCOL)
    return buf.getvalue()


@pytest.fixture(autouse=True)
def _clear_caches():
    _metrics_cache.clear()
    _results_cache.clear()
    yield
    _metrics_cache.clear()
    _results_cache.clear()


def test_skip_unpickler_recovers_metrics_and_drops_model():
    """The skip unpickler returns metrics intact and replaces the model with a placeholder."""
    blob = _gzip_pickle(_make_results_with_model())

    loaded = _SkipModelsUnpickler(gzip.GzipFile(fileobj=io.BytesIO(blob))).load()

    # Model reconstruction was skipped: it is NOT a real Keras model.
    from tensorflow import keras
    assert not isinstance(loaded["model"], keras.Model)
    # Metrics in the same pickle survive untouched.
    assert loaded["evaluation_metrics"]["df_eval"]["load"]["MAE"] == [1.1, 2.2]
    assert loaded["evaluation_metrics"]["df_eval"]["load"]["RMSE"] == [3.3, 4.4]


def test_skip_unpickler_keeps_non_keras_objects():
    """numpy arrays and plain dicts (non-keras) must deserialize normally."""
    import numpy as np
    blob = _gzip_pickle({"arr": np.array([1.0, 2.0, 3.0]), "meta": {"k": "v"}})

    loaded = _SkipModelsUnpickler(gzip.GzipFile(fileobj=io.BytesIO(blob))).load()

    assert list(loaded["arr"]) == [1.0, 2.0, 3.0]
    assert loaded["meta"]["k"] == "v"


@patch("utils.training_storage.get_supabase_admin_client")
def test_download_metrics_only_returns_df_eval(mock_admin):
    """End-to-end: a model-bearing blob in storage yields recoverable metrics."""
    blob = _gzip_pickle(_make_results_with_model())
    mock_storage = Mock()
    mock_storage.download.return_value = blob
    mock_admin.return_value.storage.from_.return_value = mock_storage

    results = download_training_results_metrics_only("sess-abc/training_results_x.pkl.gz")

    assert results["evaluation_metrics"]["df_eval"]["load"]["MAE"] == [1.1, 2.2]
    mock_storage.download.assert_called_once_with("sess-abc/training_results_x.pkl.gz")


@patch("utils.training_storage.get_supabase_admin_client")
def test_download_metrics_only_caches(mock_admin):
    """Second call for the same file hits the metrics cache (no second download)."""
    blob = _gzip_pickle(_make_results_with_model())
    mock_storage = Mock()
    mock_storage.download.return_value = blob
    mock_admin.return_value.storage.from_.return_value = mock_storage

    path = "sess-abc/training_results_x.pkl.gz"
    download_training_results_metrics_only(path)
    download_training_results_metrics_only(path)

    mock_storage.download.assert_called_once()


def test_skipped_keras_object_unpickle_model_returns_none():
    """The placeholder's _unpickle_model is the no-op that replaces Keras' loader."""
    assert _SkippedKerasObject._unpickle_model(b"anything") is None
