"""S3: LGBMR must handle >1 output feature.

train_lgbmr flattened train_y to (n_samples, n_timesteps), which only works for a
single output feature; >=2 output features raised a numpy reshape ValueError.
"""
import numpy as np
import pandas as pd
from types import SimpleNamespace

from domains.training.ml.trainer import train_lgbmr


def _mdl():
    return SimpleNamespace(N_ESTIMATORS=10, LEARNING_RATE=0.1, MAX_DEPTH=3)


def _predict_width(model, train_x):
    n_samples = train_x.shape[0]
    x_flat = train_x.reshape(n_samples, -1)
    cols = [f"x_{k}" for k in range(x_flat.shape[1])]
    return model.predict(pd.DataFrame(x_flat, columns=cols))


def test_lgbmr_two_output_features_trains_and_predicts_correct_width():
    n_samples, n_timesteps, n_feat_in, n_feat_out = 20, 4, 3, 2
    rng = np.random.RandomState(0)
    train_x = rng.rand(n_samples, n_timesteps, n_feat_in)
    train_y = rng.rand(n_samples, n_timesteps, n_feat_out)

    model = train_lgbmr(train_x, train_y, _mdl())

    pred = _predict_width(model, train_x)
    # Flattened width must be timesteps * output features so it reshapes to (n, ts, feat).
    assert pred.shape == (n_samples, n_timesteps * n_feat_out)
    pred.reshape(n_samples, n_timesteps, n_feat_out)  # must not raise


def test_lgbmr_single_output_feature_still_works():
    n_samples, n_timesteps, n_feat_in = 20, 4, 3
    rng = np.random.RandomState(1)
    train_x = rng.rand(n_samples, n_timesteps, n_feat_in)
    train_y = rng.rand(n_samples, n_timesteps, 1)

    model = train_lgbmr(train_x, train_y, _mdl())

    pred = _predict_width(model, train_x)
    assert pred.shape == (n_samples, n_timesteps * 1)
