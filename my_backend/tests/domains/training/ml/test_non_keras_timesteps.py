"""Guard: non-Keras models (SVR/LIN/LGBMR) require input_timesteps == output_timesteps.

These models flatten (samples x timesteps) into the regression row axis, so unequal
input/output steps produced a cryptic sklearn error:
  "Found input variables with inconsistent numbers of samples: [1276224, 638112]"
(1276224 = n*96 input steps, 638112 = n*48 output steps). The guard converts that
into a clear, actionable message before training. Keras models handle seq2seq fine.
"""
from domains.training.ml.exact import non_keras_timestep_error


def test_lin_rejects_unequal_timesteps():
    err = non_keras_timestep_error("LIN", 96, 48)
    assert err is not None and "96" in err and "48" in err


def test_svr_dir_rejects_unequal_timesteps():
    assert non_keras_timestep_error("SVR_dir", 96, 48) is not None


def test_svr_mimo_rejects_unequal_timesteps():
    assert non_keras_timestep_error("SVR_MIMO", 96, 48) is not None


def test_lgbmr_rejects_unequal_timesteps():
    assert non_keras_timestep_error("LGBMR", 10, 5) is not None


def test_non_keras_accepts_equal_timesteps():
    assert non_keras_timestep_error("LIN", 48, 48) is None
    assert non_keras_timestep_error("SVR_dir", 96, 96) is None


def test_keras_models_allow_unequal_timesteps():
    assert non_keras_timestep_error("Dense", 96, 48) is None
    assert non_keras_timestep_error("LSTM", 96, 48) is None
    assert non_keras_timestep_error("CNN", 96, 48) is None
