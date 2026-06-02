from unittest.mock import MagicMock, patch
import numpy as np
from domains.training.ml.exact import train_non_keras_model


@patch("domains.training.ml.exact.train_linear_model", return_value="LINMODEL")
def test_non_keras_emits_coarse_progress_and_returns_model(mock_lin):
    tracker = MagicMock()
    cfg = MagicMock()
    cfg.MODE = "LIN"
    trn_x = np.zeros((4, 2, 1))
    trn_y = np.zeros((4, 1))

    model = train_non_keras_model(cfg, trn_x, trn_y, progress_tracker=tracker)

    assert model == "LINMODEL"
    mock_lin.assert_called_once()
    # The bar must move before the synchronous fit (was frozen at 0% then jumped to 50%)
    assert tracker.emit.called
    statuses = [c.args[2] for c in tracker.emit.call_args_list]  # emit(progress, message, status)
    assert "processing" in statuses


def test_non_keras_is_safe_without_tracker():
    cfg = MagicMock()
    cfg.MODE = "LIN"
    with patch("domains.training.ml.exact.train_linear_model", return_value="M"):
        assert train_non_keras_model(cfg, np.zeros((4, 2, 1)), np.zeros((4, 1)), progress_tracker=None) == "M"
