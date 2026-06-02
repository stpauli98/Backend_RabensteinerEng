"""Adversarial-hardening tests for train-models model-config validation.

Covers BREAK 3/4 from the 2026-06-02 adversarial session: unbounded params
(e.g. N=999999999) must be rejected BEFORE a training thread is spawned, so they
can't reach Keras and OOM-kill the worker.
"""
from domains.training.api.training_routes import validate_model_config


def test_rejects_oom_neuron_count():
    # N=999999999 made Keras attempt a ~192GB allocation -> worker SIGKILL.
    assert validate_model_config({'MODE': 'Dense', 'N': 999999999}) is not None


def test_rejects_unknown_model_type():
    assert validate_model_config({'MODE': 'TOTALLY_INVALID_MODEL'}) is not None


def test_rejects_negative_layers():
    assert validate_model_config({'MODE': 'Dense', 'LAY': -5}) is not None


def test_rejects_out_of_range_epochs():
    assert validate_model_config({'MODE': 'Dense', 'EP': 10_000_000}) is not None


def test_coerces_string_numbers_for_range_check():
    # The UI may send numeric params as strings.
    assert validate_model_config({'MODE': 'Dense', 'N': '512'}) is None
    assert validate_model_config({'MODE': 'Dense', 'N': '999999999'}) is not None


def test_accepts_sane_dense_config():
    assert validate_model_config(
        {'MODE': 'Dense', 'LAY': 3, 'N': 512, 'EP': 100, 'ACTF': 'relu'}
    ) is None


def test_accepts_none_optional_params_for_non_keras():
    # SVR/LIN leave LAY/N/EP unset.
    assert validate_model_config(
        {'MODE': 'LIN', 'LAY': None, 'N': None, 'EP': None}
    ) is None
