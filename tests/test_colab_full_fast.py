"""Regression tests for the cumulative-moment full-history predictor."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


COLAB = Path(__file__).resolve().parents[1] / "colab"
if str(COLAB) not in sys.path:
    sys.path.insert(0, str(COLAB))

from oyosuri_all_in_one import (  # noqa: E402
    _predict_sequence_with_params_legacy,
    predict_sequence,
    predict_sequence_with_params,
)


def _walk(steps) -> list[int]:
    return [0, *np.cumsum(steps).astype(int).tolist()]


def _assert_results_exact(old, new) -> None:
    for old_values, new_values in zip(old, new):
        np.testing.assert_array_equal(
            np.asarray(old_values, dtype=float),
            np.asarray(new_values, dtype=float),
        )


@pytest.mark.parametrize(
    "steps",
    [
        [1] * 12,
        [-1] * 12,
        [1, -1] * 6,
        [1, 1, -1, -1] * 3,
    ],
)
def test_full_fast_matches_legacy_on_structured_paths(steps):
    x = _walk(np.asarray(steps))
    old = _predict_sequence_with_params_legacy(x, grid_size=11)
    new = predict_sequence_with_params(x, grid_size=11)
    _assert_results_exact(old, new)


@pytest.mark.parametrize("seed", [0, 1, 812])
def test_full_fast_matches_legacy_on_random_paths(seed):
    rng = np.random.default_rng(seed)
    x = _walk(rng.choice([-1, 1], size=35))
    old = _predict_sequence_with_params_legacy(x, grid_size=5)
    new = predict_sequence_with_params(x, grid_size=5)
    _assert_results_exact(old, new)


def test_full_fast_symmetric_mode_matches_legacy():
    x = _walk(np.asarray([1, 1, -1, 1, -1, -1, 1] * 5))
    old = _predict_sequence_with_params_legacy(
        x, grid_size=5, symmetric=True
    )
    new = predict_sequence_with_params(
        x, grid_size=5, symmetric=True
    )
    _assert_results_exact(old, new)


def test_predict_sequence_uses_same_fast_predictions():
    rng = np.random.default_rng(23)
    x = _walk(rng.choice([-1, 1], size=25))
    with_params = predict_sequence_with_params(x, grid_size=5)[0]
    predictions = predict_sequence(x, grid_size=5)
    np.testing.assert_array_equal(predictions, with_params)


def test_empty_sequence_keeps_legacy_result():
    assert predict_sequence_with_params([], grid_size=5) == ([], [], [], [])
    assert predict_sequence([], grid_size=5) == []
