"""Regression tests for the fast W=10/30/60 rolling engine."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


COLAB = Path(__file__).resolve().parents[1] / "colab"
if str(COLAB) not in sys.path:
    sys.path.insert(0, str(COLAB))

from oyosuri_all_in_one import minimize_Vn  # noqa: E402
from oyosuri_moments import get_moment_grid, minimize_Vn_from_moments  # noqa: E402
from oyosuri_rolling import (  # noqa: E402
    _predict_sequence_with_params_rolling_legacy,
    predict_sequence_with_params_rolling,
    predict_sequences_with_params_rolling,
)


def _walk(steps: list[int]) -> list[int]:
    return [0, *np.cumsum(steps).astype(int).tolist()]


def _assert_result_arrays_equal(old, new) -> None:
    for old_values, new_values in zip(old, new):
        np.testing.assert_array_equal(
            np.asarray(old_values, dtype=float),
            np.asarray(new_values, dtype=float),
        )


@pytest.mark.parametrize("window", [10, 30, 60])
@pytest.mark.parametrize(
    "kind",
    ["up", "down", "alternating", "random"],
)
def test_fast_minimizer_matches_legacy_for_target_windows(window, kind):
    if kind == "up":
        steps = [1] * (window - 1)
    elif kind == "down":
        steps = [-1] * (window - 1)
    elif kind == "alternating":
        steps = [1 if i % 2 == 0 else -1 for i in range(window - 1)]
    else:
        rng = np.random.default_rng(20260723 + window)
        steps = rng.choice([-1, 1], size=window - 1).astype(int).tolist()
    x = _walk(steps)

    old = minimize_Vn(x, grid_size=11)
    moments = get_moment_grid(window, 11, False)
    new = minimize_Vn_from_moments(x, moments)

    assert new.is_constant is old.is_constant
    assert new.V_min == old.V_min
    assert new.argmins == old.argmins


@pytest.mark.parametrize("window", [10, 30, 60])
def test_fast_rolling_outputs_are_exactly_legacy_outputs(window):
    rng = np.random.default_rng(812)
    x = _walk(rng.choice([-1, 1], size=70).astype(int).tolist())
    old = _predict_sequence_with_params_rolling_legacy(
        x, window=window, grid_size=3
    )
    new = predict_sequence_with_params_rolling(
        x, window=window, grid_size=3
    )
    _assert_result_arrays_equal(old, new)


def test_multi_window_engine_matches_individual_calls():
    rng = np.random.default_rng(156)
    x = _walk(rng.choice([-1, 1], size=75).astype(int).tolist())
    combined = predict_sequences_with_params_rolling(
        x, windows=[10, 30, 60], grid_size=5
    )
    for window in (10, 30, 60):
        individual = predict_sequence_with_params_rolling(
            x, window=window, grid_size=5
        )
        _assert_result_arrays_equal(individual, combined[window])


def test_symmetric_mode_is_legacy_compatible():
    x = _walk([1, 1, -1, 1, -1, -1, 1] * 5)
    old = _predict_sequence_with_params_rolling_legacy(
        x, window=30, grid_size=5, symmetric=True
    )
    new = predict_sequence_with_params_rolling(
        x, window=30, grid_size=5, symmetric=True
    )
    _assert_result_arrays_equal(old, new)
