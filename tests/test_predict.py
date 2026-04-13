"""Tests for 作成書 §8 items 5, 6, 7."""

from __future__ import annotations

import pytest

from oyosuri import predict_next, predict_sequence


def test_predict_n0_is_x0():
    """§6.1: n = 0 gives V_0 ≡ 0, so x_1* = x_0."""
    assert predict_next([0]) == pytest.approx(0.0, abs=1e-12)
    assert predict_next([5]) == pytest.approx(5.0, abs=1e-12)


def test_constant_Vn_returns_xn():
    """§8 item 6: when V_n is constant, return x_n."""
    # x_1 = 0 ⇒ V_1 ≡ 1 (constant).
    assert predict_next([0, 0]) == pytest.approx(0.0, abs=1e-12)


def test_left_right_reflection_symmetry():
    """§8 item 7: flipping the sign of x should flip the sign of the prediction.

    Transforming (p, q, α) → (q, p, 1-α) maps mu_t(x) → mu_t(-x), which
    implies the prediction flips sign.
    """
    x = [0, 1, 2, 1, 2, 3]
    x_neg = [-v for v in x]
    # Use a coarse grid for speed — symmetry holds for any grid that is
    # itself symmetric under s ↦ 1 − s (np.linspace(0, 1, k) satisfies this).
    preds = predict_sequence(x, grid_size=11)
    preds_neg = predict_sequence(x_neg, grid_size=11)
    assert len(preds) == len(preds_neg) == len(x) - 1
    for p, q in zip(preds, preds_neg):
        assert p == pytest.approx(-q, abs=1e-9)


def test_predict_sequence_length():
    """predict_sequence({x_0,…,x_N}) returns N predictions [x_1*,…,x_N*]."""
    x = [0, 1, 2, 3]
    preds = predict_sequence(x, grid_size=11)
    assert len(preds) == 3


def test_generalized_D2_uptrend_prediction():
    """{0, 1, 2} is the canonical uptrend — x_3* must be 3 under the
    generalized minimizer (see §6.4 note)."""
    pred = predict_next([0, 1, 2])
    assert pred == pytest.approx(3.0, abs=1e-12)


def test_generalized_D2_downtrend_prediction():
    pred = predict_next([0, -1, -2])
    assert pred == pytest.approx(-3.0, abs=1e-12)
