"""Tests for 作成書 §6.4 / §8 item 3 (textbook reproduction under q = p)."""

from __future__ import annotations

import pytest

from oyosuri import minimize_Vn, predict_next


def test_D2_uptrend_matches_textbook():
    """D_2 = {0, 1, 2} ⇒ (p*, α*) = (1, 0), x_3* = 3."""
    res = minimize_Vn([0, 1, 2], symmetric=True)
    assert res.V_min == pytest.approx(0.0, abs=1e-12)
    assert not res.is_constant
    # Under q = p the unique minimum is (p=1, q=1, α=0).
    unique = {(round(p, 10), round(q, 10), round(a, 10)) for p, q, a in res.argmins}
    assert unique == {(1.0, 1.0, 0.0)}

    pred = predict_next([0, 1, 2], symmetric=True)
    assert pred == pytest.approx(3.0, abs=1e-12)


def test_D2_reversal_matches_textbook_regression():
    """D_2 = {0, 1, 0} under q = p.

    The spec says (p_2*, α_2*) = (0, 1) and points at textbook §5.2
    table 5.2 for the numerical x_3*. Because the walk {0, 1, 0} sits
    exactly on the deterministic alternating path produced by (p=0,
    q=0, α=1), V_2 = 0 is achieved uniquely there, the next forced
    step is +1, so x_3* = 1 (regression-locked from the model itself).
    """
    res = minimize_Vn([0, 1, 0], symmetric=True)
    assert res.V_min == pytest.approx(0.0, abs=1e-12)
    assert not res.is_constant
    unique = {(round(p, 10), round(q, 10), round(a, 10)) for p, q, a in res.argmins}
    assert unique == {(0.0, 0.0, 1.0)}

    pred = predict_next([0, 1, 0], symmetric=True)
    assert pred == pytest.approx(1.0, abs=1e-12)


def test_symmetric_uptrend_also_predicts_three_in_generalized_mode():
    """For {0, 1, 2} the generalized (q ≠ p) argmin shrinks to the
    segment {(1, 0, α) : α ∈ [0, 1]}, and E[S̃_3] = 3 for every α. So
    the prediction matches the symmetric case (3.0)."""
    pred = predict_next([0, 1, 2], symmetric=False)
    assert pred == pytest.approx(3.0, abs=1e-12)
