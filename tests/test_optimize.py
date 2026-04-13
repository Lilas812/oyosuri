"""Tests for 作成書 §6.1 and §6.2 (Vn argmin sanity)."""

from __future__ import annotations

import pytest

from oyosuri import Vn_value, minimize_Vn, predict_next


def test_n0_is_constant():
    res = minimize_Vn([0])
    assert res.is_constant is True
    assert res.V_min == pytest.approx(0.0, abs=1e-12)


def test_n1_positive_x_minimum_is_zero():
    """For x_1 = 1, V_1 min = (x_1 - 1)^2 = 0."""
    res = minimize_Vn([0, 1])
    assert res.V_min == pytest.approx(0.0, abs=1e-12)
    assert not res.is_constant
    # The canonical interior solution (p=1, q=0, any α) must be in the
    # argmin set (spec §6.2).
    argmin_pq = {(round(p, 10), round(q, 10)) for p, q, _ in res.argmins}
    assert (1.0, 0.0) in argmin_pq
    # Every argmin point must actually give V_1 = 0.
    for p, q, a in res.argmins:
        assert Vn_value([0, 1], p, q, a) == pytest.approx(0.0, abs=1e-12)


def test_n1_positive_x_prediction_is_positive():
    """Direction preservation: x_1 > 0 ⇒ x_2* > 0."""
    pred = predict_next([0, 1])
    assert pred > 0


def test_n1_negative_x_minimum_is_zero():
    res = minimize_Vn([0, -1])
    assert res.V_min == pytest.approx(0.0, abs=1e-12)
    argmin_pq = {(round(p, 10), round(q, 10)) for p, q, _ in res.argmins}
    assert (0.0, 1.0) in argmin_pq


def test_n1_negative_x_prediction_is_negative():
    pred = predict_next([0, -1])
    assert pred < 0


def test_n1_zero_x_is_constant():
    """x_1 = 0 ⇒ V_1 ≡ 1."""
    res = minimize_Vn([0, 0])
    assert res.is_constant is True
    assert res.V_min == pytest.approx(1.0, abs=1e-12)


def test_larger_x_gives_nonzero_min():
    """For x_1 = 2, V_1 min = (x_1 - 1)^2 = 1."""
    res = minimize_Vn([0, 2])
    assert res.V_min == pytest.approx(1.0, abs=1e-12)
    argmin_pq = {(round(p, 10), round(q, 10)) for p, q, _ in res.argmins}
    assert (1.0, 0.0) in argmin_pq


def test_symmetric_mode_n1_positive_gives_unit_prediction():
    """Under q=p, the argmin at x_1 = 1 shrinks to {(0, 0, 1), (1, 1, 0)}
    whose E[S̃_2] values are 0 and 2, so the mean is 1.0 (classic textbook)."""
    pred = predict_next([0, 1], symmetric=True)
    assert pred == pytest.approx(1.0, abs=1e-12)
