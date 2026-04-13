"""Tests for 作成書 §8 item 1 and item 2 (probability and V_1 checks)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from oyosuri import (
    Vn_analytic_n1,
    Vn_value,
    coin_matrices,
    expected_position,
    mu_from_psi,
    psi_forward,
)


def test_mu_0_is_delta():
    psi = psi_forward(n=0, p=0.3, q=0.4, alpha=0.7)
    mu = mu_from_psi(psi)
    assert mu.shape == (1, 1)
    assert mu[0, 0] == pytest.approx(1.0, abs=1e-12)


def test_mu_1_matches_closed_form_at_034_04_07():
    p, q, alpha = 0.3, 0.4, 0.7
    psi = psi_forward(n=1, p=p, q=q, alpha=alpha)
    mu = mu_from_psi(psi)
    # offset = n = 1, so mu[1, 0] is mu_1(-1) and mu[1, 2] is mu_1(1).
    mu_minus = mu[1, 0]
    mu_plus = mu[1, 2]
    expected_minus = q * alpha + (1 - p) * (1 - alpha)
    expected_plus = (1 - q) * alpha + p * (1 - alpha)
    assert mu_minus == pytest.approx(expected_minus, abs=1e-12)
    assert mu_plus == pytest.approx(expected_plus, abs=1e-12)
    assert mu_minus + mu_plus == pytest.approx(1.0, abs=1e-12)


def test_mu_1_normalizes_for_many_params():
    rng = np.random.default_rng(0)
    for _ in range(50):
        p, q, alpha = rng.random(3)
        psi = psi_forward(n=1, p=float(p), q=float(q), alpha=float(alpha))
        mu = mu_from_psi(psi)
        assert mu[1].sum() == pytest.approx(1.0, abs=1e-12)


def test_Vn_value_matches_analytic_n1():
    rng = np.random.default_rng(1)
    for _ in range(30):
        p, q, alpha = rng.random(3)
        x1 = int(rng.integers(-3, 4))
        v_num = Vn_value([0, x1], float(p), float(q), float(alpha))
        v_ana = Vn_analytic_n1(x1, float(p), float(q), float(alpha))
        assert v_num == pytest.approx(v_ana, abs=1e-12)


def test_Vn_analytic_reduces_to_textbook_when_q_equals_p():
    # Textbook (5.6): V_1 = x_1^2 + 2(2p-1)(2α-1) x_1 + 1.
    for p in [0.0, 0.25, 0.5, 0.75, 1.0]:
        for alpha in [0.0, 0.3, 0.5, 0.8, 1.0]:
            for x1 in [-2, -1, 0, 1, 2]:
                v_gen = Vn_analytic_n1(x1, p, p, alpha)
                v_sym = (
                    x1 * x1
                    + 2.0 * (2 * p - 1) * (2 * alpha - 1) * x1
                    + 1.0
                )
                assert v_gen == pytest.approx(v_sym, abs=1e-12)


def test_coin_matrices_column_sums_are_one():
    P, Q = coin_matrices(0.3, 0.4)
    A = P + Q
    col_sums = A.sum(axis=0)
    assert col_sums == pytest.approx(np.array([1.0, 1.0]), abs=1e-12)


def test_expected_position_at_n1_matches_mu_diff():
    p, q, alpha = 0.2, 0.8, 0.5
    ev = expected_position(1, p, q, alpha)
    mu_plus = (1 - q) * alpha + p * (1 - alpha)
    mu_minus = q * alpha + (1 - p) * (1 - alpha)
    assert ev == pytest.approx(mu_plus - mu_minus, abs=1e-12)


def test_psi_normalization_at_larger_n():
    # Sum of mu_n(x) over x should always be 1 for any n, (p, q, alpha).
    for n in [2, 3, 4, 5]:
        psi = psi_forward(n=n, p=0.3, q=0.7, alpha=0.4)
        mu = mu_from_psi(psi)
        for t in range(n + 1):
            assert mu[t].sum() == pytest.approx(1.0, abs=1e-10)
