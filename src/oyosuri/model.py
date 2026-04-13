"""Correlated random walk (generalized 2-parameter coin matrix) core.

Implements 作成書 §3 and §4:
    A(p, q) = [[q,   1-p],
               [1-q, p  ]]
    phi(alpha) = [alpha, 1-alpha]^T

    Psi_{n+1}(x) = P(p, q) Psi_n(x+1) + Q(p, q) Psi_n(x-1)
    mu_n(x)      = |Psi_n^L(x)| + |Psi_n^R(x)|
    V_n(p, q, a) = sum_{t=0..n} sum_x (x - x_t)^2 * mu_t(x; p, q, a)
    E[S_n]       = sum_x x * mu_n(x; p, q, a)
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def coin_matrices(p: float, q: float) -> tuple[np.ndarray, np.ndarray]:
    """Return the left/right transition matrices (P, Q) from 作成書 §3.1."""
    P = np.array([[q, 1.0 - p], [0.0, 0.0]], dtype=float)
    Q = np.array([[0.0, 0.0], [1.0 - q, p]], dtype=float)
    return P, Q


def psi_forward(n: int, p: float, q: float, alpha: float) -> np.ndarray:
    """Compute Psi_t(x) for t = 0..n, x ∈ [-t, t].

    Returns an array ``psi`` of shape (n+1, 2n+1, 2) where
    ``psi[t, x + n, :] == Psi_t(x)`` for ``|x| <= t`` and zero elsewhere.

    Using a common offset of ``n`` (rather than ``t``) gives a single
    rectangular buffer and avoids per-step reallocation. Values outside
    ``|x| <= t`` remain identically zero.
    """
    if n < 0:
        raise ValueError("n must be >= 0")
    P, Q = coin_matrices(p, q)
    width = 2 * n + 1
    psi = np.zeros((n + 1, width, 2), dtype=float)
    # Initial condition: Psi_0(0) = phi(alpha) = (alpha, 1-alpha)
    psi[0, n, 0] = alpha
    psi[0, n, 1] = 1.0 - alpha
    for t in range(n):
        # Psi_{t+1}(x) = P @ Psi_t(x+1) + Q @ Psi_t(x-1)
        # Vectorize over the x axis. x+1 means shifting left, x-1 means shifting right.
        prev = psi[t]  # shape (width, 2)
        # Contribution from Psi_t(x+1): shift the prev array one slot to the left.
        left_shift = np.zeros_like(prev)
        left_shift[:-1] = prev[1:]
        # Contribution from Psi_t(x-1): shift the prev array one slot to the right.
        right_shift = np.zeros_like(prev)
        right_shift[1:] = prev[:-1]
        # Apply matrices: result[x] = P @ left_shift[x] + Q @ right_shift[x]
        # Use einsum for clarity: 'ij,xj->xi'
        psi[t + 1] = left_shift @ P.T + right_shift @ Q.T
    return psi


def mu_from_psi(psi: np.ndarray) -> np.ndarray:
    """Given a full ``psi`` array from :func:`psi_forward`, return ``mu``
    with shape ``(n+1, 2n+1)``.

    ``mu[t, x + n] == mu_t(x)`` (L1 norm over the 2-vector component axis).
    Values for ``|x| > t`` are zero.
    """
    return np.abs(psi[..., 0]) + np.abs(psi[..., 1])


def Vn_value(
    x_obs: Sequence[int],
    p: float,
    q: float,
    alpha: float,
) -> float:
    """Evaluate V_n(p, q, alpha) for the observation ``x_obs = {x_0,…,x_n}``.

    Implements 作成書 §4:
        V_n = sum_{t=0..n} sum_{x=-t..t} (x - x_t)^2 * mu_t(x; p, q, alpha)
    """
    x_arr = np.asarray(x_obs, dtype=float)
    n = x_arr.size - 1
    if n < 0:
        raise ValueError("x_obs must have at least one element")
    psi = psi_forward(n, p, q, alpha)
    mu = mu_from_psi(psi)  # shape (n+1, 2n+1)
    # x grid: [-n, -n+1, ..., n]
    x_grid = np.arange(-n, n + 1, dtype=float)
    total = 0.0
    for t in range(n + 1):
        diff = x_grid - x_arr[t]
        total += float(np.sum(diff * diff * mu[t]))
    return total


def expected_position(
    n_target: int,
    p: float,
    q: float,
    alpha: float,
) -> float:
    """Return E[S̃_{n_target}] under the model (作成書 §5 note).

    ``E[S̃_{n_target}] = sum_x x * mu_{n_target}(x; p, q, alpha)``.
    """
    if n_target < 0:
        raise ValueError("n_target must be >= 0")
    psi = psi_forward(n_target, p, q, alpha)
    mu = mu_from_psi(psi)[n_target]
    x_grid = np.arange(-n_target, n_target + 1, dtype=float)
    return float(np.sum(x_grid * mu))


def Vn_analytic_n1(x1: float, p: float, q: float, alpha: float) -> float:
    """Analytic V_1 formula (eq. 5.6').

    V_1(p, q, alpha) = x1^2 + 2[alpha(2q-1) + (1-alpha)(1-2p)] x1 + 1.

    Used purely for cross-checking :func:`Vn_value` at n=1 (see §8 item 2).
    """
    coeff = alpha * (2.0 * q - 1.0) + (1.0 - alpha) * (1.0 - 2.0 * p)
    return float(x1 * x1 + 2.0 * coeff * x1 + 1.0)
