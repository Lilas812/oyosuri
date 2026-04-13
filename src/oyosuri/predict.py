"""Prediction algorithm — 作成書 §5 step 2 and §7 Step E."""

from __future__ import annotations

from typing import Sequence

from .model import expected_position
from .optimize import minimize_Vn


def predict_next(
    x_obs: Sequence[int],
    *,
    grid_size: int = 21,
    symmetric: bool = False,
    tol: float = 1e-10,
) -> float:
    """Return the one-step prediction ``x_{n+1}*`` given ``{x_0,…,x_n}``.

    Follows 作成書 §5 step 2:
        (a) If V_n is constant in (p, q, alpha), return x_n.
        (b) If argmin is unique, return E[S̃_{n+1} | p*, q*, alpha*].
        (c) Otherwise, return the arithmetic mean of E[S̃_{n+1}] over all
            argmin candidates.
    """
    x_list = list(x_obs)
    if not x_list:
        raise ValueError("x_obs must be non-empty")
    n = len(x_list) - 1
    res = minimize_Vn(
        x_list, grid_size=grid_size, symmetric=symmetric, tol=tol
    )
    if res.is_constant:
        return float(x_list[-1])
    expectations = [
        expected_position(n + 1, p, q, alpha) for (p, q, alpha) in res.argmins
    ]
    return float(sum(expectations) / len(expectations))


def predict_sequence(
    x_obs: Sequence[int],
    *,
    grid_size: int = 21,
    symmetric: bool = False,
    tol: float = 1e-10,
) -> list[float]:
    """Return the full one-step-ahead prediction sequence.

    Given ``x_obs = {x_0, x_1, …, x_N}``, for ``n = 0, 1, …, N-1`` the
    function calls :func:`predict_next` with the prefix ``x_obs[: n + 1]``
    and returns ``[x_1*, x_2*, …, x_N*]`` (作成書 §7 Step E).
    """
    x_list = list(x_obs)
    N = len(x_list) - 1
    preds: list[float] = []
    for n in range(N):
        preds.append(
            predict_next(
                x_list[: n + 1],
                grid_size=grid_size,
                symmetric=symmetric,
                tol=tol,
            )
        )
    return preds
