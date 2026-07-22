"""Fast, legacy-compatible moment engine for rolling predictions.

The original implementation builds the complete spatial probability
distribution for every grid point.  The objective only needs the first two
moments, however::

    sum_x (x - y_t)^2 mu_t(x)
      = E[S_t^2] - 2 y_t E[S_t] + y_t^2.

For the current parameter domain ``p, q, alpha in [0, 1]`` all components of
``Psi`` are non-negative, so ``mu = |Psi_L| + |Psi_R|`` is an ordinary
probability distribution and the moment recurrence below is algebraically
identical to the spatial recurrence.

Long-double arithmetic is used for the fast screening.  The small set of
possible minimizers is then evaluated again with the original ``Vn_value``.
Consequently the public rolling predictor keeps the legacy grid/tolerance
semantics while avoiding almost all full-distribution evaluations.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence

import numpy as np

from oyosuri_all_in_one import OptResult, Vn_value


@dataclass(frozen=True)
class MomentGrid:
    """Parameter grid and moment tables through ``max_horizon``."""

    p: np.ndarray
    q: np.ndarray
    alpha: np.ndarray
    mean: np.ndarray
    second: np.ndarray
    max_horizon: int


def _parameter_grid(
    grid_size: int,
    symmetric: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return parameters in exactly the same order as legacy ``minimize_Vn``."""
    if grid_size < 2:
        raise ValueError("grid_size must be >= 2")
    grid = np.linspace(0.0, 1.0, grid_size, dtype=float)
    if symmetric:
        p = np.repeat(grid, grid_size)
        q = p.copy()
        alpha = np.tile(grid, grid_size)
    else:
        p = np.repeat(grid, grid_size * grid_size)
        q = np.tile(np.repeat(grid, grid_size), grid_size)
        alpha = np.tile(grid, grid_size * grid_size)
    return p, q, alpha


@lru_cache(maxsize=16)
def get_moment_grid(
    max_horizon: int,
    grid_size: int = 11,
    symmetric: bool = False,
) -> MomentGrid:
    """Precompute ``E[S_t]`` and ``E[S_t^2]`` for the complete grid.

    The result is independent of the observed price path and is cached.  Thus
    B values and rolling windows can reuse the same table in one Python
    session.
    """
    if max_horizon < 0:
        raise ValueError("max_horizon must be >= 0")
    p64, q64, alpha64 = _parameter_grid(grid_size, symmetric)
    dtype = np.longdouble
    p = p64.astype(dtype)
    q = q64.astype(dtype)
    alpha = alpha64.astype(dtype)

    one = dtype(1.0)
    two = dtype(2.0)
    mass_l = alpha.copy()
    mass_r = one - alpha
    first_l = np.zeros_like(alpha)
    first_r = np.zeros_like(alpha)
    second_l = np.zeros_like(alpha)
    second_r = np.zeros_like(alpha)

    mean = np.empty((max_horizon + 1, alpha.size), dtype=dtype)
    second = np.empty_like(mean)
    mean[0] = 0.0
    second[0] = 0.0

    for t in range(max_horizon):
        next_mass_l = q * mass_l + (one - p) * mass_r
        next_mass_r = (one - q) * mass_l + p * mass_r

        # A transition into L moves the position by -1; into R by +1.
        next_first_l = q * (first_l - mass_l) + (one - p) * (
            first_r - mass_r
        )
        next_first_r = (one - q) * (first_l + mass_l) + p * (
            first_r + mass_r
        )
        next_second_l = q * (second_l - two * first_l + mass_l) + (
            one - p
        ) * (second_r - two * first_r + mass_r)
        next_second_r = (one - q) * (
            second_l + two * first_l + mass_l
        ) + p * (second_r + two * first_r + mass_r)

        mass_l, mass_r = next_mass_l, next_mass_r
        first_l, first_r = next_first_l, next_first_r
        second_l, second_r = next_second_l, next_second_r
        mean[t + 1] = first_l + first_r
        second[t + 1] = second_l + second_r

    return MomentGrid(
        p=p64,
        q=q64,
        alpha=alpha64,
        mean=mean,
        second=second,
        max_horizon=max_horizon,
    )


def objective_values_from_moments(
    x_obs: Sequence[int],
    moments: MomentGrid,
) -> np.ndarray:
    """Evaluate ``V_n`` for every grid point using the moment identity."""
    x = np.asarray(x_obs, dtype=np.longdouble)
    if x.size == 0:
        raise ValueError("x_obs must have at least one element")
    n = x.size - 1
    if n > moments.max_horizon:
        raise ValueError("moment table is shorter than x_obs")
    return np.sum(
        moments.second[: n + 1]
        - 2.0 * x[:, None] * moments.mean[: n + 1]
        + x[:, None] * x[:, None],
        axis=0,
        dtype=np.longdouble,
    )


def _screening_margin(values: np.ndarray, n: int) -> np.longdouble:
    """Conservative margin before rechecking contenders with legacy floats."""
    scale = max(
        1.0,
        abs(float(np.min(values))),
        abs(float(np.max(values))),
    )
    roundoff = 512.0 * np.finfo(float).eps * scale * max(n + 1, 1)
    return np.longdouble(max(1e-8, roundoff))


def minimize_Vn_from_moments(
    x_obs: Sequence[int],
    moments: MomentGrid,
    *,
    tol: float = 1e-10,
) -> OptResult:
    """Return the legacy-compatible grid minimum using moment screening.

    Long-double moments identify a conservative contender set.  Those
    contenders are re-evaluated by the original ``Vn_value`` and the original
    absolute-tolerance rule is applied to those legacy values.  If the
    constant/non-constant boundary is numerically ambiguous, all candidates
    are rechecked.
    """
    x = [int(v) for v in x_obs]
    if not x:
        raise ValueError("x_obs must have at least one element")
    n = len(x) - 1
    values = objective_values_from_moments(x, moments)
    best_fast = np.min(values)
    worst_fast = np.max(values)
    spread_fast = worst_fast - best_fast
    margin = _screening_margin(values, n)

    if spread_fast == 0:
        # Exactly constant in long-double arithmetic.  V_0 and x_1=0 are the
        # usual cases; one legacy evaluation preserves the reported V_min.
        indices = np.arange(values.size)
        best = Vn_value(
            x,
            float(moments.p[0]),
            float(moments.q[0]),
            float(moments.alpha[0]),
        )
        return OptResult(
            V_min=best,
            argmins=[
                (
                    float(moments.p[i]),
                    float(moments.q[i]),
                    float(moments.alpha[i]),
                )
                for i in indices
            ],
            is_constant=True,
        )

    checked_full_grid = spread_fast <= np.longdouble(tol) + margin
    if checked_full_grid:
        contender_indices = np.arange(values.size)
    else:
        contender_indices = np.flatnonzero(
            values - best_fast <= np.longdouble(tol) + margin
        )

    legacy_values: list[tuple[float, int]] = []
    for i in contender_indices:
        v = Vn_value(
            x,
            float(moments.p[i]),
            float(moments.q[i]),
            float(moments.alpha[i]),
        )
        legacy_values.append((v, int(i)))

    best = min(v for v, _ in legacy_values)
    argmin_indices = [i for v, i in legacy_values if v - best <= tol]
    is_constant = (
        (max(v for v, _ in legacy_values) - best) < tol
        if checked_full_grid
        else False
    )
    return OptResult(
        V_min=float(best),
        argmins=[
            (
                float(moments.p[i]),
                float(moments.q[i]),
                float(moments.alpha[i]),
            )
            for i in argmin_indices
        ],
        is_constant=bool(is_constant),
    )
