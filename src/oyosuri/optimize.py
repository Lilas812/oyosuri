"""Grid minimizer for V_n(p, q, alpha) — 作成書 §5 step 1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .model import Vn_value


@dataclass(frozen=True)
class OptResult:
    """Result of :func:`minimize_Vn`.

    Attributes
    ----------
    V_min : float
        Minimum value of ``V_n`` over the grid.
    argmins : list[tuple[float, float, float]]
        All ``(p, q, alpha)`` grid points whose ``V_n`` value is within
        ``tol`` of ``V_min``. At least one entry is always present.
    is_constant : bool
        ``True`` when ``V_max - V_min < tol`` on the grid, i.e. ``V_n``
        is effectively constant in ``(p, q, alpha)``. Triggers 作成書
        §5 step 2 case (a).
    """

    V_min: float
    argmins: list[tuple[float, float, float]]
    is_constant: bool


def minimize_Vn(
    x_obs: Sequence[int],
    *,
    grid_size: int = 21,
    symmetric: bool = False,
    tol: float = 1e-10,
) -> OptResult:
    """Search ``[0, 1]^3`` (or ``[0, 1]^2`` with ``q = p`` when
    ``symmetric=True``) for the minimum of ``V_n(p, q, alpha)``.

    Parameters
    ----------
    x_obs :
        Observed random walk ``{x_0, …, x_n}``.
    grid_size :
        Number of grid points per axis (inclusive of 0 and 1). Default 21
        gives step 0.05, which includes the canonical boundary points
        ``{0, 0.5, 1}``.
    symmetric :
        If ``True``, enforce ``q = p`` (textbook §5.2 mode).
    tol :
        Absolute tolerance for: (a) detecting a constant ``V_n`` and
        (b) collecting near-ties as argmins.
    """
    if grid_size < 2:
        raise ValueError("grid_size must be >= 2")
    grid = np.linspace(0.0, 1.0, grid_size)

    best = np.inf
    worst = -np.inf
    values: list[tuple[float, tuple[float, float, float]]] = []

    if symmetric:
        for p in grid:
            for alpha in grid:
                v = Vn_value(x_obs, float(p), float(p), float(alpha))
                values.append((v, (float(p), float(p), float(alpha))))
                if v < best:
                    best = v
                if v > worst:
                    worst = v
    else:
        for p in grid:
            for q in grid:
                for alpha in grid:
                    v = Vn_value(x_obs, float(p), float(q), float(alpha))
                    values.append((v, (float(p), float(q), float(alpha))))
                    if v < best:
                        best = v
                    if v > worst:
                        worst = v

    is_constant = (worst - best) < tol
    argmins = [pt for v, pt in values if v - best <= tol]
    return OptResult(V_min=float(best), argmins=argmins, is_constant=is_constant)
