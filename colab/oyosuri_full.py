"""Fast, legacy-compatible predictor for the full-history model."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from oyosuri_all_in_one import expected_position
from oyosuri_moments import get_moment_grid, minimize_Vn_from_values


def predict_sequence_with_params_full_fast(
    x_obs: Sequence[int],
    *,
    grid_size: int = 21,
    symmetric: bool = False,
    tol: float = 1e-10,
) -> tuple[list[float], list[float], list[float], list[float]]:
    """Return full-history predictions using cumulative moment objectives.

    The objective for prefix ``0..n`` is obtained by adding only the term at
    time ``n``.  Candidate minimizers are rechecked with the original
    ``Vn_value`` by :func:`minimize_Vn_from_values`, and final expectations are
    still calculated by the original ``expected_position``.  Grid order,
    tolerance handling, multiple-minimum averaging and returned floats thus
    remain compatible with the legacy implementation.
    """
    x = [int(v) for v in x_obs]
    N = len(x) - 1
    if N < 0:
        return [], [], [], []

    moments = get_moment_grid(N, grid_size, symmetric)
    objective = np.zeros(moments.p.size, dtype=np.longdouble)
    preds: list[float] = []
    ps: list[float] = []
    qs: list[float] = []
    alphas: list[float] = []

    for n in range(N):
        xn = np.longdouble(x[n])
        objective += (
            moments.second[n]
            - 2.0 * xn * moments.mean[n]
            + xn * xn
        )
        prefix = x[: n + 1]
        res = minimize_Vn_from_values(prefix, objective, moments, tol=tol)

        if res.is_constant:
            preds.append(float(prefix[-1]))
            ps.append(float("nan"))
            qs.append(float("nan"))
            alphas.append(float("nan"))
            continue

        # Keep the legacy expectation calculation, including its summation
        # order, so public prediction floats do not change.
        expectations = [
            expected_position(n + 1, p, q, alpha)
            for p, q, alpha in res.argmins
        ]
        preds.append(float(sum(expectations) / len(expectations)))
        k = len(res.argmins)
        ps.append(sum(point[0] for point in res.argmins) / k)
        qs.append(sum(point[1] for point in res.argmins) / k)
        alphas.append(sum(point[2] for point in res.argmins) / k)

    return preds, ps, qs, alphas
