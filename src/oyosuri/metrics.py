"""Evaluation metrics — 作成書 §7.

All functions expect aligned sequences ``actual`` and ``pred`` containing
``{x_1, …, x_N}`` and ``{x_1*, …, x_N*}`` respectively.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def _to_arrays(actual: Sequence[float], pred: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(actual, dtype=float)
    p = np.asarray(pred, dtype=float)
    if a.shape != p.shape:
        raise ValueError(
            f"actual and pred must have the same length: {a.shape} vs {p.shape}"
        )
    return a, p


def mae(actual: Sequence[float], pred: Sequence[float]) -> float:
    """Mean absolute error."""
    a, p = _to_arrays(actual, pred)
    if a.size == 0:
        return 0.0
    return float(np.mean(np.abs(a - p)))


def rmse(actual: Sequence[float], pred: Sequence[float]) -> float:
    """Root mean squared error."""
    a, p = _to_arrays(actual, pred)
    if a.size == 0:
        return 0.0
    return float(np.sqrt(np.mean((a - p) ** 2)))


def hit_rate(actual: Sequence[float], pred: Sequence[float]) -> float:
    """Directional hit rate.

    ``HIT = (1/(N-1)) * Σ_{n=2..N} 1[sign(x_n - x_{n-1}) == sign(x_n* - x_{n-1})]``
    where ``x_{n-1}`` comes from the ``actual`` sequence. Returns ``0.0``
    when fewer than 2 samples are supplied.
    """
    a, p = _to_arrays(actual, pred)
    if a.size < 2:
        return 0.0
    prev = a[:-1]
    actual_step = np.sign(a[1:] - prev)
    pred_step = np.sign(p[1:] - prev)
    hits = actual_step == pred_step
    return float(np.mean(hits))
