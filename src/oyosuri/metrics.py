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


def mse(actual: Sequence[float], pred: Sequence[float]) -> float:
    """Mean squared error — 「偏差（予測と実測の差）」の二乗平均.

    ``MSE = (1/N) Σ_n (x_n − x_n*)^2``。ここで偏差 ``x_n − x_n*`` は
    予測誤差であって、平均からの偏差（分散）ではない。値は :func:`rmse`
    の二乗に等しい。

    本モデルでは観測ウォークが毎ステップ ±1 しか動かないため、
    「動かない」予測 (``x_n* = x_{n-1}``) の MSE は恒等的に **1** になる。
    したがって MSE < 1 で初めてモデルが方向情報を抽出できていると言える
    （MSE = 1 は無情報、MSE > 1 は何もしないより悪い）。
    """
    a, p = _to_arrays(actual, pred)
    if a.size == 0:
        return 0.0
    return float(np.mean((a - p) ** 2))


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


def mse_skill(
    actual: Sequence[float],
    pred: Sequence[float],
    baseline: Sequence[float] | None = None,
) -> float:
    """ベースライン比のスキルスコア ``1 − MSE_model / MSE_base``.

    ``baseline`` はベースライン予測列。``None`` のときは ±1 ウォークにおける
    no-change 予測の MSE（恒等的に 1）を基準にする（このとき
    ``skill = 1 − MSE_model``）。

    返り値が正ならベースラインより良く、0 は同等、負ならベースラインより
    悪い。``base_mse == 0`` のときは ``0.0`` を返す。
    """
    model_mse = mse(actual, pred)
    base_mse = 1.0 if baseline is None else mse(actual, baseline)
    if base_mse == 0.0:
        return 0.0
    return float(1.0 - model_mse / base_mse)
