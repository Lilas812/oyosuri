"""参照期間（ローリング窓 W）の比較スクリプト — 卒論 §7 用.

``oyosuri_all_in_one.py`` と ``oyosuri_rolling.py`` の機能を import し、
**同一データ・同一ボックス幅 B** に対して複数の参照期間 W で 1 ステップ先予測を回す。

提供する2つの出口:
  - ``run_mae_sweep``      … 各 W の MAE（平均絶対誤差）を数値で表示する。
  - ``run_multi_w_overlay``… 最大3つの W の予測を実測と同一グラフに重ねて出力する。

判断はあくまで重ね描きグラフを目視して行い、MAE は補助（飾り）として併記する。
MAE = (1/N) Σ_n |x_n − x_n*|（既存の ``mae()`` を使用）。±1 ウォークなので
no-change 予測の MAE は恒等的に 1（MAE<1 で「動かない予測」に優る目安）。

Colab での使い方 (別セルで順に %run):

    %run colab/oyosuri_all_in_one.py
    %run colab/oyosuri_rolling.py
    %run colab/oyosuri_experiment.py

    # (1) 数値だけ
    res = run_mae_sweep(path, B=4, windows=[2, 3, 5, 10, 20, 40],
                        start="2026-05-19-15:00", end="2026-05-19-23:58",
                        tz="Asia/Tokyo")

    # (2) 最大3つの W を同一グラフに重ねて目視（判断はこの図で）
    fig, r = run_multi_w_overlay(path, B=4, windows=[10, 30, 60],
                                 start="2026-05-19-15:00", end="2026-05-19-23:58",
                                 tz="Asia/Tokyo")
    # windows に None を入れると全期間版になる（予備実験用: windows=[2, None]）。
"""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from oyosuri_all_in_one import (
    generate_mean_renko_from_ohlc,
    load_tradingview_csv,
    mae,
    predict_sequence,
    slice_by_time,
    walk_from_bricks,
)
from oyosuri_rolling import predict_sequence_with_params_rolling


def _preds_for_window(
    x: Sequence[int],
    W: int | None,
    *,
    grid_size: int = 11,
    symmetric: bool = False,
) -> list[float]:
    """参照期間 W の 1 ステップ先予測列を返す.

    ``W`` が ``None`` または ``>= N`` のときは全期間版
    (``predict_sequence``)、そうでなければ直近 W bricks の rolling 版。
    """
    N = len(list(x)) - 1
    if W is None or int(W) >= N:
        return predict_sequence(x, grid_size=grid_size, symmetric=symmetric)
    return predict_sequence_with_params_rolling(
        x, window=int(W), grid_size=grid_size, symmetric=symmetric
    )[0]


def mae_by_window(
    x: Sequence[int],
    windows: Sequence[int],
    *,
    include_full: bool = True,
    grid_size: int = 11,
    symmetric: bool = False,
) -> dict[str, float]:
    """各参照期間 W で 1 ステップ先予測し、MAE を ``{W: MAE}`` で返す.

    ``1 <= W < N`` の W は直近 W bricks だけで当てはめるローリング窓版、
    ``include_full=True`` なら全履歴版（label ``"full"``）も加える。
    """
    x = list(x)
    N = len(x) - 1
    if N < 1:
        raise ValueError("x must contain at least 2 points")
    actual = [float(v) for v in x[1:]]

    out: dict[str, float] = {}
    for W in windows:
        w = int(W)
        if 1 <= w < N:
            preds = predict_sequence_with_params_rolling(
                x, window=w, grid_size=grid_size, symmetric=symmetric
            )[0]
            out[str(w)] = mae(actual, preds)
    if include_full:
        preds = predict_sequence(x, grid_size=grid_size, symmetric=symmetric)
        out["full"] = mae(actual, preds)
    return out


def run_mae_sweep(
    src,
    B: float,
    *,
    windows: Sequence[int] = (2, 3, 5, 10, 20, 40),
    include_full: bool = True,
    start=None,
    end=None,
    tz: str | None = None,
    grid_size: int = 11,
    symmetric: bool = False,
) -> dict:
    """TradingView CSV → 平均練行足 → 参照期間ごとの MAE を数値表示（グラフなし）.

    Returns
    -------
    dict
        {"x", "bricks", "N", "mae": {W: MAE}}
    """
    df = load_tradingview_csv(src)
    df = slice_by_time(df, start=start, end=end, tz=tz)
    if isinstance(df.index, pd.DatetimeIndex) and len(df) > 0:
        range_str = f"{df.index[0]} → {df.index[-1]}"
    else:
        range_str = f"rows [0, {len(df)})"
    print(f"当てはめ範囲: {range_str}  (bars={len(df)})")

    bricks, _ = generate_mean_renko_from_ohlc(df, B=B)
    x = walk_from_bricks(bricks)
    N = len(bricks)
    print(f"N_bricks={N}, B={B}")
    if N < 2:
        raise ValueError(
            f"ブリック数が少なすぎます (N={N})。B を小さくするか期間を広げてください。"
        )

    result = mae_by_window(
        x, windows, include_full=include_full,
        grid_size=grid_size, symmetric=symmetric,
    )

    print(f"{'W':>6} | {'MAE':>8}")
    print("-" * 18)
    for label, value in result.items():
        print(f"{label:>6} | {value:>8.4f}")

    return {"x": x, "bricks": bricks, "N": N, "mae": result}


def plot_walk_multi_w(
    x: Sequence[int],
    preds_by_w: dict[str, Sequence[float]],
    *,
    mae_by_w: dict[str, float] | None = None,
    title: str | None = None,
):
    """実測ウォーク x_n と、最大3つの W の予測 x_n* を同一図に重ねて返す.

    判断はこの図を目視で行う。凡例には W と（あれば）補助の MAE を併記する。
    """
    x = list(x)
    N = max(len(x) - 1, 0)
    fig, ax = plt.subplots(figsize=(13, 5))

    walk_n = np.arange(len(x))
    ax.step(
        walk_n, x, where="post",
        color="#333333", linewidth=1.8, label="x_n (observed)",
    )

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    markers = ["o", "s", "^"]
    for i, (label, preds) in enumerate(preds_by_w.items()):
        preds = list(preds)
        pred_n = np.arange(1, 1 + len(preds))
        leg = f"W={label}"
        if mae_by_w is not None and label in mae_by_w:
            leg += f"  (MAE={mae_by_w[label]:.3f})"
        ax.plot(
            pred_n, preds,
            color=colors[i % len(colors)], linewidth=1.4,
            marker=markers[i % len(markers)], markersize=3, label=leg,
        )

    ax.set_title(title or "Observed walk vs predictions (multiple W)")
    ax.set_xlabel("brick index n")
    ax.set_ylabel("x_n  (box units)")
    ax.set_xlim(-0.5, max(N, 1) + 0.5)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8,
              borderaxespad=0.0)
    fig.tight_layout()
    return fig


def run_multi_w_overlay(
    src,
    B: float,
    *,
    windows: Sequence[int | None] = (10, 30, 60),
    start=None,
    end=None,
    tz: str | None = None,
    grid_size: int = 11,
    symmetric: bool = False,
    title: str | None = None,
    show: bool = True,
):
    """CSV → 平均練行足 → 最大3つの W の予測を実測と同一グラフに重ねて出力.

    ``windows`` は最大3つ。要素に ``None`` を入れるとその系列は全期間版になる
    （予備実験で W=2 と全期間を比べたいときは ``windows=[2, None]``）。

    Returns
    -------
    (fig, result)
        fig    : matplotlib Figure（重ね描き）。
        result : {"x", "bricks", "N", "preds": {W: preds}, "mae": {W: MAE}}
    """
    ws = list(windows)
    if not 1 <= len(ws) <= 3:
        raise ValueError("windows は1〜3個（同一グラフに重ねるため最大3つ）")

    df = load_tradingview_csv(src)
    df = slice_by_time(df, start=start, end=end, tz=tz)
    if isinstance(df.index, pd.DatetimeIndex) and len(df) > 0:
        range_str = f"{df.index[0]} → {df.index[-1]}"
    else:
        range_str = f"rows [0, {len(df)})"
    print(f"当てはめ範囲: {range_str}  (bars={len(df)})")

    bricks, _ = generate_mean_renko_from_ohlc(df, B=B)
    x = walk_from_bricks(bricks)
    N = len(bricks)
    print(f"N_bricks={N}, B={B}")
    if N < 2:
        raise ValueError(
            f"ブリック数が少なすぎます (N={N})。B を小さくするか期間を広げてください。"
        )
    actual = [float(v) for v in x[1:]]

    preds_by_w: dict[str, list[float]] = {}
    mae_by_w: dict[str, float] = {}
    for W in ws:
        is_full = (W is None) or (int(W) >= N)
        label = "full" if is_full else str(int(W))
        preds = _preds_for_window(
            x, None if is_full else int(W),
            grid_size=grid_size, symmetric=symmetric,
        )
        preds_by_w[label] = preds
        mae_by_w[label] = mae(actual, preds)

    print("MAE (参考):", {k: round(v, 4) for k, v in mae_by_w.items()})
    fig = plot_walk_multi_w(x, preds_by_w, mae_by_w=mae_by_w, title=title)
    if show:
        plt.show()
    return fig, {
        "x": x, "bricks": bricks, "N": N,
        "preds": preds_by_w, "mae": mae_by_w,
    }
