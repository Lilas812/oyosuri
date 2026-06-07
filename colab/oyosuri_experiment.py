"""参照期間（ローリング窓 W）ごとの MAE を数値で出すだけの最小スクリプト — 卒論 §7 用.

``oyosuri_all_in_one.py`` と ``oyosuri_rolling.py`` の機能を import し、
**同一データ・同一ボックス幅 B** に対して複数の参照期間 W で 1 ステップ先予測を回し、
各 W の **MAE（平均絶対誤差）だけ** を数値で表示する。グラフは作らない。

MAE = (1/N) Σ_n |x_n − x_n*|  ……既存の ``mae()`` をそのまま使う。

Colab での使い方 (別セルで順に %run):

    %run colab/oyosuri_all_in_one.py
    %run colab/oyosuri_rolling.py
    %run colab/oyosuri_experiment.py

    res = run_mae_sweep(
        path, B=4,
        windows=[2, 3, 5, 10, 20, 40],
        start="2026-05-19-15:00",
        end="2026-05-19-23:58",
        tz="Asia/Tokyo",
    )
    # res["mae"] が {W: MAE} の辞書（全履歴は "full"）。
"""

from __future__ import annotations

from typing import Sequence

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
