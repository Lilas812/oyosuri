"""直近 W bricks だけで fit する rolling-window 版.

``oyosuri_all_in_one.py`` の全機能を import し、``window`` 個ぶんの
直近観測だけで V_n を最小化する版の予測関数とランナーを提供する。

典型的な使い方 (Colab で別セル):

    %run colab/oyosuri_all_in_one.py     # まず all-in-one を読み込む
    %run colab/oyosuri_rolling.py        # 次に rolling 拡張を読み込む

    figs, res = run_on_tradingview_rolling(
        path, B=4, window=30,
        start="2026-05-19-15:00",
        end="2026-05-19-23:58",
        tz="Asia/Tokyo",
    )

``window`` を小さくすると最近のレジームに敏感に追従するがノイズが増え、
大きくすると履歴全体を使う従来版 (``oyosuri_all_in_one`` の
``run_on_tradingview``) に近づく。
"""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import pandas as pd

from oyosuri_all_in_one import (
    expected_position,
    generate_mean_renko_from_ohlc,
    load_tradingview_csv,
    minimize_Vn,
    plot_pq,
    plot_price,
    plot_walk_vs_prediction,
    slice_by_time,
    walk_from_bricks,
)


def predict_sequence_with_params_rolling(
    x_obs: Sequence[int],
    *,
    window: int,
    grid_size: int = 11,
    symmetric: bool = False,
    tol: float = 1e-10,
) -> tuple[list[float], list[float], list[float], list[float]]:
    """直近 ``window`` bricks だけを使って各ステップを fit する.

    各ステップ ``n`` で

        prefix = x_obs[max(0, n + 1 - window) : n + 1]

    を取り、モデルが想定する ``x_0 = 0`` 規約に合わせて先頭値を引いて
    再ゼロ化してから ``minimize_Vn`` を呼ぶ。予測値は元の絶対水準に戻す。

    現在利用可能なステップ数が ``window`` 未満の区間では、その時点で
    利用可能な全履歴を使う (= 序盤は従来版と同じ振る舞い)。

    返り値は ``oyosuri_all_in_one.predict_sequence_with_params`` と同じく
    ``(preds, ps, qs, alphas)`` の 4-tuple。
    """
    if window < 1:
        raise ValueError("window must be >= 1")
    x_list = list(x_obs)
    N = len(x_list) - 1
    preds: list[float] = []
    ps: list[float] = []
    qs: list[float] = []
    alphas: list[float] = []
    for n in range(N):
        start = max(0, n + 1 - window)
        sub = x_list[start : n + 1]
        base = sub[0]
        sub_rezeroed = [int(v - base) for v in sub]
        m = len(sub_rezeroed) - 1  # m + 1 = predict horizon in model frame
        res = minimize_Vn(
            sub_rezeroed, grid_size=grid_size, symmetric=symmetric, tol=tol
        )
        if res.is_constant:
            preds.append(float(sub_rezeroed[-1] + base))
            ps.append(float("nan"))
            qs.append(float("nan"))
            alphas.append(float("nan"))
            continue
        expectations = [
            expected_position(m + 1, p, q, a) for (p, q, a) in res.argmins
        ]
        pred_local = float(sum(expectations) / len(expectations))
        preds.append(pred_local + base)
        k = len(res.argmins)
        ps.append(sum(pt[0] for pt in res.argmins) / k)
        qs.append(sum(pt[1] for pt in res.argmins) / k)
        alphas.append(sum(pt[2] for pt in res.argmins) / k)
    return preds, ps, qs, alphas


def run_on_tradingview_rolling(
    src,
    B: float,
    *,
    window: int,
    start=None,
    end=None,
    tz: str | None = None,
    grid_size: int = 11,
    symmetric: bool = False,
    title: str | None = None,
    show: bool = True,
):
    """TradingView CSV → 平均練行足 → rolling-window 予測 → 3 つの Figure.

    Parameters
    ----------
    window :
        直近 W bricks だけを使って各ステップ fit する (W = ``window``)。
        小さい W ほど最近のレジームに敏感、大きい W ほど履歴全体を使う
        従来版に近い挙動になる。
    その他のパラメータは ``oyosuri_all_in_one.run_on_tradingview`` と同義。

    Returns
    -------
    (figs, result_dict)
        figs : {"price", "walk", "params"} の 3 Figure。
        result_dict : ``oyosuri_all_in_one`` 版のキーに加えて ``"window"``
        を含む。
    """
    df = load_tradingview_csv(src)
    df = slice_by_time(df, start=start, end=end, tz=tz)
    if isinstance(df.index, pd.DatetimeIndex) and len(df) > 0:
        range_str = f"{df.index[0]} → {df.index[-1]}"
    else:
        range_str = f"rows [0, {len(df)})"
    print(f"当てはめ範囲: {range_str}  (bars={len(df)})")
    bricks, origins = generate_mean_renko_from_ohlc(df, B=B)
    x = walk_from_bricks(bricks)
    preds, ps, qs, alphas = predict_sequence_with_params_rolling(
        x, window=window, grid_size=grid_size, symmetric=symmetric
    )
    print(f"N_bricks={len(bricks)}, window={window}")

    price_title = (
        f"{title} (window={window})"
        if title
        else f"Price (window={window})"
    )
    fig_price = plot_price(df, len(bricks), title=price_title)
    fig_walk = plot_walk_vs_prediction(x, preds)
    fig_params = plot_pq(ps, qs, alphas=alphas, symmetric=symmetric)
    figs = {"price": fig_price, "walk": fig_walk, "params": fig_params}

    if show:
        plt.show()
    if isinstance(df.index, pd.DatetimeIndex) and len(df) > 0:
        used_start, used_end = df.index[0], df.index[-1]
    else:
        used_start, used_end = 0, len(df)
    return figs, {
        "bricks": bricks,
        "origins": origins,
        "x": x,
        "preds": preds,
        "ps": ps,
        "qs": qs,
        "alphas": alphas,
        "window": window,
        "start": used_start, "end": used_end,
    }
