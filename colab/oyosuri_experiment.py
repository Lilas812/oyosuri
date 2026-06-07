"""チャートパターン別の予測図 — 卒論 §7 用.

``oyosuri_all_in_one.py`` と ``oyosuri_rolling.py`` を import し、ある区間
（1チャートパターン）について 1 ステップ先予測を行い、実測ウォーク x_n と
予測 x_n* を同一図に重ねて出力する。判断はこの図を目視で行い、平均絶対誤差
MAE は補助（飾り）として併記する。

**参照期間は自由に選べる**（``run_multi_w_overlay`` の ``windows`` で指定）:
  - 整数 W      … 直近 W 本だけで当てはめる（rolling 窓）
  - "full"/None … その時点までの過去全期間で当てはめる（第5章の元モデル）
``windows`` は最大3つまで同一図に重ねられる。

    windows=[10, 30, 60]      → 直近10/30/60本 の3本を重ねる
    windows=["full"]          → 全期間参照の予測1本だけ
    windows=[10, 30, "full"]  → 直近10・直近30・全期間 を重ねる（混在）

論文の白黒印刷でも判別できるよう、実測＝太い黒実線、各系列＝破線/点線/一点鎖線
＋ ○/□/△ の白抜きマーカーで区別する（色に依存しない）。

MAE = (1/N) Σ_n |x_n − x_n*|。±1 ウォークなので「動かない予測」の MAE は
恒等的に 1（MAE<1 で「動かない予測」に優る目安）。

Colab で GitHub から実行する場合:

    # 1) クローンして colab/ を import パスに追加
    !git clone https://github.com/Lilas812/oyosuri.git
    import sys; sys.path.insert(0, "/content/oyosuri/colab")

    # 2) import（依存は自動解決）
    from oyosuri_experiment import run_multi_w_overlay, run_mae_sweep

    # 3) CSV アップロード
    from google.colab import files
    up = files.upload(); path = next(iter(up))

    # 4a) 3つの W を重ねる
    fig, r = run_multi_w_overlay(path, B=4, windows=[10, 30, 60],
                                 start="2026-05-19-15:00", end="2026-05-19-23:58",
                                 tz="Asia/Tokyo")
    # 4b) 全期間参照の予測1本だけ
    fig, r = run_multi_w_overlay(path, B=4, windows=["full"],
                                 start="2026-05-19-15:00", end="2026-05-19-23:58",
                                 tz="Asia/Tokyo")
    fig.savefig("pat_uptrend.png", dpi=150, bbox_inches="tight")  # 論文用に保存

（ローカル/旧来の %run でも可:
    %run colab/oyosuri_all_in_one.py → %run colab/oyosuri_rolling.py →
    %run colab/oyosuri_experiment.py）
"""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# このファイルのあるディレクトリ（colab/）を import パスへ追加し、GitHub clone 後に
# どの作業ディレクトリからでも兄弟モジュール (oyosuri_all_in_one / _rolling) を解決する。
import os as _os
import sys as _sys
try:
    _HERE = _os.path.dirname(_os.path.abspath(__file__))
except NameError:  # __file__ が無い実行形態へのフォールバック
    _HERE = _os.getcwd()
if _HERE not in _sys.path:
    _sys.path.insert(0, _HERE)

from oyosuri_all_in_one import (
    generate_mean_renko_from_ohlc,
    load_tradingview_csv,
    mae,
    predict_sequence,
    slice_by_time,
    walk_from_bricks,
)
from oyosuri_rolling import predict_sequence_with_params_rolling


_FULL_TOKENS = ("full", "all", "∞", "inf")


def _is_full(W, N: int) -> bool:
    """W が「過去全期間参照」を意味するか（None / "full" / N 以上の整数）."""
    if W is None:
        return True
    if isinstance(W, str):
        return W.strip().lower() in _FULL_TOKENS
    return int(W) >= N


def _preds_for_window(
    x: Sequence[int],
    W,
    *,
    grid_size: int = 11,
    symmetric: bool = False,
) -> list[float]:
    """参照期間 W の 1 ステップ先予測列を返す.

    W が full（None / "full" / N 以上）のときは全期間版 (``predict_sequence``)、
    そうでなければ直近 W bricks の rolling 版。
    """
    N = len(list(x)) - 1
    if _is_full(W, N):
        return predict_sequence(x, grid_size=grid_size, symmetric=symmetric)
    return predict_sequence_with_params_rolling(
        x, window=int(W), grid_size=grid_size, symmetric=symmetric
    )[0]


def mae_by_window(
    x: Sequence[int],
    windows: Sequence,
    *,
    include_full: bool = True,
    grid_size: int = 11,
    symmetric: bool = False,
) -> dict[str, float]:
    """各参照期間 W の MAE を ``{W: MAE}`` で返す（数値だけ欲しいとき）.

    ``windows`` の各要素は整数（rolling）または "full"/None（全期間）。
    ``include_full=True`` なら全期間版（label ``"full"``）も必ず加える。
    """
    x = list(x)
    N = len(x) - 1
    if N < 1:
        raise ValueError("x must contain at least 2 points")
    actual = [float(v) for v in x[1:]]

    out: dict[str, float] = {}
    for W in windows:
        label = "full" if _is_full(W, N) else str(int(W))
        if label in out:
            continue
        out[label] = mae(actual, _preds_for_window(
            x, W, grid_size=grid_size, symmetric=symmetric))
    if include_full and "full" not in out:
        out["full"] = mae(actual, predict_sequence(
            x, grid_size=grid_size, symmetric=symmetric))
    return out


def run_mae_sweep(
    src,
    B: float,
    *,
    windows: Sequence = (2, 3, 5, 10, 20, 40),
    include_full: bool = True,
    start=None,
    end=None,
    tz: str | None = None,
    grid_size: int = 11,
    symmetric: bool = False,
) -> dict:
    """CSV → 平均練行足 → 参照期間ごとの MAE を数値表示（グラフなし）.

    Returns {"x", "bricks", "N", "mae": {W: MAE}}.
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
    """実測ウォーク x_n と、最大3つの系列の予測 x_n* を同一図に重ねて返す.

    論文の白黒印刷でも判別できるよう、系列は色ではなく「線種＋マーカー形状」で
    区別する（実測＝太い黒実線・マーカー無し、各系列＝破線/点線/一点鎖線 ＋
    ○/□/△ の白抜きマーカー）。判断はこの図を目視で行い、凡例に W と
    （あれば）補助の MAE を併記する。
    """
    x = list(x)
    N = max(len(x) - 1, 0)
    fig, ax = plt.subplots(figsize=(13, 5))

    walk_n = np.arange(len(x))
    ax.step(
        walk_n, x, where="post",
        color="black", linewidth=2.2, label="x_n (observed)", zorder=2,
    )

    # 白黒で判別するための「線種・マーカー形状」の組（色には依存しない）
    linestyles = ["--", ":", "-."]
    markers = ["o", "s", "^"]
    grays = ["black", "0.45", "black"]
    me = max(1, N // 15)  # マーカーを間引いて重なりを避ける
    for i, (label, preds) in enumerate(preds_by_w.items()):
        preds = list(preds)
        pred_n = np.arange(1, 1 + len(preds))
        leg = "full (whole history)" if label == "full" else f"W={label}"
        if mae_by_w is not None and label in mae_by_w:
            leg += f"  (MAE={mae_by_w[label]:.3f})"
        ax.plot(
            pred_n, preds,
            color=grays[i % len(grays)],
            linestyle=linestyles[i % len(linestyles)],
            marker=markers[i % len(markers)],
            markersize=5, markerfacecolor="white", markeredgewidth=1.1,
            markevery=me, linewidth=1.5, label=leg, zorder=3,
        )

    ax.set_title(title or "Observed walk vs prediction")
    ax.set_xlabel("brick index n")
    ax.set_ylabel("x_n  (box units)")
    ax.set_xlim(-0.5, max(N, 1) + 0.5)
    ax.grid(alpha=0.3, linestyle=":")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=9,
              borderaxespad=0.0)
    fig.tight_layout()
    return fig


def run_multi_w_overlay(
    src,
    B: float,
    *,
    windows: Sequence = (10, 30, 60),
    start=None,
    end=None,
    tz: str | None = None,
    grid_size: int = 11,
    symmetric: bool = False,
    title: str | None = None,
    show: bool = True,
):
    """CSV → 平均練行足 → 指定した参照期間の予測を実測と同一グラフに重ねて出力.

    ``windows`` は最大3つ。各要素は整数（直近 W 本＝rolling）または
    "full"/None（過去全期間参照）。
      - windows=[10, 30, 60]     … 直近10/30/60本 を3本重ね
      - windows=["full"]         … 全期間参照の予測1本だけ
      - windows=[10, 30, "full"] … 混在

    Returns
    -------
    (fig, result)
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
        label = "full" if _is_full(W, N) else str(int(W))
        if label in preds_by_w:
            continue
        preds = _preds_for_window(x, W, grid_size=grid_size, symmetric=symmetric)
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
