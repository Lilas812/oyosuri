"""参照期間（ローリング窓 W）スイープ実験 — 卒論 §7 用.

``oyosuri_all_in_one.py`` と ``oyosuri_rolling.py`` の機能を import し、
**同一データ・同一ボックス幅 B** に対して複数の参照期間 W で予測を回し、
予測精度（MSE / hit_rate / MAE）を **W の関数** として比較する。

狙い（§7 の物語の背骨）:
    まず両極端の W を見せる ―― W が極小だと直近に過敏でノイジー、
    W=全履歴だとレジーム転換に鈍い。どちらも MSE が無情報基準 1 付近〜超で
    「効いていない」。だから中間の W を探す、という流れを 1 枚の図で示す。

評価指標の約束:
    「偏差」＝ 予測と実測値の差（予測誤差 ``x_n − x_n*``）。
    MSE = (1/N) Σ (x_n − x_n*)^2。±1 ウォークなので no-change 予測の MSE は
    恒等的に 1。よって MSE < 1 で初めて方向情報を取れている。

Colab での典型的な使い方 (別セルで順に %run):

    %run colab/oyosuri_all_in_one.py
    %run colab/oyosuri_rolling.py
    %run colab/oyosuri_experiment.py

    figs, res = run_window_sweep(
        path, B=4,
        windows=[2, 3, 5, 10, 20, 40],   # 両極端を含めると物語が立つ
        start="2026-05-19-15:00",
        end="2026-05-19-23:58",
        tz="Asia/Tokyo",
    )
    # res["rows"] に各 W の指標、res["best"] に最良 W が入る。
"""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from oyosuri_all_in_one import (
    generate_mean_renko_from_ohlc,
    hit_rate,
    load_tradingview_csv,
    mae,
    mse,
    mse_skill,
    plot_walk_vs_prediction,
    predict_sequence_with_params,
    slice_by_time,
    walk_from_bricks,
)
from oyosuri_rolling import predict_sequence_with_params_rolling


# ----------------------------------------------------------------------------
# ベースライン予測（無情報の比較対象）
# ----------------------------------------------------------------------------

def baseline_preds(x: Sequence[int]) -> tuple[list[float], list[float]]:
    """no-change と momentum のベースライン予測列を返す.

    どちらも actual = x[1:]（= x_1..x_N）に整列した長さ N の列。
      - no-change : x_{n+1}* = x_n                （MSE は ±1 walk で恒等的に 1）
      - momentum  : x_{n+1}* = x_n + (x_n − x_{n-1})（直前の動きをそのまま延長）
    momentum の初手は直前ブリックが無いので no-change で代用する。
    """
    x = list(x)
    N = len(x) - 1
    nochange = [float(x[n]) for n in range(N)]
    momentum: list[float] = []
    for n in range(N):
        if n == 0:
            momentum.append(float(x[0]))
        else:
            momentum.append(float(2 * x[n] - x[n - 1]))
    return nochange, momentum


# ----------------------------------------------------------------------------
# 窓スイープ本体
# ----------------------------------------------------------------------------

def evaluate_windows(
    x: Sequence[int],
    windows: Sequence[int | None],
    *,
    grid_size: int = 11,
    symmetric: bool = False,
) -> tuple[list[dict], dict]:
    """各参照期間 W で 1 ステップ先予測を回し、指標行のリストを返す.

    ``windows`` の要素が ``None`` または ``>= N`` のときは全履歴版
    (``predict_sequence_with_params``) を使い、label を ``"full"`` とする。

    Returns
    -------
    (rows, baselines)
        rows : 各 W の dict
            {"label", "window", "mse", "hit", "mae", "skill",
             "preds", "ps", "qs", "alphas"}
        baselines : {"actual", "nochange", "momentum",
                     "base_mse", "momentum_mse"}
    """
    x = list(x)
    N = len(x) - 1
    if N < 1:
        raise ValueError("x must contain at least 2 points")
    actual = [float(v) for v in x[1:]]
    nochange, momentum = baseline_preds(x)
    base_mse = mse(actual, nochange)  # ±1 walk なら厳密に 1.0

    rows: list[dict] = []
    seen_labels: set[str] = set()
    for W in windows:
        if W is None or int(W) >= N:
            preds, ps, qs, alphas = predict_sequence_with_params(
                x, grid_size=grid_size, symmetric=symmetric
            )
            label, wval = "full", N
        else:
            wv = int(W)
            if wv < 1:
                continue
            preds, ps, qs, alphas = predict_sequence_with_params_rolling(
                x, window=wv, grid_size=grid_size, symmetric=symmetric
            )
            label, wval = str(wv), wv
        if label in seen_labels:
            continue
        seen_labels.add(label)
        m = mse(actual, preds)
        rows.append(
            {
                "label": label,
                "window": wval,
                "mse": m,
                "hit": hit_rate(actual, preds),
                "mae": mae(actual, preds),
                "skill": mse_skill(actual, preds, nochange),
                "preds": preds,
                "ps": ps,
                "qs": qs,
                "alphas": alphas,
            }
        )
    rows.sort(key=lambda r: r["window"])
    baselines = {
        "actual": actual,
        "nochange": nochange,
        "momentum": momentum,
        "base_mse": base_mse,
        "momentum_mse": mse(actual, momentum),
    }
    return rows, baselines


def print_summary(rows: list[dict], baselines: dict) -> None:
    """W ごとの指標とベースラインをそろえて表示する."""
    print(f"{'W':>6} | {'MSE':>8} | {'skill':>7} | {'hit':>6} | {'MAE':>7}")
    print("-" * 46)
    best = min(rows, key=lambda r: r["mse"]) if rows else None
    for r in rows:
        mark = "  <- best" if r is best else ""
        print(
            f"{r['label']:>6} | {r['mse']:>8.4f} | {r['skill']:>7.3f} | "
            f"{r['hit']:>6.3f} | {r['mae']:>7.4f}{mark}"
        )
    print("-" * 46)
    print(
        f"{'base':>6} | {baselines['base_mse']:>8.4f} |   (no-change, 基準=1)"
    )
    print(
        f"{'momtm':>6} | {baselines['momentum_mse']:>8.4f} |   (momentum)"
    )


def plot_window_sweep(
    rows: list[dict],
    baselines: dict,
    *,
    title: str | None = None,
):
    """MSE と hit_rate を参照期間 W の関数として描いた Figure を返す（§7.4 の図）.

    左軸 = MSE（赤）, 右軸 = hit rate（青）。
    水平線: MSE=1（no-change 基準）, momentum の MSE, hit=0.5（コイン投げ）。
    """
    pos = np.arange(len(rows))
    labels = [r["label"] for r in rows]
    mses = [r["mse"] for r in rows]
    hits = [r["hit"] for r in rows]

    fig, ax1 = plt.subplots(figsize=(11, 5))
    c_mse, c_hit = "#d62728", "#1f77b4"

    ax1.plot(pos, mses, color=c_mse, marker="o", linewidth=1.6, label="MSE")
    ax1.axhline(1.0, color=c_mse, linestyle=":", linewidth=1.0, alpha=0.7,
                label="MSE=1 (no-change)")
    ax1.axhline(baselines["momentum_mse"], color="#9467bd", linestyle="--",
                linewidth=1.0, alpha=0.7, label="MSE (momentum)")
    ax1.set_xlabel("reference period W  (rolling window;  'full' = whole history)")
    ax1.set_ylabel("MSE", color=c_mse)
    ax1.tick_params(axis="y", labelcolor=c_mse)
    ax1.set_xticks(pos)
    ax1.set_xticklabels(labels)
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(pos, hits, color=c_hit, marker="s", linewidth=1.6, label="hit rate")
    ax2.axhline(0.5, color=c_hit, linestyle=":", linewidth=1.0, alpha=0.6,
                label="hit=0.5 (coin flip)")
    ax2.set_ylabel("hit rate", color=c_hit)
    ax2.tick_params(axis="y", labelcolor=c_hit)
    ax2.set_ylim(0.0, 1.0)

    ax1.set_title(title or "Prediction quality vs reference period W")
    lines1, lab1 = ax1.get_legend_handles_labels()
    lines2, lab2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, lab1 + lab2, loc="upper left",
               bbox_to_anchor=(1.08, 1.0), fontsize=8, borderaxespad=0.0)
    fig.tight_layout()
    return fig


def _select_overlays(rows: list[dict]) -> dict[str, dict]:
    """walk-vs-prediction を重ねて見せる代表 W を選ぶ: 極小 / 最良 / 全履歴."""
    out: dict[str, dict] = {}
    finite = [r for r in rows if r["label"] != "full"]
    if finite:
        out["smallest_W"] = min(finite, key=lambda r: r["window"])
    if rows:
        out["best_W"] = min(rows, key=lambda r: r["mse"])
    full = [r for r in rows if r["label"] == "full"]
    if full:
        out["full"] = full[0]
    # 同じ行が複数タグに入ったら最初のタグだけ残す
    seen: set[str] = set()
    deduped: dict[str, dict] = {}
    for tag, r in out.items():
        if r["label"] in seen:
            continue
        seen.add(r["label"])
        deduped[tag] = r
    return deduped


def run_window_sweep(
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
    title: str | None = None,
    show: bool = True,
):
    """TradingView CSV → 平均練行足 → 参照期間 W スイープ → 図と指標表.

    Parameters
    ----------
    windows :
        評価する参照期間（ローリング窓）W のリスト。両極端（極小と
        全履歴に近い大きな値）を含めると §7 の物語が立つ。
    include_full :
        True なら全履歴版（W=N）も評価して比較に加える。
    その他のパラメータは ``oyosuri_all_in_one.run_on_tradingview`` と同義。

    Returns
    -------
    (figs, result)
        figs : {"sweep": Figure, "walk_<tag>": Figure, ...}
            "sweep" が §7.4 の MSE/hit vs W 図。walk_* は代表 W の
            観測 vs 予測オーバーレイ（極小 / 最良 / 全履歴）。
        result : {"x", "bricks", "rows", "baselines", "best", "N"}
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
    N = len(bricks)
    print(f"N_bricks={N}, B={B}")
    if N < 2:
        raise ValueError(
            f"ブリック数が少なすぎます (N={N})。B を小さくするか期間を広げてください。"
        )

    win_list: list[int | None] = [int(w) for w in windows if 1 <= int(w) < N]
    if include_full:
        win_list.append(None)
    rows, baselines = evaluate_windows(
        x, win_list, grid_size=grid_size, symmetric=symmetric
    )
    print_summary(rows, baselines)
    best = min(rows, key=lambda r: r["mse"]) if rows else None
    if best is not None:
        print(
            f"\n最良の参照期間: W={best['label']} "
            f"(MSE={best['mse']:.4f}, skill={best['skill']:.3f}, "
            f"hit={best['hit']:.3f})"
        )

    sweep_title = (
        f"{title} — MSE/hit vs W" if title
        else "Prediction quality vs reference period W"
    )
    figs: dict = {"sweep": plot_window_sweep(rows, baselines, title=sweep_title)}
    for tag, r in _select_overlays(rows).items():
        fig = plot_walk_vs_prediction(x, r["preds"])
        fig.axes[0].set_title(f"Integer walk vs prediction  (W={r['label']})")
        figs[f"walk_{tag}"] = fig

    if show:
        plt.show()
    return figs, {
        "x": x,
        "bricks": bricks,
        "origins": origins,
        "rows": rows,
        "baselines": baselines,
        "best": best,
        "N": N,
    }
