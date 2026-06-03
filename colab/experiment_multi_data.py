"""10 種類の多様な合成データに rolling-window 予測を当てはめ、
window=15/30/60 の予測を 1 枚に重ねて比較する実験スクリプト。
"""
import sys
import time

sys.path.insert(0, "colab")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from oyosuri_all_in_one import generate_mean_renko_from_ohlc, walk_from_bricks
from oyosuri_rolling import predict_sequence_with_params_rolling


def ohlc_from_closes(closes, seed):
    rng = np.random.default_rng(seed + 9999)
    closes = np.asarray(closes, dtype=float)
    highs = closes + rng.uniform(0.1, 1.0, size=len(closes))
    lows = closes - rng.uniform(0.1, 1.0, size=len(closes))
    opens = np.concatenate([[closes[0]], closes[:-1]])
    idx = pd.date_range("2024-01-01", periods=len(closes), freq="h", tz="UTC")
    return pd.DataFrame(
        {"Open": opens, "High": highs, "Low": lows, "Close": closes}, index=idx
    )


def gen_phases(phases, seed, base=1800.0):
    """phases: list of (n_bars, drift, sigma)."""
    rng = np.random.default_rng(seed)
    closes = [base]
    for n_bars, drift, sigma in phases:
        for _ in range(n_bars):
            closes.append(closes[-1] + rng.normal(drift, sigma))
    return ohlc_from_closes(closes, seed)


# 10 種類の多様なデータ定義 (name, phases, B)
DATASETS = [
    ("01 強い上昇トレンド",      [(400, 0.30, 0.7)], 2.0),
    ("02 強い下降トレンド",      [(400, -0.30, 0.7)], 2.0),
    ("03 レンジ (ドリフト無)",   [(500, 0.0, 0.8)], 1.5),
    ("04 V字 (下→上)",          [(220, -0.28, 0.7), (220, 0.28, 0.7)], 1.8),
    ("05 逆V字 (上→下)",        [(220, 0.28, 0.7), (220, -0.28, 0.7)], 1.8),
    ("06 ブレイクアウト (レンジ→上昇)", [(260, 0.0, 0.7), (220, 0.32, 0.7)], 1.8),
    ("07 トレンド終焉 (上昇→レンジ)",   [(220, 0.32, 0.7), (260, 0.0, 0.7)], 1.8),
    ("08 高ボラ・ノイジー",      [(420, 0.0, 1.5)], 2.5),
    ("09 低ボラ・滑らか上昇",    [(450, 0.18, 0.35)], 1.2),
    ("10 複数レジーム切替",      [(120, 0.0, 0.7), (110, -0.26, 0.7),
                                  (110, 0.24, 0.7), (120, 0.0, 0.7)], 1.6),
]

WINDOWS = [15, 30, 60]
WCOLORS = {15: "#ff7f0e", 30: "#2ca02c", 60: "#d62728"}

GRID = 11


def run_one(name, phases, B, seed):
    df = gen_phases(phases, seed)
    bricks, _ = generate_mean_renko_from_ohlc(df, B=B)
    x = walk_from_bricks(bricks)
    N = len(x) - 1

    fig, ax = plt.subplots(figsize=(13, 5))
    walk_n = np.arange(len(x))
    ax.step(walk_n, x, where="post", color="#1f77b4", linewidth=1.5,
            label="x_n (observed)", zorder=2)
    for w in WINDOWS:
        preds, _, _, _ = predict_sequence_with_params_rolling(
            x, window=w, grid_size=GRID
        )
        pred_n = np.arange(1, 1 + len(preds))
        ax.plot(pred_n, preds, color=WCOLORS[w], linewidth=1.3,
                marker="o", markersize=2.5, alpha=0.85,
                label=f"x_n* (W={w})", zorder=3)
    ax.set_title(f"{name}  (N_bricks={N})")
    ax.set_ylabel("x_n  (box units)")
    ax.set_xlabel("brick index n")
    ax.grid(alpha=0.3)
    ax.set_xlim(-0.5, max(N, 1) + 0.5)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=9,
              borderaxespad=0.0)
    fig.tight_layout()
    safe = name.split()[0]
    out = f"/tmp/multi_{safe}.png"
    fig.savefig(out, dpi=105, bbox_inches="tight")
    plt.close(fig)
    return out, N


if __name__ == "__main__":
    for i, (name, phases, B) in enumerate(DATASETS):
        t0 = time.time()
        out, N = run_one(name, phases, B, seed=100 + i)
        print(f"{name}: N={N}  ({time.time()-t0:.0f}s)  -> {out}", flush=True)
    print("ALL DONE")
