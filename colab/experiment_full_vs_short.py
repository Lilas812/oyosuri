"""全期間参照 vs 極端に短い W=3 / W=10 を 1 枚に重ねた比較図を生成。
22 個 (DATASETS + EXTRA_DATASETS) すべてに対して走らせる。
"""
import sys
import time

sys.path.insert(0, "colab")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiment_multi_data import (
    DATASETS,
    EXTRA_DATASETS,
    gen_phases,
)
from oyosuri_all_in_one import (
    generate_mean_renko_from_ohlc,
    predict_sequence_with_params,
    walk_from_bricks,
)
from oyosuri_rolling import predict_sequence_with_params_rolling

GRID = 11
SHORT_WS = [3, 10]
COLOR_FULL = "#d62728"   # red
COLOR_W3 = "#ff7f0e"     # orange
COLOR_W10 = "#2ca02c"    # green


def run_cmp(name, df, B):
    bricks, _ = generate_mean_renko_from_ohlc(df, B=B)
    x = walk_from_bricks(bricks)
    N = len(x) - 1

    t0 = time.time()
    preds_w3, *_ = predict_sequence_with_params_rolling(x, window=3, grid_size=GRID)
    t_w3 = time.time() - t0
    t0 = time.time()
    preds_w10, *_ = predict_sequence_with_params_rolling(x, window=10, grid_size=GRID)
    t_w10 = time.time() - t0
    t0 = time.time()
    preds_full, *_ = predict_sequence_with_params(x, grid_size=GRID)
    t_full = time.time() - t0

    fig, ax = plt.subplots(figsize=(13, 5))
    walk_n = np.arange(len(x))
    ax.step(walk_n, x, where="post", color="#1f77b4", linewidth=1.5,
            label="x_n (observed)", zorder=2)
    pred_n = np.arange(1, 1 + N)
    ax.plot(pred_n, preds_w3, color=COLOR_W3, linewidth=1.2,
            marker="o", markersize=2.2, alpha=0.85,
            label="x_n* (W=3)", zorder=3)
    ax.plot(pred_n, preds_w10, color=COLOR_W10, linewidth=1.2,
            marker="o", markersize=2.2, alpha=0.85,
            label="x_n* (W=10)", zorder=4)
    ax.plot(pred_n, preds_full, color=COLOR_FULL, linewidth=1.4,
            marker="o", markersize=2.4, alpha=0.9,
            label="x_n* (全期間)", zorder=5)
    ax.set_title(f"{name}  (N_bricks={N})  全期間 vs 短期 W=3/10")
    ax.set_ylabel("x_n  (box units)")
    ax.set_xlabel("brick index n")
    ax.grid(alpha=0.3)
    ax.set_xlim(-0.5, max(N, 1) + 0.5)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=9,
              borderaxespad=0.0)
    fig.tight_layout()
    safe = name.split()[0]
    out = f"/tmp/multi_cmp_{safe}.png"
    fig.savefig(out, dpi=105, bbox_inches="tight")
    plt.close(fig)
    return out, N, t_w3, t_w10, t_full


if __name__ == "__main__":
    for i, (name, phases, B) in enumerate(DATASETS):
        df = gen_phases(phases, seed=100 + i)
        out, N, t3, t10, tf = run_cmp(name, df, B)
        print(f"{name}: N={N}  W3={t3:.0f}s W10={t10:.0f}s full={tf:.0f}s  -> {out}",
              flush=True)
    for i, (name, builder, B) in enumerate(EXTRA_DATASETS):
        df = builder(200 + i)
        out, N, t3, t10, tf = run_cmp(name, df, B)
        print(f"{name}: N={N}  W3={t3:.0f}s W10={t10:.0f}s full={tf:.0f}s  -> {out}",
              flush=True)
    print("ALL DONE")
