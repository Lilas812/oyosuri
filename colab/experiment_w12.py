"""極端に短い W=1 と W=2 だけを観測ウォークと重ねた図を全 22 データに対し生成。
"""
import sys
import time

sys.path.insert(0, "colab")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiment_multi_data import DATASETS, EXTRA_DATASETS, gen_phases
from oyosuri_all_in_one import generate_mean_renko_from_ohlc, walk_from_bricks
from oyosuri_rolling import predict_sequence_with_params_rolling

GRID = 11
COLOR_W1 = "#9467bd"   # purple
COLOR_W2 = "#e377c2"   # pink


def run_w12(name, df, B):
    bricks, _ = generate_mean_renko_from_ohlc(df, B=B)
    x = walk_from_bricks(bricks)
    N = len(x) - 1

    t0 = time.time()
    preds_w1, *_ = predict_sequence_with_params_rolling(x, window=1, grid_size=GRID)
    t_w1 = time.time() - t0
    t0 = time.time()
    preds_w2, *_ = predict_sequence_with_params_rolling(x, window=2, grid_size=GRID)
    t_w2 = time.time() - t0

    fig, ax = plt.subplots(figsize=(13, 5))
    walk_n = np.arange(len(x))
    ax.step(walk_n, x, where="post", color="#1f77b4", linewidth=1.5,
            label="x_n (observed)", zorder=2)
    pred_n = np.arange(1, 1 + N)
    ax.plot(pred_n, preds_w1, color=COLOR_W1, linewidth=1.2,
            marker="o", markersize=2.2, alpha=0.85,
            label="x_n* (W=1)", zorder=3)
    ax.plot(pred_n, preds_w2, color=COLOR_W2, linewidth=1.2,
            marker="o", markersize=2.2, alpha=0.85,
            label="x_n* (W=2)", zorder=4)
    ax.set_title(f"{name}  (N_bricks={N})  W=1 vs W=2")
    ax.set_ylabel("x_n  (box units)")
    ax.set_xlabel("brick index n")
    ax.grid(alpha=0.3)
    ax.set_xlim(-0.5, max(N, 1) + 0.5)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=9,
              borderaxespad=0.0)
    fig.tight_layout()
    safe = name.split()[0]
    out = f"/tmp/multi_w12_{safe}.png"
    fig.savefig(out, dpi=105, bbox_inches="tight")
    plt.close(fig)
    return out, N, t_w1, t_w2


if __name__ == "__main__":
    for i, (name, phases, B) in enumerate(DATASETS):
        df = gen_phases(phases, seed=100 + i)
        out, N, t1, t2 = run_w12(name, df, B)
        print(f"{name}: N={N}  W1={t1:.0f}s W2={t2:.0f}s  -> {out}", flush=True)
    for i, (name, builder, B) in enumerate(EXTRA_DATASETS):
        df = builder(200 + i)
        out, N, t1, t2 = run_w12(name, df, B)
        print(f"{name}: N={N}  W1={t1:.0f}s W2={t2:.0f}s  -> {out}", flush=True)
    print("ALL DONE")
