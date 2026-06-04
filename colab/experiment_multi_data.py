"""10 種類の多様な合成データに rolling-window 予測を当てはめ、
window=15/30/60 の予測を 1 枚に重ねて比較する実験スクリプト。
"""
import sys
import time

sys.path.insert(0, "colab")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd
from matplotlib import rcParams

# 日本語フォント (Noto CJK) を登録。無ければ既定フォントのまま。
for _fp in (
    "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
):
    try:
        fm.fontManager.addfont(_fp)
        rcParams["font.family"] = fm.FontProperties(fname=_fp).get_name()
        break
    except Exception:
        pass
rcParams["axes.unicode_minus"] = False

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


def gen_phases_var_sigma(segments, seed, base=1800.0):
    """各 segment 内で sigma が時間変化する版 (volatility cluster 等)。
    segments: list of (n_bars, drift, sigma_start, sigma_end)
    """
    rng = np.random.default_rng(seed)
    closes = [base]
    for n_bars, drift, s0, s1 in segments:
        sigmas = np.linspace(s0, s1, n_bars)
        for s in sigmas:
            closes.append(closes[-1] + rng.normal(drift, s))
    return ohlc_from_closes(closes, seed)


def gen_phases_var_drift(segments, seed, base=1800.0):
    """各 segment 内で drift が時間変化する版 (加速/減衰トレンド)。
    segments: list of (n_bars, drift_start, drift_end, sigma)
    """
    rng = np.random.default_rng(seed)
    closes = [base]
    for n_bars, d0, d1, sigma in segments:
        drifts = np.linspace(d0, d1, n_bars)
        for d in drifts:
            closes.append(closes[-1] + rng.normal(d, sigma))
    return ohlc_from_closes(closes, seed)


def gen_ou(n_bars, kappa, sigma, seed, base=1800.0, mean=1800.0):
    """Ornstein-Uhlenbeck 風: x_{t+1} = x_t + kappa*(mean - x_t) + N(0,sigma)."""
    rng = np.random.default_rng(seed)
    closes = [base]
    for _ in range(n_bars):
        prev = closes[-1]
        closes.append(prev + kappa * (mean - prev) + rng.normal(0, sigma))
    return ohlc_from_closes(closes, seed)


def gen_trend_sine(n_bars, drift, amp, period, sigma, seed, base=1800.0):
    """線形トレンド + 正弦波 + ノイズ。"""
    rng = np.random.default_rng(seed)
    t = np.arange(n_bars + 1)
    noise = np.concatenate([[0.0], rng.normal(0.0, sigma, n_bars)])
    closes = base + drift * t + amp * np.sin(2.0 * np.pi * t / period) + np.cumsum(noise)
    return ohlc_from_closes(list(closes), seed)


def gen_jumps(phases, jump_at, jump_size, seed, base=1800.0):
    """phases に従って生成しつつ、指定 step で離散的ジャンプを差し込む。
    jump_at: list[int] バー番号、jump_size: list[float] 同長
    """
    rng = np.random.default_rng(seed)
    closes = [base]
    step = 0
    for n_bars, drift, sigma in phases:
        for _ in range(n_bars):
            jump = 0.0
            for ja, js in zip(jump_at, jump_size):
                if step == ja:
                    jump = js
            closes.append(closes[-1] + rng.normal(drift, sigma) + jump)
            step += 1
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

# 追加 12 種類: 性格の異なる時系列
EXTRA_DATASETS = [
    ("11 緩やか上昇 (低S/N)",
     lambda s: gen_phases([(450, 0.06, 0.7)], s), 1.4),
    ("12 急騰&急落",
     lambda s: gen_phases([(150, 0.0, 0.6), (60, 0.55, 0.6),
                            (220, -0.22, 0.6)], s), 1.5),
    ("13 階段状上昇 (consol/impulse 交互)",
     lambda s: gen_phases([(70, 0.0, 0.5), (40, 0.40, 0.5),
                            (70, 0.0, 0.5), (40, 0.40, 0.5),
                            (70, 0.0, 0.5), (40, 0.40, 0.5)], s), 1.2),
    ("14 加速トレンド",
     lambda s: gen_phases_var_drift([(400, 0.02, 0.45, 0.7)], s), 1.8),
    ("15 減衰トレンド",
     lambda s: gen_phases_var_drift([(400, 0.45, 0.02, 0.7)], s), 1.8),
    ("16 平均回帰 (OU)",
     lambda s: gen_ou(500, kappa=0.05, sigma=1.0, seed=s), 2.0),
    ("17 トレンド+周期",
     lambda s: gen_trend_sine(500, drift=0.10, amp=8.0, period=120,
                               sigma=0.5, seed=s), 1.6),
    ("18 ジャンプ拡散",
     lambda s: gen_jumps([(450, 0.0, 0.7)],
                          jump_at=[120, 260, 360],
                          jump_size=[20.0, -25.0, 18.0], seed=s), 1.8),
    ("19 W字 (ダブルボトム)",
     lambda s: gen_phases([(100, -0.30, 0.6), (100, 0.30, 0.6),
                            (100, -0.30, 0.6), (100, 0.30, 0.6)], s), 1.6),
    ("20 M字 (ダブルトップ)",
     lambda s: gen_phases([(100, 0.30, 0.6), (100, -0.30, 0.6),
                            (100, 0.30, 0.6), (100, -0.30, 0.6)], s), 1.6),
    ("21 ボラクラスタ",
     lambda s: gen_phases_var_sigma([(200, 0.0, 0.4, 0.4),
                                      (120, 0.0, 0.4, 1.6),
                                      (200, 0.0, 1.6, 0.4)], s), 1.8),
    ("22 純粋ランダムウォーク",
     lambda s: gen_phases([(500, 0.0, 1.0)], s), 1.8),
]


WINDOWS = [15, 30, 60]
WCOLORS = {15: "#ff7f0e", 30: "#2ca02c", 60: "#d62728"}

GRID = 11


def _run_with_df(name, df, B):
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


def run_one(name, phases, B, seed):
    df = gen_phases(phases, seed)
    return _run_with_df(name, df, B)


def run_one_extra(name, df_builder, B, seed):
    df = df_builder(seed)
    return _run_with_df(name, df, B)


if __name__ == "__main__":
    import os

    run_extra_only = os.environ.get("RUN_EXTRA_ONLY") == "1"
    if not run_extra_only:
        for i, (name, phases, B) in enumerate(DATASETS):
            t0 = time.time()
            out, N = run_one(name, phases, B, seed=100 + i)
            print(f"{name}: N={N}  ({time.time()-t0:.0f}s)  -> {out}", flush=True)
    for i, (name, builder, B) in enumerate(EXTRA_DATASETS):
        t0 = time.time()
        out, N = run_one_extra(name, builder, B, seed=200 + i)
        print(f"{name}: N={N}  ({time.time()-t0:.0f}s)  -> {out}", flush=True)
    print("ALL DONE")
