"""発表スライド（docs/slides/）専用の図を生成するスクリプト.

生成物（既定の保存先は docs/slides/figures/）:

  dist_compare_rw_crw.png
      … 通常のランダムウォークと相関付きランダムウォークの確率分布
        $P(\\tilde{S}_n = x)$ を重ねた比較図（2 パネル: n が小さい場合と
        大きい場合）。1 歩あたりの上下の確率はどれも 1/2 で等しく，
        違いは「直前の向きとの相関」だけである。それでも分布の形が
        はっきり変わることを 1 枚で見せるための図。

対称（時間発展対称）CRW は本実装の ``(p, q, alpha) = (a, a, 1/2)`` に対応し，
``a = 1/2`` が通常の対称ランダムウォークそのものになる（このとき分布は
二項分布に一致する）。極限分散は $\\sigma^2 = a/(1-a)$（卒論 §4.3）。

使い方:

    python colab/make_slide_figs.py                 # 既定の n=10, 50
    python colab/make_slide_figs.py --n 8 40        # パネルの n を変える
"""

from __future__ import annotations

import argparse
import os as _os
import sys as _sys

import matplotlib

matplotlib.use("Agg")  # 画面の無い環境でも保存だけ行う

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

try:
    _HERE = _os.path.dirname(_os.path.abspath(__file__))
except NameError:  # __file__ が無い実行形態へのフォールバック
    _HERE = _os.getcwd()
if _HERE not in _sys.path:
    _sys.path.insert(0, _HERE)

from oyosuri_experiment import _setup_jp_font
from oyosuri_limit_figs import crw_distribution

DPI = 150

# (a, 線色, 線種, マーカー, 日本語ラベル, 英語ラベル)
SPECS = (
    (2.0 / 3.0, "black", "-", "^", "持続性：$a=2/3$", "persistent: a=2/3"),
    (0.5, "black", "--", "o", "通常のランダムウォーク：$a=1/2$",
     "ordinary RW: a=1/2"),
    (1.0 / 3.0, "0.55", "-", "s", "反転傾向：$a=1/3$", "anti-persistent: a=1/3"),
)


def plot_dist_compare(n_small: int = 10, n_large: int = 50, alpha: float = 0.5):
    """通常 RW と相関付き RW の分布を 2 つの時刻について並べた比較図を返す."""
    jp = _setup_jp_font()
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6))

    for ax, n in zip(axes, (n_small, n_large)):
        peak = 0.0
        x_edge = 2.0
        for a, color, ls, marker, lab_jp, lab_en in SPECS:
            x, mu = crw_distribution(a, n=n, alpha=alpha)
            ax.plot(
                x, mu, color=color, linestyle=ls, marker=marker,
                markersize=6.0, markerfacecolor="white", markeredgecolor=color,
                linewidth=1.9, label=(lab_jp if jp else lab_en), zorder=3,
            )
            peak = max(peak, float(mu.max()))
            # 確率がほぼ 0 の裾まで描くと図が横に間延びするので，
            # 見える範囲（1/1000 以上）を全系列でとった最大値に合わせる。
            visible = np.abs(x[mu > 1e-3])
            if visible.size:
                x_edge = max(x_edge, float(visible.max()))
        ax.set_title(f"$n={n}$", fontsize=13)
        ax.set_xlabel("位置 $x$" if jp else "position x", fontsize=12)
        ax.set_ylabel("確率 $P(\\tilde{S}_n = x)$" if jp else
                      "probability P(S_n = x)", fontsize=12)
        ax.set_xlim(-x_edge - 1.0, x_edge + 1.0)
        ax.set_ylim(0.0, peak * 1.18)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.tick_params(labelsize=11)
        ax.grid(alpha=0.3, linestyle=":")

    # 凡例は 1 つだけ（2 パネルで系列は共通）。裾が空く左上に置き，
    # 山と重ならないよう，そのパネルだけ上を広めにとる。
    axes[1].set_ylim(top=axes[1].get_ylim()[1] * 1.22)
    axes[1].legend(loc="upper left", fontsize=11)
    fig.tight_layout()
    return fig


def main(argv: list[str] | None = None) -> None:
    default_dir = _os.path.join(
        _os.path.dirname(_HERE), "docs", "slides", "figures")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--save-dir", default=default_dir)
    ap.add_argument("--n", nargs=2, type=int, default=[10, 50],
                    metavar=("N_SMALL", "N_LARGE"),
                    help="2 パネルそれぞれの時刻 n（既定: 10 50）")
    args = ap.parse_args(argv)

    _os.makedirs(args.save_dir, exist_ok=True)
    fig = plot_dist_compare(n_small=args.n[0], n_large=args.n[1])
    out = _os.path.join(args.save_dir, "dist_compare_rw_crw.png")
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
