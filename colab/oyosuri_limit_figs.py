"""相関付きランダムウォークの分布図（n=100 ヒストグラム＋標準正規）— 卒論 §4 用.

弱収束極限定理（§4.3）

    S̃_n / √n  ⇒  N(0, a/(1-a))     (n → ∞)

を目視で裏づける図を生成する。時刻 ``n``（既定 100）における CRW の厳密分布
``μ_n(x) = P(S̃_n = x)`` を ``oyosuri_all_in_one.psi_forward`` で計算し，
再スケール変数 ``u = S̃_n/√n`` 上のヒストグラム（棒）として描く。これに

  - 標準正規 ``N(0, 1)``       … 通常ランダムウォークの極限（σ²=1）。破線。
  - CRW 自身の極限 ``N(0, a/(1-a))`` … 収束先。実線。

を重ねることで，持続性 ``a > 1/2`` では分布が標準正規より広がり（σ²>1），
反転傾向 ``a < 1/2`` では狭まる（σ²<1）ことが一目で分かる。

対称（時間発展対称）CRW は本実装の ``(p, q, α) = (a, a, 1/2)`` に対応する
（``coin_matrices`` の P, Q が論文 §4 の対称 P, Q に一致）。

Colab での使い方:

    !git clone https://github.com/Lilas812/oyosuri.git
    import sys; sys.path.insert(0, "/content/oyosuri/colab")
    !pip install -q matplotlib-fontja          # 日本語ラベル（任意）

    from oyosuri_limit_figs import plot_crw_hist_vs_normal, run_limit_figs

    fig = plot_crw_hist_vs_normal(a=2/3, n=100)     # 1枚（持続性）
    fig.savefig("crw_limit_persistent.png", dpi=150, bbox_inches="tight")

    figs = run_limit_figs(n=100, save_dir=".")       # 持続性/反転/重ね の3枚
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

# このファイルのあるディレクトリ（colab/）を import パスへ追加し、GitHub clone 後に
# どの作業ディレクトリからでも兄弟モジュールを解決する。
import os as _os
import sys as _sys
try:
    _HERE = _os.path.dirname(_os.path.abspath(__file__))
except NameError:  # __file__ が無い実行形態へのフォールバック
    _HERE = _os.getcwd()
if _HERE not in _sys.path:
    _sys.path.insert(0, _HERE)

from oyosuri_all_in_one import mu_from_psi, psi_forward
from oyosuri_experiment import _setup_jp_font


def crw_distribution(a: float, n: int = 100, alpha: float = 0.5):
    """対称 CRW の時刻 n における厳密分布を返す.

    Returns
    -------
    (x_even, mu_even)
        ``x_even`` は到達可能な位置（n と同じ偶奇の整数），
        ``mu_even`` は対応する確率質量 ``μ_n(x)``（和が 1）。
    """
    if not 0.0 < a < 1.0:
        raise ValueError("a must satisfy 0 < a < 1")
    if n < 1:
        raise ValueError("n must be >= 1")
    psi = psi_forward(n, p=a, q=a, alpha=alpha)
    mu_full = mu_from_psi(psi)[n]                 # 長さ 2n+1（x = -n..n）
    x_full = np.arange(-n, n + 1, dtype=float)
    reachable = ((np.arange(2 * n + 1) + n) % 2) == 0  # n と同じ偶奇のみ非ゼロ
    return x_full[reachable], mu_full[reachable]


def _normal_pdf(u: np.ndarray, var: float) -> np.ndarray:
    return np.exp(-u * u / (2.0 * var)) / np.sqrt(2.0 * np.pi * var)


def plot_crw_hist_vs_normal(
    a: float,
    n: int = 100,
    *,
    alpha: float = 0.5,
    show_own_limit: bool = True,
    title: str | None = None,
):
    """時刻 n の CRW 分布を u=S̃_n/√n 上のヒストグラムで描き，正規分布を重ねる.

    棒は確率質量を密度に直して（ビン幅 ``2/√n`` で割って）描くので，
    重ねた正規分布の密度曲線と縦軸スケールが一致する。
    """
    jp = _setup_jp_font()
    x_even, mu = crw_distribution(a, n=n, alpha=alpha)
    sqrt_n = np.sqrt(n)
    u = x_even / sqrt_n                 # 再スケール位置
    bin_w = 2.0 / sqrt_n               # 到達可能 x の間隔 2 を再スケール
    dens = mu / bin_w                  # 質量 → 密度

    sigma2 = a / (1.0 - a)            # CRW の極限分散
    grid = np.linspace(u.min() - 0.3, u.max() + 0.3, 600)

    obs_lab = "CRW の分布" if jp else "CRW dist."
    std_lab = "標準正規 $N(0,1)$" if jp else "$N(0,1)$"
    lim_lab = (f"極限 $N(0,{sigma2:.2f})$" if jp else f"limit $N(0,{sigma2:.2f})$")

    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    ax.bar(
        u, dens, width=bin_w * 0.92,
        color="0.78", edgecolor="0.35", linewidth=0.4,
        label=f"{obs_lab}（$n={n}$）" if jp else f"{obs_lab} (n={n})",
        zorder=2,
    )
    ax.plot(grid, _normal_pdf(grid, 1.0), color="black", linestyle="--",
            linewidth=1.8, label=std_lab, zorder=3)
    if show_own_limit:
        ax.plot(grid, _normal_pdf(grid, sigma2), color="black", linestyle="-",
                linewidth=1.6, label=lim_lab, zorder=3)

    if title is None:
        kind = ("持続性" if a > 0.5 else "反転傾向" if a < 0.5 else "対称") if jp \
            else ("persistent" if a > 0.5 else "anti-persistent" if a < 0.5 else "symmetric")
        title = (f"相関付きランダムウォークの分布（{kind}：$a={a:.3g}$，$\\sigma^2={sigma2:.2f}$）"
                 if jp else
                 f"CRW distribution ({kind}: a={a:.3g}, sigma^2={sigma2:.2f})")
    ax.set_title(title)
    ax.set_xlabel(r"$u = \tilde{S}_n/\sqrt{n}$")
    ax.set_ylabel("確率密度" if jp else "density")
    ax.grid(alpha=0.3, linestyle=":")
    ax.legend(loc="upper right", fontsize=10)
    fig.tight_layout()
    return fig


def run_limit_figs(
    n: int = 100,
    *,
    a_persistent: float = 2.0 / 3.0,
    a_antipersistent: float = 1.0 / 3.0,
    save_dir: str | None = None,
    dpi: int = 150,
    show: bool = True,
):
    """持続性・反転傾向の2枚＋両者重ねの計3枚を生成する.

    ``save_dir`` 指定時は ``crw_limit_persistent.png`` /
    ``crw_limit_antipersistent.png`` / ``crw_hist_n100.png`` で保存（前2つは
    既存 ``\\includegraphics`` と同名なので tex 無修正で差し替え可能）。
    """
    figs: dict[str, plt.Figure] = {}
    figs["persistent"] = plot_crw_hist_vs_normal(a_persistent, n=n)
    figs["antipersistent"] = plot_crw_hist_vs_normal(a_antipersistent, n=n)
    figs["combined"] = _plot_combined(a_persistent, a_antipersistent, n=n)

    if save_dir is not None:
        names = {
            "persistent": "crw_limit_persistent.png",
            "antipersistent": "crw_limit_antipersistent.png",
            "combined": "crw_hist_n100.png",
        }
        for key, fig in figs.items():
            out = _os.path.join(save_dir, names[key])
            fig.savefig(out, dpi=dpi, bbox_inches="tight")
            print(f"saved: {out}")
    if show:
        plt.show()
    elif save_dir is not None:
        for fig in figs.values():
            plt.close(fig)
    return figs


def _plot_combined(a_hi: float, a_lo: float, n: int = 100, alpha: float = 0.5):
    """持続性・反転傾向の CRW 分布を，標準正規を挟んで1枚に重ねる比較図."""
    jp = _setup_jp_font()
    sqrt_n = np.sqrt(n)
    bin_w = 2.0 / sqrt_n
    fig, ax = plt.subplots(figsize=(8.6, 5.0))

    grid = np.linspace(-4.0, 4.0, 600)
    ax.plot(grid, _normal_pdf(grid, 1.0), color="black", linestyle="--",
            linewidth=1.8, zorder=4,
            label="標準正規 $N(0,1)$" if jp else "$N(0,1)$")

    specs = [
        (a_hi, "0.45", ("持続性" if jp else "persistent")),
        (a_lo, "0.78", ("反転傾向" if jp else "anti-persistent")),
    ]
    for a, color, kind in specs:
        x_even, mu = crw_distribution(a, n=n, alpha=alpha)
        u = x_even / sqrt_n
        sigma2 = a / (1.0 - a)
        ax.bar(u, mu / bin_w, width=bin_w * 0.92, color=color,
               edgecolor="0.3", linewidth=0.3, alpha=0.65, zorder=2,
               label=(f"{kind}：$a={a:.3g}$（$\\sigma^2={sigma2:.2f}$）" if jp
                      else f"{kind}: a={a:.3g} (s2={sigma2:.2f})"))

    ax.set_title("相関付きランダムウォークの分布の比較（$n=%d$）" % n if jp
                 else "CRW distribution comparison (n=%d)" % n)
    ax.set_xlabel(r"$u = \tilde{S}_n/\sqrt{n}$")
    ax.set_ylabel("確率密度" if jp else "density")
    ax.set_xlim(-4.0, 4.0)
    ax.grid(alpha=0.3, linestyle=":")
    ax.legend(loc="upper right", fontsize=10)
    fig.tight_layout()
    return fig
