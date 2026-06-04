"""図 crw_limit_simulation を生成するスクリプト.

時間発展対称な相関付きランダムウォークの確率ベクトルの時間発展
    Psi_{n+1}(x) = P Psi_n(x+1) + Q Psi_n(x-1)
を用いて、初期確率ベクトル phi = (1/2, 1/2)^T のもとで時刻 n = 1000 までの
分布 P(S_n = x) を厳密に計算する。得られた分布をスケーリング S_n/sqrt(n) の
もとで密度に換算し、極限正規分布 N(0, a/(1-a)) と重ねて比較する。

時間発展対称な場合のコイン行列
    A = [[a, 1-a],
         [1-a, a ]]
は、モデルの一般化2パラメータ型 A(p, q) = [[q, 1-p], [1-q, p]] において
p = q = a とおいたものに対応する（src/oyosuri/model.py を参照）。

実行:
    python paper/make_crw_limit_simulation.py
出力:
    paper/crw_limit_simulation.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def coin_matrices(a: float) -> tuple[np.ndarray, np.ndarray]:
    """時間発展対称な左/右遷移行列 (P, Q) を返す（p = q = a）。"""
    P = np.array([[a, 1.0 - a], [0.0, 0.0]], dtype=float)
    Q = np.array([[0.0, 0.0], [1.0 - a, a]], dtype=float)
    return P, Q


def mu_at_n(n: int, a: float, alpha: float = 0.5) -> np.ndarray:
    """時刻 n における確率分布 mu_n(x) = ||Psi_n(x)||_1 を厳密計算して返す。

    返り値 ``mu`` は長さ ``2n+1`` の配列で ``mu[x + n] == P(S_n = x)``。
    メモリ節約のため現時刻の Psi のみを保持して漸化的に更新する。
    """
    P, Q = coin_matrices(a)
    width = 2 * n + 1
    psi = np.zeros((width, 2), dtype=float)
    psi[n, 0] = alpha
    psi[n, 1] = 1.0 - alpha
    for _ in range(n):
        left_shift = np.zeros_like(psi)
        left_shift[:-1] = psi[1:]      # Psi_t(x+1)
        right_shift = np.zeros_like(psi)
        right_shift[1:] = psi[:-1]     # Psi_t(x-1)
        psi = left_shift @ P.T + right_shift @ Q.T
    return np.abs(psi[:, 0]) + np.abs(psi[:, 1])


def scaled_density(n: int, a: float):
    """S_n/sqrt(n) のスケーリングのもとでの (y, 密度) を返す。

    時刻 n の到達可能位置 x は n と同じ偶奇をもち間隔 2 なので、
    y = x/sqrt(n) における密度は f(y) ~= P(S_n=x) / (2/sqrt(n)) で近似する。
    """
    mu = mu_at_n(n, a)
    x = np.arange(-n, n + 1)
    mask = mu > 0.0                    # 到達可能（n と同じ偶奇）な位置のみ
    y = x[mask] / np.sqrt(n)
    dens = mu[mask] * np.sqrt(n) / 2.0
    return y, dens


def normal_pdf(x: np.ndarray, var: float) -> np.ndarray:
    return np.exp(-x * x / (2.0 * var)) / np.sqrt(2.0 * np.pi * var)


def main() -> None:
    n = 1000
    cases = [
        ("persistent", 2.0 / 3.0),       # a = 2/3, sigma^2 = 2
        ("anti-persistent", 1.0 / 3.0),  # a = 1/3, sigma^2 = 1/2
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, (label, a) in zip(axes, cases):
        var = a / (1.0 - a)
        y, dens = scaled_density(n, a)

        ax.fill_between(y, dens, step=None, alpha=0.35, color="tab:blue",
                        label=fr"exact  $\tilde S_n/\sqrt{{n}}$  ($n={n}$)")
        ax.plot(y, dens, color="tab:blue", lw=0.8)

        grid = np.linspace(-4.0 * np.sqrt(var), 4.0 * np.sqrt(var), 400)
        ax.plot(grid, normal_pdf(grid, var), color="tab:red", lw=2.0,
                label=fr"$N(0,\,a/(1-a))=N(0,{var:.3g})$")

        ax.set_title(fr"{label}:  $a={a:.4g}$,  $\sigma^2=a/(1-a)={var:.3g}$")
        ax.set_xlabel(r"$x = \tilde S_n/\sqrt{n}$")
        ax.set_ylabel("density")
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(alpha=0.25)
        ax.set_xlim(grid[0], grid[-1])

    fig.tight_layout()
    out = Path(__file__).resolve().parent / "crw_limit_simulation.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"saved {out}")

    # 数値的な一致のチェック（標準誤差ではなく分散の一致を確認）
    for label, a in cases:
        mu = mu_at_n(n, a)
        x = np.arange(-n, n + 1)
        mean = float(np.sum(x * mu))
        var_emp = float(np.sum((x - mean) ** 2 * mu)) / n   # Var(S_n)/n -> a/(1-a)
        print(f"{label:16s} a={a:.4f}  Var(S_n)/n={var_emp:.5f}  "
              f"a/(1-a)={a/(1-a):.5f}  sum(mu)={mu.sum():.6f}")


if __name__ == "__main__":
    main()
