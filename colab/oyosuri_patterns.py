"""チャートパターン別の合成ウォーク生成 — 卒論 §7 用.

実データからパターン区間を切り出すと，区間ごとに足の本数 N がバラついて
図の横軸スケールが揃わない。本モジュールは代わりに，4つの代表パターン

    uptrend  … 上昇継続
    range2up … レンジ→上昇
    up2range … 上昇→レンジ
    up2down  … 上昇→下降

の ±1 ブリック列を **同一の長さ（既定 150 本）** で人工的に生成し，
``oyosuri_experiment`` と同じ体裁（実測＝太い黒実線，各系列＝実線＋
○/□/△ マーカー，凡例に MAE 併記）で 1 ステップ先予測を重ね描きする。

生成規則（実データの練行足の見た目に合わせた簡易モデル）:

  - トレンド局面（up / down）: 直前と同じ方向を確率 ``stay`` で継続し，
    逆方向からは確率 ``pull`` で引き戻される 2 状態マルコフ連鎖。
    既定値 (stay, pull) = (0.70, 0.58) で 1 本あたり約 +0.3 のドリフト
    （150 本の上昇継続でおよそ +45）。
  - レンジ局面（range）: 局面開始時の水準 anchor への平均回帰
    ``P(up) = clip(0.5 − kappa·(x − anchor), clip_lo, clip_hi)``。
    既定 kappa=0.09 でおおよそ ±4〜6 の帯にとどまる。

2 局面パターンは前半/後半で半々に分割する。乱数 seed を変えれば
同じパターンの別インスタンスが得られる（卒論用は各パターン 2 本）。

Colab での使い方:

    !git clone https://github.com/Lilas812/oyosuri.git
    import sys; sys.path.insert(0, "/content/oyosuri/colab")
    !pip install -q matplotlib-fontja   # 凡例・タイトルの日本語表示

    from oyosuri_patterns import (
        run_pattern_overlay, run_all_patterns,
        run_pattern_gallery, run_all_pattern_galleries,
    )

    # 1枚だけ（上昇→下降，150本，W=2/10/60）
    fig, r = run_pattern_overlay("up2down", n_bricks=150, seed=0)
    fig.savefig("pat_up2down_1.png", dpi=150, bbox_inches="tight")

    # 4パターン × seed 2つ = 8枚を一括生成して保存
    results = run_all_patterns(n_bricks=150, seeds=(0, 1), save_dir=".")

    # 「同じパターンをもっと多く」見たいとき（コンタクトシート）:
    #   1パターン 6 例を 1 枚のグリッド図に（W=2/10/60）
    fig, res = run_pattern_gallery("uptrend", n_instances=6)
    #   4パターンぶんを一括生成・保存（gallery_uptrend_w.png など）
    galleries = run_all_pattern_galleries(n_instances=6, save_dir=".")
"""

from __future__ import annotations

from typing import Sequence

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

from oyosuri_all_in_one import mae, walk_from_bricks
from oyosuri_experiment import (
    _is_full,
    _preds_for_window,
    _setup_jp_font,
    draw_walk_multi_w,
    plot_walk_multi_w,
)


# パターン名 → 局面の並び（2局面は前半/後半で半々）
PATTERNS: dict[str, tuple[str, ...]] = {
    "uptrend": ("up",),
    "range2up": ("range", "up"),
    "up2range": ("up", "range"),
    "up2down": ("up", "down"),
}

JP_TITLES: dict[str, str] = {
    "uptrend": "上昇継続",
    "range2up": "レンジ→上昇",
    "up2range": "上昇→レンジ",
    "up2down": "上昇→下降",
}


def make_pattern_bricks(
    pattern: str,
    n_bricks: int = 150,
    seed: int = 0,
    *,
    stay: float = 0.70,
    pull: float = 0.58,
    kappa: float = 0.09,
    clip_lo: float = 0.12,
    clip_hi: float = 0.88,
) -> list[int]:
    """指定パターンの ±1 ブリック列（長さ n_bricks ちょうど）を生成する.

    Parameters
    ----------
    pattern : "uptrend" / "range2up" / "up2range" / "up2down"
    stay : トレンド局面で直前と同じ方向を継続する確率（>0.5 で持続性）。
    pull : トレンド局面で逆方向からトレンド方向へ引き戻す確率。
    kappa : レンジ局面の平均回帰の強さ。大きいほど帯が狭い。
    """
    if pattern not in PATTERNS:
        raise ValueError(f"pattern must be one of {sorted(PATTERNS)}, got {pattern!r}")
    if n_bricks < 2:
        raise ValueError("n_bricks must be >= 2")
    if not (0.0 < pull <= stay < 1.0):
        raise ValueError("require 0 < pull <= stay < 1")

    rng = np.random.default_rng(seed)
    phases = PATTERNS[pattern]
    k = len(phases)
    lengths = [n_bricks // k] * k
    lengths[-1] += n_bricks - sum(lengths)

    bricks: list[int] = []
    x = 0
    prev = 1 if rng.random() < 0.5 else -1
    for regime, length in zip(phases, lengths):
        anchor = x  # レンジ局面は開始水準を中心に往復する
        for _ in range(length):
            if regime == "up":
                p_up = stay if prev == 1 else pull
            elif regime == "down":
                p_up = (1.0 - stay) if prev == -1 else (1.0 - pull)
            else:  # range
                p_up = min(clip_hi, max(clip_lo, 0.5 - kappa * (x - anchor)))
            b = 1 if rng.random() < p_up else -1
            bricks.append(b)
            x += b
            prev = b
    return bricks


def run_pattern_overlay(
    pattern: str,
    *,
    n_bricks: int = 150,
    seed: int = 0,
    windows: Sequence = (2, 10, 60),
    grid_size: int = 11,
    symmetric: bool = False,
    title: str | None = None,
    show: bool = True,
    **gen_kwargs,
):
    """合成パターン 1 本に対し，指定参照期間の予測を実測と重ねた図を返す.

    ``windows`` の各要素は整数（直近 W 本）または "full"/None（全期間）。
    最大 3 つまで同一図に重ねる（体裁は ``run_multi_w_overlay`` と同じ）。

    Returns
    -------
    (fig, result)
        result : {"pattern", "seed", "x", "bricks", "N",
                  "preds": {W: preds}, "mae": {W: MAE}}
    """
    ws = list(windows)
    if not 1 <= len(ws) <= 3:
        raise ValueError("windows は1〜3個（同一グラフに重ねるため最大3つ）")

    bricks = make_pattern_bricks(pattern, n_bricks=n_bricks, seed=seed, **gen_kwargs)
    x = walk_from_bricks(bricks)
    N = len(bricks)
    actual = [float(v) for v in x[1:]]
    print(f"pattern={pattern} ({JP_TITLES[pattern]}), N={N}, seed={seed}")

    preds_by_w: dict[str, list[float]] = {}
    mae_by_w: dict[str, float] = {}
    for W in ws:
        label = "full" if _is_full(W, N) else str(int(W))
        if label in preds_by_w:
            continue
        preds = _preds_for_window(x, W, grid_size=grid_size, symmetric=symmetric)
        preds_by_w[label] = list(preds)
        mae_by_w[label] = mae(actual, preds)

    print("MAE (参考):", {k: round(v, 4) for k, v in mae_by_w.items()})
    if title is None:
        title = f"{JP_TITLES[pattern]}（合成データ，N={N}）"
    fig = plot_walk_multi_w(x, preds_by_w, mae_by_w=mae_by_w, title=title)
    if show:
        plt.show()
    return fig, {
        "pattern": pattern, "seed": seed,
        "x": x, "bricks": bricks, "N": N,
        "preds": preds_by_w, "mae": mae_by_w,
    }


def run_all_patterns(
    *,
    n_bricks: int = 150,
    seeds: Sequence[int] | dict[str, Sequence[int]] = (0, 1),
    windows: Sequence = (2, 10, 60),
    grid_size: int = 11,
    symmetric: bool = False,
    save_dir: str | None = None,
    fname_fmt: str = "pat_{pattern}_{k}.png",
    dpi: int = 150,
    show: bool = False,
    **gen_kwargs,
):
    """4パターン × 各 seed の図を一括生成する.

    ``seeds`` は全パターン共通のシーケンスか，``{パターン名: シーケンス}``
    の辞書（パターンごとに形の良い seed を選びたいとき）。
    ``save_dir`` を指定すると ``fname_fmt``（``{pattern}``・``{k}``・
    ``{seed}`` を展開，k は 1 始まり）で保存する。卒論の図名に合わせる
    なら seed 1 つ＋ ``fname_fmt="pat_{pattern}_w.png"``。
    戻り値は ``{(pattern, seed): result}``。
    """
    results: dict[tuple[str, int], dict] = {}
    for pattern in PATTERNS:
        pat_seeds = seeds[pattern] if isinstance(seeds, dict) else seeds
        for k, seed in enumerate(pat_seeds, start=1):
            fig, res = run_pattern_overlay(
                pattern,
                n_bricks=n_bricks, seed=seed, windows=windows,
                grid_size=grid_size, symmetric=symmetric,
                title=f"{JP_TITLES[pattern]}（合成データ {k}，N={n_bricks}）",
                show=show, **gen_kwargs,
            )
            results[(pattern, seed)] = res
            if save_dir is not None:
                fname = fname_fmt.format(pattern=pattern, k=k, seed=seed)
                out = _os.path.join(save_dir, fname)
                fig.savefig(out, dpi=dpi, bbox_inches="tight")
                print(f"saved: {out}")
            if not show:
                plt.close(fig)
    return results


def run_pattern_gallery(
    pattern: str,
    *,
    n_instances: int = 6,
    seeds: Sequence[int] | None = None,
    n_bricks: int = 150,
    windows: Sequence = (2, 10, 60),
    ncols: int = 3,
    grid_size: int = 11,
    symmetric: bool = False,
    title: str | None = None,
    show: bool = True,
    **gen_kwargs,
):
    """1 パターンの複数インスタンスを 1 枚のグリッド図（コンタクトシート）にまとめる.

    同じパターン（例：上昇継続）でも乱数 seed を変えると別の形が出る。
    それを ``n_instances`` 本ぶん生成し，各サブプロットに実測と
    ``windows``（既定 W=2/10/60）の予測を重ねて並べる。1 枚を眺めるだけで
    「同じパターンでも W ごとの追従の傾向がどれだけ安定しているか」を
    まとめて目視できる。

    Parameters
    ----------
    n_instances : 生成するインスタンス数（``seeds`` 未指定なら 0..n-1）。
    seeds : 使う seed の並び（指定すると ``n_instances`` は無視）。
    ncols : グリッドの列数。行数は自動。

    Returns
    -------
    (fig, results)
        results : [{"seed", "N", "mae": {W: MAE}}, ...]
    """
    if pattern not in PATTERNS:
        raise ValueError(f"pattern must be one of {sorted(PATTERNS)}, got {pattern!r}")
    ws = list(windows)
    if not 1 <= len(ws) <= 3:
        raise ValueError("windows は1〜3個（同一グラフに重ねるため最大3つ）")
    if seeds is None:
        seeds = list(range(n_instances))
    else:
        seeds = list(seeds)
    if not seeds:
        raise ValueError("seeds/n_instances が空です")

    jp = _setup_jp_font()
    ncols = max(1, min(ncols, len(seeds)))
    nrows = -(-len(seeds) // ncols)  # 切り上げ
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(6.0 * ncols, 3.2 * nrows), squeeze=False,
    )

    results: list[dict] = []
    for idx, seed in enumerate(seeds):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        bricks = make_pattern_bricks(pattern, n_bricks=n_bricks, seed=seed, **gen_kwargs)
        x = walk_from_bricks(bricks)
        N = len(bricks)
        actual = [float(v) for v in x[1:]]

        preds_by_w: dict[str, list[float]] = {}
        mae_by_w: dict[str, float] = {}
        for W in ws:
            label = "full" if _is_full(W, N) else str(int(W))
            if label in preds_by_w:
                continue
            preds = _preds_for_window(x, W, grid_size=grid_size, symmetric=symmetric)
            preds_by_w[label] = list(preds)
            mae_by_w[label] = mae(actual, preds)

        draw_walk_multi_w(
            ax, x, preds_by_w, mae_by_w=mae_by_w,
            title=f"seed={seed}", jp=jp,
            legend=(idx == 0), legend_outside=False, markersize=4.5,
        )
        results.append({"seed": seed, "N": N, "mae": mae_by_w})
        print(f"  {pattern} seed={seed}: MAE="
              f"{ {k: round(v, 3) for k, v in mae_by_w.items()} }")

    # 余ったセルは消す
    for idx in range(len(seeds), nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].axis("off")

    if title is None:
        title = f"{JP_TITLES[pattern]}（合成データ {len(seeds)} 例，N={n_bricks}，W={'/'.join(str(w) for w in ws)}）"
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    if show:
        plt.show()
    return fig, results


def run_all_pattern_galleries(
    *,
    n_instances: int = 6,
    n_bricks: int = 150,
    windows: Sequence = (2, 10, 60),
    ncols: int = 3,
    grid_size: int = 11,
    symmetric: bool = False,
    save_dir: str | None = None,
    fname_fmt: str = "gallery_{pattern}_w.png",
    dpi: int = 150,
    show: bool = False,
    **gen_kwargs,
):
    """4 パターンぶんのコンタクトシートを一括生成する.

    ``save_dir`` を指定すると ``fname_fmt``（``{pattern}`` を展開）で保存する。
    戻り値は ``{pattern: results}``（各 results は ``run_pattern_gallery`` の
    2 要素目）。
    """
    out: dict[str, list[dict]] = {}
    for pattern in PATTERNS:
        print(f"=== {pattern} ({JP_TITLES[pattern]}) ===")
        fig, res = run_pattern_gallery(
            pattern,
            n_instances=n_instances, n_bricks=n_bricks, windows=windows,
            ncols=ncols, grid_size=grid_size, symmetric=symmetric,
            show=show, **gen_kwargs,
        )
        out[pattern] = res
        if save_dir is not None:
            _os.makedirs(save_dir, exist_ok=True)
            fname = fname_fmt.format(pattern=pattern)
            path = _os.path.join(save_dir, fname)
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
            print(f"saved: {path}")
        if not show:
            plt.close(fig)
    return out
