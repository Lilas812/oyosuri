"""卒論 §7（実験結果）の図表一式を生成するスクリプト.

生成物（既定の保存先は docs/thesis/）:

  [最終版]   pat_uptrend_w.png / pat_range2up_w.png / pat_up2range_w.png /
             pat_up2down_w.png
             … 局面別分析（§7.4）。合成データによる図であり，論文本文にも
               その旨を明記してあるため，実データ入手後も差し替え不要。

  [仮置き]   w_sweep_summary.png … W スイープの 2 パネル要約図（§7.2）
             w_overlay.png       … W=2/10/60 の重ね描き（§7.3）
             … 実データ（XAUUSD の CSV）が無い間の動作確認・体裁確認用。
               実データ入手後は ``oyosuri_experiment.run_w_sweep`` /
               ``run_multi_w_overlay`` を CSV に対して実行して差し替える。

  標準出力   tab:w_sweep / tab:pattern_summary に貼る LaTeX 行，
             足系列の記述統計（上昇比率・反転率 r̂・平均連長），
             W=2 とナイーブ順張り（直前と同方向）予測の比較（考察用）。

仮置きの全系列は，§7.4 と同じ合成生成規則（oyosuri_patterns）で
レンジ→上昇→下降 の複数局面をつないだ N=200 の符号列を用いる。

使い方:

  一括実行（Colab など時間制限の無い環境）:
      python colab/make_thesis_figs.py

  分割実行（1 プロセスの実行時間に制限がある環境向け。
  各ステージは結果を JSON でキャッシュし，最後に finalize が統合する）:
      python colab/make_thesis_figs.py --stage sweep --windows 2,3,5,10,20
      python colab/make_thesis_figs.py --stage sweep --windows 30,60
      python colab/make_thesis_figs.py --stage sweep --windows full
      python colab/make_thesis_figs.py --stage pattern --pattern uptrend
      ...（4 パターンぶん）...
      python colab/make_thesis_figs.py --stage finalize
"""

from __future__ import annotations

import argparse
import glob
import json
import os as _os
import sys as _sys

try:
    _HERE = _os.path.dirname(_os.path.abspath(__file__))
except NameError:
    _HERE = _os.getcwd()
if _HERE not in _sys.path:
    _sys.path.insert(0, _HERE)

import matplotlib

matplotlib.use("Agg")  # 画面の無い環境でも保存だけ行う

from oyosuri_all_in_one import mae, walk_from_bricks
from oyosuri_experiment import (
    brick_stats,
    plot_w_sweep,
    plot_walk_multi_w,
    print_w_sweep,
    sweep_by_window,
    w_sweep_to_latex,
)
from oyosuri_patterns import JP_TITLES, PATTERNS, make_pattern_bricks, run_pattern_overlay

GRID_SIZE = 11          # 論文記載どおり p,q,α ∈ {0, 0.1, …, 1}
SWEEP_WINDOWS = ("2", "3", "5", "10", "20", "30", "60", "full")
OVERLAY_WINDOWS = ("2", "10", "60")   # 重ね描き・局面別で使う代表 3 通り
PATTERN_SEED = 0
DPI = 150


def make_standin_series() -> list[int]:
    """仮置き用の複数局面（レンジ→上昇→下降）合成符号列 N=200 を返す."""
    return (
        make_pattern_bricks("range2up", n_bricks=100, seed=2)
        + make_pattern_bricks("up2down", n_bricks=100, seed=3)
    )


def naive_momentum_preds(x: list[int]) -> list[float]:
    """ナイーブ順張り（直前と同方向）予測列.

    x*_{n+1} = x_n + (x_n - x_{n-1})。最初のステップは方向情報が無いので
    「変化なし」x*_1 = x_0 とする。
    """
    preds = [float(x[0])]
    for n in range(1, len(x) - 1):
        preds.append(float(x[n] + (x[n] - x[n - 1])))
    return preds


# ----------------------------------------------------------------------
# ステージ実装
# ----------------------------------------------------------------------

def stage_sweep(windows: list[str], cache_dir: str) -> None:
    """指定した参照期間ぶんだけスイープし，結果を JSON にキャッシュする."""
    bricks = make_standin_series()
    x = walk_from_bricks(bricks)
    table = sweep_by_window(x, windows, include_full=False, grid_size=GRID_SIZE)
    print_w_sweep(table)
    _os.makedirs(cache_dir, exist_ok=True)
    out = _os.path.join(cache_dir, f"sweep_{'_'.join(windows)}.json")
    with open(out, "w") as f:
        json.dump(table, f)
    print(f"cached: {out}")


def stage_pattern(pattern: str, save_dir: str, cache_dir: str) -> None:
    """1 パターンぶんの図（§7.4 最終版）を生成し，MAE をキャッシュする."""
    fig, res = run_pattern_overlay(
        pattern, n_bricks=150, seed=PATTERN_SEED,
        windows=[int(w) for w in OVERLAY_WINDOWS],
        grid_size=GRID_SIZE, title="", show=False,
    )
    _os.makedirs(save_dir, exist_ok=True)
    out = _os.path.join(save_dir, f"pat_{pattern}_w.png")
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    print(f"saved: {out}")
    _os.makedirs(cache_dir, exist_ok=True)
    cache = _os.path.join(cache_dir, f"pattern_{pattern}.json")
    with open(cache, "w") as f:
        json.dump({"pattern": pattern, "mae": res["mae"]}, f)
    print(f"cached: {cache}")


def stage_finalize(save_dir: str, cache_dir: str) -> None:
    """キャッシュを統合して図・表・考察用の数値を出力する."""
    _os.makedirs(save_dir, exist_ok=True)
    bricks = make_standin_series()
    x = walk_from_bricks(bricks)
    N = len(bricks)
    stats = brick_stats(bricks)
    print(f"\n=== 仮置き全系列（合成・複数局面） N={N} ===")
    print("足系列の記述統計:", {k: (round(v, 4) if isinstance(v, float) else v)
                               for k, v in stats.items()})

    # --- スイープ結果の統合（§7.2） ---
    merged: dict[str, dict] = {}
    for path in sorted(glob.glob(_os.path.join(cache_dir, "sweep_*.json"))):
        with open(path) as f:
            merged.update(json.load(f))
    table = {lb: merged[lb] for lb in SWEEP_WINDOWS if lb in merged}
    missing = [lb for lb in SWEEP_WINDOWS if lb not in merged]
    if missing:
        print(f"warning: スイープ未実行の W があります: {missing}")
    if table:
        print_w_sweep(table)
        maes = [row["mae"] for row in table.values()]
        times = [row["time_s"] for row in table.values()]
        mono_mae = all(a <= b + 1e-12 for a, b in zip(maes, maes[1:]))
        mono_time = all(a <= b + 1e-12 for a, b in zip(times, times[1:]))
        print(f"単調性チェック: MAE 単調増加={mono_mae}, 計算時間 単調増加={mono_time}")
        print("\n--- tab:w_sweep 用 LaTeX（仮置き値） ---")
        print(w_sweep_to_latex(table))
        fig = plot_w_sweep(table, N=N)
        out = _os.path.join(save_dir, "w_sweep_summary.png")
        fig.savefig(out, dpi=DPI, bbox_inches="tight")
        print(f"saved: {out}")

    # --- W=2/10/60 の重ね描き（§7.3）— スイープの予測列を再利用 ---
    if all(lb in table for lb in OVERLAY_WINDOWS):
        actual = [float(v) for v in x[1:]]
        preds_by_w = {lb: table[lb]["preds"] for lb in OVERLAY_WINDOWS}
        mae_by_w = {lb: table[lb]["mae"] for lb in OVERLAY_WINDOWS}
        fig = plot_walk_multi_w(x, preds_by_w, mae_by_w=mae_by_w)
        out = _os.path.join(save_dir, "w_overlay.png")
        fig.savefig(out, dpi=DPI, bbox_inches="tight")
        print(f"saved: {out}")

        # --- 考察用: W=2 とナイーブ順張りの比較 ---
        naive = naive_momentum_preds(x)
        naive_mae = mae(actual, naive)
        r_hat = stats["reversal_rate"]
        diffs = [abs(p - nv)
                 for p, nv in zip(table["2"]["preds"][1:], naive[1:])]
        print("\n--- 考察用: W=2 vs ナイーブ順張り ---")
        print(f"反転率 r̂ = {r_hat:.4f},  2*r̂ = {2 * r_hat:.4f}")
        print(f"ナイーブ順張り MAE = {naive_mae:.4f}")
        print(f"モデル W=2 の MAE  = {table['2']['mae']:.4f}")
        print(f"|W=2予測 − 順張り予測| の平均 = {sum(diffs) / len(diffs):.4f},"
              f" 最大 = {max(diffs):.4f}")

    # --- 局面別 MAE 表（§7.4） ---
    rows = []
    for pattern in PATTERNS:
        path = _os.path.join(cache_dir, f"pattern_{pattern}.json")
        if not _os.path.exists(path):
            print(f"warning: パターン未実行: {pattern}")
            continue
        with open(path) as f:
            rows.append(json.load(f))
    if rows:
        print("\n--- tab:pattern_summary 用 LaTeX ---")
        for row in rows:
            cells = " & ".join(
                f"{row['mae'][w]:.3f}" for w in OVERLAY_WINDOWS)
            print(f"    {JP_TITLES[row['pattern']]} & {cells} \\\\")


def main(argv: list[str] | None = None) -> None:
    default_dir = _os.path.join(_os.path.dirname(_HERE), "docs", "thesis")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--save-dir", default=default_dir)
    ap.add_argument("--cache-dir", default="/tmp/oyosuri_thesis_cache")
    ap.add_argument("--stage", default="all",
                    choices=["all", "sweep", "pattern", "finalize"])
    ap.add_argument("--windows", default=",".join(SWEEP_WINDOWS),
                    help="sweep ステージで計算する W（カンマ区切り，full 可）")
    ap.add_argument("--pattern", default=None, choices=sorted(PATTERNS))
    args = ap.parse_args(argv)

    if args.stage in ("all", "sweep"):
        stage_sweep([w.strip() for w in args.windows.split(",") if w.strip()],
                    args.cache_dir)
    if args.stage == "pattern":
        if args.pattern is None:
            raise SystemExit("--stage pattern には --pattern が必要です")
        stage_pattern(args.pattern, args.save_dir, args.cache_dir)
    if args.stage == "all":
        for pattern in PATTERNS:
            stage_pattern(pattern, args.save_dir, args.cache_dir)
    if args.stage in ("all", "finalize"):
        stage_finalize(args.save_dir, args.cache_dir)


if __name__ == "__main__":
    main()
