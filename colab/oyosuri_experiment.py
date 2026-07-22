"""チャートパターン別の予測図 — 卒論 §7 用.

``oyosuri_all_in_one.py`` と ``oyosuri_rolling.py`` を import し、ある区間
（1チャートパターン）について 1 ステップ先予測を行い、実測ウォーク x_n と
予測 x_n* を同一図に重ねて出力する。判断はこの図を目視で行い、平均絶対誤差
MAE は補助（飾り）として併記する。

**参照期間は自由に選べる**（``run_multi_w_overlay`` の ``windows`` で指定）:
  - 整数 W      … 直近 W 本だけで当てはめる（rolling 窓）
  - "full"/None … その時点までの過去全期間で当てはめる（第5章の元モデル）
``windows`` は最大3つまで同一図に重ねられる。

    windows=[10, 30, 60]      → 直近10/30/60本 の3本を重ねる
    windows=["full"]          → 全期間参照の予測1本だけ
    windows=[10, 30, "full"]  → 直近10・直近30・全期間 を重ねる（混在）

凡例は日本語（実測値／全期間／w=30）で、白黒印刷でも判別できるよう色ではなく
マーカー形状で区別する（線はすべて実線。実測＝太い黒実線、各系列＝実線＋
○/□/△ の白抜きマーカー。点線は重なると追えないため実線にした）。日本語フォントが無い環境では英語ラベルに自動
フォールバックする（□に潰れるのを防ぐ）。

MAE = (1/N) Σ_n |x_n − x_n*|。±1 ウォークなので「動かない予測」の MAE は
恒等的に 1（MAE<1 で「動かない予測」に優る目安）。

Colab で GitHub から実行する場合:

    # 1) クローンして colab/ を import パスに追加
    !git clone https://github.com/Lilas812/oyosuri.git
    import sys; sys.path.insert(0, "/content/oyosuri/colab")

    # 1') 凡例の日本語（実測値/全期間）を表示したいとき一度だけ実行
    !pip install -q matplotlib-fontja

    # 2) import（依存は自動解決）
    from oyosuri_experiment import run_multi_w_overlay, run_mae_sweep

    # 3) CSV アップロード
    from google.colab import files
    up = files.upload(); path = next(iter(up))

    # 4a) 3つの W を重ねる / 4b) 全期間参照の予測1本だけ
    fig, r = run_multi_w_overlay(path, B=4, windows=[10, 30, 60],
                                 start="2026-05-19-15:00", end="2026-05-19-23:58",
                                 tz="Asia/Tokyo")
    fig.savefig("pat_uptrend.png", dpi=150, bbox_inches="tight")  # 論文用に保存

（ローカル/旧来の %run でも可:
    %run colab/oyosuri_all_in_one.py → %run colab/oyosuri_rolling.py →
    %run colab/oyosuri_experiment.py）
"""

from __future__ import annotations

import time
from collections import Counter
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# このファイルのあるディレクトリ（colab/）を import パスへ追加し、GitHub clone 後に
# どの作業ディレクトリからでも兄弟モジュール (oyosuri_all_in_one / _rolling) を解決する。
import os as _os
import sys as _sys
try:
    _HERE = _os.path.dirname(_os.path.abspath(__file__))
except NameError:  # __file__ が無い実行形態へのフォールバック
    _HERE = _os.getcwd()
if _HERE not in _sys.path:
    _sys.path.insert(0, _HERE)

from oyosuri_all_in_one import (
    generate_mean_renko_from_ohlc,
    hit_rate,
    load_tradingview_csv,
    mae,
    predict_sequence,
    slice_by_time,
    walk_from_bricks,
)
from oyosuri_rolling import (
    predict_sequence_with_params_rolling,
    predict_sequences_with_params_rolling,
)


_FULL_TOKENS = ("full", "all", "∞", "inf")


def _setup_jp_font() -> bool:
    """凡例の日本語（実測値・全期間 等）を表示できるようフォントを設定する.

    Colab では ``!pip install matplotlib-fontja``（または
    ``!apt-get -y install fonts-ipafont-gothic`` 後に matplotlib のフォント
    キャッシュ再構築）を一度実行しておくと日本語が出る。日本語フォントが
    見つからない場合は False を返し、呼び出し側は英語ラベルにフォール
    バックする（□で潰れるのを防ぐ）。
    """
    import matplotlib
    from matplotlib import font_manager

    # 1) matplotlib-fontja / japanize-matplotlib が入っていれば使う
    for _mod in ("matplotlib_fontja", "japanize_matplotlib"):
        try:
            __import__(_mod)
            matplotlib.rcParams["axes.unicode_minus"] = False
            return True
        except Exception:
            pass
    # 2) システムにある日本語フォントを探して設定
    candidates = [
        "Noto Sans CJK JP", "Noto Sans JP", "IPAexGothic", "IPAGothic",
        "IPAPGothic", "TakaoGothic", "VL Gothic", "Yu Gothic", "Meiryo",
        "Hiragino Sans", "MS Gothic",
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            matplotlib.rcParams["font.family"] = name
            matplotlib.rcParams["axes.unicode_minus"] = False
            return True
    return False


def _is_full(W, N: int) -> bool:
    """W が「過去全期間参照」を意味するか（None / "full" / N 以上の整数）."""
    if W is None:
        return True
    if isinstance(W, str):
        return W.strip().lower() in _FULL_TOKENS
    return int(W) >= N


def _preds_for_window(
    x: Sequence[int],
    W,
    *,
    grid_size: int = 11,
    symmetric: bool = False,
) -> list[float]:
    """参照期間 W の 1 ステップ先予測列を返す.

    W が full（None / "full" / N 以上）のときは全期間版 (``predict_sequence``)、
    そうでなければ直近 W bricks の rolling 版。
    """
    N = len(list(x)) - 1
    if _is_full(W, N):
        return predict_sequence(x, grid_size=grid_size, symmetric=symmetric)
    return predict_sequence_with_params_rolling(
        x, window=int(W), grid_size=grid_size, symmetric=symmetric
    )[0]


def mae_by_window(
    x: Sequence[int],
    windows: Sequence,
    *,
    include_full: bool = True,
    grid_size: int = 11,
    symmetric: bool = False,
) -> dict[str, float]:
    """各参照期間 W の MAE を ``{W: MAE}`` で返す（数値だけ欲しいとき）.

    ``windows`` の各要素は整数（rolling）または "full"/None（全期間）。
    ``include_full=True`` なら全期間版（label ``"full"``）も必ず加える。
    """
    x = list(x)
    N = len(x) - 1
    if N < 1:
        raise ValueError("x must contain at least 2 points")
    actual = [float(v) for v in x[1:]]

    out: dict[str, float] = {}
    for W in windows:
        label = "full" if _is_full(W, N) else str(int(W))
        if label in out:
            continue
        out[label] = mae(actual, _preds_for_window(
            x, W, grid_size=grid_size, symmetric=symmetric))
    if include_full and "full" not in out:
        out["full"] = mae(actual, predict_sequence(
            x, grid_size=grid_size, symmetric=symmetric))
    return out


def run_mae_sweep(
    src,
    B: float,
    *,
    windows: Sequence = (2, 3, 5, 10, 20, 40),
    include_full: bool = True,
    start=None,
    end=None,
    tz: str | None = None,
    grid_size: int = 11,
    symmetric: bool = False,
) -> dict:
    """CSV → 平均練行足 → 参照期間ごとの MAE を数値表示（グラフなし）.

    Returns {"x", "bricks", "N", "mae": {W: MAE}}.
    """
    df = load_tradingview_csv(src)
    df = slice_by_time(df, start=start, end=end, tz=tz)
    if isinstance(df.index, pd.DatetimeIndex) and len(df) > 0:
        range_str = f"{df.index[0]} → {df.index[-1]}"
    else:
        range_str = f"rows [0, {len(df)})"
    print(f"当てはめ範囲: {range_str}  (bars={len(df)})")

    bricks, _ = generate_mean_renko_from_ohlc(df, B=B)
    x = walk_from_bricks(bricks)
    N = len(bricks)
    print(f"N_bricks={N}, B={B}")
    if N < 2:
        raise ValueError(
            f"ブリック数が少なすぎます (N={N})。B を小さくするか期間を広げてください。"
        )

    result = mae_by_window(
        x, windows, include_full=include_full,
        grid_size=grid_size, symmetric=symmetric,
    )
    print(f"{'W':>6} | {'MAE':>8}")
    print("-" * 18)
    for label, value in result.items():
        print(f"{label:>6} | {value:>8.4f}")
    return {"x": x, "bricks": bricks, "N": N, "mae": result}


def plot_walk_multi_w(
    x: Sequence[int],
    preds_by_w: dict[str, Sequence[float]],
    *,
    mae_by_w: dict[str, float] | None = None,
    title: str | None = None,
):
    """実測ウォーク x_n と、最大3つの系列の予測 x_n* を同一図に重ねて返す.

    凡例は日本語（実測値／全期間／w=30）。白黒印刷でも判別できるよう、系列は
    色ではなくマーカー形状で区別する（線はすべて実線。実測＝太い黒実線・
    マーカー無し、各系列＝実線＋○/□/△ の白抜きマーカー）。判断はこの図を
    目視で行い、凡例に補助の MAE を併記する。日本語フォントが無ければ英語に
    自動フォールバックする。
    """
    jp = _setup_jp_font()
    obs_label = "実測値" if jp else "observed"
    full_label = "全期間" if jp else "full"

    x = list(x)
    N = max(len(x) - 1, 0)
    fig, ax = plt.subplots(figsize=(13, 5))

    walk_n = np.arange(len(x))
    ax.step(
        walk_n, x, where="post",
        color="black", linewidth=2.2, label=obs_label, zorder=2,
    )

    # すべて実線。白黒では「マーカー形状＋濃淡」で区別する
    # （点線は重なると追えないため、線は実線にしてマーカーで識別する）
    markers = ["o", "s", "^"]
    grays = ["black", "0.50", "black"]
    me = max(1, N // 18)  # マーカーは追える程度の間隔で配置
    for i, (label, preds) in enumerate(preds_by_w.items()):
        preds = list(preds)
        pred_n = np.arange(1, 1 + len(preds))
        leg = full_label if label == "full" else f"w={label}"
        if mae_by_w is not None and label in mae_by_w:
            leg += f"  (MAE={mae_by_w[label]:.3f})"
        ax.plot(
            pred_n, preds,
            color=grays[i % len(grays)],
            linestyle="-",
            marker=markers[i % len(markers)],
            markersize=6, markerfacecolor="white", markeredgewidth=1.2,
            markevery=me, linewidth=1.4, label=leg, zorder=3,
        )

    if title:
        ax.set_title(title)
    ax.set_xlabel("n")
    ax.set_ylabel("$x_n$")
    ax.set_xlim(-0.5, max(N, 1) + 0.5)
    ax.grid(alpha=0.3, linestyle=":")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=9,
              borderaxespad=0.0)
    fig.tight_layout()
    return fig


def run_multi_w_overlay(
    src,
    B: float,
    *,
    windows: Sequence = (10, 30, 60),
    start=None,
    end=None,
    tz: str | None = None,
    grid_size: int = 11,
    symmetric: bool = False,
    title: str | None = None,
    show: bool = True,
    save_path: str | None = None,
):
    """CSV → 平均練行足 → 指定した参照期間の予測を実測と同一グラフに重ねて出力.

    ``windows`` は最大3つ。各要素は整数（直近 W 本＝rolling）または
    "full"/None（過去全期間参照）。
      - windows=[10, 30, 60]     … 直近10/30/60本 を3本重ね
      - windows=["full"]         … 全期間参照の予測1本だけ
      - windows=[10, 30, "full"] … 混在

    MAE と方向的中率 HIT は図には載せず，ログに W | MAE | HIT の表で
    出力する。``save_path`` を指定すると図を PNG として保存する
    （Colab で ``fig.savefig`` を別途書かなくて済む）。

    Returns
    -------
    (fig, result)
        result : {"x", "bricks", "N", "preds": {W: preds},
                  "mae": {W: MAE}, "hit": {W: HIT}}
    """
    ws = list(windows)
    if not 1 <= len(ws) <= 3:
        raise ValueError("windows は1〜3個（同一グラフに重ねるため最大3つ）")

    df = load_tradingview_csv(src)
    df = slice_by_time(df, start=start, end=end, tz=tz)
    if isinstance(df.index, pd.DatetimeIndex) and len(df) > 0:
        range_str = f"{df.index[0]} → {df.index[-1]}"
    else:
        range_str = f"rows [0, {len(df)})"
    print(f"当てはめ範囲: {range_str}  (bars={len(df)})")

    bricks, _ = generate_mean_renko_from_ohlc(df, B=B)
    x = walk_from_bricks(bricks)
    N = len(bricks)
    print(f"N_bricks={N}, B={B}")
    if N < 2:
        raise ValueError(
            f"ブリック数が少なすぎます (N={N})。B を小さくするか期間を広げてください。"
        )
    actual = [float(v) for v in x[1:]]

    preds_by_w: dict[str, list[float]] = {}
    mae_by_w: dict[str, float] = {}
    hit_by_w: dict[str, float] = {}

    rolling_windows = list(dict.fromkeys(
        int(W) for W in ws if not _is_full(W, N)
    ))
    rolling_results = (
        predict_sequences_with_params_rolling(
            x,
            windows=rolling_windows,
            grid_size=grid_size,
            symmetric=symmetric,
        )
        if rolling_windows
        else {}
    )
    for W in ws:
        label = "full" if _is_full(W, N) else str(int(W))
        if label in preds_by_w:
            continue
        if label == "full":
            preds = predict_sequence(
                x, grid_size=grid_size, symmetric=symmetric
            )
        else:
            preds = rolling_results[int(W)][0]
        preds_by_w[label] = preds
        mae_by_w[label] = mae(actual, preds)
        hit_by_w[label] = hit_rate(actual, preds)

    # 指標は図には載せず，ログに表で出す
    print(f"{'W':>6} | {'MAE':>8} | {'HIT':>8}")
    print("-" * 30)
    for label in preds_by_w:
        print(f"{label:>6} | {mae_by_w[label]:>8.4f} | {hit_by_w[label]:>8.4f}")

    fig = plot_walk_multi_w(x, preds_by_w, title=title)
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"図を保存: {save_path}")
    if show:
        plt.show()
    return fig, {
        "x": x, "bricks": bricks, "N": N,
        "preds": preds_by_w, "mae": mae_by_w, "hit": hit_by_w,
    }


# ============================================================================
# 参照期間 W の感度分析（MAE ＋ 計算時間） — 卒論 §7.2 用
# ============================================================================

def timed_preds_for_window(
    x: Sequence[int],
    W,
    *,
    grid_size: int = 11,
    symmetric: bool = False,
) -> tuple[list[float], float]:
    """参照期間 W の予測列と所要時間（秒）を返す.

    時間は ``_preds_for_window`` 全体（全ステップの fit ＋予測）の
    wall-clock。同一マシン・同一系列で W だけ変えて比較する用途を想定。
    """
    t0 = time.perf_counter()
    preds = _preds_for_window(x, W, grid_size=grid_size, symmetric=symmetric)
    return preds, time.perf_counter() - t0


def sweep_by_window(
    x: Sequence[int],
    windows: Sequence,
    *,
    include_full: bool = True,
    grid_size: int = 11,
    symmetric: bool = False,
) -> dict[str, dict]:
    """各参照期間 W の MAE と計算時間を ``{W: {...}}`` で返す.

    ``mae_by_window`` の計時付き版。返り値の各エントリは

        {"mae": float, "time_s": float, "time_per_step_ms": float,
         "preds": list[float]}

    で，``preds`` は重ね描き（``plot_walk_multi_w``）への再利用用。
    """
    x = list(x)
    N = len(x) - 1
    if N < 1:
        raise ValueError("x must contain at least 2 points")
    actual = [float(v) for v in x[1:]]

    labels: list[tuple[str, object]] = []
    for W in windows:
        label = "full" if _is_full(W, N) else str(int(W))
        if label not in (lb for lb, _ in labels):
            labels.append((label, W))
    if include_full and "full" not in (lb for lb, _ in labels):
        labels.append(("full", "full"))

    out: dict[str, dict] = {}
    for label, W in labels:
        preds, secs = timed_preds_for_window(
            x, W, grid_size=grid_size, symmetric=symmetric)
        out[label] = {
            "mae": mae(actual, preds),
            "time_s": secs,
            "time_per_step_ms": 1000.0 * secs / max(len(preds), 1),
            "preds": preds,
        }
    return out


def print_w_sweep(table: dict[str, dict]) -> None:
    """``sweep_by_window`` の結果を整形して表示する."""
    print(f"{'W':>6} | {'MAE':>8} | {'time[s]':>9} | {'ms/step':>8}")
    print("-" * 42)
    for label, row in table.items():
        print(
            f"{label:>6} | {row['mae']:>8.4f} | {row['time_s']:>9.2f}"
            f" | {row['time_per_step_ms']:>8.1f}"
        )


def w_sweep_to_latex(table: dict[str, dict]) -> str:
    """``sweep_by_window`` の結果から卒論の表（tab:w_sweep）の中身を作る.

    列 = 各参照期間（"full" は 全期間），行 = MAE／総計算時間／1ステップ
    当たり時間。出力をそのまま tabular 環境に貼り付けられる。
    """
    labels = list(table.keys())
    heads = ["全期間" if lb == "full" else f"$W={lb}$" for lb in labels]
    col_spec = "l|" + "c" * len(labels)
    lines = [
        f"\\begin{{tabular}}{{{col_spec}}}\\hline",
        "    指標 & " + " & ".join(heads) + " \\\\\\hline",
        "    MAE & "
        + " & ".join(f"{table[lb]['mae']:.3f}" for lb in labels)
        + " \\\\",
        "    総計算時間 [s] & "
        + " & ".join(f"{table[lb]['time_s']:.1f}" for lb in labels)
        + " \\\\",
        "    1ステップ当たり [ms] & "
        + " & ".join(f"{table[lb]['time_per_step_ms']:.0f}" for lb in labels)
        + " \\\\\\hline",
        "\\end{tabular}",
    ]
    return "\n".join(lines)


def plot_w_sweep(
    table: dict[str, dict],
    *,
    N: int | None = None,
    title: str | None = None,
):
    """W スイープの 2 パネル要約図（左: MAE，右: 1ステップ当たり時間）を返す.

    横軸は参照期間 W（対数軸）。"full"（全期間）は W=N の位置に置き，
    目盛りラベルを「全期間」とする（``N`` 必須）。左パネルには
    「変化なし」予測の基準 MAE=1 を破線で示す。白黒印刷を想定して
    黒の白抜きマーカー＋実線で描く。
    """
    jp = _setup_jp_font()
    full_label = "全期間" if jp else "full"
    xlab = "参照期間 $W$" if jp else "reference window $W$"
    base_label = "「変化なし」予測 (MAE=1)" if jp else "no-change (MAE=1)"

    ws: list[float] = []
    ticks: list[str] = []
    for label in table:
        if label == "full":
            if N is None:
                raise ValueError("table に 'full' を含む場合は N を指定する")
            ws.append(float(N))
            ticks.append(full_label)
        else:
            ws.append(float(int(label)))
            ticks.append(label)
    maes = [table[lb]["mae"] for lb in table]
    ms = [table[lb]["time_per_step_ms"] for lb in table]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    for ax, ys in ((ax1, maes), (ax2, ms)):
        ax.plot(
            ws, ys,
            color="black", linestyle="-", marker="o",
            markersize=6, markerfacecolor="white", markeredgewidth=1.2,
            linewidth=1.4,
        )
        ax.set_xscale("log")
        ax.set_xticks(ws)
        ax.set_xticklabels(ticks)
        ax.minorticks_off()
        ax.set_xlabel(xlab)
        ax.grid(alpha=0.3, linestyle=":")
    ax1.axhline(1.0, color="0.4", linestyle="--", linewidth=1.0,
                label=base_label)
    ax1.set_ylabel("MAE")
    ax1.legend(loc="best", fontsize=9)
    ax2.set_yscale("log")
    ax2.set_ylabel("1ステップ当たり計算時間 [ms]" if jp
                   else "time per step [ms]")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig


def run_w_sweep(
    src,
    B: float,
    *,
    windows: Sequence = (2, 3, 5, 10, 20, 30, 60),
    include_full: bool = True,
    start=None,
    end=None,
    tz: str | None = None,
    grid_size: int = 11,
    symmetric: bool = False,
    title: str | None = None,
    show: bool = True,
):
    """CSV → 平均練行足 → W スイープ（MAE＋計算時間）→ 2 パネル要約図.

    ``run_mae_sweep`` の計時付き・図付き版。卒論 §7.2（参照期間 W に
    対する感度分析）の表と図をこれ 1 つで作る。

    Returns
    -------
    (fig, result)
        result : {"x", "bricks", "N", "table"}（table は
        ``sweep_by_window`` の返り値で，W ごとの予測列も含む）
    """
    df = load_tradingview_csv(src)
    df = slice_by_time(df, start=start, end=end, tz=tz)
    if isinstance(df.index, pd.DatetimeIndex) and len(df) > 0:
        range_str = f"{df.index[0]} → {df.index[-1]}"
    else:
        range_str = f"rows [0, {len(df)})"
    print(f"当てはめ範囲: {range_str}  (bars={len(df)})")

    bricks, _ = generate_mean_renko_from_ohlc(df, B=B)
    x = walk_from_bricks(bricks)
    N = len(bricks)
    print(f"N_bricks={N}, B={B}")
    if N < 2:
        raise ValueError(
            f"ブリック数が少なすぎます (N={N})。B を小さくするか期間を広げてください。"
        )

    table = sweep_by_window(
        x, windows, include_full=include_full,
        grid_size=grid_size, symmetric=symmetric,
    )
    print_w_sweep(table)
    fig = plot_w_sweep(table, N=N, title=title)
    if show:
        plt.show()
    return fig, {"x": x, "bricks": bricks, "N": N, "table": table}


def brick_stats(bricks: Sequence[int]) -> dict:
    """練行足符号列の記述統計（卒論 §7.1・考察用）.

    Returns
    -------
    dict
        N（足の本数）, up_ratio（+1 の比率）,
        reversal_rate（直前と逆符号になった遷移の比率 r̂）,
        mean_run / max_run（同符号が連続する長さ＝連長の平均・最大）,
        run_hist（連長の度数分布）。
        ナイーブな「直前と同方向」予測の MAE は 2*reversal_rate になる。
    """
    b = [int(v) for v in bricks]
    N = len(b)
    if N < 2:
        raise ValueError("bricks must contain at least 2 elements")
    runs: list[int] = []
    cur = 1
    for prev, nxt in zip(b, b[1:]):
        if nxt == prev:
            cur += 1
        else:
            runs.append(cur)
            cur = 1
    runs.append(cur)
    reversals = len(runs) - 1
    return {
        "N": N,
        "up_ratio": sum(1 for v in b if v == 1) / N,
        "reversal_rate": reversals / (N - 1),
        "mean_run": N / len(runs),
        "max_run": max(runs),
        "run_hist": dict(sorted(Counter(runs).items())),
    }
