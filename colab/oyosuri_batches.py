"""150 ブリックごとに区切って区間データを大量に集めるバッチ収集モジュール.

長い TradingView CSV を平均練行足のブリック列に変換し，**150 本ずつ
（重複なし・連続）** のバッチに区切って，各区間の **生データ** をそのまま
保存する。後から自由に再分析・再描画できるアーカイブを作るのが目的。

参照期間（window）は本家 ``oyosuri_experiment.run_multi_w_overlay`` と
同じく **1〜3 個まで選べる**。各要素は

  - 整数 W       … 直近 W 本だけで当てはめる（rolling 窓）
  - "full"/None  … その区間の全期間で当てはめる（第5章の元モデル）

各バッチで保存する生データ（モデルの x_0 = 0 規約に合わせて再ゼロ化した
「モデルフレーム」で揃える。絶対水準は manifest の ``base`` で復元できる）:

  - n            … 区間内のブリック index 0..L（L = batch_size）
  - x            … 実測ウォーク x_n（再ゼロ化済み，x_0 = 0）
  - pred_<W>     … 参照期間 <W> での 1 ステップ先予測 x_n*（n=0 は NaN）
  - p_<W>, q_<W>, alpha_<W> … その予測を出した最適化結果（n=0 は NaN）

（``<W>`` は整数窓ならその数字，全期間なら ``full``。例: windows=[10, 30,
"full"] → pred_10 / pred_30 / pred_full の 3 系列ぶんの列が並ぶ。）

さらに全区間 × 各参照期間を 1 行 = 1 区間でまとめた ``manifest.csv`` と，
**各区間の重ね図 PNG**（実測 x_n × 各参照期間の予測 x_n*，
``batch_0000.png`` …），まとめてダウンロードするための ``batches.zip``
を出力する。描画は本家 ``oyosuri_experiment.plot_walk_multi_w`` を再利用
する（実測＝黒太線，各系列＝○/□/△ の白抜きマーカー，凡例に MAE 併記）。

Colab での使い方:

    # 1) クローンして colab/ を import パスに追加
    !git clone https://github.com/Lilas812/oyosuri.git
    import sys; sys.path.insert(0, "/content/oyosuri/colab")

    # 1') 凡例の日本語（実測値/全期間）を表示したいとき一度だけ実行
    !pip install -q matplotlib-fontja

    # 2) import（依存は自動解決）
    from oyosuri_batches import run_batch_collection

    # 3) CSV アップロード
    from google.colab import files
    up = files.upload(); path = next(iter(up))

    # 4) 150 本ずつ区切って，参照期間 10/30/全期間 の 3 系列で全区間を収集
    #    → CSV ＋ 区間ごとの重ね図 PNG を保存（make_images=True が既定）
    batches, manifest = run_batch_collection(
        path, B=4, batch_size=150,
        windows=[10, 30, "full"],     # 本家と同じく 1〜3 個（整数=rolling, "full"=全期間）
        out_dir="oyosuri_batches", symmetric=False,
    )
    manifest          # 1 行 = 1 区間 × 各参照期間のサマリ表（DataFrame）

    # 5) 生成された zip（CSV ＋ PNG 入り）をダウンロード
    from google.colab import files
    files.download("oyosuri_batches.zip")

メモリ上に残った ``batches`` は区間ごとの生データ dict のリストなので，
``plot_batch(batches[0])`` 等でその場で再描画できる。保存済みの
``batch_XXXX.csv`` を ``load_saved_batches`` で読み戻した素の DataFrame
もそのまま ``plot_batch`` に渡せる。

区切らず **全期間を 1 枚の図** にしたいだけなら，本家
``oyosuri_experiment.run_multi_w_overlay(path, B=4, windows=[10, 30, "full"])``
＋ ``fig.savefig(...)`` を使う（本モジュールは不要）。
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# このファイルのあるディレクトリ（colab/）を import パスへ追加し、GitHub clone 後に
# どの作業ディレクトリからでも兄弟モジュール (oyosuri_all_in_one) を解決する。
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
    predict_sequence_with_params,
    rmse,
    slice_by_time,
    walk_from_bricks,
)
from oyosuri_experiment import plot_walk_multi_w
from oyosuri_rolling import predict_sequence_with_params_rolling


# 本家 oyosuri_experiment と同じ「全期間参照」トークン。
_FULL_TOKENS = ("full", "all", "∞", "inf")


def _is_full(W) -> bool:
    """W が「全期間参照」を意味するか（None / "full"/"all"/"inf" 等）."""
    if W is None:
        return True
    if isinstance(W, str):
        return W.strip().lower() in _FULL_TOKENS
    return False


def _label_for(W) -> str:
    """列名・サマリ用のラベル（整数窓 → その数字, 全期間 → "full"）."""
    return "full" if _is_full(W) else str(int(W))


def _preds_params_for_window(
    sub: Sequence[int],
    W,
    *,
    grid_size: int,
    symmetric: bool,
) -> tuple[list[float], list[float], list[float], list[float]]:
    """参照期間 ``W`` での予測列と (p*, q*, α*) 列を返す.

    全期間（None / "full"）なら ``predict_sequence_with_params``、整数なら
    直近 W 本の rolling 版 ``predict_sequence_with_params_rolling``。
    どちらも (preds, ps, qs, alphas) の 4-tuple を返す。
    """
    if _is_full(W):
        return predict_sequence_with_params(
            sub, grid_size=grid_size, symmetric=symmetric
        )
    return predict_sequence_with_params_rolling(
        sub, window=int(W), grid_size=grid_size, symmetric=symmetric
    )


def split_walk_into_batches(
    x: Sequence[int],
    *,
    batch_size: int = 150,
    drop_last: bool = True,
) -> list[tuple[int, int]]:
    """整数ウォーク ``x`` を ``batch_size`` 本ずつの非重複バッチに区切る.

    返すのは各バッチの **ブリック index 範囲** ``(start, end)``（end は
    排他的，すなわち区間は bricks[start:end] の ``end - start`` 本）。
    対応するウォーク部分列は ``x[start : end + 1]``（端点を共有するため
    要素数は ``end - start + 1``）。

    ``drop_last=True`` のとき，端数（``batch_size`` 未満）の最後のバッチは
    捨てる（全サンプルを同じ長さに揃えるため。既定）。``False`` なら
    端数も 1 区間として残す。
    """
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    N = len(list(x)) - 1  # ブリック総数
    out: list[tuple[int, int]] = []
    start = 0
    while start < N:
        end = min(start + batch_size, N)
        if (end - start) < batch_size and drop_last:
            break
        out.append((start, end))
        start = end
    return out


def _dedup_window_labels(windows: Sequence) -> list:
    """1〜3 個の windows を，ラベル重複を除いて順序を保ったまま返す."""
    ws = list(windows)
    if not 1 <= len(ws) <= 3:
        raise ValueError("windows は1〜3個（本家と同じく同時に最大3つ）")
    out: list = []
    seen: set[str] = set()
    for W in ws:
        label = _label_for(W)
        if label in seen:
            continue
        seen.add(label)
        out.append(W)
    return out


def collect_batch_data(
    x: Sequence[int],
    *,
    batch_size: int = 150,
    windows: Sequence = (10, 30, "full"),
    grid_size: int = 11,
    symmetric: bool = False,
    drop_last: bool = True,
) -> list[dict]:
    """ウォーク ``x`` を区切り，各区間の生データ（dict）のリストを返す.

    ``windows`` は参照期間（本家と同じく 1〜3 個）。各要素は整数 W（直近
    W 本の rolling）または "full"/None（区間の全期間）。

    各 dict のキー:
      - ``batch_id``                  … 0 始まりの通し番号
      - ``start_brick`` / ``end_brick`` … 元ウォーク上のブリック index 範囲
      - ``n_bricks``                  … 区間のブリック本数（通常 batch_size）
      - ``base``                      … 再ゼロ化に使った絶対水準 x[start]
      - ``labels``                    … 参照期間ラベルのリスト（例 ["10","30","full"]）
      - ``frame``                     … 区間生データの DataFrame
                                        (列 n, x, 各 W ごとの pred/p/q/alpha)
      - ``metrics``                   … {label: {"mae","rmse","hit"}}（参考）

    ``frame`` はモデルフレーム（x_0 = 0）で揃えてある。絶対水準に戻すには
    ``frame["x"] + base`` とすればよい。
    """
    x_list = list(x)
    windows = _dedup_window_labels(windows)
    ranges = split_walk_into_batches(
        x_list, batch_size=batch_size, drop_last=drop_last
    )
    nan = float("nan")
    batches: list[dict] = []
    for bid, (s, e) in enumerate(ranges):
        base = x_list[s]
        sub = [int(v - base) for v in x_list[s : e + 1]]  # 再ゼロ化（x_0=0）
        L = len(sub) - 1  # = e - s 本
        actual = [float(v) for v in sub[1:]]
        # n=0..L で 1 枚の表に揃える。pred/p/q/alpha は予測ステップ n=1..L に
        # 対応するので先頭 n=0 は NaN で詰める。
        cols: dict[str, list] = {"n": list(np.arange(L + 1)), "x": sub}
        labels: list[str] = []
        metrics: dict[str, dict] = {}
        for W in windows:
            label = _label_for(W)
            labels.append(label)
            preds, ps, qs, alphas = _preds_params_for_window(
                sub, W, grid_size=grid_size, symmetric=symmetric
            )
            cols[f"pred_{label}"] = [nan] + list(preds)
            cols[f"p_{label}"] = [nan] + list(ps)
            cols[f"q_{label}"] = [nan] + list(qs)
            cols[f"alpha_{label}"] = [nan] + list(alphas)
            metrics[label] = {
                "mae": mae(actual, preds),
                "rmse": rmse(actual, preds),
                "hit": hit_rate(actual, preds),
            }
        batches.append(
            {
                "batch_id": bid,
                "start_brick": s,
                "end_brick": e,
                "n_bricks": e - s,
                "base": float(base),
                "labels": labels,
                "frame": pd.DataFrame(cols),
                "metrics": metrics,
            }
        )
    return batches


def build_manifest(batches: list[dict]) -> pd.DataFrame:
    """区間ごとの生データ list から 1 行 = 1 区間のサマリ表を作る.

    各参照期間 ``<label>`` ごとに mae_<label> / rmse_<label> / hit_<label> と，
    区間内（V_n が定数でない予測ステップ）の平均 p_mean_<label> /
    q_mean_<label> / alpha_mean_<label> を列に展開する。
    """
    rows = []
    for b in batches:
        f = b["frame"]
        row = {
            "batch_id": b["batch_id"],
            "start_brick": b["start_brick"],
            "end_brick": b["end_brick"],
            "n_bricks": b["n_bricks"],
            "base": b["base"],
            "x_end": float(f["x"].iloc[-1]),       # 区間終端の到達点（再ゼロ）
        }
        for label in b["labels"]:
            m = b["metrics"][label]
            row[f"mae_{label}"] = m["mae"]
            row[f"rmse_{label}"] = m["rmse"]
            row[f"hit_{label}"] = m["hit"]
            row[f"p_mean_{label}"] = float(np.nanmean(f[f"p_{label}"]))
            row[f"q_mean_{label}"] = float(np.nanmean(f[f"q_{label}"]))
            row[f"alpha_mean_{label}"] = float(np.nanmean(f[f"alpha_{label}"]))
        rows.append(row)
    return pd.DataFrame(rows)


def plot_batch(batch, *, title: str | None = None):
    """1 区間の実測 x_n と各参照期間の予測 x_n* を重ね図にして fig を返す.

    描画本体は本家 ``oyosuri_experiment.plot_walk_multi_w``（実測＝黒太線，
    各系列＝○/□/△ の白抜きマーカー，凡例に MAE 併記）。

    Parameters
    ----------
    batch :
        ``collect_batch_data`` が返す区間 dict（``frame`` / ``labels`` /
        ``metrics`` 等を持つ），または保存済み ``batch_XXXX.csv`` を読み
        戻した素の DataFrame。DataFrame の場合は ``pred_<label>`` 列名から
        参照期間ラベルを自動検出し，MAE はその場で再計算する。
    title :
        図のタイトル。省略時は区間 dict なら
        ``batch 0003  bricks [450, 600)`` の形式，DataFrame ならタイトル無し。
    """
    if isinstance(batch, pd.DataFrame):
        frame = batch
        labels = [c[len("pred_"):] for c in frame.columns
                  if c.startswith("pred_")]
        metrics = None
        default_title = None
    else:
        frame = batch["frame"]
        labels = batch["labels"]
        metrics = batch.get("metrics")
        default_title = (
            f"batch {batch['batch_id']:04d}  "
            f"bricks [{batch['start_brick']}, {batch['end_brick']})"
        )
    if not labels:
        raise ValueError("pred_<label> 列が見つかりません")
    x = [float(v) for v in frame["x"]]
    actual = x[1:]
    preds_by_w: dict[str, list[float]] = {}
    mae_by_w: dict[str, float] = {}
    for label in labels:
        # 先頭 n=0 は NaN 詰め（予測は n=1..L に対応）なので落とす
        preds = [float(v) for v in frame[f"pred_{label}"].iloc[1:]]
        preds_by_w[label] = preds
        if metrics is not None and label in metrics:
            mae_by_w[label] = metrics[label]["mae"]
        else:
            mae_by_w[label] = mae(actual, preds)
    return plot_walk_multi_w(
        x, preds_by_w, mae_by_w=mae_by_w,
        title=default_title if title is None else title,
    )


def save_batch_images(
    batches: list[dict],
    out_dir,
    *,
    dpi: int = 150,
) -> list[Path]:
    """各区間の重ね図を ``out_dir/batch_XXXX.png`` として保存する.

    ``batch_0000.csv`` と同じ連番で対応が取れる。図は保存後すぐ閉じる
    （区間数が多くてもメモリを食わないように）。
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for b in batches:
        fig = plot_batch(b)
        path = out / f"batch_{b['batch_id']:04d}.png"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    print(f"{len(paths)} 枚の重ね図 PNG を {out}/ に保存しました。")
    return paths


def save_batches(
    batches: list[dict],
    out_dir,
    *,
    meta: dict | None = None,
    make_zip: bool = True,
    make_images: bool = True,
    dpi: int = 150,
) -> pd.DataFrame:
    """区間ごとの生データを ``out_dir`` に書き出す.

    出力:
      - ``out_dir/batch_0000.csv`` …各区間の生データ (n, x, pred, p, q, alpha)
      - ``out_dir/batch_0000.png`` …各区間の重ね図（``make_images=True`` 時）
      - ``out_dir/manifest.csv``   …1 行 = 1 区間のサマリ表
      - ``out_dir/meta.json``      …収集条件 (B, batch_size, 期間 など)
      - ``out_dir.zip``            …上記をまとめた zip（``make_zip=True`` 時）

    Returns
    -------
    manifest : pd.DataFrame
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for b in batches:
        b["frame"].to_csv(out / f"batch_{b['batch_id']:04d}.csv", index=False)
    if make_images:
        save_batch_images(batches, out, dpi=dpi)
    manifest = build_manifest(batches)
    manifest.to_csv(out / "manifest.csv", index=False)
    if meta is not None:
        (out / "meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
    if make_zip:
        archive = shutil.make_archive(str(out), "zip", root_dir=str(out))
        print(f"zip 出力: {archive}")
    print(f"{len(batches)} 区間を {out}/ に保存しました。")
    return manifest


def run_batch_collection(
    src,
    B: float,
    *,
    batch_size: int = 150,
    windows: Sequence = (10, 30, "full"),
    start=None,
    end=None,
    tz: str | None = None,
    grid_size: int = 11,
    symmetric: bool = False,
    drop_last: bool = True,
    out_dir="oyosuri_batches",
    make_zip: bool = True,
    make_images: bool = True,
    dpi: int = 150,
) -> tuple[list[dict], pd.DataFrame]:
    """TradingView CSV → 平均練行足 → 150 本ずつ区切って区間生データを収集・保存.

    Parameters
    ----------
    src :
        CSV ファイルのパス / URL / pandas DataFrame。
    B :
        平均練行足のボックス幅（価格単位）。小さいほどブリックが増え，
        同じ期間からより多くの区間が取れる。
    batch_size :
        1 区間あたりのブリック本数（既定 150）。
    windows :
        参照期間（本家 ``run_multi_w_overlay`` と同じく 1〜3 個）。各要素は
        整数 W（直近 W 本の rolling）または "full"/None（区間の全期間）。
        例: [10, 30, "full"] / [20] / [10, 30, 60]。
    start, end, tz :
        当てはめ対象の時間範囲。``oyosuri_all_in_one.run_on_tradingview``
        と同義（DatetimeIndex なら文字列日時，通常 index なら行番号）。
    grid_size, symmetric :
        V_n 最小化の設定（``predict_sequence_with_params`` と同義）。
    drop_last :
        端数区間（batch_size 未満）を捨てるか（既定 True）。
    out_dir :
        出力先ディレクトリ名。``None`` なら保存せずメモリ上の結果のみ返す。
    make_zip :
        True なら ``out_dir.zip`` も作る（Colab でまとめて DL する用）。
    make_images :
        True（既定）なら各区間の重ね図 ``batch_XXXX.png`` も保存する
        （zip にも同梱される）。
    dpi :
        重ね図 PNG の解像度（既定 150）。

    Returns
    -------
    (batches, manifest)
        batches  : 区間ごとの生データ dict のリスト（``collect_batch_data``）。
        manifest : 1 行 = 1 区間 × 各参照期間のサマリ表（DataFrame）。
    """
    df = load_tradingview_csv(src)
    df = slice_by_time(df, start=start, end=end, tz=tz)
    if isinstance(df.index, pd.DatetimeIndex) and len(df) > 0:
        range_str = f"{df.index[0]} → {df.index[-1]}"
        used_start, used_end = str(df.index[0]), str(df.index[-1])
    else:
        range_str = f"rows [0, {len(df)})"
        used_start, used_end = 0, len(df)
    print(f"当てはめ範囲: {range_str}  (bars={len(df)})")

    windows = _dedup_window_labels(windows)
    labels = [_label_for(W) for W in windows]
    bricks, _ = generate_mean_renko_from_ohlc(df, B=B)
    x = walk_from_bricks(bricks)
    N = len(bricks)
    n_full = N // batch_size
    print(
        f"N_bricks={N}, B={B}, batch_size={batch_size} "
        f"→ 非重複で {n_full} 区間（端数 {N - n_full * batch_size} 本）"
    )
    print(f"参照期間: {labels}")
    if N < batch_size:
        raise ValueError(
            f"ブリック数 N={N} が batch_size={batch_size} 未満です。"
            "B を小さくするか，期間を広げてください。"
        )

    batches = collect_batch_data(
        x, batch_size=batch_size, windows=windows, grid_size=grid_size,
        symmetric=symmetric, drop_last=drop_last,
    )
    manifest = build_manifest(batches)
    print(f"収集した区間数: {len(batches)}")
    for label in labels:
        print(
            f"  [{label:>4}] MAE 平均 = "
            f"{manifest[f'mae_{label}'].mean():.4f}  /  "
            f"HIT 平均 = {manifest[f'hit_{label}'].mean():.4f}"
        )

    if out_dir is not None:
        meta = {
            "B": B,
            "batch_size": batch_size,
            "windows": [("full" if _is_full(W) else int(W)) for W in windows],
            "grid_size": grid_size,
            "symmetric": symmetric,
            "drop_last": drop_last,
            "N_bricks": N,
            "n_batches": len(batches),
            "fit_start": used_start,
            "fit_end": used_end,
            "tz": tz,
            "make_images": make_images,
            "dpi": dpi,
        }
        save_batches(
            batches, out_dir, meta=meta, make_zip=make_zip,
            make_images=make_images, dpi=dpi,
        )

    return batches, manifest


# ---------------------------------------------------------------------------
# 保存済みデータの読み戻し（再分析・再描画用）
# ---------------------------------------------------------------------------

def load_saved_batches(out_dir) -> tuple[list[pd.DataFrame], pd.DataFrame]:
    """``save_batches`` / ``run_batch_collection`` の出力を読み戻す.

    Returns
    -------
    (frames, manifest)
        frames   : batch_id 順の区間 DataFrame リスト。
        manifest : manifest.csv を読み込んだ DataFrame。
    """
    out = Path(out_dir)
    manifest = pd.read_csv(out / "manifest.csv")
    frames = [
        pd.read_csv(out / f"batch_{int(bid):04d}.csv")
        for bid in manifest["batch_id"]
    ]
    return frames, manifest


if __name__ == "__main__":
    # 合成 XAUUSD で動作確認（CSV が無くても試せる）。
    from oyosuri_all_in_one import _make_synthetic_xauusd

    demo_df = _make_synthetic_xauusd(n_bars=4000, seed=0)
    batches, manifest = run_batch_collection(
        demo_df, B=0.5, batch_size=150, windows=[10, 30, "full"],
        grid_size=7, out_dir="/tmp/oyosuri_batches_demo",
    )
    print(manifest.to_string(index=False))
