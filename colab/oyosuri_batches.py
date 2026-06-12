"""150 ブリックごとに区切って区間データを大量に集めるバッチ収集モジュール.

長い TradingView CSV を平均練行足のブリック列に変換し，**150 本ずつ
（重複なし・連続）** のバッチに区切って，各区間の **生データ** をそのまま
保存する。後から自由に再分析・再描画できるアーカイブを作るのが目的。

各バッチで保存する生データ（モデルの x_0 = 0 規約に合わせて再ゼロ化した
「モデルフレーム」で揃える。絶対水準は manifest の ``base`` で復元できる）:

  - n          … 区間内のブリック index 0..L（L = batch_size）
  - x          … 実測ウォーク x_n（再ゼロ化済み，x_0 = 0）
  - pred       … 1 ステップ先予測 x_n*（n=0 は NaN）
  - p, q, alpha … その予測を出した最適化結果（n=0 は NaN）

さらに全区間を 1 行 = 1 区間でまとめた ``manifest.csv`` と，まとめて
ダウンロードするための ``batches.zip`` を出力する。

Colab での使い方:

    # 1) クローンして colab/ を import パスに追加
    !git clone https://github.com/Lilas812/oyosuri.git
    import sys; sys.path.insert(0, "/content/oyosuri/colab")

    # 2) import（依存は自動解決）
    from oyosuri_batches import run_batch_collection

    # 3) CSV アップロード
    from google.colab import files
    up = files.upload(); path = next(iter(up))

    # 4) 150 本ずつ区切って全区間の生データを収集 → 保存
    batches, manifest = run_batch_collection(
        path, B=4, batch_size=150,
        out_dir="oyosuri_batches", symmetric=False,
    )
    manifest          # 1 行 = 1 区間のサマリ表（DataFrame）

    # 5) 生成された zip をダウンロード
    from google.colab import files
    files.download("oyosuri_batches.zip")

メモリ上に残った ``batches`` は区間ごとの DataFrame のリストなので，
そのまま ``batches[0]`` 等で再描画・再分析できる。
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Sequence

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


def collect_batch_data(
    x: Sequence[int],
    *,
    batch_size: int = 150,
    grid_size: int = 11,
    symmetric: bool = False,
    drop_last: bool = True,
) -> list[dict]:
    """ウォーク ``x`` を区切り，各区間の生データ（dict）のリストを返す.

    各 dict のキー:
      - ``batch_id``                  … 0 始まりの通し番号
      - ``start_brick`` / ``end_brick`` … 元ウォーク上のブリック index 範囲
      - ``n_bricks``                  … 区間のブリック本数（通常 batch_size）
      - ``base``                      … 再ゼロ化に使った絶対水準 x[start]
      - ``frame``                     … 区間生データの DataFrame
                                        (列 n, x, pred, p, q, alpha)
      - ``mae`` / ``rmse`` / ``hit``  … 区間内の予測評価（参考）

    ``frame`` はモデルフレーム（x_0 = 0）で揃えてある。絶対水準に戻すには
    ``frame["x"] + base`` とすればよい。
    """
    x_list = list(x)
    ranges = split_walk_into_batches(
        x_list, batch_size=batch_size, drop_last=drop_last
    )
    batches: list[dict] = []
    for bid, (s, e) in enumerate(ranges):
        base = x_list[s]
        sub = [int(v - base) for v in x_list[s : e + 1]]  # 再ゼロ化（x_0=0）
        preds, ps, qs, alphas = predict_sequence_with_params(
            sub, grid_size=grid_size, symmetric=symmetric
        )
        L = len(sub) - 1  # = e - s 本
        # n=0..L で 1 枚の表に揃える。pred/p/q/alpha は予測ステップ n=1..L に
        # 対応するので先頭 n=0 は NaN で詰める。
        nan = float("nan")
        frame = pd.DataFrame(
            {
                "n": np.arange(L + 1),
                "x": sub,
                "pred": [nan] + list(preds),
                "p": [nan] + list(ps),
                "q": [nan] + list(qs),
                "alpha": [nan] + list(alphas),
            }
        )
        actual = [float(v) for v in sub[1:]]
        batches.append(
            {
                "batch_id": bid,
                "start_brick": s,
                "end_brick": e,
                "n_bricks": e - s,
                "base": float(base),
                "frame": frame,
                "mae": mae(actual, preds),
                "rmse": rmse(actual, preds),
                "hit": hit_rate(actual, preds),
            }
        )
    return batches


def build_manifest(batches: list[dict]) -> pd.DataFrame:
    """区間ごとの生データ list から 1 行 = 1 区間のサマリ表を作る.

    p*, q*, α* は区間内（V_n が定数でない予測ステップ）の平均。
    """
    rows = []
    for b in batches:
        f = b["frame"]
        rows.append(
            {
                "batch_id": b["batch_id"],
                "start_brick": b["start_brick"],
                "end_brick": b["end_brick"],
                "n_bricks": b["n_bricks"],
                "base": b["base"],
                "x_end": float(f["x"].iloc[-1]),       # 区間終端の到達点（再ゼロ）
                "mae": b["mae"],
                "rmse": b["rmse"],
                "hit": b["hit"],
                "p_mean": float(np.nanmean(f["p"])),
                "q_mean": float(np.nanmean(f["q"])),
                "alpha_mean": float(np.nanmean(f["alpha"])),
            }
        )
    return pd.DataFrame(rows)


def save_batches(
    batches: list[dict],
    out_dir,
    *,
    meta: dict | None = None,
    make_zip: bool = True,
) -> pd.DataFrame:
    """区間ごとの生データを ``out_dir`` に書き出す.

    出力:
      - ``out_dir/batch_0000.csv`` …各区間の生データ (n, x, pred, p, q, alpha)
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
    start=None,
    end=None,
    tz: str | None = None,
    grid_size: int = 11,
    symmetric: bool = False,
    drop_last: bool = True,
    out_dir="oyosuri_batches",
    make_zip: bool = True,
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

    Returns
    -------
    (batches, manifest)
        batches  : 区間ごとの生データ dict のリスト（``collect_batch_data``）。
        manifest : 1 行 = 1 区間のサマリ表（DataFrame）。
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

    bricks, _ = generate_mean_renko_from_ohlc(df, B=B)
    x = walk_from_bricks(bricks)
    N = len(bricks)
    n_full = N // batch_size
    print(
        f"N_bricks={N}, B={B}, batch_size={batch_size} "
        f"→ 非重複で {n_full} 区間（端数 {N - n_full * batch_size} 本）"
    )
    if N < batch_size:
        raise ValueError(
            f"ブリック数 N={N} が batch_size={batch_size} 未満です。"
            "B を小さくするか，期間を広げてください。"
        )

    batches = collect_batch_data(
        x, batch_size=batch_size, grid_size=grid_size,
        symmetric=symmetric, drop_last=drop_last,
    )
    manifest = build_manifest(batches)
    print(f"収集した区間数: {len(batches)}")
    print(
        "MAE 平均 = "
        f"{manifest['mae'].mean():.4f}  /  "
        f"HIT 平均 = {manifest['hit'].mean():.4f}"
    )

    if out_dir is not None:
        meta = {
            "B": B,
            "batch_size": batch_size,
            "grid_size": grid_size,
            "symmetric": symmetric,
            "drop_last": drop_last,
            "N_bricks": N,
            "n_batches": len(batches),
            "fit_start": used_start,
            "fit_end": used_end,
            "tz": tz,
        }
        save_batches(batches, out_dir, meta=meta, make_zip=make_zip)

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
        demo_df, B=0.5, batch_size=150, out_dir="/tmp/oyosuri_batches_demo",
    )
    print(manifest.to_string(index=False))
