"""データ全体を 150 ステップずつ連続区切りで回す一括収集器 — データ量を稼ぐ版.

``oyosuri_all_in_one.py`` / ``oyosuri_rolling.py`` を import し，平均練行足
ウォーク ``x_n`` を **先頭から ``seg_len`` (既定 150) ステップずつ重複なしで
区切り**，各セグメントを 1 本の独立ウォークとして当てはめ・予測する。
これを「データの始めから終わりまで」繰り返すことで，1 本の CSV から
``⌊N/seg_len⌋`` 本ぶんの予測サンプルを一気に集める。

各セグメントで集める情報（現行 all-in-one の出力に揃えたうえで方向一致を追加）:

  - 実測ウォーク ``x_n`` と 1 ステップ先予測 ``x_n*``
  - 当てはめ最適パラメータ ``p*, q*, α*``（各ステップ）
  - 評価指標 ``MAE`` / ``RMSE`` / ``HIT``（方向的中率）
  - **各ステップで上下の方向が合っているか** ``dir_match``（新規）
      方向は「直前の実測値 ``x_{n-1}`` から見た符号」で判定する:
        実測方向 = sign(x_n   − x_{n-1})   （±1 ウォークなので必ず ±1）
        予測方向 = sign(x_n*  − x_{n-1})   （動かない予測のときは 0）
      両者が一致したステップを的中（``dir_match = 1``）とする。

返り値は long 形式 1 行 = 1 予測ステップの ``rows`` と，セグメント単位の
``summary``（pandas があれば DataFrame，無ければ ``list[dict]``）。

Colab での使い方:

    !git clone https://github.com/Lilas812/oyosuri.git
    import sys; sys.path.insert(0, "/content/oyosuri/colab")

    from oyosuri_segments import collect_segments

    res = collect_segments(path, B=4, seg_len=150,
                           start="2026-05-19-15:00", end="2026-05-19-23:58",
                           tz="Asia/Tokyo")
    res["summary"]                 # セグメント別の MAE / 方向的中率など
    res["rows"]                    # 1 行 = 1 ステップ（実測・予測・方向一致）
    res["rows"].to_csv("dump.csv", index=False)   # まとめて保存したいとき

直接ウォークを渡したいとき（CSV 不要）は ``collect_from_walk(x, seg_len)``。
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

# このファイルのあるディレクトリ（colab/）を import パスへ追加し，GitHub clone 後に
# どの作業ディレクトリからでも兄弟モジュールを解決する。
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
    load_tradingview_csv,
    mae,
    rmse,
    slice_by_time,
    walk_from_bricks,
)
from oyosuri_rolling import predict_sequence_with_params_rolling

try:  # pandas は集計の見やすさのためだけに使う（無くても動く）
    import pandas as _pd
except Exception:  # pragma: no cover - pandas は依存に含まれるが保険
    _pd = None


def direction_match(
    x: Sequence[float], preds: Sequence[float]
) -> tuple[list[int], list[int], list[int]]:
    """各ステップで実測方向と予測方向が一致したかを返す.

    ``x = [x_0,…,x_N]``，``preds = [x_1*,…,x_N*]`` を受け取り，各 n=1..N で

        実測方向 d_n  = sign(x_n  − x_{n-1})
        予測方向 d_n* = sign(x_n* − x_{n-1})

    を計算し，一致フラグ ``1[d_n == d_n*]`` を返す。

    Returns
    -------
    (dir_actual, dir_pred, match)
        いずれも長さ ``N`` の int リスト。``match`` は 0/1。
    """
    x = list(x)
    preds = list(preds)
    if len(preds) != len(x) - 1:
        raise ValueError(
            f"preds length {len(preds)} must equal len(x)-1 = {len(x) - 1}"
        )
    prev = np.asarray(x[:-1], dtype=float)
    actual = np.asarray(x[1:], dtype=float)
    pred = np.asarray(preds, dtype=float)
    d_actual = np.sign(actual - prev).astype(int)
    d_pred = np.sign(pred - prev).astype(int)
    match = (d_actual == d_pred).astype(int)
    return d_actual.tolist(), d_pred.tolist(), match.tolist()


def predict_segment(
    seg_x: Sequence[int],
    *,
    grid_size: int = 11,
    symmetric: bool = False,
) -> dict:
    """1 セグメント（0 始まりのウォーク）を全期間当てはめで予測し情報を集める.

    ``predict_sequence_with_params_rolling`` を ``window = len(seg_x)``（=各
    ステップで利用可能な全履歴）で呼ぶことで，第5章の全期間モデルと同じ予測に
    加えて各ステップの最適パラメータ ``p*, q*, α*`` も同時に得る。

    Returns
    -------
    dict
        ``preds, ps, qs, alphas``（各長さ ``n_steps``），
        ``mae, rmse, hit``（スカラ），
        ``dir_actual, dir_pred, dir_match``（各長さ ``n_steps``）。
    """
    x = [int(v) for v in seg_x]
    n_steps = len(x) - 1
    if n_steps < 1:
        raise ValueError("segment must contain at least 2 points")
    preds, ps, qs, alphas = predict_sequence_with_params_rolling(
        x, window=len(x), grid_size=grid_size, symmetric=symmetric
    )
    actual = [float(v) for v in x[1:]]
    d_actual, d_pred, match = direction_match(x, preds)
    return {
        "preds": preds,
        "ps": ps,
        "qs": qs,
        "alphas": alphas,
        "mae": mae(actual, preds),
        "rmse": rmse(actual, preds),
        "hit": float(np.mean(match)) if match else 0.0,
        "dir_actual": d_actual,
        "dir_pred": d_pred,
        "dir_match": match,
    }


def collect_from_walk(
    x: Sequence[int],
    *,
    seg_len: int = 150,
    grid_size: int = 11,
    symmetric: bool = False,
    drop_last: bool = True,
    brick_offset: int = 0,
):
    """ウォーク ``x`` を ``seg_len`` ステップずつ重複なしで区切り一括収集する.

    Parameters
    ----------
    x :
        ``[x_0,…,x_N]`` の練行足ウォーク（``walk_from_bricks`` の出力）。
    seg_len :
        1 セグメントのステップ数（ブリック数）。既定 150。
    drop_last :
        ``True`` なら端数（``seg_len`` 未満になる末尾）を捨て，全セグメントを
        ちょうど ``seg_len`` ステップに揃える。``False`` なら末尾も短いまま含める。
    brick_offset :
        ブリック通し番号の起点（CSV 全体での位置合わせ用）。

    Returns
    -------
    dict
        ``rows``    : 1 行 = 1 予測ステップ（long 形式）。
        ``summary`` : 1 行 = 1 セグメント。
        ``segments``: セグメント別 ``predict_segment`` 出力のリスト（生データ）。
        pandas があれば ``rows``/``summary`` は DataFrame。
    """
    if seg_len < 2:
        raise ValueError("seg_len must be >= 2 (need at least one step)")
    x = [int(v) for v in x]
    n_bricks = len(x) - 1  # x_0 を除いたステップ総数
    if n_bricks < 1:
        raise ValueError("walk must contain at least 2 points")

    n_full = n_bricks // seg_len
    n_segments = n_full if drop_last else -(-n_bricks // seg_len)  # ceil
    dropped = n_bricks - n_full * seg_len if drop_last else 0
    if n_segments == 0:
        raise ValueError(
            f"ブリック数 {n_bricks} が seg_len={seg_len} に満たないため，"
            f"完全な 150 ステップ・セグメントを作れません。"
            f"seg_len を下げるか期間を広げてください。"
        )

    rows: list[dict] = []
    summary: list[dict] = []
    segments: list[dict] = []

    for s in range(n_segments):
        lo = s * seg_len                          # ブリック index（x のステップ）
        hi = min(lo + seg_len, n_bricks)
        # x のスライスは点 x_lo..x_hi（hi-lo ステップ）。先頭を 0 に再ゼロ化。
        sub = x[lo : hi + 1]
        base = sub[0]
        seg_x = [v - base for v in sub]
        info = predict_segment(
            seg_x, grid_size=grid_size, symmetric=symmetric
        )
        info["segment"] = s
        info["brick_start"] = brick_offset + lo
        info["brick_end"] = brick_offset + hi
        info["n_steps"] = hi - lo
        info["x"] = seg_x
        segments.append(info)

        for i in range(hi - lo):
            n_local = i + 1                       # セグメント内のステップ番号
            rows.append({
                "segment": s,
                "n": n_local,
                "brick": brick_offset + lo + n_local,
                "x_prev": seg_x[i],
                "x_actual": seg_x[i + 1],
                "x_pred": info["preds"][i],
                "p": info["ps"][i],
                "q": info["qs"][i],
                "alpha": info["alphas"][i],
                "abs_err": abs(seg_x[i + 1] - info["preds"][i]),
                "dir_actual": info["dir_actual"][i],
                "dir_pred": info["dir_pred"][i],
                "dir_match": info["dir_match"][i],
            })

        summary.append({
            "segment": s,
            "brick_start": brick_offset + lo,
            "brick_end": brick_offset + hi,
            "n_steps": hi - lo,
            "mae": info["mae"],
            "rmse": info["rmse"],
            "hit": info["hit"],          # 方向的中率（dir_match の平均）
            "n_dir_match": int(sum(info["dir_match"])),
        })

    if _pd is not None:
        rows = _pd.DataFrame(rows)
        summary = _pd.DataFrame(summary)

    return {
        "rows": rows,
        "summary": summary,
        "segments": segments,
        "n_segments": n_segments,
        "seg_len": seg_len,
        "dropped_bricks": dropped,
    }


def collect_segments(
    src,
    B: float,
    *,
    seg_len: int = 150,
    start=None,
    end=None,
    tz: str | None = None,
    grid_size: int = 11,
    symmetric: bool = False,
    drop_last: bool = True,
    verbose: bool = True,
) -> dict:
    """CSV → 平均練行足 → ``seg_len`` ステップ連続区切りで全区間を一括収集.

    ``oyosuri_all_in_one.run_on_tradingview`` と同じ入口（CSV・B・期間指定）から，
    データ全体を ``seg_len`` ステップずつ重複なしで割り，各セグメントの予測・
    最適パラメータ・MAE/RMSE・方向一致をまとめて返す。

    Returns
    -------
    dict
        ``collect_from_walk`` の戻り値に ``bricks`` / ``x`` を加えたもの。
    """
    df = load_tradingview_csv(src)
    df = slice_by_time(df, start=start, end=end, tz=tz)
    is_dt = _pd is not None and isinstance(df.index, _pd.DatetimeIndex)
    if verbose:
        if is_dt and len(df) > 0:
            range_str = f"{df.index[0]} → {df.index[-1]}"
        else:
            range_str = f"rows [0, {len(df)})"
        print(f"当てはめ範囲: {range_str}  (bars={len(df)})")

    bricks, _ = generate_mean_renko_from_ohlc(df, B=B)
    x = walk_from_bricks(bricks)
    N = len(bricks)
    if verbose:
        print(f"N_bricks={N}, B={B}, seg_len={seg_len}")

    out = collect_from_walk(
        x, seg_len=seg_len, grid_size=grid_size, symmetric=symmetric,
        drop_last=drop_last,
    )
    out["bricks"] = bricks
    out["x"] = x

    if verbose:
        n_seg = out["n_segments"]
        print(
            f"集めたセグメント数: {n_seg}（=各 {seg_len} ステップ）"
            f"  / 末尾切り捨て {out['dropped_bricks']} ブリック"
        )
        summary = out["summary"]
        if _pd is not None and len(summary) > 0:
            overall_hit = float(summary["n_dir_match"].sum()) / float(
                summary["n_steps"].sum()
            )
            print(
                f"全体: MAE平均={summary['mae'].mean():.4f}  "
                f"RMSE平均={summary['rmse'].mean():.4f}  "
                f"方向的中率(全ステップ)={overall_hit:.4f}  "
                f"総ステップ数={int(summary['n_steps'].sum())}"
            )
    return out
