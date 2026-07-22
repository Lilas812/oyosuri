"""Google Colab 用 単一ファイル版.

XAUUSD × 平均練行足 × 相関付きランダムウォーク（一般化2パラメータ型）
の全実装と検証テストを 1 ファイルにまとめたもの。Colab のセルに貼り付けて
そのまま実行できる（numpy / pandas / matplotlib は Colab に同梱済み）。

作成書 §2〜§8 に準拠。構成:
    1. モデル中核 (coin 行列 / Ψ_n 漸化式 / μ_n / V_n / E[S̃_n])
    2. 平均練行足生成器
    3. V_n 最小化器 ([0,1]^3 グリッド探索)
    4. 予測器 (§5 step 2 の (a)(b)(c))
    5. 評価指標 (MAE / RMSE / HIT)
    6. 検証テスト (作成書 §6, §8)
    7. TradingView CSV ローダー + matplotlib チャート描画
    8. デモ実行

Colab での典型的な使い方:

    # TradingView で「Export chart data」→ CSV を保存 → Colab にアップロード
    from google.colab import files
    up = files.upload()
    path = next(iter(up))
    run_on_tradingview(path, B=2.0)        # B はボックス幅（価格単位）

    # 既に pandas DataFrame がある場合はそのまま渡せる
    run_on_tradingview(df, B=2.0)

    # 当てはめたい時間範囲を限定する場合 (start/end は両端を含む)
    run_on_tradingview(path, B=2.0,
                       start="2024-03-01", end="2024-05-31")
    # 日本時間で指定したいとき (TradingView の CSV は UTC なので tz を渡す)
    run_on_tradingview(path, B=2.0,
                       start="2026-05-13-00:24",
                       end="2026-05-13-12:07",
                       tz="Asia/Tokyo")
    # index が DatetimeIndex でないときは行番号 (int) でも指定できる
    run_on_tradingview(df, B=2.0, start=500, end=1500)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================================
# 1. モデル中核 — 作成書 §3, §4
# ============================================================================

def coin_matrices(p: float, q: float) -> tuple[np.ndarray, np.ndarray]:
    """左/右遷移行列 (P, Q) を返す。A(p, q) = P + Q (§3.1)."""
    P = np.array([[q, 1.0 - p], [0.0, 0.0]], dtype=float)
    Q = np.array([[0.0, 0.0], [1.0 - q, p]], dtype=float)
    return P, Q


def psi_forward(n: int, p: float, q: float, alpha: float) -> np.ndarray:
    """Psi_t(x) を t = 0..n, x ∈ [-t, t] で返す (§3.3).

    返り値 shape = (n+1, 2n+1, 2)。psi[t, x+n, :] = Psi_t(x)。
    """
    if n < 0:
        raise ValueError("n must be >= 0")
    P, Q = coin_matrices(p, q)
    width = 2 * n + 1
    psi = np.zeros((n + 1, width, 2), dtype=float)
    psi[0, n, 0] = alpha
    psi[0, n, 1] = 1.0 - alpha
    for t in range(n):
        prev = psi[t]
        left_shift = np.zeros_like(prev)
        left_shift[:-1] = prev[1:]   # Psi_t(x+1)
        right_shift = np.zeros_like(prev)
        right_shift[1:] = prev[:-1]  # Psi_t(x-1)
        psi[t + 1] = left_shift @ P.T + right_shift @ Q.T
    return psi


def mu_from_psi(psi: np.ndarray) -> np.ndarray:
    """|L| + |R| を component 軸に沿って合計し mu_t(x) を返す."""
    return np.abs(psi[..., 0]) + np.abs(psi[..., 1])


def Vn_value(x_obs: Sequence[int], p: float, q: float, alpha: float) -> float:
    """V_n(p, q, alpha) = Σ_{t=0..n} Σ_x (x - x_t)^2 · mu_t(x) (§4)."""
    x_arr = np.asarray(x_obs, dtype=float)
    n = x_arr.size - 1
    if n < 0:
        raise ValueError("x_obs must have at least one element")
    psi = psi_forward(n, p, q, alpha)
    mu = mu_from_psi(psi)
    x_grid = np.arange(-n, n + 1, dtype=float)
    total = 0.0
    for t in range(n + 1):
        diff = x_grid - x_arr[t]
        total += float(np.sum(diff * diff * mu[t]))
    return total


def expected_position(n_target: int, p: float, q: float, alpha: float) -> float:
    """E[S̃_{n_target}] = Σ_x x · mu_{n_target}(x) (§5 注記)."""
    if n_target < 0:
        raise ValueError("n_target must be >= 0")
    psi = psi_forward(n_target, p, q, alpha)
    mu = mu_from_psi(psi)[n_target]
    x_grid = np.arange(-n_target, n_target + 1, dtype=float)
    return float(np.sum(x_grid * mu))


def Vn_analytic_n1(x1: float, p: float, q: float, alpha: float) -> float:
    """V_1 解析式 (5.6'): x1^2 + 2[α(2q-1) + (1-α)(1-2p)] x1 + 1."""
    coeff = alpha * (2.0 * q - 1.0) + (1.0 - alpha) * (1.0 - 2.0 * p)
    return float(x1 * x1 + 2.0 * coeff * x1 + 1.0)


# ============================================================================
# 2. 平均練行足生成器 — 作成書 §2
# ============================================================================

def generate_mean_renko(
    prices: Sequence[float],
    B: float,
    base_price: float | None = None,
) -> tuple[list[int], list[float]]:
    """平均練行足のブリック列と原点列を返す.

    反転にも B 単位しか要らない「対称」練行足。複数ボックス分一気に動いた
    場合は同方向に k 本まとめて生成する (§2)。
    """
    if B <= 0:
        raise ValueError("B must be positive")
    price_list = list(prices)
    if not price_list:
        return [], [base_price if base_price is not None else 0.0]

    o = float(base_price if base_price is not None else price_list[0])
    bricks: list[int] = []
    origins: list[float] = [o]

    for P in price_list:
        diff = float(P) - o
        if diff >= B:
            k = int(diff // B)
            for _ in range(k):
                bricks.append(+1)
                o += B
                origins.append(o)
        elif diff <= -B:
            k = int((-diff) // B)
            for _ in range(k):
                bricks.append(-1)
                o -= B
                origins.append(o)
    return bricks, origins


def generate_mean_renko_from_ohlc(
    ohlc: pd.DataFrame,
    B: float,
    base_price: float | None = None,
) -> tuple[list[int], list[float]]:
    """OHLC DataFrame から平均練行足を生成する. バー内は Open → 近い極値
    → 遠い極値 → Close の順で走査 (古典的近似)."""
    required = {"Open", "High", "Low", "Close"}
    missing = required - set(ohlc.columns)
    if missing:
        raise ValueError(f"ohlc is missing columns: {sorted(missing)}")
    pts: list[float] = []
    for row in ohlc.itertuples(index=False):
        o = float(row.Open); h = float(row.High)
        lo = float(row.Low); c = float(row.Close)
        if abs(h - o) <= abs(lo - o):
            seq = [o, h, lo, c]
        else:
            seq = [o, lo, h, c]
        pts.extend(seq)
    return generate_mean_renko(pts, B=B, base_price=base_price)


def walk_from_bricks(bricks: Sequence[int]) -> list[int]:
    """±1 ブリック列を x_0=0, x_n = x_{n-1} + b_n で累積 (§1)."""
    x: list[int] = [0]
    running = 0
    for b in bricks:
        if b not in (-1, 1):
            raise ValueError(f"brick must be ±1, got {b!r}")
        running += int(b)
        x.append(running)
    return x


# ============================================================================
# 3. V_n 最小化器 — 作成書 §5 step 1
# ============================================================================

@dataclass(frozen=True)
class OptResult:
    V_min: float
    argmins: list[tuple[float, float, float]]
    is_constant: bool


def minimize_Vn(
    x_obs: Sequence[int],
    *,
    grid_size: int = 21,
    symmetric: bool = False,
    tol: float = 1e-10,
) -> OptResult:
    """[0,1]^3 (symmetric=True なら q=p の [0,1]^2) を grid_size 分割した
    グリッド上で V_n を全点評価し、最小値と全 argmin 候補を返す (§5 step 1)."""
    if grid_size < 2:
        raise ValueError("grid_size must be >= 2")
    grid = np.linspace(0.0, 1.0, grid_size)

    best = np.inf
    worst = -np.inf
    values: list[tuple[float, tuple[float, float, float]]] = []

    if symmetric:
        for p in grid:
            for alpha in grid:
                v = Vn_value(x_obs, float(p), float(p), float(alpha))
                values.append((v, (float(p), float(p), float(alpha))))
                if v < best: best = v
                if v > worst: worst = v
    else:
        for p in grid:
            for q in grid:
                for alpha in grid:
                    v = Vn_value(x_obs, float(p), float(q), float(alpha))
                    values.append((v, (float(p), float(q), float(alpha))))
                    if v < best: best = v
                    if v > worst: worst = v

    is_constant = (worst - best) < tol
    argmins = [pt for v, pt in values if v - best <= tol]
    return OptResult(V_min=float(best), argmins=argmins, is_constant=is_constant)


# ============================================================================
# 4. 予測器 — 作成書 §5 step 2 / §7 Step E
# ============================================================================

def predict_next(
    x_obs: Sequence[int],
    *,
    grid_size: int = 21,
    symmetric: bool = False,
    tol: float = 1e-10,
) -> float:
    """次時刻の予測値 x_{n+1}* を返す (§5 step 2).

    (a) V_n 定数      → x_n
    (b) argmin 一意   → E[S̃_{n+1} | p*, q*, α*]
    (c) argmin 複数   → 上記 E の相加平均
    """
    x_list = list(x_obs)
    if not x_list:
        raise ValueError("x_obs must be non-empty")
    n = len(x_list) - 1
    res = minimize_Vn(x_list, grid_size=grid_size, symmetric=symmetric, tol=tol)
    if res.is_constant:
        return float(x_list[-1])
    expectations = [
        expected_position(n + 1, p, q, alpha) for (p, q, alpha) in res.argmins
    ]
    return float(sum(expectations) / len(expectations))


def predict_sequence(
    x_obs: Sequence[int],
    *,
    grid_size: int = 21,
    symmetric: bool = False,
    tol: float = 1e-10,
) -> list[float]:
    """x_obs = {x_0,…,x_N} から [x_1*, x_2*, …, x_N*] を生成 (§7 Step E)."""
    x_list = list(x_obs)
    N = len(x_list) - 1
    preds: list[float] = []
    for n in range(N):
        preds.append(
            predict_next(
                x_list[: n + 1],
                grid_size=grid_size, symmetric=symmetric, tol=tol,
            )
        )
    return preds


# ============================================================================
# 5. 評価指標 — 作成書 §7
# ============================================================================

def _to_arrays(actual, pred):
    a = np.asarray(actual, dtype=float)
    p = np.asarray(pred, dtype=float)
    if a.shape != p.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {p.shape}")
    return a, p


def mae(actual, pred) -> float:
    a, p = _to_arrays(actual, pred)
    return 0.0 if a.size == 0 else float(np.mean(np.abs(a - p)))


def rmse(actual, pred) -> float:
    a, p = _to_arrays(actual, pred)
    return 0.0 if a.size == 0 else float(np.sqrt(np.mean((a - p) ** 2)))


def hit_rate(actual, pred) -> float:
    a, p = _to_arrays(actual, pred)
    if a.size < 2:
        return 0.0
    prev = a[:-1]
    return float(np.mean(np.sign(a[1:] - prev) == np.sign(p[1:] - prev)))


# ============================================================================
# 6. 検証テスト — 作成書 §6, §8
# ============================================================================

def _approx(a: float, b: float, tol: float = 1e-12) -> bool:
    return abs(a - b) <= tol


def run_verification_tests() -> None:
    """作成書 §6 / §8 の検証チェックを一通り実行."""
    passed = 0
    failed: list[str] = []

    def check(name: str, cond: bool) -> None:
        nonlocal passed
        if cond:
            passed += 1
            print(f"  [OK]  {name}")
        else:
            failed.append(name)
            print(f"  [NG]  {name}")

    print("--- §8 item 1: 初期確率の数値一致 ---")
    psi0 = psi_forward(0, 0.3, 0.4, 0.7)
    check("mu_0(0) == 1", _approx(mu_from_psi(psi0)[0, 0], 1.0))
    psi1 = psi_forward(1, 0.3, 0.4, 0.7)
    mu1 = mu_from_psi(psi1)
    expected_minus = 0.4 * 0.7 + (1 - 0.3) * (1 - 0.7)
    expected_plus = (1 - 0.4) * 0.7 + 0.3 * (1 - 0.7)
    check("mu_1(-1) closed form", _approx(mu1[1, 0], expected_minus))
    check("mu_1(+1) closed form", _approx(mu1[1, 2], expected_plus))
    check("mu_1 sum == 1", _approx(mu1[1].sum(), 1.0))

    print("--- §8 item 2: V_1 解析式との一致 ---")
    rng = np.random.default_rng(0)
    all_match = True
    for _ in range(30):
        p, q, a = rng.random(3)
        x1 = int(rng.integers(-3, 4))
        v_num = Vn_value([0, x1], float(p), float(q), float(a))
        v_ana = Vn_analytic_n1(x1, float(p), float(q), float(a))
        if not _approx(v_num, v_ana):
            all_match = False
            break
    check("V_1 numeric == analytic (30 random params)", all_match)

    print("--- §8 item 3: 対称制約下での教科書再現 ---")
    red_ok = True
    for p in [0.0, 0.25, 0.5, 0.75, 1.0]:
        for alpha in [0.0, 0.3, 0.5, 0.8, 1.0]:
            for x1 in [-2, -1, 0, 1, 2]:
                v_gen = Vn_analytic_n1(x1, p, p, alpha)
                v_sym = x1 * x1 + 2.0 * (2 * p - 1) * (2 * alpha - 1) * x1 + 1.0
                if not _approx(v_gen, v_sym):
                    red_ok = False
    check("q=p で V_1 が教科書 (5.6) に一致", red_ok)

    res = minimize_Vn([0, 1, 2], symmetric=True)
    check("D_2={0,1,2} symmetric V_min == 0", _approx(res.V_min, 0.0))
    uniq = {(round(p, 6), round(q, 6), round(a, 6)) for p, q, a in res.argmins}
    check("D_2={0,1,2} symmetric argmin == {(1,1,0)}", uniq == {(1.0, 1.0, 0.0)})
    check("D_2={0,1,2} predict_next == 3", _approx(predict_next([0, 1, 2], symmetric=True), 3.0))

    res = minimize_Vn([0, 1, 0], symmetric=True)
    check("D_2={0,1,0} symmetric V_min == 0", _approx(res.V_min, 0.0))
    uniq = {(round(p, 6), round(q, 6), round(a, 6)) for p, q, a in res.argmins}
    check("D_2={0,1,0} symmetric argmin == {(0,0,1)}", uniq == {(0.0, 0.0, 1.0)})
    check("D_2={0,1,0} predict_next == 1", _approx(predict_next([0, 1, 0], symmetric=True), 1.0))

    print("--- §8 item 4: 平均練行足の対称性 ---")
    up_bricks, _ = generate_mean_renko([100, 101], B=1.0)
    down_bricks, _ = generate_mean_renko([100, 99], B=1.0)
    check("上昇 1 box で 1 本生成", up_bricks == [1])
    check("下降 1 box で 1 本生成", down_bricks == [-1])
    rev_bricks, _ = generate_mean_renko([100, 101, 100, 101], B=1.0)
    check("1 box 反転で新足 (classic renko ではない)", rev_bricks == [1, -1, 1])
    jump_bricks, jump_origins = generate_mean_renko([100, 103], B=1.0)
    check("3 box 一気移動で 3 本", jump_bricks == [1, 1, 1])
    check("多 box jump の原点更新", jump_origins == [100.0, 101.0, 102.0, 103.0])

    print("--- §8 item 5: 複数 argmin 時の相加平均 ---")
    # x_1 = 1 の symmetric 版では argmin = {(0,0,1), (1,1,0)}。
    # E[S̃_2 | 0,0,1] = 0, E[S̃_2 | 1,1,0] = 2。平均 1.0。
    check(
        "symmetric predict_next([0,1]) == 1 (=(0+2)/2)",
        _approx(predict_next([0, 1], symmetric=True), 1.0),
    )

    print("--- §8 item 6: V_n ≡ const 判定 ---")
    check("V_0 is constant", minimize_Vn([0]).is_constant)
    check("x_1=0 の V_1 is constant", minimize_Vn([0, 0]).is_constant)
    check("x_1=0 の予測 == x_0", _approx(predict_next([0, 0]), 0.0))

    print("--- §8 item 7: 左右反転対称性 ---")
    x = [0, 1, 2, 1, 2, 3]
    x_neg = [-v for v in x]
    preds = predict_sequence(x, grid_size=11)
    preds_neg = predict_sequence(x_neg, grid_size=11)
    sym_ok = all(_approx(p, -q, tol=1e-9) for p, q in zip(preds, preds_neg))
    check("predict_sequence(x) == -predict_sequence(-x)", sym_ok)

    print(f"\n==> passed {passed}, failed {len(failed)}")
    if failed:
        print("failed tests:")
        for name in failed:
            print(f"  - {name}")
        raise AssertionError(f"{len(failed)} test(s) failed")


# ============================================================================
# 7. TradingView CSV ローダー + matplotlib チャート描画
# ============================================================================

_COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "time":  ("time", "date", "datetime", "timestamp"),
    "Open":  ("open", "o"),
    "High":  ("high", "h"),
    "Low":   ("low", "l"),
    "Close": ("close", "c", "last"),
}


def _read_csv_any_encoding(src) -> pd.DataFrame:
    """エンコーディングを自動判別して CSV を読む.

    UTF-8 以外（日本語環境のエクスポートに多い cp932 / UTF-16 等）でも
    ``UnicodeDecodeError`` で落ちないように，
      1. ローカルファイルなら先頭バイトの BOM で判別
         （FF FE / FE FF → utf-16, EF BB BF → utf-8-sig）
      2. 判別できなければ utf-8 → cp932 → utf-16 → latin-1 の順に試す
    で読み込む。全て失敗したら最後の例外を再送出する。
    """
    from pathlib import Path as _Path

    encodings = ["utf-8", "cp932", "utf-16", "latin-1"]
    try:
        is_file = _Path(str(src)).is_file()
    except OSError:
        is_file = False
    if is_file:
        with open(str(src), "rb") as f:
            head = f.read(4)
        if head[:2] in (b"\xff\xfe", b"\xfe\xff"):
            encodings = ["utf-16"]
        elif head[:3] == b"\xef\xbb\xbf":
            encodings = ["utf-8-sig"]

    last_err: Exception | None = None
    for enc in encodings:
        try:
            return pd.read_csv(str(src), encoding=enc)
        except (UnicodeDecodeError, UnicodeError) as e:
            last_err = e
    raise last_err


def load_tradingview_csv(src) -> pd.DataFrame:
    """TradingView エクスポート CSV / DataFrame を正規化.

    対応入力:
      - str / pathlib.Path: CSV ファイルのパスまたは URL → pd.read_csv
        （エンコーディングは utf-8 / cp932 / utf-16 等を自動判別）
      - pd.DataFrame: そのまま使う（コピー）

    列名は大文字小文字を無視して次のエイリアスで解決:
      time  ← time / date / datetime / timestamp
      Open  ← open  / O
      High  ← high  / H
      Low   ← low   / L
      Close ← close / C / last

    time が数値なら unix 秒 or ミリ秒を自動判定して DatetimeIndex に
    変換する。欠損列は Close で埋める。

    Returns
    -------
    pd.DataFrame
        index = DatetimeIndex (time 列が無ければ RangeIndex)
        columns = ['Open', 'High', 'Low', 'Close']
    """
    if isinstance(src, pd.DataFrame):
        raw = src.copy()
    else:
        raw = _read_csv_any_encoding(src)

    # lower-case → original のマップを作る
    lower_map = {str(c).strip().lower(): c for c in raw.columns}

    def find(canon_key: str) -> str | None:
        for alias in _COLUMN_ALIASES[canon_key]:
            if alias in lower_map:
                return lower_map[alias]
        return None

    close_col = find("Close")
    if close_col is None:
        raise ValueError(
            f"close 列が見つかりません (columns={list(raw.columns)})"
        )
    close = pd.to_numeric(raw[close_col], errors="coerce")

    open_col = find("Open")
    high_col = find("High")
    low_col = find("Low")

    out = pd.DataFrame(
        {
            "Open": pd.to_numeric(raw[open_col], errors="coerce") if open_col else close,
            "High": pd.to_numeric(raw[high_col], errors="coerce") if high_col else close,
            "Low":  pd.to_numeric(raw[low_col],  errors="coerce") if low_col  else close,
            "Close": close,
        }
    )
    out = out.dropna(subset=["Close"]).reset_index(drop=True)

    time_col = find("time")
    if time_col is not None:
        t_raw = raw[time_col].iloc[: len(out)]
        if pd.api.types.is_numeric_dtype(t_raw):
            # TradingView は秒もミリ秒もあり得る。1e11 超ならミリ秒と推定。
            unit = "ms" if float(t_raw.max()) > 1e11 else "s"
            idx = pd.to_datetime(t_raw, unit=unit, utc=True)
        else:
            idx = pd.to_datetime(t_raw, utc=True, errors="coerce")
        out.index = pd.DatetimeIndex(idx, name="time")
    return out


def predict_sequence_with_params(
    x_obs: Sequence[int],
    *,
    grid_size: int = 21,
    symmetric: bool = False,
    tol: float = 1e-10,
) -> tuple[list[float], list[float], list[float], list[float]]:
    """x_obs = {x_0,…,x_N} から予測列と各ステップの (p*, q*, α*) を返す.

    argmin が複数あるときは各成分を相加平均、V_n が定数のときは NaN.
    """
    x_list = list(x_obs)
    N = len(x_list) - 1
    preds: list[float] = []
    ps: list[float] = []
    qs: list[float] = []
    alphas: list[float] = []
    for n in range(N):
        prefix = x_list[: n + 1]
        res = minimize_Vn(
            prefix, grid_size=grid_size, symmetric=symmetric, tol=tol
        )
        if res.is_constant:
            preds.append(float(prefix[-1]))
            ps.append(float("nan"))
            qs.append(float("nan"))
            alphas.append(float("nan"))
            continue
        expectations = [
            expected_position(n + 1, p, q, alpha)
            for (p, q, alpha) in res.argmins
        ]
        preds.append(float(sum(expectations) / len(expectations)))
        k = len(res.argmins)
        ps.append(sum(pt[0] for pt in res.argmins) / k)
        qs.append(sum(pt[1] for pt in res.argmins) / k)
        alphas.append(sum(pt[2] for pt in res.argmins) / k)
    return preds, ps, qs, alphas


# Keep the original implementations as a regression oracle.  Public sequence
# calls below use the cumulative-moment engine, while ``predict_next`` and the
# mathematical core remain unchanged.
_predict_sequence_legacy = predict_sequence
_predict_sequence_with_params_legacy = predict_sequence_with_params


def predict_sequence_with_params(
    x_obs: Sequence[int],
    *,
    grid_size: int = 21,
    symmetric: bool = False,
    tol: float = 1e-10,
) -> tuple[list[float], list[float], list[float], list[float]]:
    """高速な全期間予測列と各ステップの (p*, q*, alpha*) を返す.

    旧実装と同じグリッド・許容誤差・複数最小解処理を保ったまま、各prefixの
    評価値を一次・二次モーメントから累積更新する。
    """
    from oyosuri_full import predict_sequence_with_params_full_fast

    return predict_sequence_with_params_full_fast(
        x_obs,
        grid_size=grid_size,
        symmetric=symmetric,
        tol=tol,
    )


def predict_sequence(
    x_obs: Sequence[int],
    *,
    grid_size: int = 21,
    symmetric: bool = False,
    tol: float = 1e-10,
) -> list[float]:
    """高速な全期間1ステップ先予測列を返す."""
    return predict_sequence_with_params(
        x_obs,
        grid_size=grid_size,
        symmetric=symmetric,
        tol=tol,
    )[0]


def plot_price(
    df: pd.DataFrame,
    N: int,
    *,
    title: str | None = None,
):
    """元の終値ラインのみを描画した Figure を返す."""
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.set_title(title if title else "Price")
    if len(df) >= 2 and N >= 1:
        price_x = np.linspace(0, N, num=len(df))
        ax.plot(
            price_x, df["Close"].to_numpy(),
            color="#888888", linewidth=0.9, label="Close",
        )
    ax.set_xlim(-0.5, max(N, 1) + 0.5)
    ax.set_ylabel("Price")
    ax.set_xlabel("brick index n")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8,
              borderaxespad=0.0)
    fig.tight_layout()
    return fig


def plot_walk_vs_prediction(
    x_walk: Sequence[int],
    preds: Sequence[float],
):
    """整数ウォーク x_n と予測 x_n* のみを描画した Figure を返す.

    凡例は軸の外側 (右上) に置き、データと重ならないようにする.
    """
    x_walk = list(x_walk)
    preds = list(preds)
    N = max(len(x_walk) - 1, 0)

    fig, ax = plt.subplots(figsize=(13, 5))
    walk_n = np.arange(len(x_walk))
    ax.step(
        walk_n, x_walk, where="post",
        color="#1f77b4", linewidth=1.6, label="x_n (observed)",
    )
    if preds:
        pred_n = np.arange(1, 1 + len(preds))
        ax.plot(
            pred_n, preds,
            color="#ff7f0e", linewidth=1.4, marker="o", markersize=3,
            label="x_n* (predicted)",
        )
    ax.set_title("Integer walk vs prediction")
    ax.set_ylabel("x_n  (box units)")
    ax.set_xlabel("brick index n")
    ax.grid(alpha=0.3)
    ax.set_xlim(-0.5, max(N, 1) + 0.5)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8,
              borderaxespad=0.0)
    fig.tight_layout()
    return fig


def plot_pq(
    ps: Sequence[float],
    qs: Sequence[float],
    *,
    alphas: Sequence[float] | None = None,
    symmetric: bool = False,
):
    """各ステップの最適化結果 (p*, q*, α*) を描画した Figure を返す."""
    ps = list(ps)
    qs = list(qs)
    M = len(ps)
    n_axis = np.arange(1, M + 1)

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(
        n_axis, ps, color="#2ca02c", linewidth=1.4, marker="o", markersize=3,
        label="p*",
    )
    if symmetric:
        ax.set_title("Optimal parameter p* (= q*) per step")
    else:
        ax.plot(
            n_axis, qs, color="#d62728", linewidth=1.4, marker="s",
            markersize=3, label="q*",
        )
        ax.set_title("Optimal parameters p*, q* per step")
    if alphas is not None:
        ax.plot(
            n_axis, list(alphas), color="#9467bd", linewidth=1.0,
            marker="^", markersize=3, linestyle="--", label="α*",
        )
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(0.5, max(M, 1) + 0.5)
    ax.set_xlabel("brick index n")
    ax.set_ylabel("parameter value")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8,
              borderaxespad=0.0)
    fig.tight_layout()
    return fig


def plot_tradingview_result(
    df: pd.DataFrame,
    bricks: Sequence[int],
    origins: Sequence[float],
    x_walk: Sequence[int],
    preds: Sequence[float],
    B: float,
    title: str | None = None,
    *,
    ps: Sequence[float] | None = None,
    qs: Sequence[float] | None = None,
    alphas: Sequence[float] | None = None,
    symmetric: bool = False,
) -> dict:
    """価格 / ウォーク vs 予測 / p,q を 3 つの別 Figure として返す.

    Returns
    -------
    dict
        {"price": Figure, "walk": Figure, "params": Figure or None}
        ``params`` は ``ps`` / ``qs`` が与えられたときのみ含まれる.
    """
    N = len(list(bricks))
    fig_price = plot_price(df, N, title=title)
    fig_walk = plot_walk_vs_prediction(x_walk, preds)
    fig_params = None
    if ps is not None and qs is not None:
        fig_params = plot_pq(ps, qs, alphas=alphas, symmetric=symmetric)
    return {"price": fig_price, "walk": fig_walk, "params": fig_params}


import re as _re

_DASH_DT_RE = _re.compile(
    r"^(\d{4}-\d{1,2}-\d{1,2})[-_T ](\d{1,2}:\d{1,2}(?::\d{1,2})?)\s*$"
)


def _normalize_datetime_str(v):
    """"2026-05-13-00:24" のような区切りも許容して pandas が読める形に直す."""
    if isinstance(v, str):
        m = _DASH_DT_RE.match(v.strip())
        if m:
            return f"{m.group(1)} {m.group(2)}"
    return v


def slice_by_time(
    df: pd.DataFrame,
    start=None,
    end=None,
    tz: str | None = None,
) -> pd.DataFrame:
    """DataFrame を [start, end] の範囲に絞り込む.

    - DatetimeIndex の場合: start/end は文字列 ("2024-03-01",
      "2026-05-13 00:24", "2026-05-13-00:24" など) / pd.Timestamp /
      datetime のいずれでも可。
    - tz : タイムゾーン名 ("Asia/Tokyo" など)。start/end が tz 情報を
      持たない文字列のとき、その tz で解釈してから index の tz
      (TradingView は UTC) に変換する。tz 付き文字列を渡せばそちらが優先。
    - それ以外 (RangeIndex 等): start/end は行番号 (int) として扱う。
    - start/end が None の側はその端まで含む。
    """
    if start is None and end is None:
        return df

    if isinstance(df.index, pd.DatetimeIndex):
        idx_tz = df.index.tz

        def _conv(v):
            if v is None:
                return None
            ts = pd.Timestamp(_normalize_datetime_str(v))
            if ts.tzinfo is None and tz is not None:
                ts = ts.tz_localize(tz)
            if idx_tz is not None:
                if ts.tzinfo is None:
                    ts = ts.tz_localize(idx_tz)
                else:
                    ts = ts.tz_convert(idx_tz)
            elif ts.tzinfo is not None:
                ts = ts.tz_localize(None)
            return ts

        s = _conv(start)
        e = _conv(end)
        out = df.loc[s:e]
    else:
        n = len(df)
        s = 0 if start is None else int(start)
        e = n if end is None else int(end)
        out = df.iloc[s:e]

    if out.empty:
        raise ValueError(
            f"指定範囲にデータがありません (start={start!r}, end={end!r})"
        )
    return out


def run_on_tradingview(
    src,
    B: float,
    *,
    start=None,
    end=None,
    tz: str | None = None,
    grid_size: int = 11,
    symmetric: bool = False,
    title: str | None = None,
    show: bool = True,
):
    """TradingView CSV / DataFrame 入力 → 練行足 → 予測 → チャート描画.

    Parameters
    ----------
    src :
        CSV ファイルのパス, URL, または pandas DataFrame.
    B :
        平均練行足のボックス幅（価格単位）。
    start, end :
        当てはめ対象の時間範囲（両端を含む）。
        DatetimeIndex なら "2024-03-01", "2026-05-13 00:24",
        "2026-05-13-00:24" などの文字列 / pd.Timestamp,
        通常 index なら行番号 (int) で指定する。None ならその端まで。
    tz :
        start/end を解釈するタイムゾーン (例 "Asia/Tokyo")。
        TradingView の CSV は UTC なので、日本時間で指定したいときは
        tz="Asia/Tokyo" を渡せば自動で UTC に変換してから絞り込む。
    grid_size :
        V_n 最小化の 1 軸あたりグリッド点数。11 で十分な場合が多い。
    symmetric :
        True なら教科書互換の q=p 制約モード。
    title :
        Panel 1 のタイトル上書き。
    show :
        True なら plt.show() を呼ぶ（Colab では return された figure が
        暗黙表示されるのでどちらでも可）。

    Returns
    -------
    (figs, result_dict)
        figs : dict[str, matplotlib.figure.Figure]
            {"price", "walk", "params"} の 3 つの Figure.
        result_dict : {"bricks", "origins", "x", "preds", "ps", "qs",
                       "alphas", "start", "end"}
    """
    df = load_tradingview_csv(src)
    df = slice_by_time(df, start=start, end=end, tz=tz)
    if isinstance(df.index, pd.DatetimeIndex) and len(df) > 0:
        range_str = f"{df.index[0]} → {df.index[-1]}"
    else:
        range_str = f"rows [0, {len(df)})"
    print(f"当てはめ範囲: {range_str}  (bars={len(df)})")
    bricks, origins = generate_mean_renko_from_ohlc(df, B=B)
    x = walk_from_bricks(bricks)
    preds, ps, qs, alphas = predict_sequence_with_params(
        x, grid_size=grid_size, symmetric=symmetric
    )
    print(f"N_bricks={len(bricks)}")

    figs = plot_tradingview_result(
        df, bricks, origins, x, preds, B=B, title=title,
        ps=ps, qs=qs, alphas=alphas, symmetric=symmetric,
    )
    if show:
        plt.show()
    if isinstance(df.index, pd.DatetimeIndex) and len(df) > 0:
        used_start, used_end = df.index[0], df.index[-1]
    else:
        used_start, used_end = 0, len(df)
    return figs, {
        "bricks": bricks,
        "origins": origins,
        "x": x,
        "preds": preds,
        "ps": ps,
        "qs": qs,
        "alphas": alphas,
        "start": used_start, "end": used_end,
    }


# ============================================================================
# 8. デモ実行
# ============================================================================

def _make_synthetic_xauusd(n_bars: int = 120, seed: int = 0) -> pd.DataFrame:
    """ダミー XAUUSD OHLC を生成 (ランダムウォーク + ちょいトレンド)."""
    rng = np.random.default_rng(seed)
    base = 1800.0
    closes = [base]
    for _ in range(n_bars - 1):
        closes.append(closes[-1] + rng.normal(0.05, 0.8))
    closes_arr = np.asarray(closes)
    highs = closes_arr + rng.uniform(0.1, 1.0, size=n_bars)
    lows = closes_arr - rng.uniform(0.1, 1.0, size=n_bars)
    opens = np.concatenate([[closes_arr[0]], closes_arr[:-1]])
    idx = pd.date_range("2024-01-01", periods=n_bars, freq="h", tz="UTC")
    return pd.DataFrame(
        {"Open": opens, "High": highs, "Low": lows, "Close": closes_arr},
        index=idx,
    )


def run_demo() -> None:
    """合成 XAUUSD OHLC を TradingView 相当の DataFrame として流し、
    チャートを描画するデモ."""
    print("\n=== Demo: 合成 XAUUSD OHLC での一気通貫実行 ===")
    print(
        "# Colab で実データを使う場合:\n"
        "#   from google.colab import files\n"
        "#   up = files.upload()                 # TradingView の CSV を選択\n"
        "#   path = next(iter(up))\n"
        "#   run_on_tradingview(path, B=2.0)\n"
    )
    df = _make_synthetic_xauusd(n_bars=120, seed=0)
    print(f"OHLC shape={df.shape}, range {df.index[0]} → {df.index[-1]}")
    figs, _ = run_on_tradingview(
        df, B=1.0, grid_size=11,
        title="Synthetic XAUUSD demo (B=1.0)",
        show=False,
    )
    # Colab ではセル出力に figure が自動表示される。ローカル実行時は
    # 明示的に savefig したい場合に利用。
    for name, fig in figs.items():
        if fig is None:
            continue
        out_path = Path(f"/tmp/oyosuri_demo_{name}.png")
        try:
            fig.savefig(out_path, dpi=110, bbox_inches="tight")
            print(f"(figure '{name}' saved to {out_path})")
        except Exception as e:
            print(f"(figure '{name}' save skipped: {e})")


if __name__ == "__main__":
    run_verification_tests()
    run_demo()
