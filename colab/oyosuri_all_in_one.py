"""Google Colab 用 単一ファイル版.

XAUUSD × 平均練行足 × 相関付きランダムウォーク（一般化2パラメータ型）
の全実装と検証テストを 1 ファイルにまとめたもの。Colab のセルに貼り付けて
そのまま実行できる（numpy / pandas / matplotlib は Colab に同梱済み）。

作成書 §2〜§8 に準拠。構成:
    1. モデル中核 (coin 行列 / Ψ_n 漸化式 / μ_n / V_n / E[S̃_n])
    2. 平均練行足生成器
    3. V_n 最小化器 ([0,1]^3 グリッド探索)
    4. 予測器 (§5 step 2 の (a)(b)(c))
    5. 評価指標 (ラン長分布 KS 検定 / ACF 比較)
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
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle


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
# 5. 評価指標 — ラン長分布 KS 検定 / ACF 比較
# ============================================================================
# CRW の構造仮定を直接テストする 2 指標:
#   (A) ラン長 KS 検定: 経験ラン長 vs 幾何分布 Geom(1-p), Geom(1-q)
#       → 持続性構造 (モデルの妥当性) の goodness-of-fit
#   (B) ACF 比較: 経験 ACF vs 理論 (p+q-1)^k
#       → 1 次マルコフ仮定が何 lag まで保つか (モデルの限界)
# p, q は遷移回数から直接最尤推定 (CRW の MLE).

def estimate_pq_mle(bricks: Sequence[int]) -> tuple[float, float]:
    """遷移カウントから p, q を最尤推定.

    p = P(b_{i+1}=+1 | b_i=+1),  q = P(b_{i+1}=-1 | b_i=-1).
    該当遷移がゼロの場合は 0.5 (情報なし) を返す.
    """
    arr = np.asarray(list(bricks), dtype=int)
    if arr.size < 2:
        return 0.5, 0.5
    prev = arr[:-1]
    nxt = arr[1:]
    n_plus = int(np.sum(prev == 1))
    n_minus = int(np.sum(prev == -1))
    n_plus_cont = int(np.sum((prev == 1) & (nxt == 1)))
    n_minus_cont = int(np.sum((prev == -1) & (nxt == -1)))
    p_hat = n_plus_cont / n_plus if n_plus > 0 else 0.5
    q_hat = n_minus_cont / n_minus if n_minus > 0 else 0.5
    return float(p_hat), float(q_hat)


def compute_signed_runs(bricks: Sequence[int]) -> tuple[list[int], list[int]]:
    """±1 列を走査し, +1 ラン長と -1 ラン長を分離して返す."""
    runs_plus: list[int] = []
    runs_minus: list[int] = []
    bricks_list = list(bricks)
    if not bricks_list:
        return runs_plus, runs_minus
    cur_sign = bricks_list[0]
    cur_len = 1
    for b in bricks_list[1:]:
        if b == cur_sign:
            cur_len += 1
        else:
            (runs_plus if cur_sign == 1 else runs_minus).append(cur_len)
            cur_sign = b
            cur_len = 1
    (runs_plus if cur_sign == 1 else runs_minus).append(cur_len)
    return runs_plus, runs_minus


def _discrete_geom_ks_stat(runs: Sequence[int], param: float) -> float:
    """離散 KS 統計量 D = sup_k |F_emp(k) - F_geom(k; param)|.

    scipy.stats.kstest は連続分布向けで, 離散分布ではサポート上の
    ジャンプのせいで D が系統的に膨らむ (真に幾何分布のデータで
    D ≈ 0.5 になる). ここではサポート {1, 2, ...} 上で CDF を直接
    比較する正しい離散 KS を実装する.
    """
    arr = np.asarray(list(runs), dtype=int)
    if arr.size == 0:
        return 0.0
    param = float(np.clip(param, 1e-6, 1.0 - 1e-6))
    kmax = int(arr.max())
    k_vals = np.arange(1, kmax + 1)
    F_emp = np.array([float(np.mean(arr <= k)) for k in k_vals])
    F_theo = 1.0 - (1.0 - param) ** k_vals
    return float(np.max(np.abs(F_emp - F_theo)))


def _mc_pvalue_geom_ks(
    runs: Sequence[int], param: float, *, n_boot: int = 999, seed: int = 0
) -> float:
    """H0: Geom(param) からの無作為抽出で D' >= D_obs となる割合 (MC p 値).

    離散 KS は閉じた形の p 値が無いので, 同サイズの擬似データを大量に
    生成して経験的に p 値を評価する (999 回で十分安定).
    """
    arr = np.asarray(list(runs), dtype=int)
    if arr.size < 2:
        return 1.0
    param = float(np.clip(param, 1e-6, 1.0 - 1e-6))
    D_obs = _discrete_geom_ks_stat(arr, param)
    rng = np.random.default_rng(seed)
    count_ge = 0
    for _ in range(n_boot):
        synth = rng.geometric(param, size=arr.size)
        if _discrete_geom_ks_stat(synth, param) >= D_obs:
            count_ge += 1
    return float((count_ge + 1) / (n_boot + 1))


def run_length_ks_test(
    bricks: Sequence[int], p_hat: float, q_hat: float
) -> dict:
    """ラン長分布の離散 KS 検定.

    +1 ラン長 vs Geom(1-p_hat),  -1 ラン長 vs Geom(1-q_hat).
    幾何分布 P(L=k) = p^(k-1)(1-p) に対して, scipy 仕様の
    param = 1 - p (reversal prob) を渡す.
    """
    runs_plus, runs_minus = compute_signed_runs(bricks)
    param_plus = 1.0 - p_hat
    param_minus = 1.0 - q_hat
    D_plus = _discrete_geom_ks_stat(runs_plus, param_plus)
    D_minus = _discrete_geom_ks_stat(runs_minus, param_minus)
    pval_plus = _mc_pvalue_geom_ks(runs_plus, param_plus)
    pval_minus = _mc_pvalue_geom_ks(runs_minus, param_minus)
    return {
        "D_plus": D_plus, "pval_plus": pval_plus,
        "D_minus": D_minus, "pval_minus": pval_minus,
        "runs_plus": runs_plus, "runs_minus": runs_minus,
    }


def empirical_acf(bricks: Sequence[int], max_lag: int) -> np.ndarray:
    """±1 列の経験自己相関 r_k を k = 0..max_lag で返す."""
    arr = np.asarray(list(bricks), dtype=float)
    if arr.size == 0:
        return np.zeros(max_lag + 1)
    arr = arr - arr.mean()
    denom = float(np.dot(arr, arr))
    acf = np.zeros(max_lag + 1)
    acf[0] = 1.0
    if denom <= 0.0:
        return acf
    upper = min(max_lag, arr.size - 1)
    for k in range(1, upper + 1):
        acf[k] = float(np.dot(arr[:-k], arr[k:]) / denom)
    return acf


def theoretical_acf(p_hat: float, q_hat: float, max_lag: int) -> np.ndarray:
    """理論 ACF: lambda^k, lambda = p + q - 1 (対称 q=p で (2p-1)^k)."""
    lam = p_hat + q_hat - 1.0
    k = np.arange(max_lag + 1)
    return np.power(lam, k)


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


def load_tradingview_csv(src) -> pd.DataFrame:
    """TradingView エクスポート CSV / DataFrame を正規化.

    対応入力:
      - str / pathlib.Path: CSV ファイルのパスまたは URL → pd.read_csv
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
        raw = pd.read_csv(str(src))

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


def plot_tradingview_result(
    df: pd.DataFrame,
    bricks: Sequence[int],
    origins: Sequence[float],
    x_walk: Sequence[int],
    preds: Sequence[float],
    B: float,
    p_hat: float,
    q_hat: float,
    ks: dict,
    emp_acf: np.ndarray,
    theo_acf: np.ndarray,
    title: str | None = None,
):
    """4 パネルチャートを matplotlib で描画して Figure を返す.

    Panel 1 : 元の終値ライン + 平均練行足ブリックの box 重ね描き
    Panel 2 : 整数ウォーク x_n (観測) と予測 x_n* (モデル) の折れ線
    Panel 3 : ラン長分布 — 経験ヒストグラム vs 幾何分布 (符号別 2 サブ軸)
    Panel 4 : ACF 比較 — 経験 ACF (棒) vs 理論 (p+q-1)^k (ライン)
    """
    bricks = list(bricks)
    origins = list(origins)
    x_walk = list(x_walk)
    preds = list(preds)
    N = len(bricks)

    fig = plt.figure(figsize=(11, 12))
    gs = fig.add_gridspec(
        4, 2, height_ratios=[3, 2, 2, 2], hspace=0.45, wspace=0.25
    )
    ax_price = fig.add_subplot(gs[0, :])
    ax_walk = fig.add_subplot(gs[1, :])
    ax_run_plus = fig.add_subplot(gs[2, 0])
    ax_run_minus = fig.add_subplot(gs[2, 1])
    ax_acf = fig.add_subplot(gs[3, :])

    # ---- Panel 1: 元の終値 + 練行足ブリック -----------------------------
    if title:
        ax_price.set_title(title)
    else:
        ax_price.set_title(f"Price & Mean Renko bricks (B={B})")

    if len(df) >= 2 and N >= 1:
        price_x = np.linspace(0, N, num=len(df))
        ax_price.plot(
            price_x, df["Close"].to_numpy(),
            color="#888888", linewidth=0.9, label="Close",
        )

    for i, (b, o_start) in enumerate(zip(bricks, origins[:-1])):
        color = "#2ca02c" if b > 0 else "#d62728"
        y_bottom = o_start if b > 0 else o_start - B
        rect = Rectangle(
            (i + 0.05, y_bottom), 0.9, B,
            facecolor=color, edgecolor="black", linewidth=0.4, alpha=0.7,
        )
        ax_price.add_patch(rect)

    ax_price.set_xlim(-0.5, max(N, 1) + 0.5)
    if origins:
        pad = B * 2
        ax_price.set_ylim(min(origins) - pad, max(origins) + pad)
    ax_price.set_ylabel("Price")
    ax_price.grid(alpha=0.3)
    ax_price.legend(loc="upper left", fontsize=8)

    # ---- Panel 2: walk x_n と predict x_n* ------------------------------
    walk_n = np.arange(len(x_walk))
    ax_walk.step(
        walk_n, x_walk, where="post",
        color="#1f77b4", linewidth=1.6, label="x_n (observed)",
    )
    if preds:
        pred_n = np.arange(1, 1 + len(preds))
        ax_walk.plot(
            pred_n, preds,
            color="#ff7f0e", linewidth=1.4, marker="o", markersize=3,
            label="x_n* (predicted)",
        )
    ax_walk.set_title("Integer walk vs prediction")
    ax_walk.set_ylabel("x_n  (box units)")
    ax_walk.set_xlabel("brick index n")
    ax_walk.grid(alpha=0.3)
    ax_walk.legend(loc="upper left", fontsize=8)
    ax_walk.set_xlim(-0.5, max(N, 1) + 0.5)

    # ---- Panel 3: ラン長分布 (符号別) ----------------------------------
    def _plot_run_hist(ax, runs, param, color_emp, color_theo, label):
        ax.set_xlabel("run length L")
        ax.set_ylabel("probability")
        ax.grid(alpha=0.3)
        if len(runs) == 0:
            ax.set_title(f"{label}: (no runs)")
            return
        max_L = max(runs)
        bins = np.arange(1, max_L + 2) - 0.5
        ax.hist(
            runs, bins=bins, density=True,
            color=color_emp, alpha=0.6, edgecolor="black",
            label="empirical",
        )
        ks_clip = float(np.clip(param, 1e-6, 1.0 - 1e-6))
        ks_L = np.arange(1, max_L + 1)
        theo_pmf = (1.0 - ks_clip) ** (ks_L - 1) * ks_clip
        ax.plot(
            ks_L, theo_pmf, marker="o", color=color_theo, linewidth=1.5,
            label=f"Geom(1-param={1-ks_clip:.3f})",
        )
        ax.legend(fontsize=8)

    _plot_run_hist(
        ax_run_plus, ks["runs_plus"], 1.0 - p_hat,
        color_emp="#2ca02c", color_theo="#0a6b0a", label="+1 runs",
    )
    ax_run_plus.set_title(
        f"+1 run length   KS D={ks['D_plus']:.3f}  p={ks['pval_plus']:.3f}"
    )
    _plot_run_hist(
        ax_run_minus, ks["runs_minus"], 1.0 - q_hat,
        color_emp="#d62728", color_theo="#7a0f10", label="-1 runs",
    )
    ax_run_minus.set_title(
        f"-1 run length   KS D={ks['D_minus']:.3f}  p={ks['pval_minus']:.3f}"
    )

    # ---- Panel 4: ACF 比較 ---------------------------------------------
    max_lag = len(emp_acf) - 1
    lags = np.arange(max_lag + 1)
    ax_acf.bar(
        lags, emp_acf, width=0.75,
        color="#1f77b4", alpha=0.7, edgecolor="black", linewidth=0.4,
        label="empirical",
    )
    ax_acf.plot(
        lags, theo_acf, marker="o", color="#ff7f0e", linewidth=1.6,
        label=f"theoretical  (p+q-1)^k",
    )
    ax_acf.axhline(0.0, color="black", linewidth=0.6)
    ax_acf.set_xlabel("lag k")
    ax_acf.set_ylabel("ACF")
    l1 = float(np.sum(np.abs(emp_acf[1:] - theo_acf[1:])))
    ax_acf.set_title(
        f"ACF: empirical vs theoretical   "
        f"p̂={p_hat:.3f}  q̂={q_hat:.3f}  L1(lag1..{max_lag})={l1:.3f}"
    )
    ax_acf.grid(alpha=0.3)
    ax_acf.legend(loc="upper right", fontsize=8)

    return fig


def run_on_tradingview(
    src,
    B: float,
    *,
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
    (fig, result_dict)
        fig: matplotlib Figure
        result_dict: {
            "bricks", "origins", "x", "preds",
            "p_hat", "q_hat",
            "ks": {"D_plus", "pval_plus", "D_minus", "pval_minus",
                   "runs_plus", "runs_minus"},
            "emp_acf", "theo_acf", "acf_l1",
        }
    """
    df = load_tradingview_csv(src)
    bricks, origins = generate_mean_renko_from_ohlc(df, B=B)
    x = walk_from_bricks(bricks)
    preds = predict_sequence(x, grid_size=grid_size, symmetric=symmetric)

    p_hat, q_hat = estimate_pq_mle(bricks)
    ks = run_length_ks_test(bricks, p_hat, q_hat)
    max_lag = min(20, max(len(bricks) - 1, 0))
    emp_acf = empirical_acf(bricks, max_lag=max_lag)
    theo_acf = theoretical_acf(p_hat, q_hat, max_lag=max_lag)
    acf_l1 = float(np.sum(np.abs(emp_acf[1:] - theo_acf[1:]))) if max_lag >= 1 else 0.0

    print(
        f"N_bricks={len(bricks)}  p_hat={p_hat:.4f}  q_hat={q_hat:.4f}\n"
        f"RunLen KS:  D_+={ks['D_plus']:.4f} (p={ks['pval_plus']:.4f})   "
        f"D_-={ks['D_minus']:.4f} (p={ks['pval_minus']:.4f})\n"
        f"ACF L1 dist (lag 1..{max_lag}): {acf_l1:.4f}"
    )

    fig = plot_tradingview_result(
        df, bricks, origins, x, preds, B=B,
        p_hat=p_hat, q_hat=q_hat, ks=ks,
        emp_acf=emp_acf, theo_acf=theo_acf,
        title=title,
    )
    if show:
        plt.show()
    return fig, {
        "bricks": bricks,
        "origins": origins,
        "x": x,
        "preds": preds,
        "p_hat": p_hat,
        "q_hat": q_hat,
        "ks": ks,
        "emp_acf": emp_acf,
        "theo_acf": theo_acf,
        "acf_l1": acf_l1,
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
    fig, _ = run_on_tradingview(
        df, B=1.0, grid_size=11,
        title="Synthetic XAUUSD demo (B=1.0)",
        show=False,
    )
    # Colab ではセル出力に figure が自動表示される。ローカル実行時は
    # 明示的に savefig したい場合に利用。
    out_path = Path("/tmp/oyosuri_demo.png")
    try:
        fig.savefig(out_path, dpi=110)
        print(f"(figure saved to {out_path})")
    except Exception as e:
        print(f"(figure save skipped: {e})")


if __name__ == "__main__":
    run_verification_tests()
    run_demo()
