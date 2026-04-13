"""Google Colab 用 単一ファイル版.

XAUUSD × 平均練行足 × 相関付きランダムウォーク（一般化2パラメータ型）
の全実装と検証テストを 1 ファイルにまとめたもの。Colab のセルに貼り付けて
そのまま実行できる（numpy / pandas は Colab に同梱済み）。

作成書 §2〜§8 に準拠。構成:
    1. モデル中核 (coin 行列 / Ψ_n 漸化式 / μ_n / V_n / E[S̃_n])
    2. 平均練行足生成器
    3. V_n 最小化器 ([0,1]^3 グリッド探索)
    4. 予測器 (§5 step 2 の (a)(b)(c))
    5. 評価指標 (MAE / RMSE / HIT)
    6. 検証テスト (作成書 §6, §8)
    7. デモ実行
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

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
# 7. デモ実行
# ============================================================================

def run_demo() -> None:
    """ミニ価格列に対して平均練行足 → 予測 → 評価指標 まで一気通貫で表示."""
    print("\n=== Demo: 合成価格列での一気通貫実行 ===")
    prices = [1800, 1802, 1801, 1803, 1805, 1804, 1806, 1807, 1805, 1806]
    print(f"prices: {prices}")

    bricks, origins = generate_mean_renko(prices, B=1.0)
    x = walk_from_bricks(bricks)
    print(f"bricks: {bricks}")
    print(f"walk x: {x}")

    preds = predict_sequence(x, grid_size=11)
    print(f"preds : {[round(v, 4) for v in preds]}")

    actual = x[1:]  # {x_1, …, x_N}
    print(f"actual: {actual}")
    print(
        f"MAE = {mae(actual, preds):.4f}, "
        f"RMSE = {rmse(actual, preds):.4f}, "
        f"HIT = {hit_rate(actual, preds):.4f}"
    )


if __name__ == "__main__":
    run_verification_tests()
    run_demo()
