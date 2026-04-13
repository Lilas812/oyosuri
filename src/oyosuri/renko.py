"""Mean Renko (対称練行足) generator — 作成書 §2.

Key property: a reversal only needs a single box, not two. Both continuation
and reversal emit a new brick after B units of price movement, measured from
the *current* brick's opening price ``o_i``.
"""

from __future__ import annotations

from typing import Sequence


def generate_mean_renko(
    prices: Sequence[float],
    B: float,
    base_price: float | None = None,
) -> tuple[list[int], list[float]]:
    """Generate mean renko bricks from a 1D price series.

    Parameters
    ----------
    prices :
        Sequence of price observations in chronological order.
    B :
        Box size (``B > 0``) in the same units as ``prices``.
    base_price :
        Initial reference price ``o_1`` (作成書 §2). When ``None`` the
        first element of ``prices`` is used.

    Returns
    -------
    bricks :
        Sequence of ±1 values. ``+1`` means an up brick, ``-1`` a down
        brick.
    origins :
        The opening prices ``o_i``. ``origins[0]`` is the initial base
        price, ``origins[-1]`` is the opening price of the *next* (not
        yet generated) brick. Thus ``len(origins) == len(bricks) + 1``.
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
        # How many whole boxes past the current origin?
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
        # else: no brick generated
    return bricks, origins


def generate_mean_renko_from_ohlc(
    ohlc,
    B: float,
    base_price: float | None = None,
) -> tuple[list[int], list[float]]:
    """Generate mean renko bricks from an OHLC DataFrame.

    Within each bar we traverse ``Open → nearer extreme → farther extreme
    → Close``. This is the classical heuristic for expanding an OHLC bar
    into an intrabar price path when only OHLC is available.
    """
    required = {"Open", "High", "Low", "Close"}
    missing = required - set(ohlc.columns)
    if missing:
        raise ValueError(f"ohlc is missing columns: {sorted(missing)}")
    pts: list[float] = []
    for row in ohlc.itertuples(index=False):
        o = float(row.Open)
        h = float(row.High)
        lo = float(row.Low)
        c = float(row.Close)
        # Nearer extreme to open first (deterministic heuristic).
        if abs(h - o) <= abs(lo - o):
            seq = [o, h, lo, c]
        else:
            seq = [o, lo, h, c]
        pts.extend(seq)
    return generate_mean_renko(pts, B=B, base_price=base_price)


def walk_from_bricks(bricks: Sequence[int]) -> list[int]:
    """Accumulate ±1 bricks into the integer random-walk ``{x_0, …, x_N}``.

    ``x_0 = 0``, ``x_n = x_{n-1} + b_n``.
    """
    x: list[int] = [0]
    running = 0
    for b in bricks:
        if b not in (-1, 1):
            raise ValueError(f"brick must be ±1, got {b!r}")
        running += int(b)
        x.append(running)
    return x
