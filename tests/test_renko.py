"""Tests for 作成書 §8 item 4 (mean renko symmetry)."""

from __future__ import annotations

import pandas as pd

from oyosuri import (
    generate_mean_renko,
    generate_mean_renko_from_ohlc,
    walk_from_bricks,
)


def test_monotone_up():
    bricks, origins = generate_mean_renko([100, 101, 102, 103], B=1.0)
    assert bricks == [1, 1, 1]
    assert origins == [100.0, 101.0, 102.0, 103.0]


def test_monotone_down():
    bricks, origins = generate_mean_renko([100, 99, 98, 97], B=1.0)
    assert bricks == [-1, -1, -1]
    assert origins == [100.0, 99.0, 98.0, 97.0]


def test_one_box_reversal_generates_new_brick():
    # Core "mean renko" property: a single-box reversal already emits a brick.
    bricks, _ = generate_mean_renko([100, 101, 100, 101], B=1.0)
    assert bricks == [1, -1, 1]


def test_walk_from_bricks_matches_cumsum():
    bricks = [1, 1, -1, 1, -1, -1]
    assert walk_from_bricks(bricks) == [0, 1, 2, 1, 2, 1, 0]


def test_multi_box_jump_collapses_to_multiple_bricks():
    # A single observation three boxes above the origin should emit three
    # bricks in one shot, with origins advancing accordingly.
    bricks, origins = generate_mean_renko([100, 103], B=1.0)
    assert bricks == [1, 1, 1]
    assert origins == [100.0, 101.0, 102.0, 103.0]


def test_symmetry_of_required_price_change():
    # Same price delta in both directions must produce the same brick count
    # — the defining property vs classic renko where reversal needs 2B.
    up_bricks, _ = generate_mean_renko([100, 101], B=1.0)
    down_bricks, _ = generate_mean_renko([100, 99], B=1.0)
    assert len(up_bricks) == 1
    assert len(down_bricks) == 1


def test_sub_box_movement_does_not_emit():
    bricks, origins = generate_mean_renko([100, 100.5, 100.9], B=1.0)
    assert bricks == []
    assert origins == [100.0]


def test_from_ohlc_produces_bricks():
    df = pd.DataFrame(
        [
            {"Open": 100.0, "High": 101.5, "Low": 99.5, "Close": 101.0},
            {"Open": 101.0, "High": 102.2, "Low": 100.1, "Close": 100.3},
        ]
    )
    bricks, _ = generate_mean_renko_from_ohlc(df, B=1.0)
    # Exact sequence depends on the deterministic near-extreme heuristic,
    # but we should at least get some bricks and they should all be ±1.
    assert len(bricks) > 0
    assert all(b in (-1, 1) for b in bricks)
