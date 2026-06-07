"""Tests for evaluation metrics — 作成書 §7 + MSE / skill 追加分."""

from __future__ import annotations

import pytest

from oyosuri import hit_rate, mae, mse, mse_skill, rmse, walk_from_bricks


def test_mse_equals_rmse_squared():
    """MSE は RMSE の二乗に等しい."""
    actual = [0, 1, 2, 1, 2, 3]
    pred = [0.2, 0.8, 2.1, 1.4, 1.9, 2.7]
    assert mse(actual, pred) == pytest.approx(rmse(actual, pred) ** 2, abs=1e-12)


def test_mse_zero_when_exact():
    actual = [1.0, -2.0, 3.5]
    assert mse(actual, actual) == pytest.approx(0.0, abs=1e-12)


def test_mse_empty_is_zero():
    assert mse([], []) == 0.0


def test_no_change_baseline_mse_is_one():
    """±1 ウォークでは no-change 予測 (x_n*=x_{n-1}) の MSE は厳密に 1.

    これが MSE を読むときの「無情報の基準値」になる（§7 のキー判断）。
    """
    bricks = [1, 1, -1, 1, -1, -1, 1, 1, 1, -1]
    x = walk_from_bricks(bricks)  # x_0=0, 以降 ±1 ずつ
    actual = x[1:]                # x_1..x_N
    nochange = x[:-1]             # x_0..x_{N-1}  (= x_n* で x_{n+1} を予測)
    assert mse(actual, nochange) == pytest.approx(1.0, abs=1e-12)


def test_mse_skill_default_baseline_is_one_minus_mse():
    """baseline=None のとき skill = 1 − MSE_model（基準 MSE = 1）."""
    actual = [0, 1, 0, 1]
    pred = [0.3, 0.6, 0.1, 0.9]
    assert mse_skill(actual, pred) == pytest.approx(
        1.0 - mse(actual, pred), abs=1e-12
    )


def test_mse_skill_perfect_prediction_is_one():
    actual = [0, 1, 0, 1]
    assert mse_skill(actual, actual) == pytest.approx(1.0, abs=1e-12)


def test_mse_skill_against_explicit_baseline():
    """完全予測ならどんな（非自明な）ベースラインに対しても skill = 1."""
    actual = [0.0, 2.0, 4.0]
    baseline = [1.0, 1.0, 1.0]  # 一定予測（base_mse > 0）
    assert mse_skill(actual, actual, baseline) == pytest.approx(1.0, abs=1e-12)


def test_mse_skill_zero_baseline_returns_zero():
    """base_mse == 0（ベースラインが完全）のときは 0.0 を返す."""
    actual = [1.0, 2.0]
    pred = [1.5, 2.5]
    assert mse_skill(actual, pred, baseline=list(actual)) == 0.0


def test_mae_and_hit_rate_still_work():
    """既存指標が回帰していないことの軽いスモークテスト."""
    actual = [0, 1, 2, 1]
    pred = [0.0, 1.0, 1.5, 1.2]
    assert mae(actual, pred) == pytest.approx((0 + 0 + 0.5 + 0.2) / 4, abs=1e-12)
    # actual の向き: +,+,- / pred の向き(前の actual 基準): +,+,-
    assert 0.0 <= hit_rate(actual, pred) <= 1.0
