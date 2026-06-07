"""XAUUSD mean renko × correlated random walk (2-parameter generalization).

Public API mirrors 作成書 §9 ("想定する成果物"):
    - Mean renko generator
    - Psi_n(x; p, q, alpha) recurrence
    - V_n(p, q, alpha) evaluator
    - Argmin over [0, 1]^3 (preserving multiple solutions)
    - E[S̃_{n+1} | p, q, alpha] calculator
    - Prediction sequence generator
    - Metrics: MAE, MSE, RMSE, hit rate, skill score
"""

from .model import (
    Vn_analytic_n1,
    Vn_value,
    coin_matrices,
    expected_position,
    mu_from_psi,
    psi_forward,
)
from .optimize import OptResult, minimize_Vn
from .predict import predict_next, predict_sequence
from .renko import (
    generate_mean_renko,
    generate_mean_renko_from_ohlc,
    walk_from_bricks,
)
from .metrics import mae, rmse, mse, hit_rate, mse_skill

__all__ = [
    "Vn_analytic_n1",
    "Vn_value",
    "coin_matrices",
    "expected_position",
    "mu_from_psi",
    "psi_forward",
    "OptResult",
    "minimize_Vn",
    "predict_next",
    "predict_sequence",
    "generate_mean_renko",
    "generate_mean_renko_from_ohlc",
    "walk_from_bricks",
    "mae",
    "rmse",
    "mse",
    "hit_rate",
    "mse_skill",
]
