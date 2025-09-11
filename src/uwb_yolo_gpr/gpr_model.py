# src/uwb_yolo_gpr/gpr_model.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import joblib
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel


@dataclass
class GPConfig:
    # Kernel & core hyperparams
    kernel: str = "RBF"        # "RBF" | "Matern"
    ard: bool = True           # per-feature length-scale (Automatic Relevance Determination)
    length_scale: float | list = 1.0
    noise: float = 1e-3        # WhiteKernel noise level
    alpha: float = 1e-6        # numerical nugget (ridge) for stability
    matern_nu: float = 1.5     # only used if kernel == "Matern"

    # ★ Added: stability/robustness controls (with safe defaults)
    length_scale_bounds: Tuple[float, float] = (1e-4, 1e4)
    noise_bounds: Tuple[float, float] = (1e-6, 1e1)
    normalize_y: bool = True
    n_restarts_optimizer: int = 3


def _make_kernel(cfg: GPConfig, n_features: int):
    """
    Build kernel with ARD (vector length_scale) support and sensible bounds to
    avoid degeneracy (length_scale collapsing to tiny values, etc.).
    """
    # ARD: broadcast per-feature length_scale
    if cfg.ard:
        ls = np.broadcast_to(cfg.length_scale, (n_features,)).astype(float)
    else:
        ls = float(cfg.length_scale)

    # Base kernel with bounds
    if cfg.kernel.upper() == "MATERN":
        base = Matern(
            length_scale=ls,
            nu=cfg.matern_nu,
            length_scale_bounds=cfg.length_scale_bounds,
        )
    else:  # RBF default
        base = RBF(
            length_scale=ls,
            length_scale_bounds=cfg.length_scale_bounds,
        )

    # Additive white noise with bounds
    return base + WhiteKernel(
        noise_level=cfg.noise,
        noise_level_bounds=cfg.noise_bounds,
    )


class GPR2D:
    """
    Two independent Gaussian Processes for dx and dy.
    Trains and predicts correction components separately, then stacks them.
    """
    def __init__(self, cfg: GPConfig, n_features: int):
        k = _make_kernel(cfg, n_features)
        # ★ Added: normalize_y + multiple optimizer restarts for stability
        self.gp_dx = GaussianProcessRegressor(
            kernel=k,
            alpha=cfg.alpha,
            normalize_y=cfg.normalize_y,
            n_restarts_optimizer=cfg.n_restarts_optimizer,
        )
        self.gp_dy = GaussianProcessRegressor(
            kernel=k,
            alpha=cfg.alpha,
            normalize_y=cfg.normalize_y,
            n_restarts_optimizer=cfg.n_restarts_optimizer,
        )
        self.n_features = n_features

    def fit(self, X: np.ndarray, y_dx: np.ndarray, y_dy: np.ndarray):
        assert X.shape[1] == self.n_features, \
            f"Expected {self.n_features} features, got {X.shape[1]}"
        self.gp_dx.fit(X, y_dx)
        self.gp_dy.fit(X, y_dy)

    def predict(
        self,
        X: np.ndarray,
        return_std: bool = True
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        mu_dx, std_dx = self.gp_dx.predict(X, return_std=True)
        mu_dy, std_dy = self.gp_dy.predict(X, return_std=True)
        mu = np.stack([mu_dx, mu_dy], axis=1)
        if return_std:
            std = np.stack([std_dx, std_dy], axis=1)
            return mu, std
        return mu, None

    def save(self, out_dir: str | Path):
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.gp_dx, out / "gp_dx.joblib")
        joblib.dump(self.gp_dy, out / "gp_dy.joblib")

    @staticmethod
    def load(in_dir: str | Path) -> "GPR2D":
        in_dir = Path(in_dir)
        gp_dx = joblib.load(in_dir / "gp_dx.joblib")
        gp_dy = joblib.load(in_dir / "gp_dy.joblib")

        model = object.__new__(GPR2D)  # bypass __init__
        model.gp_dx = gp_dx
        model.gp_dy = gp_dy

        # Infer input dimensionality from the fitted kernel (k1 = base kernel)
        # length_scale can be scalar or vector; ensure 1D and take its length
        ls = gp_dx.kernel_.k1.length_scale
        model.n_features = len(np.atleast_1d(ls))
        return model
