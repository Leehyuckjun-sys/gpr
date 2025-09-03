from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import joblib
from typing import Tuple, Optional

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel


@dataclass
class GPConfig:
    kernel: str = "RBF"        # RBF | Matern
    ard: bool = True           # per-feature length-scale
    length_scale: float | list = 1.0
    noise: float = 1e-3        # WhiteKernel noise level
    alpha: float = 1e-6        # numerical nugget
    matern_nu: float = 1.5     # if kernel == "Matern"


def _make_kernel(cfg: GPConfig, n_features: int):
    # ARD: 피처별 length_scale (길이 n_features로 브로드캐스트)
    if cfg.ard:
        ls = np.broadcast_to(cfg.length_scale, (n_features,)).astype(float)
    else:
        ls = float(cfg.length_scale)

    base = Matern(length_scale=ls, nu=cfg.matern_nu) if cfg.kernel.upper() == "MATERN" else RBF(length_scale=ls)
    return base + WhiteKernel(noise_level=cfg.noise)


class GPR2D:
    """dx, dy 각각에 대해 독립 GP를 학습·추론"""
    def __init__(self, cfg: GPConfig, n_features: int):
        k = _make_kernel(cfg, n_features)
        self.gp_dx = GaussianProcessRegressor(kernel=k, alpha=cfg.alpha, normalize_y=False)
        self.gp_dy = GaussianProcessRegressor(kernel=k, alpha=cfg.alpha, normalize_y=False)
        self.n_features = n_features

    def fit(self, X: np.ndarray, y_dx: np.ndarray, y_dy: np.ndarray):
        assert X.shape[1] == self.n_features
        self.gp_dx.fit(X, y_dx)
        self.gp_dy.fit(X, y_dy)

    def predict(self, X: np.ndarray, return_std: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
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
        model = object.__new__(GPR2D)
        model.gp_dx = gp_dx
        model.gp_dy = gp_dy
        # 입력 차원: 커널의 length_scale 길이로 추정 (k1=RBF/Matern, k2=White)
        ls = gp_dx.kernel_.k1.length_scale
        model.n_features = len(np.atleast_1d(ls))
        return model
