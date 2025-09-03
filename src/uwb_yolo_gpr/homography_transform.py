from __future__ import annotations
import numpy as np
from pathlib import Path

def compute_homography(src_px: np.ndarray, dst_m: np.ndarray) -> np.ndarray:
    """
    src_px: (N,2) in pixels  [(u,v)]
    dst_m : (N,2) in meters  [(x,y)]
    Returns 3x3 H (dst ≈ H * src), using DLT (N>=4).
    """
    assert src_px.shape == dst_m.shape and src_px.shape[0] >= 4
    A = []
    for (u, v), (x, y) in zip(src_px, dst_m):
        A.append([u, v, 1, 0, 0, 0, -x*u, -x*v, -x])
        A.append([0, 0, 0, u, v, 1, -y*u, -y*v, -y])
    A = np.asarray(A, dtype=float)
    _, _, Vt = np.linalg.svd(A)
    H = Vt[-1].reshape(3, 3)
    return H / H[2, 2]

def apply_homography(H: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """uv: (N,2) pixels -> (N,2) meters"""
    ones = np.ones((uv.shape[0], 1))
    P = np.concatenate([uv, ones], axis=1)  # (N,3)
    XYW = (H @ P.T).T
    xy = XYW[:, :2] / XYW[:, 2:3]
    return xy

def save_H(H: np.ndarray, path: str | Path):
    path = Path(path)
    np.save(path, H)

def load_H(path: str | Path) -> np.ndarray:
    path = Path(path)
    return np.load(path)
