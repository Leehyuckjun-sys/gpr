from __future__ import annotations
import argparse, json
from pathlib import Path
import yaml
import numpy as np
import pandas as pd
import joblib
from typing import Optional, List

from .gpr_model import GPR2D
from .homography_transform import load_H, apply_homography

def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def _load_bundle(models_root: Path, group_name: str):
    d = models_root / group_name
    scaler = joblib.load(d / "scaler.joblib")
    model = GPR2D.load(d)
    feat_path = d / "features.json"
    feat_cols = json.loads(feat_path.read_text(encoding="utf-8")) if feat_path.exists() else None
    return scaler, model, feat_cols

def correct_block(df: pd.DataFrame, feat_cols: List[str], scaler, model: GPR2D) -> pd.DataFrame:
    X = df[feat_cols].to_numpy(dtype=float)
    Xs = scaler.transform(X)
    mu, std = model.predict(Xs, return_std=True)
    out = df.copy()
    out["dx_hat"], out["dy_hat"] = mu[:, 0], mu[:, 1]
    out["std_dx"], out["std_dy"] = std[:, 0], std[:, 1]
    out["x_corr"] = out["x_ip"] + out["dx_hat"]
    out["y_corr"] = out["y_ip"] + out["dy_hat"]
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="CSV path for inference")
    ap.add_argument("--config", default="configs/example.yaml")
    ap.add_argument("--models_dir", default="models")
    ap.add_argument("--out", required=True, help="output CSV path")
    args = ap.parse_args()

    cfg = load_config(args.config)
    df = pd.read_csv(args.data)

    # 필요 시 px->m 변환
    if {"x_ip", "y_ip"}.isdisjoint(df.columns) and {"u", "v"}.issubset(df.columns):
        H = load_H(cfg["homography_path"])
        xy = apply_homography(H, df[["u", "v"]].to_numpy())
        df["x_ip"], df["y_ip"] = xy[:, 0], xy[:, 1]

    models_root = Path(args.models_dir)
    group_by = cfg.get("group_by", None)

    if group_by:
        parts = []
        for keys, g in df.groupby(group_by, dropna=False):
            name = "_".join([f"{k}={v}" for k, v in zip(group_by, (keys if isinstance(keys, tuple) else (keys,)))])
            d = models_root / name
            if not d.exists():
                d = models_root / "global"
            scaler, model, feat_cols = _load_bundle(models_root, d.name)
            if feat_cols is None:
                feat_cols = cfg["feature_cols"]
            parts.append(correct_block(g, feat_cols, scaler, model))
        out = pd.concat(parts).sort_index()
    else:
        scaler, model, feat_cols = _load_bundle(models_root, "global")
        if feat_cols is None:
            feat_cols = cfg["feature_cols"]
        out = correct_block(df, feat_cols, scaler, model)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"Saved: {args.out}")

if __name__ == "__main__":
    main()
