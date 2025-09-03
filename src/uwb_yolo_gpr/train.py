from __future__ import annotations
import argparse, json
from pathlib import Path
import yaml
import numpy as np
import pandas as pd
from typing import List, Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import joblib

from .gpr_model import GPConfig, GPR2D
from .sync_and_resample import resample_uniform, add_speed_accel, build_targets
from .homography_transform import load_H, apply_homography


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def prepare_df(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    t_col = cfg.get("time_col", "t_ms")
    hz = cfg.get("resample_hz", 10)

    # 1) 리샘플
    df = resample_uniform(df, t_col=t_col, hz=hz)

    # 2) x_ip/y_ip 없고 (u,v)만 있으면 호모그래피 적용
    if "x_ip" not in df.columns and {"u", "v"}.issubset(df.columns):
        H_path = cfg.get("homography_path", None)
        if not H_path:
            raise ValueError("x_ip/y_ip가 없고 (u,v)만 있는 경우 homography_path가 필요합니다.")
        H = load_H(H_path)
        xy = apply_homography(H, df[["u", "v"]].to_numpy())
        df["x_ip"], df["y_ip"] = xy[:, 0], xy[:, 1]

    # 3) YOLO 궤적 기반 속도(없으면 자동 생성)
    if "v_ip" not in df.columns:
        tmp = add_speed_accel(df, x_col="x_ip", y_col="y_ip", t_col=t_col, speed_col="v_ip", accel_col="v_ip_accel")
        df = tmp

    # 4) 타깃 만들기
    df = build_targets(df, uwb_x="uwb_x", uwb_y="uwb_y", x_ip="x_ip", y_ip="y_ip",
                       out_dx="target_dx", out_dy="target_dy")
    return df

def validate_features(df: pd.DataFrame, feat_cols: List[str]):
    missing = [c for c in feat_cols if c not in df.columns]
    if missing:
        raise KeyError(f"feature_cols에 명시된 컬럼이 CSV에 없습니다: {missing}")

def select_features(df: pd.DataFrame, cfg: dict) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    feat_cols: List[str] = cfg["feature_cols"]
    validate_features(df, feat_cols)
    X = df[feat_cols].to_numpy(dtype=float)
    y = df[["target_dx", "target_dy"]].to_numpy(dtype=float)
    return X, y, feat_cols

def train_one_group(df: pd.DataFrame, cfg: dict, out_dir: Path):
    X, y, feat_cols = select_features(df, cfg)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    Xtr, Xte, ytr, yte = train_test_split(Xs, y, test_size=0.2, random_state=cfg.get("seed", 42))

    mcfg = GPConfig(kernel=cfg["model"]["kernel"],
                    ard=cfg["model"]["ard"],
                    length_scale=cfg["model"]["length_scale"],
                    noise=cfg["model"]["noise"],
                    alpha=cfg["model"]["alpha"],
                    matern_nu=cfg["model"]["matern_nu"])
    model = GPR2D(mcfg, n_features=Xs.shape[1])
    model.fit(Xtr, ytr[:, 0], ytr[:, 1])

    mu, std = model.predict(Xte, return_std=True)
    mae = mean_absolute_error(yte, mu, multioutput="raw_values")  # [mae_dx, mae_dy]
    p90 = np.percentile(np.hypot(yte[:, 0] - mu[:, 0], yte[:, 1] - mu[:, 1]), 90)

    out_dir.mkdir(parents=True, exist_ok=True)
    model.save(out_dir)
    joblib.dump(scaler, out_dir / "scaler.joblib")

    # 재현 가능한 추론을 위해 feature 목록 저장
    (out_dir / "features.json").write_text(json.dumps(feat_cols, ensure_ascii=False, indent=2), encoding="utf-8")

    with open(out_dir / "metrics.txt", "w", encoding="utf-8") as f:
        f.write(f"MAE_dx={mae[0]:.4f}, MAE_dy={mae[1]:.4f}, P90_norm={p90:.4f}\n")
        f.write(f"Features: {feat_cols}\n")

    print(f"[{out_dir.name}] MAE_dx={mae[0]:.3f}, MAE_dy={mae[1]:.3f}, P90={p90:.3f}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="CSV path (must contain t_ms, uwb_x, uwb_y and either x_ip/y_ip or u/v)")
    ap.add_argument("--config", default="configs/example.yaml")
    ap.add_argument("--models_dir", default="models")
    args = ap.parse_args()

    cfg = load_config(args.config)
    df = pd.read_csv(args.data)
    df = prepare_df(df, cfg)

    group_by = cfg.get("group_by", None)
    if group_by:
        groups = df.groupby(group_by, dropna=False)
        for keys, gdf in groups:
            if len(gdf) < cfg.get("min_samples_per_group", 800):
                if cfg.get("fallback_to_global", True):
                    print(f"skip group={keys} (samples {len(gdf)}). Use global model instead.")
                    continue
            name = "_".join([f"{k}={v}" for k, v in zip(group_by, (keys if isinstance(keys, tuple) else (keys,)))])
            out_dir = Path(args.models_dir) / name
            train_one_group(gdf, cfg, out_dir)
    else:
        out_dir = Path(args.models_dir) / "global"
        train_one_group(df, cfg, out_dir)

if __name__ == "__main__":
    main()
