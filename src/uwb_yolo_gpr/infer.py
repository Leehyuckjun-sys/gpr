from __future__ import annotations
import argparse, json
from pathlib import Path
import yaml
import numpy as np
import pandas as pd
import joblib
from typing import List, Optional

from .gpr_model import GPR2D
from .homography_transform import load_H, apply_homography
from .sync_and_resample import merge_yolo_uwb, add_speed_accel

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
    # 단일 CSV 모드
    ap.add_argument("--data", help="단일 CSV 경로(통합 파일)")
    # 두 CSV 모드
    ap.add_argument("--yolo_csv", help="YOLO CSV 경로")
    ap.add_argument("--uwb_csv", help="UWB CSV 경로 (평가/로그용; 미제공 시 순수 보정만)")
    ap.add_argument("--config", default="configs/example.yaml")
    ap.add_argument("--models_dir", default="models")
    ap.add_argument("--out", required=True, help="output CSV path")
    args = ap.parse_args()

    cfg = load_config(args.config)
    models_root = Path(args.models_dir)

    # 입력 준비
    if args.data:
        df = pd.read_csv(args.data)
        # x_ip/y_ip 없으면 호모그래피 적용
        if {"x_ip","y_ip"}.isdisjoint(df.columns) and {"u","v"}.issubset(df.columns):
            H = load_H(cfg["homography_path"])
            xy = apply_homography(H, df[["u","v"]].to_numpy())
            df["x_ip"], df["y_ip"] = xy[:,0], xy[:,1]
        if "v_ip" not in df.columns:
            t_col = cfg.get("time_col","t_ms")
            df = add_speed_accel(df, x_col="x_ip", y_col="y_ip", t_col=t_col, speed_col="v_ip", accel_col="v_ip_accel")
    else:
        if not args.yolo_csv:
            raise ValueError("두 CSV 모드: 최소한 --yolo_csv 는 필요합니다.")
        yolo_df = pd.read_csv(args.yolo_csv)
        if args.uwb_csv:
            uwb_df  = pd.read_csv(args.uwb_csv)
            df = merge_yolo_uwb(yolo_df, uwb_df, cfg)  # 평가 가능(uwb_x/uwb_y 포함)
        else:
            # 순수 보정만: YOLO만 준비
            from .sync_and_resample import _ensure_yolo_xy, _ensure_v_ip, resample_uniform  # 내부 유틸 재사용
            yolo_df = _ensure_yolo_xy(yolo_df, cfg)
            yolo_df = _ensure_v_ip(yolo_df, cfg)
            df = resample_uniform(yolo_df, t_col=cfg.get("time_col","t_ms"), hz=cfg.get("resample_hz",10))

    # 모델 로딩(그룹 사용 안 함 = global)
    scaler, model, feat_cols = _load_bundle(models_root, "global")
    if feat_cols is None:
        feat_cols = cfg["feature_cols"]

    out = correct_block(df, feat_cols, scaler, model)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"Saved: {args.out}")

if __name__ == "__main__":
    main()
