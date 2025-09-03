# scripts/infer_real.py
from __future__ import annotations
import argparse, os, sys, subprocess
from pathlib import Path

def run():
    parser = argparse.ArgumentParser(description="Infer/correct YOLO with trained GPR (real data).")
    parser.add_argument("--yolo", required=True, help="YOLO CSV path for inference")
    parser.add_argument("--uwb",  default=None, help="(optional) UWB CSV for evaluation")
    parser.add_argument("--config", default=str(Path("configs") / "example.yaml"))
    parser.add_argument("--models", default=str(Path("models")))
    parser.add_argument("--out",    default=str(Path("data") / "corrected_real.csv"))
    parser.add_argument("--python", default=sys.executable, help="Python interpreter to use")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    src_path  = repo_root / "src"
    cfg_path  = Path(args.config)
    models_dir = Path(args.models)
    out_path   = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 하위 프로세스에도 src를 인식시키기
    env = os.environ.copy()
    env["PYTHONPATH"] = (str(src_path) + os.pathsep + env.get("PYTHONPATH", ""))

    cmd = [
        args.python, "-m", "uwb_yolo_gpr.infer",
        "--yolo_csv", str(Path(args.yolo)),
        "--config",   str(cfg_path),
        "--models_dir", str(models_dir),
        "--out",      str(out_path),
    ]
    if args.uwb:
        cmd += ["--uwb_csv", str(Path(args.uwb))]

    print(f"[info] python     : {args.python}")
    print(f"[info] repo_root  : {repo_root}")
    print(f"[info] PYTHONPATH : {env['PYTHONPATH']}")
    print(f"[info] run        : {' '.join([str(c) for c in cmd])}")

    subprocess.run(cmd, check=True, env=env)

    print(f"[done] saved -> {out_path}")

    # (선택) UWB가 있으면 개선율 계산
    if args.uwb:
        try:
            import pandas as pd
            import numpy as np
            df = pd.read_csv(out_path)
            cols = {"uwb_x","uwb_y","x_ip","y_ip","x_corr","y_corr"}
            if cols.issubset(df.columns):
                err_raw  = np.hypot(df["uwb_x"]-df["x_ip"],  df["uwb_y"]-df["y_ip"])
                err_corr = np.hypot(df["uwb_x"]-df["x_corr"],df["uwb_y"]-df["y_corr"])
                rmse_raw  = float((err_raw**2).mean()**0.5)
                rmse_corr = float((err_corr**2).mean()**0.5)
                improve   = 100.0*(1 - rmse_corr/rmse_raw) if rmse_raw > 0 else 0.0
                print(f"[eval] RMSE raw={rmse_raw:.4f}  RMSE corr={rmse_corr:.4f}  Improve={improve:.1f}%")
            else:
                print(f"[warn] columns missing for RMSE check: need {sorted(cols)}")
        except Exception as e:
            print(f"[warn] failed to compute improvement: {e}")

if __name__ == "__main__":
    run()
