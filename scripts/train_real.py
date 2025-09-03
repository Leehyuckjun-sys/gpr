# scripts/train_real.py
from __future__ import annotations
import argparse, os, sys, subprocess
from pathlib import Path

def ensure_exists(p: Path, name: str):
    if not p.exists():
        raise FileNotFoundError(f"{name} not found: {p}")

def run():
    parser = argparse.ArgumentParser(description="Train GPR with YOLO+UWB CSVs (real data).")
    parser.add_argument("--yolo", required=True, help="YOLO CSV path (x_ip,y_ip or u,v)")
    parser.add_argument("--uwb",  required=True, help="UWB CSV path (uwb_x,uwb_y)")
    parser.add_argument("--config", default=str(Path("configs") / "example.yaml"))
    parser.add_argument("--models", default=str(Path("models")))
    # 고급: 다른 파이썬으로 실행하고 싶으면 지정(기본: 현재 파이썬)
    parser.add_argument("--python", default=sys.executable, help="Python interpreter to use")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    src_path  = repo_root / "src"
    cfg_path  = Path(args.config)
    models_dir = Path(args.models)

    # 존재 확인
    ensure_exists(Path(args.yolo), "YOLO CSV")
    ensure_exists(Path(args.uwb),  "UWB  CSV")
    ensure_exists(cfg_path,        "config (yaml)")
    models_dir.mkdir(parents=True, exist_ok=True)

    # 하위 프로세스에도 src를 인식시키기
    env = os.environ.copy()
    env["PYTHONPATH"] = (str(src_path) + os.pathsep + env.get("PYTHONPATH", ""))

    cmd = [
        args.python, "-m", "uwb_yolo_gpr.train",
        "--yolo_csv", str(Path(args.yolo)),
        "--uwb_csv",  str(Path(args.uwb)),
        "--config",   str(cfg_path),
        "--models_dir", str(models_dir),
    ]
    print(f"[info] python     : {args.python}")
    print(f"[info] repo_root  : {repo_root}")
    print(f"[info] PYTHONPATH : {env['PYTHONPATH']}")
    print(f"[info] run        : {' '.join([str(c) for c in cmd])}")

    subprocess.run(cmd, check=True, env=env)

    # 결과 요약
    metrics = models_dir / "global" / "metrics.txt"
    if metrics.exists():
        print(f"\n[metrics] {metrics}")
        print(metrics.read_text(encoding="utf-8"))
    else:
        print("[warn] metrics.txt not found. Check the training logs above.")

if __name__ == "__main__":
    run()
