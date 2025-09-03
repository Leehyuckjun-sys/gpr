from __future__ import annotations
import numpy as np
import pandas as pd

# ===== 기본 리샘플/특성 유틸 =====

def resample_uniform(df: pd.DataFrame, t_col: str = "t_ms", hz: int | None = 10) -> pd.DataFrame:
    """균일 시간격자 리샘플. hz가 None/0이면 그대로 반환."""
    if not hz:
        return df.sort_values(t_col).reset_index(drop=True)
    dt_ms = 1000.0 / hz
    grid = np.arange(df[t_col].min(), df[t_col].max() + 0.5 * dt_ms, dt_ms)
    out = (
        df.drop_duplicates(t_col)
          .set_index(t_col)
          .reindex(grid)
          .interpolate()
          .reset_index()
          .rename(columns={"index": t_col})
    )
    return out

def add_speed_accel(df: pd.DataFrame, x_col: str, y_col: str, t_col: str = "t_ms",
                    speed_col: str = "speed", accel_col: str = "accel") -> pd.DataFrame:
    """x,y(m) 궤적으로부터 스칼라 속도/가속도(m/s, m/s^2) 추가."""
    t = df[t_col].to_numpy(dtype=float) / 1000.0
    x = df[x_col].to_numpy(dtype=float)
    y = df[y_col].to_numpy(dtype=float)
    order = np.argsort(t)
    t, x, y = t[order], x[order], y[order]
    vx = np.gradient(x, t, edge_order=2)
    vy = np.gradient(y, t, edge_order=2)
    speed = np.hypot(vx, vy)
    accel = np.gradient(speed, t, edge_order=2)
    df = df.iloc[order].copy()
    df[speed_col] = speed
    df[accel_col] = accel
    return df.reset_index(drop=True)

def build_targets(df: pd.DataFrame,
                  uwb_x: str = "uwb_x", uwb_y: str = "uwb_y",
                  x_ip: str = "x_ip", y_ip: str = "y_ip",
                  out_dx: str = "target_dx", out_dy: str = "target_dy") -> pd.DataFrame:
    """
    타깃 정의: YOLO(px->m)를 UWB 기준으로 보정
    target_dx = uwb_x - x_ip, target_dy = uwb_y - y_ip
    """
    if out_dx not in df.columns:
        df[out_dx] = df[uwb_x] - df[x_ip]
    if out_dy not in df.columns:
        df[out_dy] = df[uwb_y] - df[y_ip]
    return df

# ====== 새로 추가: YOLO/UWB 두 CSV 병합 유틸 ======

def _ensure_yolo_xy(yolo: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """YOLO CSV가 (u,v)만 있으면 호모그래피로 x_ip,y_ip 생성. 이미 있으면 통과."""
    if {"x_ip", "y_ip"}.issubset(yolo.columns):
        return yolo.copy()
    if {"u", "v"}.issubset(yolo.columns):
        # 지연 import로 순환의존 방지
        from .homography_transform import load_H, apply_homography
        H_path = cfg.get("homography_path", None)
        if not H_path:
            raise ValueError("YOLO CSV에 x_ip/y_ip가 없고 (u,v)만 있습니다. homography_path가 필요합니다.")
        H = load_H(H_path)
        xy = apply_homography(H, yolo[["u", "v"]].to_numpy())
        yolo = yolo.copy()
        yolo["x_ip"], yolo["y_ip"] = xy[:, 0], xy[:, 1]
        return yolo
    raise KeyError("YOLO CSV는 x_ip,y_ip 또는 u,v 칼럼 중 하나를 포함해야 합니다.")

def _ensure_v_ip(yolo: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """YOLO 궤적 기반 속도 v_ip가 없으면 x_ip,y_ip로 계산."""
    if "v_ip" in yolo.columns:
        return yolo.copy()
    t_col = cfg.get("time_col", "t_ms")
    tmp = add_speed_accel(yolo, x_col="x_ip", y_col="y_ip", t_col=t_col,
                          speed_col="v_ip", accel_col="v_ip_accel")
    return tmp

def _ensure_uwb_speed_accel(uwb: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """UWB에 speed/accel이 없으면 uwb_x,uwb_y로 계산."""
    if {"speed", "accel"}.issubset(uwb.columns):
        return uwb.copy()
    t_col = cfg.get("time_col", "t_ms")
    tmp = add_speed_accel(uwb, x_col="uwb_x", y_col="uwb_y", t_col=t_col,
                          speed_col="speed", accel_col="accel")
    return tmp

def merge_yolo_uwb(yolo_df: pd.DataFrame, uwb_df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    두 CSV를 하나로 병합해 학습/추론에 바로 쓸 수 있는 DF 생성.
    요구 컬럼:
      YOLO: t_ms + (x_ip,y_ip) 또는 (u,v) [+ v_ip(없으면 자동 계산)]
      UWB : t_ms + uwb_x, uwb_y [+ speed, accel(없으면 자동 계산)]
    처리:
      1) YOLO 시간축에 yolo_time_offset_ms 적용
      2) 공통 시간 구간에 대해 resample_hz 기준 균일 격자 생성
      3) 두 DF를 각각 격자에 보간 후 t_ms로 inner-join
    """
    t_col = cfg.get("time_col", "t_ms")
    hz = cfg.get("resample_hz", 10)
    offset = cfg.get("yolo_time_offset_ms", 0)

    if t_col not in yolo_df.columns or t_col not in uwb_df.columns:
        raise KeyError(f"두 CSV 모두 '{t_col}' 칼럼이 필요합니다.")

    # 0) 필수 열 구성
    yolo = _ensure_yolo_xy(yolo_df, cfg)
    yolo = _ensure_v_ip(yolo, cfg)
    uwb  = _ensure_uwb_speed_accel(uwb_df, cfg)

    # 1) YOLO 시간 오프셋 적용
    yolo = yolo.copy()
    yolo[t_col] = yolo[t_col].astype(float) + float(offset)

    # 2) 공통 시간대 기준 균일 그리드
    y_min = max(yolo[t_col].min(), uwb[t_col].min())
    y_max = min(yolo[t_col].max(), uwb[t_col].max())
    if y_max <= y_min:
        raise ValueError("YOLO/UWB 시간 구간의 겹치는 부분이 없습니다. yolo_time_offset_ms를 조정하세요.")
    if not hz:
        hz = 10
    dt_ms = 1000.0 / hz
    grid = np.arange(y_min, y_max + 0.5 * dt_ms, dt_ms)

    def _to_grid(df: pd.DataFrame) -> pd.DataFrame:
        return (
            df.drop_duplicates(t_col)
              .set_index(t_col)
              .reindex(grid)
              .interpolate()
              .reset_index()
              .rename(columns={"index": t_col})
        )

    yolo_g = _to_grid(yolo)
    uwb_g  = _to_grid(uwb)

    # 3) 같은 t_ms로 병합 (동일 격자이므로 바로 merge)
    #    충돌 가능성이 있는 컬럼명은 미리 설계상 분리(v_ip vs speed)
    out = pd.merge(yolo_g, uwb_g, on=t_col, how="inner", suffixes=("", "_uwb"))
    # (안전) 중복 열 정리 예: uwb_x_uwb → uwb_x로 통일
    for c in ["uwb_x", "uwb_y", "speed", "accel"]:
        cu = c + "_uwb"
        if cu in out.columns and c not in out.columns:
            out = out.rename(columns={cu: c})
        elif cu in out.columns:
            out = out.drop(columns=[cu])
    return out

