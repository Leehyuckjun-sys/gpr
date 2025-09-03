from __future__ import annotations
import numpy as np
import pandas as pd

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
