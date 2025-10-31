# utily.py — keep report schema, renew math
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Sequence, Optional

import numpy as np
import pandas as pd

# 必須列（レポート既存スキーマ）
REQUIRED_COLUMNS: Sequence[str] = (
    "Time",
    "LFx", "LFy", "LFz", "LMx", "LMy", "LMz", "LTz", "LPx", "LPy",
    "RFx", "RFy", "RFz", "RMx", "RMy", "RMz", "RTz", "RPx", "RPy",
    "MPx", "MPy", "MFx", "MFy", "MFz", "MTz",
)

# ───────── helpers ─────────
def _assert_columns(df: pd.DataFrame, cols: Sequence[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"CSVに必要な列が不足: {missing}")

def _nan_safe(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    a[~np.isfinite(a)] = np.nan
    if np.all(np.isnan(a)):  # 全NaNならそのまま
        return a
    # 線形内挿（端は最近傍保持）
    return pd.Series(a).interpolate("linear", limit_direction="both").to_numpy(dtype=float)

def _moving_mean(x: np.ndarray, win_len: int) -> np.ndarray:
    if win_len <= 1:
        return x.copy()
    win = np.ones(win_len) / win_len
    y = np.convolve(x, win, mode="same")
    return y

def _impulse(x: np.ndarray, dt: float) -> float:
    x = _nan_safe(x)
    return float(np.trapz(x, dx=dt))

def _path_length(x: np.ndarray, y: np.ndarray) -> float:
    dx = np.diff(_nan_safe(x))
    dy = np.diff(_nan_safe(y))
    return float(np.nansum(np.hypot(dx, dy)))

def _rfd_peak(x: np.ndarray, dt: float, around_idx: int | None = None, halfwin: int = 10) -> float:
    """最大立ち上がり速度（RFD）。around_idx 指定なら近傍の最大傾斜を取る。"""
    x = _nan_safe(x)
    if len(x) < 3:
        return float("nan")
    g = np.gradient(x, dt)  # 可変間隔不要 → dt固定
    if around_idx is None:
        return float(np.nanmax(np.abs(g)))
    s = max(0, around_idx - halfwin); e = min(len(g), around_idx + halfwin + 1)
    return float(np.nanmax(np.abs(g[s:e]))) if e > s else float("nan")

def _pick_axes_prefix(is_right_handed: bool) -> Tuple[str, str]:
    # 右打ち: Axis=右足, Stride=左足 / 左打ちは逆
    return ("R", "L") if is_right_handed else ("L", "R")

# ───────── swing window detection ─────────
@dataclass(frozen=True)
class SwingWindow:
    idx_start: int
    idx_peak: int
    idx_end: int
    t_start: float
    t_peak: float
    t_end: float

    @property
    def idx_range(self) -> np.ndarray:
        return np.arange(self.idx_start, self.idx_end + 1, dtype=int)

def find_swing_window_by_mtz_peak(
    df: pd.DataFrame,
    fs: float,
    smooth_sec: float = 1.0,
) -> SwingWindow:
    """MTzピークを中心に適度な長さの窓を切る（既存仕様の堅牢版）。"""
    _assert_columns(df, REQUIRED_COLUMNS)
    dt = 1.0 / fs
    N = len(df)
    mTz = _nan_safe(df["MTz"].to_numpy())
    win = max(1, int(round(smooth_sec * fs)))

    # 1) ピーク検出（平滑+絶対値ピーク）
    mTz_abs = np.abs(mTz)
    idx_peak = int(np.argmax(mTz_abs)) if mTz_abs.size else 0
    t_peak = idx_peak / fs

    # 2) 起点：ピークより前の最小（緩やかな谷）をスタート
    mTz_smooth = _moving_mean(mTz, win)
    idx_start = int(np.argmin(mTz_smooth[:max(1, idx_peak)])) if idx_peak > 0 else 0
    t_start = idx_start / fs

    # 3) 終点：ピーク+win で適度に切る（必要なら調整可）
    idx_end = min(N - 1, idx_peak + win)
    t_end = idx_end / fs

    return SwingWindow(idx_start, idx_peak, idx_end, t_start, t_peak, t_end)

# ───────── main: batting analysis (table/cards) ─────────
def analyze_fp_batting(
    df: pd.DataFrame,
    fs: float,
    is_right_handed: bool,
    body_weight: float,
    smooth_sec: float = 1.0,
) -> Dict[str, float]:
    """
    レポート本文・カード類が参照する英語キーを従来名のまま返す。
    - Fz/Fx ピーク・RFD、BW正規化、mTzピーク・RFD・インパルス、mFzインパルス
    - Fz最大時の合成CoP, 合成CoP速度最大 etc.
    """
    _assert_columns(df, REQUIRED_COLUMNS)
    dt = 1.0 / fs
    N = len(df)

    # 窓決定（mTz ピーク中心）
    win = find_swing_window_by_mtz_peak(df, fs, smooth_sec=smooth_sec)
    r = win.idx_range

    # 軸足/踏込足のプレフィックス
    axis_p, stride_p = _pick_axes_prefix(is_right_handed)

    # --- COP移動量（足内） ---
    def _cop_movement(prefix: str) -> float:
        x = _nan_safe(df[f"{prefix}Px"].to_numpy())
        y = _nan_safe(df[f"{prefix}Py"].to_numpy())
        return _path_length(x[r], y[r])

    COPmove_axis   = _cop_movement(axis_p)
    COPmove_stride = _cop_movement(stride_p)

    # --- Fz（軸/踏込） ---
    Fz_axis   = _nan_safe(df[f"{axis_p}Fz"].to_numpy())
    Fz_stride = _nan_safe(df[f"{stride_p}Fz"].to_numpy())
    Fz_axis_seg   = Fz_axis[r];   Fz_stride_seg = Fz_stride[r]
    idx_Fz_axis_pk   = int(np.argmax(Fz_axis_seg)) if Fz_axis_seg.size else 0
    idx_Fz_stride_pk = int(np.argmax(Fz_stride_seg)) if Fz_stride_seg.size else 0
    Fz_peak_axis   = float(np.max(Fz_axis_seg))   if Fz_axis_seg.size   else float("nan")
    Fz_peak_stride = float(np.max(Fz_stride_seg)) if Fz_stride_seg.size else float("nan")
    RFD_Fz_axis   = _rfd_peak(Fz_axis_seg, dt, idx_Fz_axis_pk)
    RFD_Fz_stride = _rfd_peak(Fz_stride_seg, dt, idx_Fz_stride_pk)

    # --- Fx（軸/踏込） ---
    Fx_axis   = _nan_safe(df[f"{axis_p}Fx"].to_numpy()); Fx_axis_seg = Fx_axis[r]
    Fx_stride = _nan_safe(df[f"{stride_p}Fx"].to_numpy()); Fx_stride_seg = Fx_stride[r]
    idx_Fx_axis_pk   = int(np.argmax(Fx_axis_seg)) if Fx_axis_seg.size else 0
    idx_Fx_stride_pk = int(np.argmax(Fx_stride_seg)) if Fx_stride_seg.size else 0
    Fx_peak_axis   = float(np.max(Fx_axis_seg))   if Fx_axis_seg.size   else float("nan")
    Fx_peak_stride = float(np.max(Fx_stride_seg)) if Fx_stride_seg.size else float("nan")
    RFD_Fx_axis   = _rfd_peak(Fx_axis_seg, dt, idx_Fx_axis_pk)
    RFD_Fx_stride = _rfd_peak(Fx_stride_seg, dt, idx_Fx_stride_pk)

    # --- 合力（mFz） & 回旋（mTz） ---
    mFz = _nan_safe(df["MFz"].to_numpy()); mFz_seg = mFz[r]
    mTz = _nan_safe(df["MTz"].to_numpy()); mTz_seg = mTz[r]
    idx_Tz_peak = int(np.argmax(np.abs(mTz_seg))) if mTz_seg.size else 0
    mFz_impulse = _impulse(np.maximum(mFz_seg, 0.0), dt)
    mTz_peak    = float(np.max(np.abs(mTz_seg))) if mTz_seg.size else float("nan")
    mTz_peakBW  = float(mTz_peak / body_weight) if body_weight else float("nan")
    mTz_RFD     = _rfd_peak(mTz_seg, dt, idx_Tz_peak)
    mTz_impulse = _impulse(np.abs(mTz_seg), dt)

    # --- Fz最大時の合成COP位置 & 合成CoP速度 ---
    idx_Fzmax = int(np.argmax(mFz)) if mFz.size else 0
    MPx = _nan_safe(df["MPx"].to_numpy()); MPy = _nan_safe(df["MPy"].to_numpy())
    COPX_atFzmax = float(MPx[idx_Fzmax]) if MPx.size else float("nan")
    COPY_atFzmax = float(MPy[idx_Fzmax]) if MPy.size else float("nan")
    COP_speed = np.hypot(np.diff(MPx), np.diff(MPy)) * fs
    COP_speed_max = float(np.max(COP_speed)) if COP_speed.size else float("nan")

    # まとめ（既存キーそのまま）
    out: Dict[str, float] = {
        "tStart": win.t_start, "tPeak": win.t_peak, "tEnd": win.t_end,

        "COPmove_axis": COPmove_axis, "COPmove_stride": COPmove_stride,

        "Fz_peak_axis": Fz_peak_axis, "Fz_peak_stride": Fz_peak_stride,
        "Fz_peakBW_axis": Fz_peak_axis / body_weight if body_weight else float("nan"),
        "Fz_peakBW_stride": Fz_peak_stride / body_weight if body_weight else float("nan"),
        "Fz_RFD_axis": RFD_Fz_axis, "Fz_RFD_stride": RFD_Fz_stride,

        "Fx_peak_axis": Fx_peak_axis, "Fx_peak_stride": Fx_peak_stride,
        "Fx_peakBW_axis": Fx_peak_axis / body_weight if body_weight else float("nan"),
        "Fx_peakBW_stride": Fx_peak_stride / body_weight if body_weight else float("nan"),
        "Fx_RFD_axis": RFD_Fx_axis, "Fx_RFD_stride": RFD_Fx_stride,

        "mFz_impulse": mFz_impulse,
        "mTz_peak": mTz_peak, "mTz_peakBW": mTz_peakBW, "mTz_RFD": mTz_RFD, "mTz_impulse": mTz_impulse,

        "COPX_atFzmax": COPX_atFzmax, "COPY_atFzmax": COPY_atFzmax,
        "COP_speed_max": COP_speed_max,
    }
    # NaN→0 安全化
    for k, v in list(out.items()):
        if not np.isfinite(v): out[k] = 0.0
    return out

# ───────── radar metrics (Japanese keys for the template) ─────────
def compute_cog_cop_metrics_from_fp(
    df: pd.DataFrame,
    fs: float | None = None,
    use_window: Optional[SwingWindow] = None,
) -> Dict[str, float]:
    """
    レポートの「重心移動指標」4項目（キー名は従来の日本語ラベルそのまま）:
      - 重心移動量             : MPx/MPy のパス長（全区間 or 指定ウィンドウ）
      - 足内CoP移動量（左/右） : LPx/LPy と RPx/RPy のパス長
      - ピーク時重心バランス   : Fz合計最大時の LFz/(LFz+RFz)
    """
    _assert_columns(df, REQUIRED_COLUMNS)

    # 解析範囲
    if use_window is not None:
        r = use_window.idx_range
    else:
        if fs:  # fs があれば mTzピーク±1秒を既定とする（従来に近い可視化）
            win = find_swing_window_by_mtz_peak(df, fs)
            r = win.idx_range
        else:
            r = np.arange(len(df), dtype=int)

    # 1) 重心移動量（=合成CoP軌跡の長さ）
    MPx = _nan_safe(df["MPx"].to_numpy()); MPy = _nan_safe(df["MPy"].to_numpy())
    cog_move = _path_length(MPx[r], MPy[r])

    # 2) 足内CoP移動量（左/右）
    LPx = _nan_safe(df["LPx"].to_numpy()); LPy = _nan_safe(df["LPy"].to_numpy())
    RPx = _nan_safe(df["RPx"].to_numpy()); RPy = _nan_safe(df["RPy"].to_numpy())
    move_L = _path_length(LPx[r], LPy[r])
    move_R = _path_length(RPx[r], RPy[r])

    # 3) ピーク時重心バランス（mFz最大時における左右比）
    mFz = _nan_safe(df["MFz"].to_numpy())
    idx = int(np.argmax(mFz)) if mFz.size else 0
    LFz = _nan_safe(df["LFz"].to_numpy()); RFz = _nan_safe(df["RFz"].to_numpy())
    den = float(LFz[idx] + RFz[idx])
    balance = (float(LFz[idx]) / den) if (np.isfinite(den) and den != 0.0) else 0.5  # 0=右寄り/1=左寄り ⇄ 以前の定義に合わせるならここで左右を入替

    metrics = {
        "足内CoP移動量（右）": float(move_R),
        "ピーク時重心バランス": float(np.clip(balance, 0.0, 1.0)),
        "足内CoP移動量（左）": float(move_L),
        "重心移動量": float(cog_move),
    }
    # NaN→0
    for k, v in list(metrics.items()):
        if not np.isfinite(v): metrics[k] = 0.0
    return metrics

def normalize_for_radar(metrics: Dict[str, float]) -> Dict[str, float]:
    """
    レーダーの正規化（0..1）。スケールの仮定が揺らぐケースに強いよう、
    基本は「分子/全項目の最大」で正規化（バランス項目だけはそのまま0..1）。
    """
    vals = [float(v) for v in metrics.values() if np.isfinite(v)]
    vmax = max(vals) if vals else 1.0
    vmax = vmax if vmax > 0 else 1.0
    out: Dict[str, float] = {}
    for k, v in metrics.items():
        try:
            fv = float(v)
        except Exception:
            fv = 0.0
        if "バランス" in k:
            out[k] = max(0.0, min(1.0, fv))
        else:
            out[k] = float(fv) / vmax
    return out
