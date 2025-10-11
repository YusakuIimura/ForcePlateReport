# utily.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Sequence, Optional

import numpy as np
import pandas as pd


# ========================
# 基本ユーティリティ
# ========================

REQUIRED_COLUMNS: Sequence[str] = (
    "Time",
    "LFx", "LFy", "LFz", "LMx", "LMy", "LMz", "LTz", "LPx", "LPy",
    "RFx", "RFy", "RFz", "RMx", "RMy", "RMz", "RTz", "RPx", "RPy",
    "MFx", "MFy", "MFz", "MMx", "MMy", "MMz", "MTz", "MPx", "MPy",
)

@dataclass
class SwingWindow:
    idx_start: int
    idx_peak: int
    idx_end: int
    t_start: float
    t_peak: float
    t_end: float
    idx_range: np.ndarray  # np.arange(idx_start, idx_end)


def _assert_columns(df: pd.DataFrame, cols: Sequence[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"CSVに必要な列がありません: {missing}")


def _nan_safe(values: np.ndarray | pd.Series) -> np.ndarray:
    """NaNを0に、infを有限値にクリップして返す"""
    arr = np.asarray(values, dtype=float).copy()
    arr[~np.isfinite(arr)] = 0.0
    return arr


def _moving_mean(x: np.ndarray, win: int) -> np.ndarray:
    """移動平均（端は片側）"""
    if win <= 1:
        return x
    s = pd.Series(x, dtype=float)
    return s.rolling(window=win, center=True, min_periods=1).mean().to_numpy()


def _impulse(y: np.ndarray, dt: float) -> float:
    """∫ y dt を台形則で近似"""
    return float(np.trapz(y, dx=dt))


def _path_length(x: np.ndarray, y: np.ndarray) -> float:
    """2D軌跡の総移動距離"""
    x = _nan_safe(x)
    y = _nan_safe(y)
    dx = np.diff(x)
    dy = np.diff(y)
    return float(np.sqrt(dx * dx + dy * dy).sum())


def _rfd_peak(y: np.ndarray, dt: float, idx_peak: int) -> float:
    """
    Rate of Force Development を単純差分で近似。
    y[0]→y[idx_peak] の平均立ち上がり速度。
    """
    if idx_peak <= 0:
        return float("nan")
    return float((y[idx_peak] - y[0]) / (idx_peak * dt))


def _pick_axes_prefix(is_right_handed: bool) -> Tuple[str, str]:
    """
    右利き: 右=軸足, 左=踏込足
    左利き: 左=軸足, 右=踏込足
    """
    return ("R", "L") if is_right_handed else ("L", "R")


# ========================
# 解析ウィンドウの決定
# ========================

def find_swing_window_by_mtz_peak(
    df: pd.DataFrame,
    fs: float,
    smooth_sec: float = 1.0,
) -> SwingWindow:
    """
    MTz（合成回旋モーメント）から「スイング開始～ピーク～終了」を決める。
      - ピーク: |MTz| 最大
      - 開始: ピーク前に 1秒移動平均した MTz が最小になる点
      - 終了: ピーク後に +smooth_sec だけ進めた点（範囲外は末尾に丸め）
    """
    _assert_columns(df, REQUIRED_COLUMNS)
    mTz = _nan_safe(df["MTz"].to_numpy())
    N = len(mTz)
    win = max(1, int(round(fs * smooth_sec)))

    idx_peak = int(np.argmax(np.abs(mTz)))
    t_peak = idx_peak / fs

    mTz_smooth = _moving_mean(mTz, win)
    idx_start = int(np.argmin(mTz_smooth[: max(1, idx_peak)])) if idx_peak > 0 else 0
    t_start = idx_start / fs

    idx_end = min(N - 1, idx_peak + win)
    t_end = idx_end / fs

    return SwingWindow(
        idx_start=idx_start,
        idx_peak=idx_peak,
        idx_end=idx_end,
        t_start=t_start,
        t_peak=t_peak,
        t_end=t_end,
        idx_range=np.arange(idx_start, idx_end, dtype=int),
    )


# ========================
# メイン解析（添付ロジック準拠）
# ========================

def analyze_fp_batting(
    df: pd.DataFrame,
    fs: float,
    is_right_handed: bool,
    body_weight: float,
    smooth_sec: float = 1.0,
) -> Dict[str, float]:
    """
    フォースプレート打撃解析（添付コードの意図を保持し、堅牢化）
    入力:
      df  : 指定の列名を持つDataFrame（固定スキーマ）
      fs  : サンプリング周波数 [Hz]
      is_right_handed : 右打ちなら True
      body_weight     : 体重 [N]（kg×9.806ではなく N を想定）
      smooth_sec      : MTz 平滑・ウィンドウ長（既定 1.0 s）

    出力（主な指標の辞書）：添付の項目に加えて一部派生値も含む
    """
    _assert_columns(df, REQUIRED_COLUMNS)
    dt = 1.0 / fs
    N = len(df)

    # 解析区間
    win = find_swing_window_by_mtz_peak(df, fs, smooth_sec=smooth_sec)
    r = win.idx_range

    # 軸足/踏込足のプレフィックス
    axis_p, stride_p = _pick_axes_prefix(is_right_handed)

    # --- COP移動量（足内） ---
    def _cop_movement(prefix: str) -> float:
        x = _nan_safe(df[f"{prefix}Px"].to_numpy())
        y = _nan_safe(df[f"{prefix}Py"].to_numpy())
        return _path_length(x[r], y[r])

    COPmove_axis = _cop_movement(axis_p)
    COPmove_stride = _cop_movement(stride_p)

    # --- 軸足/踏込足の Fz, Fx 系 ---
    Fz_axis = _nan_safe(df[f"{axis_p}Fz"].to_numpy())[r]
    Fx_axis = np.abs(_nan_safe(df[f"{axis_p}Fx"].to_numpy())[r])

    Fz_stride = _nan_safe(df[f"{stride_p}Fz"].to_numpy())[r]
    Fx_stride = np.abs(_nan_safe(df[f"{stride_p}Fx"].to_numpy())[r])

    Fz_peak_axis = float(np.max(Fz_axis)) if Fz_axis.size else float("nan")
    Fz_peak_stride = float(np.max(Fz_stride)) if Fz_stride.size else float("nan")
    Fx_peak_axis = float(np.max(Fx_axis)) if Fx_axis.size else float("nan")
    Fx_peak_stride = float(np.max(Fx_stride)) if Fx_stride.size else float("nan")

    idx_Fz_peak_axis = int(np.argmax(Fz_axis)) if Fz_axis.size else 0
    idx_Fz_peak_stride = int(np.argmax(Fz_stride)) if Fz_stride.size else 0

    RFD_Fz_axis = _rfd_peak(Fz_axis, dt, idx_Fz_peak_axis)
    RFD_Fz_stride = _rfd_peak(Fz_stride, dt, idx_Fz_peak_stride)

    RFD_Fx_axis = _rfd_peak(Fx_axis, dt, idx_Fz_peak_axis)
    RFD_Fx_stride = _rfd_peak(Fx_stride, dt, idx_Fz_peak_stride)

    # --- 合成（mFz, mTz） ---
    mFz = _nan_safe(df["MFz"].to_numpy())
    mTz = _nan_safe(df["MTz"].to_numpy())

    mFz_impulse = _impulse(mFz[r], dt)

    mTz_seg = mTz[r]
    idx_Tz_peak = int(np.argmax(np.abs(mTz_seg))) if mTz_seg.size else 0
    mTz_peak = float(np.abs(mTz_seg[idx_Tz_peak])) if mTz_seg.size else float("nan")
    mTz_peakBW = float(mTz_peak / body_weight) if body_weight else float("nan")
    mTz_RFD = _rfd_peak(mTz_seg, dt, idx_Tz_peak) if mTz_seg.size else float("nan")
    mTz_impulse = _impulse(mTz_seg, dt)

    # --- Fz最大時の合成COP位置 ---
    idx_Fzmax = int(np.argmax(mFz)) if mFz.size else 0
    COPx_atFzmax = float(_nan_safe(df["MPx"].to_numpy())[idx_Fzmax])
    COPy_atFzmax = float(_nan_safe(df["MPy"].to_numpy())[idx_Fzmax])
    t_Fzmax = idx_Fzmax / fs

    # --- 合成COP速度の最大 ---
    MPx = _nan_safe(df["MPx"].to_numpy())
    MPy = _nan_safe(df["MPy"].to_numpy())
    COP_speed = np.sqrt(np.diff(MPx) ** 2 + np.diff(MPy) ** 2) * fs
    COP_speed_max = float(np.max(COP_speed)) if COP_speed.size else float("nan")
    idx_speed_max = int(np.argmax(COP_speed)) if COP_speed.size else 0
    t_speed_max = idx_speed_max / fs

    # --- 結果まとめ ---
    result: Dict[str, float] = {
        # ウィンドウ
        "tStart": win.t_start, "tPeak": win.t_peak, "tEnd": win.t_end,
        # COP移動量（足内）
        "COPmove_axis": COPmove_axis, "COPmove_stride": COPmove_stride,
        # Fz, Fx ピーク & RFD
        "Fz_peak_axis": Fz_peak_axis, "Fz_peak_stride": Fz_peak_stride,
        "Fz_peakBW_axis": Fz_peak_axis / body_weight if body_weight else float("nan"),
        "Fz_peakBW_stride": Fz_peak_stride / body_weight if body_weight else float("nan"),
        "Fz_RFD_axis": RFD_Fz_axis, "Fz_RFD_stride": RFD_Fz_stride,
        "Fx_peak_axis": Fx_peak_axis, "Fx_peak_stride": Fx_peak_stride,
        "Fx_peakBW_axis": Fx_peak_axis / body_weight if body_weight else float("nan"),
        "Fx_peakBW_stride": Fx_peak_stride / body_weight if body_weight else float("nan"),
        "Fx_RFD_axis": RFD_Fx_axis, "Fx_RFD_stride": RFD_Fx_stride,
        # 合力・回旋
        "mFz_impulse": mFz_impulse,
        "mTz_peak": mTz_peak, "mTz_peakBW": mTz_peakBW,
        "mTz_RFD": mTz_RFD, "mTz_impulse": mTz_impulse,
        # Fz最大時のCOP
        "COPx_atFzmax": COPx_atFzmax, "COPy_atFzmax": COPy_atFzmax, "t_Fzmax": t_Fzmax,
        # COP速度最大
        "COP_speed_max": COP_speed_max, "t_speed_max": t_speed_max,
    }
    return result


# ========================
# レポートで先に使っていた「重心移動指標（4つ）」もここで提供
# ========================

def compute_cog_cop_metrics_from_fp(
    df: pd.DataFrame,
    fs: float | None = None,
    use_window: Optional[SwingWindow] = None,
) -> Dict[str, float]:
    """
    レポートの「重心移動指標」4項目を算出（ダミー→実値に差替え済み）
      - 重心移動量             : MPx/MPy のパス長（全区間 or 指定ウィンドウ）
      - 足内CoP移動量（左/右） : LPx/LPy と RPx/RPy のパス長
      - ピーク時重心バランス   : Fz合計最大時の LFz/(LFz+RFz)
    """
    _assert_columns(df, REQUIRED_COLUMNS)

    # 解析範囲
    if use_window is not None:
        r = use_window.idx_range
    else:
        # fs が与えられていれば、mTzピーク±1秒の範囲で切るのも有効
        if fs:
            win = find_swing_window_by_mtz_peak(df, fs)
            r = win.idx_range
        else:
            r = np.arange(len(df), dtype=int)

    MPx = _nan_safe(df["MPx"].to_numpy())[r]
    MPy = _nan_safe(df["MPy"].to_numpy())[r]
    LPx = _nan_safe(df["LPx"].to_numpy())[r]
    LPy = _nan_safe(df["LPy"].to_numpy())[r]
    RPx = _nan_safe(df["RPx"].to_numpy())[r]
    RPy = _nan_safe(df["RPy"].to_numpy())[r]

    cog_move = _path_length(MPx, MPy)
    cop_l = _path_length(LPx, LPy)
    cop_r = _path_length(RPx, RPy)

    LFz = _nan_safe(df["LFz"].to_numpy())
    RFz = _nan_safe(df["RFz"].to_numpy())
    total = LFz + RFz
    if np.any(total > 0):
        idx = int(np.argmax(total))
        balance = float(LFz[idx] / total[idx]) if total[idx] > 0 else 0.5
    else:
        balance = 0.5

    return {
        "重心移動量": float(cog_move),
        "足内CoP移動量（左）": float(cop_l),
        "足内CoP移動量（右）": float(cop_r),
        "ピーク時重心バランス": float(balance),
    }


def normalize_for_radar(metrics: Dict[str, float]) -> Dict[str, float]:
    """
    レーダー用に 0–1 正規化（「バランス」はそのまま使用）
    """
    if not metrics:
        return {}
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
