# report_core.py (simplified)
import os, re, sys, base64, tempfile, datetime as dt
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import streamlit as st

# matplotlib (画像を書き出すのみ)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 外部ユーティリティ（既存の分析・テンプレ描画は活かす）
from render_template import render_html
from utily import (
    compute_cog_cop_metrics_from_fp,
    normalize_for_radar,
    analyze_fp_batting,
)

# -------------------- 基本ユーティリティ --------------------

def log(msg: str):
    """右上トースト + セッションログ（最多200件）"""
    st.session_state.setdefault("logs", [])
    st.session_state["logs"].append(f"{dt.datetime.now():%H:%M:%S}  {msg}")
    st.session_state["logs"] = st.session_state["logs"][-200:]
    try:
        st.toast(msg, icon="🛠️")
    except Exception:
        pass

def _base_dir() -> Path:
    """PyInstaller対応のベースディレクトリ解決"""
    return Path(getattr(sys, "_MEIPASS", os.path.dirname(__file__)))

def _to_data_uri(png_path: Path) -> str:
    b = png_path.read_bytes()
    return "data:image/png;base64," + base64.b64encode(b).decode("ascii")

# -------------------- CSVロード & メタ抽出 --------------------

def load_csv_from_path(csv_path: Path) -> Tuple[pd.DataFrame, str, str, str, str]:
    """
    CSVを読み込み、計測日時/日付/時刻/継続時間(秒表記) を返す。
    - 文字コードは utf-8-sig → cp932 の順でトライ
    - ファイル名が YYYYMMDD_HHMMSS_FP.csv ならそこから日時を復元
    - Time 列があれば末尾値から duration を推定
    """
    df = None
    errors = []
    for enc in ("utf-8-sig", "cp932"):
        try:
            df = pd.read_csv(csv_path, encoding=enc, sep=None, engine="python")
            break
        except Exception as e:
            errors.append(f"{enc}: {e}")
    if df is None:
        raise RuntimeError("CSVの読み込みに失敗: \n" + "\n".join(errors))

    measured_at = date_str = time_str = ""
    m = re.match(r"(\d{8})_(\d{6})_FP\.csv$", csv_path.name)
    if m:
        ymd, hms = m.group(1), m.group(2)
        dt_obj = dt.datetime.strptime(ymd + hms, "%Y%m%d%H%M%S")
        measured_at = dt_obj.strftime("%Y-%m-%d %H:%M:%S")
        date_str    = dt_obj.strftime("%Y-%m-%d")
        time_str    = dt_obj.strftime("%H:%M:%S")

    duration_str = ""
    if "Time" in df.columns and not df["Time"].empty:
        t = pd.to_numeric(df["Time"], errors="coerce").dropna()
        if not t.empty:
            duration_str = f"{float(t.iloc[-1]):.2f} s"

    return df, measured_at, date_str, time_str, duration_str

# -------------------- 図の生成（最小） --------------------

def generate_plots(df: pd.DataFrame, out_dir: Path) -> dict:
    """
    Time/LFz/RFz/MTz があれば簡易プロットをPNG出力し、data URIを返す。
    どれか欠けていたら該当プロットはスキップ。
    """
    out = {"fz_uri": "", "tz_uri": ""}

    if "Time" not in df.columns:
        log("CSVにTime列がありません（グラフ生成スキップ）。")
        return out

    time = pd.to_numeric(df["Time"], errors="coerce")

    # Fz（左右）
    if {"LFz", "RFz"}.issubset(df.columns):
        try:
            lfz = pd.to_numeric(df["LFz"], errors="coerce")
            rfz = pd.to_numeric(df["RFz"], errors="coerce")
            fig = plt.figure(figsize=(6.0, 3.2), dpi=150)
            ax = fig.add_subplot(111)
            ax.plot(time, lfz, label="LFz")
            ax.plot(time, rfz, label="RFz")
            ax.set_xlabel("Time (s)"); ax.set_ylabel("Fz")
            ax.grid(True, alpha=0.3); ax.legend()
            fig.tight_layout()
            p = out_dir / "plot_fz.png"
            fig.savefig(p.as_posix()); plt.close(fig)
            out["fz_uri"] = _to_data_uri(p)
            log("左右Fzグラフを保存しました。")
        except Exception as e:
            log(f"左右Fzの描画に失敗: {e!s}")
    else:
        log("CSVにLFz/RFz列がありません（Fzグラフはスキップ）。")

    # Tz（全体）
    if "MTz" in df.columns:
        try:
            mtz = pd.to_numeric(df["MTz"], errors="coerce")
            fig = plt.figure(figsize=(6.0, 3.2), dpi=150)
            ax = fig.add_subplot(111)
            ax.plot(time, mtz, label="MTz")
            ax.set_xlabel("Time (s)"); ax.set_ylabel("Tz")
            ax.grid(True, alpha=0.3); ax.legend()
            fig.tight_layout()
            p = out_dir / "plot_tz.png"
            fig.savefig(p.as_posix()); plt.close(fig)
            out["tz_uri"] = _to_data_uri(p)
            log("Tzグラフを保存しました。")
        except Exception as e:
            log(f"Tzの描画に失敗: {e!s}")
    else:
        log("CSVにMTz列がありません（Tzグラフはスキップ）。")

    return out

# -------------------- レポートHTML生成 --------------------

def build_report_html_from_df(df: pd.DataFrame, meta: dict, start_img_uri: str | None = None) -> str:
    """
    単一の meta 辞書でテンプレへ渡す簡素版。
    - プロット生成（Fz/Tz）
    - COG/COP指標の計算と正規化（既存の utily を使用）
    - 打撃アナリシス（利き手・体重から計算）
    - render_html でテンプレ描画
    """
    template_path = _base_dir() / "report_template.html"
    if not template_path.exists():
        st.error(f"テンプレートが見つかりません: {template_path.as_posix()}")
        return ""

    # 図出力（CSVごとに一時フォルダへ）
    out_dir = Path(tempfile.mkdtemp(prefix="report_"))
    log(f"一時フォルダを作成: {out_dir}")
    plots = generate_plots(df, out_dir)

    # --- 指標計算
    metrics = compute_cog_cop_metrics_from_fp(df)
    metrics_fmt = {k: f"{float(v) if v is not None else 0.0:.2f}" for k, v in metrics.items()}
    radar = normalize_for_radar(metrics)
    _label_map = {
    "足内CoP移動量（左）": "足内CoP\n移動量\n（左）",
    "足内CoP移動量（右）": "足内CoP\n移動量\n（右）",
    "ピーク時重心バランス": "ピーク時\n重心バランス",
    # "重心移動量" はそのままでもOK
    }
    radar = {_label_map.get(k, k): float(v) for k, v in radar.items()}
    

    # サンプリング周波数の推定
    fs = 100.0
    if "Time" in df.columns:
        t = pd.to_numeric(df["Time"], errors="coerce").dropna().to_numpy()
        if t.size >= 2:
            dt_med = float(np.median(np.diff(t)))
            if dt_med > 0:
                fs = 1.0 / dt_med

    def _to_float(v, default=0.0):
        try:
            return float(str(v).strip())
        except Exception:
            return default

    weight_kg = _to_float(meta.get("weight_kg", ""), 0.0)
    body_weight_N = weight_kg * 9.806 if weight_kg > 0 else 700.0
    is_right = (meta.get("handedness", "右") != "左")

    res = analyze_fp_batting(df, fs=fs, is_right_handed=is_right, body_weight=body_weight_N)

    # --- テンプレへ渡すデータ（必要最小限）
    data = {
        "meta": {
            # 必須キー（テンプレは meta.* で参照）
            "filename":     meta.get("filename", ""),
            "measured_at":  meta.get("measured_at", ""),
            "date":         meta.get("date", ""),
            "time":         meta.get("time", ""),
            "duration_sec": meta.get("duration_sec", ""),
            "user_name":  meta.get("user_name", ""),
            "handedness":   meta.get("handedness", ""),
            "height_cm":    meta.get("height_cm", ""),
            "weight_kg":    meta.get("weight_kg", ""),
            # 任意項目（テンプレが使う場合のみ）
            "foot_size_cm": meta.get("foot_size_cm", ""),
            "step_width_cm":meta.get("step_width_cm", ""),
        },
        "fz_uri": plots.get("fz_uri", ""),
        "tz_uri": plots.get("tz_uri", ""),
        "cog_metrics": metrics_fmt,
        "radar": radar,
        "grf": {
            "step":  {
                "peakN":  float(res.get("Fz_peak_stride", 0.0)),
                "peakBW": float(res.get("Fz_peakBW_stride", 0.0)),
                "rfdN":   float(res.get("Fz_RFD_stride", 0.0)),
            },
            "axis":  {
                "peakN":  float(res.get("Fz_peak_axis", 0.0)),
                "peakBW": float(res.get("Fz_peakBW_axis", 0.0)),
                "rfdN":   float(res.get("Fz_RFD_axis", 0.0)),
            },
            "impulse": float(res.get("mFz_impulse", 0.0)),
        },
        "rot": {
            "peak":    float(res.get("mTz_peak", 0.0)),
            "peakBW":  float(res.get("mTz_peakBW", 0.0)),
            "rfd":     float(res.get("mTz_RFD", 0.0)),
            "impulse": float(res.get("mTz_impulse", 0.0)),
        },
        "start_img_uri": start_img_uri or "",
    }

    log("レポートHTMLを生成しています…")
    rendered_html = render_html(
        data=data,
        template_dir=template_path.parent.as_posix(),
        template_name=template_path.name,
        out_dir=Path(tempfile.mkdtemp(prefix="report_")),
    )
    log("レポートHTMLを生成しました。")
    return rendered_html

# -------------------- 印刷ツールバー付きのラッパ --------------------

def render_report_with_print_toolbar(report_html: str) -> str:
    from html import escape as html_escape
    srcdoc = html_escape(report_html, quote=True)
    return f"""
<!doctype html><html lang="ja"><head><meta charset="utf-8">
<style>
  html,body{{height:100%;margin:0}}
  .toolbar{{position:sticky;top:0;padding:8px 12px;background:#fff;border-bottom:1px solid #ddd}}
  .toolbar button{{padding:6px 12px;border-radius:8px;border:1px solid #bbb;cursor:pointer}}
  .frame-wrap{{height:calc(100% - 46px)}} iframe{{width:100%;height:100%;border:0}}
  @page{{size:A4;margin:14mm}}
  @media print{{.toolbar{{display:none}} body{{-webkit-print-color-adjust:exact;print-color-adjust:exact}}}}
</style></head><body>
  <div class="toolbar"><button onclick="(function(){{const f=document.getElementById('frame');f&&f.contentWindow&&f.contentWindow.print();}})()">A4で印刷</button></div>
  <div class="frame-wrap"><iframe id="frame" srcdoc='{srcdoc}'></iframe></div>
</body></html>""".strip()
