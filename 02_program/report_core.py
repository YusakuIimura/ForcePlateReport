import os, re, sys, json, base64, tempfile, datetime as dt
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from render_template import render_html
from utily import (
    compute_cog_cop_metrics_from_fp, normalize_for_radar, analyze_fp_batting
)

# ---- 元app.pyから移植（関数名・挙動は維持） ----
def log(msg: str):
    st.session_state.logs.append(f"{dt.datetime.now().strftime('%H:%M:%S')}  {msg}")
    if len(st.session_state.logs) > 200:
        st.session_state.logs = st.session_state.logs[-200:]
    try:
        st.toast(msg, icon="🛠️")
    except Exception:
        pass

def to_data_uri(png_path: Path) -> str:
    b = Path(png_path).read_bytes()
    return "data:image/png;base64," + base64.b64encode(b).decode("ascii")

def generate_plots(df: pd.DataFrame, out_dir: Path) -> dict:
    paths = {"fz_uri": "", "tz_uri": ""}
    if "Time" not in df.columns:
        log("CSVにTime列がありません。（グラフ生成スキップ）")
        return paths
    time = pd.to_numeric(df["Time"], errors="coerce")
    # LFz/RFz
    if set(["LFz", "RFz"]).issubset(df.columns):
        try:
            lfz = pd.to_numeric(df["LFz"], errors="coerce")
            rfz = pd.to_numeric(df["RFz"], errors="coerce")
            fig = plt.figure(figsize=(6.0, 3.2), dpi=150); ax = fig.add_subplot(111)
            ax.plot(time, lfz, label="LFz"); ax.plot(time, rfz, label="RFz")
            ax.set_xlabel("Time (s)"); ax.set_ylabel("Fz"); ax.grid(True, alpha=0.3); ax.legend()
            fig.tight_layout()
            fz_path = out_dir / "plot_fz.png"; fig.savefig(fz_path.as_posix()); plt.close(fig)
            paths["fz_uri"] = to_data_uri(fz_path)
            log("左右Fzグラフを保存しました。")
        except Exception as e:
            log(f"左右Fzの描画に失敗: {e!s}")
    else:
        log("CSVにLFz/RFz列がありません。（左右Fzグラフはスキップ）")
    # MTz
    if "MTz" in df.columns:
        try:
            mtz = pd.to_numeric(df["MTz"], errors="coerce")
            fig = plt.figure(figsize=(6.0, 3.2), dpi=150); ax = fig.add_subplot(111)
            ax.plot(time, mtz, label="MTz")
            ax.set_xlabel("Time (s)"); ax.set_ylabel("Tz"); ax.grid(True, alpha=0.3); ax.legend()
            fig.tight_layout()
            tz_path = out_dir / "plot_tz.png"; fig.savefig(tz_path.as_posix()); plt.close(fig)
            paths["tz_uri"] = to_data_uri(tz_path)
            log("Tzグラフを保存しました。")
        except Exception as e:
            log(f"Tzの描画に失敗: {e!s}")
    else:
        log("CSVにMTz列がありません。（Tzグラフはスキップ）")
    return paths

def scan_csv_dir(root: str) -> List[Path]:
    p = Path(root)
    if not p.exists() or not p.is_dir():
        return []
    files = list(p.glob("*_FP.csv"))
    files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return files

def load_csv_from_path(csv_path: Path) -> Tuple[pd.DataFrame, str, str, str, str]:
    df = None; errors = []
    for enc in ("utf-8-sig", "cp932"):
        try:
            df = pd.read_csv(csv_path, encoding=enc, sep=None, engine="python"); break
        except Exception as e: errors.append(f"{enc}: {e}")
    if df is None:
        raise RuntimeError("CSVの読み込みに失敗: \n" + "\n".join(errors))
    measured_at, date_str, time_str = "", "", ""
    m = re.match(r"(\d{8})_(\d{6})_FP\.csv$", csv_path.name)
    if m:
        ymd, hms = m.group(1), m.group(2)
        dt_obj = dt.datetime.strptime(ymd + hms, "%Y%m%d%H%M%S")
        measured_at = dt_obj.strftime("%Y-%m-%d %H:%M:%S")
        date_str = dt_obj.strftime("%Y-%m-%d")
        time_str = dt_obj.strftime("%H:%M:%S")
    duration_str = ""
    if "Time" in df.columns and not df["Time"].empty:
        _time_series = pd.to_numeric(df["Time"], errors="coerce").dropna()
        if not _time_series.empty:
            duration_val = float(_time_series.iloc[-1]); duration_str = f"{duration_val:.2f} s"
    return df, measured_at, date_str, time_str, duration_str

def _base_dir() -> Path:
    return Path(getattr(sys, "_MEIPASS", os.path.dirname(__file__)))

def _load_settings() -> dict:
    candidates = [
        _base_dir() / "settings.json",
        Path(os.environ.get("FORCEPLATE_SETTINGS", "")).resolve() if os.environ.get("FORCEPLATE_SETTINGS") else None,
        Path.cwd() / "settings.json",
    ]
    for p in [c for c in candidates if c]:
        try:
            if p.exists():
                txt = p.read_text(encoding="utf-8")
                txt = re.sub(r"//.*?$", "", txt, flags=re.MULTILINE)
                txt = re.sub(r"/\*.*?\*/", "", txt, flags=re.DOTALL)
                txt = re.sub(r",\s*(?=[}\]])", "", txt)
                return json.loads(txt)
        except Exception:
            pass
    return {}

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

# レポートHTML生成（元app.py右側のコアを1関数に）
def build_report_html_from_df(df: pd.DataFrame, basic_meta: dict, file_meta: dict) -> str:
    BASE = _base_dir(); template_path = BASE / "report_template.html"
    if not template_path.exists():
        st.error(f"テンプレートが見つかりません: {template_path.as_posix()}")
        return ""
    # プロット（CSVが変わるたびに再生成）
    out_dir = Path(tempfile.mkdtemp(prefix="report_")); log(f"一時フォルダを作成: {out_dir}")
    plots = generate_plots(df, out_dir)

    data = {
        "timeseries": {},
        "radar": {},
        "photo_uri": "",
        "fz_uri": plots.get("fz_uri", ""),
        "tz_uri": plots.get("tz_uri", ""),
        "meta": {
            **file_meta,
            "player_name":  basic_meta.get("player_name",""),
            "height_cm":    basic_meta.get("height_cm",""),
            "weight_kg":    basic_meta.get("weight_kg",""),
            "foot_size_cm": basic_meta.get("foot_size_cm",""),
            "handedness":   basic_meta.get("handedness","右"),
            "step_width_cm":basic_meta.get("step_width_cm",""),
        },
    }

    metrics = compute_cog_cop_metrics_from_fp(df)
    data["cog_metrics"] = metrics
    data["radar"] = normalize_for_radar(metrics)

    # FS推定
    fs = 1000.0
    if "Time" in df.columns:
        t = pd.to_numeric(df["Time"], errors="coerce").dropna().to_numpy()
        if t.size >= 2:
            dt_med = float(np.median(np.diff(t))); 
            if dt_med > 0: fs = 1.0 / dt_med

    def _to_float(s, default=0.0):
        try: return float(str(s).strip())
        except Exception: return default
    weight_kg = _to_float(basic_meta.get("weight_kg",""), 0.0)
    body_weight_N = weight_kg * 9.806 if weight_kg > 0 else 700.0
    is_right = (basic_meta.get("handedness","右") != "左")

    res = analyze_fp_batting(df, fs=fs, is_right_handed=is_right, body_weight=body_weight_N)
    data["grf"] = {
        "step": {"peakN": float(res.get("Fz_peak_stride",0.0)),
                 "peakBW": float(res.get("Fz_peakBW_stride",0.0)),
                 "rfdN": float(res.get("Fz_RFD_stride",0.0))},
        "axis": {"peakN": float(res.get("Fz_peak_axis",0.0)),
                 "peakBW": float(res.get("Fz_peakBW_axis",0.0)),
                 "rfdN": float(res.get("Fz_RFD_axis",0.0))},
        "impulse": float(res.get("mFz_impulse",0.0)),
    }
    data["rot"] = {
        "peak": float(res.get("mTz_peak",0.0)),
        "peakBW": float(res.get("mTz_peakBW",0.0)),
        "rfd": float(res.get("mTz_RFD",0.0)),
        "impulse": float(res.get("mTz_impulse",0.0)),
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
