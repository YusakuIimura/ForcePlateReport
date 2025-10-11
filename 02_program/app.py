# app.py — CSV読込→右枠にレポート。基本情報は「反映」ボタンで更新。グラフはCSV変更時のみ再生成。
import os
import json
import re
import sys
import base64
import tempfile
import datetime as dt
from pathlib import Path
import plotly.graph_objs as go

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from render_template import render_html
from utily import (
    compute_cog_cop_metrics_from_fp, normalize_for_radar,
    analyze_fp_batting,
)
from typing import List, Tuple

# プロット
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------- ユーティリティ ----------
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
    """
    CSVから左右Fz, TzのPNGを生成して data URI を返す。
    期待列：
      - Time, LFz, RFz
      - Time, MTz
    """
    paths = {"fz_uri": "", "tz_uri": ""}

    if "Time" not in df.columns:
        log("CSVにTime列がありません。（グラフ生成スキップ）")
        return paths

    time = pd.to_numeric(df["Time"], errors="coerce")

    # 左右Fz
    if set(["LFz", "RFz"]).issubset(df.columns):
        log("左右Fzグラフを作成しています…")
        try:
            lfz = pd.to_numeric(df["LFz"], errors="coerce")
            rfz = pd.to_numeric(df["RFz"], errors="coerce")
            fig = plt.figure(figsize=(6.0, 3.2), dpi=150)
            ax = fig.add_subplot(111)
            ax.plot(time, lfz, label="LFz")
            ax.plot(time, rfz, label="RFz")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Fz")
            ax.grid(True, alpha=0.3)
            ax.legend()
            fig.tight_layout()
            fz_path = out_dir / "plot_fz.png"
            fig.savefig(fz_path.as_posix())
            plt.close(fig)
            paths["fz_uri"] = to_data_uri(fz_path)
            log("左右Fzグラフを保存しました。")
        except Exception as e:
            log(f"左右Fzの描画に失敗: {e!s}")
    else:
        log("CSVにLFz/RFz列がありません。（左右Fzグラフはスキップ）")

    # Tz（MTz）
    if "MTz" in df.columns:
        log("Tzグラフを作成しています…")
        try:
            mtz = pd.to_numeric(df["MTz"], errors="coerce")
            fig = plt.figure(figsize=(6.0, 3.2), dpi=150)
            ax = fig.add_subplot(111)
            ax.plot(time, mtz, label="MTz")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Tz")
            ax.grid(True, alpha=0.3)
            ax.legend()
            fig.tight_layout()
            tz_path = out_dir / "plot_tz.png"
            fig.savefig(tz_path.as_posix())
            plt.close(fig)
            paths["tz_uri"] = to_data_uri(tz_path)
            log("Tzグラフを保存しました。")
        except Exception as e:
            log(f"Tzの描画に失敗: {e!s}")
    else:
        log("CSVにMTz列がありません。（Tzグラフはスキップ）")

    return paths

def scan_csv_dir(root: str) -> List[Path]:
    """共有フォルダから *_FP.csv を列挙（更新日時降順）"""
    p = Path(root)
    if not p.exists() or not p.is_dir():
        return []
    files = list(p.glob("*_FP.csv"))  # パターン: yyyyMMdd_hhmmss_FP.csv を想定
    files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return files

def load_csv_from_path(csv_path: Path) -> Tuple[pd.DataFrame, str, str, str, str]:
    """ファイルパスからCSVを読み、付帯情報を返す: df, measured_at, date, time, duration"""
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

    # ファイル名 yyyyMMdd_hhmmss_FP.csv から日時抽出
    measured_at, date_str, time_str = "", "", ""
    m = re.match(r"(\d{8})_(\d{6})_FP\.csv$", csv_path.name)
    if m:
        ymd, hms = m.group(1), m.group(2)
        dt_obj = dt.datetime.strptime(ymd + hms, "%Y%m%d%H%M%S")
        measured_at = dt_obj.strftime("%Y-%m-%d %H:%M:%S")
        date_str = dt_obj.strftime("%Y-%m-%d")
        time_str = dt_obj.strftime("%H:%M:%S")

    # 計測時間（Time末尾）
    duration_str = ""
    if "Time" in df.columns and not df["Time"].empty:
        _time_series = pd.to_numeric(df["Time"], errors="coerce").dropna()
        if not _time_series.empty:
            duration_val = float(_time_series.iloc[-1])
            duration_str = f"{duration_val:.2f} s"

    return df, measured_at, date_str, time_str, duration_str

def _base_dir() -> Path:
    # ソース実行でも EXE 実行でも同じ場所を見る
    return Path(getattr(sys, "_MEIPASS", os.path.dirname(__file__)))

def _load_settings() -> dict:
    """launcher と同じ settings.json を探して読み込む（なければ {} を返す）"""
    # 探索候補: 1) appと同じ階層 2) 環境変数 3) カレント
    candidates = [
        _base_dir() / "settings.json",
        Path(os.environ.get("FORCEPLATE_SETTINGS", "")).resolve() if os.environ.get("FORCEPLATE_SETTINGS") else None,
        Path.cwd() / "settings.json",
    ]
    for p in [c for c in candidates if c]:
        try:
            if p.exists():
                txt = p.read_text(encoding="utf-8")
                import re
                txt_relaxed = re.sub(r"//.*?$", "", txt, flags=re.MULTILINE)         # 行コメント
                txt_relaxed = re.sub(r"/\*.*?\*/", "", txt_relaxed, flags=re.DOTALL)  # ブロックコメント
                txt_relaxed = re.sub(r",\s*(?=[}\]])", "", txt_relaxed)               # 末尾カンマ
                return json.loads(txt_relaxed)
        except Exception:
            pass
    return {}

def render_report_with_print_toolbar(report_html: str) -> str:
    from html import escape as html_escape
    srcdoc = html_escape(report_html, quote=True)
    return f"""
<!doctype html>
<html lang="ja"><head><meta charset="utf-8">
<style>
  html,body{{height:100%;margin:0}}
  .toolbar{{position:sticky;top:0;padding:8px 12px;background:#fff;border-bottom:1px solid #ddd}}
  .toolbar button{{padding:6px 12px;border-radius:8px;border:1px solid #bbb;cursor:pointer}}
  .frame-wrap{{height:calc(100% - 46px)}}
  iframe{{width:100%;height:100%;border:0}}
  @page{{size:A4;margin:14mm}}
  @media print{{.toolbar{{display:none}} body{{-webkit-print-color-adjust:exact;print-color-adjust:exact}}}}
</style></head>
<body>
  <div class="toolbar">
    <button onclick="(function(){{const f=document.getElementById('frame');f&&f.contentWindow&&f.contentWindow.print();}})()">A4で印刷</button>
  </div>
  <div class="frame-wrap">
    <iframe id="frame" srcdoc='{srcdoc}'></iframe>
  </div>
</body></html>
""".strip()

# ---------- Streamlit アプリ本体 ----------
st.set_page_config(page_title="試験結果ビューア", layout="wide")

# ---------- session_state 初期化 ----------
if "df" not in st.session_state:
    st.session_state.df = None

if "report_html" not in st.session_state:
    st.session_state.report_html = None

if "meta" not in st.session_state:
    st.session_state.meta = {
        "filename": "",
        "measured_at": "",
        "date": "",
        "time": "",
        "duration_sec": "",
    }

if "report_height" not in st.session_state:
    st.session_state.report_height = 1000

if "data_dir" not in st.session_state:
    _cfg = _load_settings()
    data_root = (
        (_cfg.get("data") or {}).get("root")
        or (_cfg.get("files") or {}).get("data_dir")
        or r"C:\ForcePlateData"   # フォールバック
    )
    st.session_state.data_dir = str(data_root)

if "dir_files" not in st.session_state:
    st.session_state.dir_files = []   # パスのリスト
if "selected_path" not in st.session_state:
    st.session_state.selected_path = ""

# 基本情報（確定値）：レポートに使う本番値
if "basic" not in st.session_state:
    st.session_state.basic = {
        "player_name": "",
        "height_cm": "",
        "weight_kg": "",
        "foot_size_cm": "",
        "handedness": "右",
        "step_width_cm": "",
    }

# 基本情報（編集中）：入力欄の値。反映ボタンを押すまでレポートに使わない
if "basic_pending" not in st.session_state:
    st.session_state.basic_pending = st.session_state.basic.copy()

# 処理ログ
if "logs" not in st.session_state:
    st.session_state.logs = []

# プロットのキャッシュ（data URI）と、どのCSV用かを識別するキー
if "plots" not in st.session_state:
    st.session_state.plots = {"fz_uri": "", "tz_uri": ""}
if "plots_key" not in st.session_state:
    st.session_state.plots_key = ""  # ファイル名などで紐付け




st.title("試験結果ビューア")
st.caption("CSVを選ぶと右側にレポートが表示。基本情報は『反映』ボタンで更新。グラフはCSV変更時のみ再生成。")

# ===== レイアウト =====
left, right = st.columns([1, 2], gap="large")

with left:
    st.subheader("操作")
    uploaded = st.file_uploader("CSVをアップロード", type=["csv"], key="csv_uploader")
    st.write("")
    
    # 共有フォルダからの読み込み
    st.markdown("**共有フォルダから読み込む**")
    st.text_input("共有フォルダのパス", key="data_dir", value=st.session_state.data_dir, placeholder=r"C:\ForcePlateData")
    colA, colB = st.columns([1, 2])
    with colA:
        if st.button("更新 / 再読込", use_container_width=True):
            st.session_state.dir_files = scan_csv_dir(st.session_state.data_dir)
            if st.session_state.dir_files:
                # 直近の1件をとりあえず選択状態に
                st.session_state.selected_path = st.session_state.dir_files[0].as_posix()
            else:
                st.session_state.selected_path = ""
    with colB:
        files = st.session_state.dir_files or []
        labels = [f"{f.name}  —  {dt.datetime.fromtimestamp(f.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')}" for f in files]
        idx = 0
        if st.session_state.selected_path:
            try:
                idx = [f.as_posix() for f in files].index(st.session_state.selected_path)
            except ValueError:
                idx = 0 if files else 0
        selected = st.selectbox("CSVファイルを選択", labels, index=(idx if files else None), placeholder="更新ボタンで一覧を取得", disabled=(len(files)==0))
        # セレクトボックスが選ばれたらパスを更新
        if files and selected:
            st.session_state.selected_path = files[labels.index(selected)].as_posix()

    # 「読み込む」ボタン（選択確定）
    if st.session_state.selected_path and st.button("このCSVを読み込む", type="primary", use_container_width=True):
        try:
            csv_path = Path(st.session_state.selected_path)
            df, measured_at, date_str, time_str, duration_str = load_csv_from_path(csv_path)

            # state更新（既存のアップロード読込と同じ流れ）
            st.session_state.df = df
            st.session_state.meta = {
                "filename": csv_path.name,
                "measured_at": measured_at,
                "date": date_str,
                "time": time_str,
                "duration_sec": duration_str,
            }
            # pending初期化
            st.session_state.basic_pending = st.session_state.basic.copy()

            # プロットを作成（CSVごとに作り直し）
            out_dir = Path(tempfile.mkdtemp(prefix="report_"))
            log(f"一時フォルダを作成: {out_dir}")
            st.session_state.plots = generate_plots(df, out_dir)
            st.session_state.plots_key = csv_path.name

            st.session_state.report_html = None
            log(f"CSV読み込み成功: {df.shape[0]}行 × {df.shape[1]}列")
            st.success(f"読み込み成功: {csv_path.name}")
        except Exception as e:
            st.error("読み込みでエラーが発生しました。")
            with st.expander("エラー詳細"):
                st.code(str(e))
                
    # ------------------------------------------------------

    # 新しいファイルを選んだときだけ読み直す
    if uploaded is not None:
        current_name = uploaded.name
        already_loaded = (
            st.session_state.df is not None
            and st.session_state.meta.get("filename") == current_name
        )
        if not already_loaded:
            df = None
            errors = []
            for enc in ("utf-8-sig", "cp932"):
                try:
                    uploaded.seek(0)
                    df = pd.read_csv(uploaded, encoding=enc, sep=None, engine="python")
                    break
                except Exception as e:
                    errors.append(f"{enc}: {e}")

            if df is None:
                st.error("CSVの読み込みに失敗しました。")
                with st.expander("エラー詳細"):
                    st.code("\n".join(errors))
                st.session_state.df = None
                st.session_state.report_html = None
                st.session_state.meta = {
                    "filename": "",
                    "measured_at": "",
                    "date": "",
                    "time": "",
                    "duration_sec": "",
                }
                st.session_state.plots = {"fz_uri": "", "tz_uri": ""}
                st.session_state.plots_key = ""
                log("CSV読み込みエラー。")
            else:
                # ファイル名 yyyyMMdd_hhmmss_FP.csv
                measured_at = ""
                date_str, time_str = "", ""
                m = re.match(r"(\d{8})_(\d{6})_FP\.csv$", current_name)
                if m:
                    ymd, hms = m.group(1), m.group(2)
                    dt_obj = dt.datetime.strptime(ymd + hms, "%Y%m%d%H%M%S")
                    measured_at = dt_obj.strftime("%Y-%m-%d %H:%M:%S")
                    date_str = dt_obj.strftime("%Y-%m-%d")
                    time_str = dt_obj.strftime("%H:%M:%S")

                # 計測時間（Time末尾）
                duration_str = ""
                if "Time" in df.columns and not df["Time"].empty:
                    _time_series = pd.to_numeric(df["Time"], errors="coerce").dropna()
                    if not _time_series.empty:
                        duration_val = float(_time_series.iloc[-1])
                        duration_str = f"{duration_val:.2f} s"

                # state更新
                st.session_state.df = df
                st.session_state.meta = {
                    "filename": current_name,
                    "measured_at": measured_at,
                    "date": date_str,
                    "time": time_str,
                    "duration_sec": duration_str,
                }

                # 基本情報の編集中値を初期化（既存の確定値をコピー）
                st.session_state.basic_pending = st.session_state.basic.copy()

                # プロットはCSVごとに作り直し（キャッシュ）
                out_dir = Path(tempfile.mkdtemp(prefix="report_"))
                log(f"一時フォルダを作成: {out_dir}")
                st.session_state.plots = generate_plots(df, out_dir)
                st.session_state.plots_key = current_name

                # レポートは再生成待ち（次の描画サイクルで自動生成）
                st.session_state.report_html = None
                log(f"CSV読み込み成功: {df.shape[0]}行 × {df.shape[1]}列")

    # ファイル情報
    with st.container(border=True):
        st.markdown("**ファイル情報**")
        st.write("ファイル名:", st.session_state.meta.get("filename") or "—")
        st.write("年月日:",     st.session_state.meta.get("date") or "—")
        st.write("時刻:",       st.session_state.meta.get("time") or "—")
        st.write("計測日時:",   st.session_state.meta.get("measured_at") or "—")
        st.write("計測時間:",   st.session_state.meta.get("duration_sec") or "—")

    st.write("")
    # 基本情報（入力→反映ボタンで確定）
    with st.container(border=True):
        st.markdown("**基本情報（手入力）**")
        bp = st.session_state.basic_pending  # ショートハンド

        st.text_input("選手名", key="basic_pending_player_name", value=bp["player_name"], placeholder="山田 太郎")
        bp["player_name"] = st.session_state.get("basic_pending_player_name", "")

        c1, c2 = st.columns(2)
        with c1:
            st.text_input("身長 (cm)", key="basic_pending_height_cm", value=bp["height_cm"], placeholder="170")
            bp["height_cm"] = st.session_state.get("basic_pending_height_cm", "")
            st.text_input("足の大きさ (cm)", key="basic_pending_foot_size_cm", value=bp["foot_size_cm"], placeholder="27.0")
            bp["foot_size_cm"] = st.session_state.get("basic_pending_foot_size_cm", "")
        with c2:
            st.text_input("体重 (kg)", key="basic_pending_weight_kg", value=bp["weight_kg"], placeholder="65")
            bp["weight_kg"] = st.session_state.get("basic_pending_weight_kg", "")
            handed_idx = 0 if bp.get("handedness", "右") != "左" else 1
            st.selectbox("打ち手", ["右", "左"], key="basic_pending_handedness", index=handed_idx)
            bp["handedness"] = st.session_state.get("basic_pending_handedness", "右")

        st.text_input("ステップ幅 (cm)", key="basic_pending_step_width_cm", value=bp["step_width_cm"], placeholder="30")
        bp["step_width_cm"] = st.session_state.get("basic_pending_step_width_cm", "")

        # 反映ボタン（ここで初めてレポート更新）
        if st.button("基本情報を反映", type="primary", use_container_width=True):
            st.session_state.basic = st.session_state.basic_pending.copy()
            st.session_state.report_html = None  # レポートのみ再生成（グラフは再生成しない）
            log("基本情報をレポートに反映しました。")

    st.write("")
    # 表示オプション
    with st.container(border=True):
        st.markdown("**表示オプション**")
        st.session_state.report_height = st.slider(
            "レポート枠の高さ（px）",
            min_value=600, max_value=2000, value=st.session_state.report_height, step=50,
            help="レポートが枠からはみ出す場合は高さを上げてください。"
        )

    st.write("")
    # 処理ログ
    # with st.container(border=True):
    #     st.markdown("**処理ログ**")
    #     if not st.session_state.logs:
    #         st.caption("ここに処理の進捗が表示されます。")
    #     else:
    #         for line in st.session_state.logs[-200:]:
    #             st.code(line, language="text")

with right:
    st.subheader("レポート表示")
    report_container = st.container(height=st.session_state.report_height + 40, border=True)

    # レポート自動生成：df があり report_html が未生成なら生成
    if st.session_state.df is not None and not st.session_state.report_html:
        try:
            BASE = Path(getattr(sys, "_MEIPASS", os.path.dirname(__file__)))
            template_path = BASE / "report_template.html"
            if not template_path.exists():
                st.error(f"テンプレートが見つかりません: {template_path.as_posix()}")
            else:
                # プロットはCSVが変わった時だけ再生成（キャッシュ利用）
                if st.session_state.plots_key != st.session_state.meta.get("filename"):
                    out_dir = Path(tempfile.mkdtemp(prefix="report_"))
                    log(f"[再生成] 一時フォルダ: {out_dir}")
                    st.session_state.plots = generate_plots(st.session_state.df, out_dir)
                    st.session_state.plots_key = st.session_state.meta.get("filename", "")

                data = {
                    "timeseries": {},
                    "radar": {},
                    "photo_uri": "",
                    # 画像URIはキャッシュから
                    "fz_uri": st.session_state.plots.get("fz_uri", ""),
                    "tz_uri": st.session_state.plots.get("tz_uri", ""),
                    "meta": {
                        **st.session_state.meta,
                        # 確定済みの基本情報のみ使用（pendingは使わない）
                        "player_name":  st.session_state.basic["player_name"],
                        "height_cm":    st.session_state.basic["height_cm"],
                        "weight_kg":    st.session_state.basic["weight_kg"],
                        "foot_size_cm": st.session_state.basic["foot_size_cm"],
                        "handedness":   st.session_state.basic["handedness"],
                        "step_width_cm":st.session_state.basic["step_width_cm"],
                    },
                }
                
                # csvから各種指標を計算しレポートに埋め込み
                metrics = compute_cog_cop_metrics_from_fp(st.session_state.df)   
                data["cog_metrics"] = metrics                                   
                data["radar"] = normalize_for_radar(metrics)
                
                # FS推定（Timeの中央値差分から）
                fs = 1000.0
                if "Time" in st.session_state.df.columns:
                    t = pd.to_numeric(st.session_state.df["Time"], errors="coerce").dropna().to_numpy()
                    if t.size >= 2:
                        dt_med = float(np.median(np.diff(t)))
                        if dt_med > 0:
                            fs = 1.0 / dt_med

                # 体重[N]（未入力は700N相当を仮置き）
                def _to_float(s, default=0.0):
                    try: return float(str(s).strip())
                    except Exception: return default
                weight_kg = _to_float(st.session_state.basic.get("weight_kg", ""), 0.0)
                body_weight_N = weight_kg * 9.806 if weight_kg > 0 else 700.0

                is_right = (st.session_state.basic.get("handedness", "右") != "左")

                res = analyze_fp_batting(
                    st.session_state.df, fs=fs,
                    is_right_handed=is_right,
                    body_weight=body_weight_N,
                )

                # Jinja で扱いやすい形に整形（% は 0–1 のまま渡し、テンプレ側で×100）
                data["grf"] = {
                    "step": {
                        "peakN":  float(res.get("Fz_peak_stride", 0.0)),
                        "peakBW": float(res.get("Fz_peakBW_stride", 0.0)),  # 0-1
                        "rfdN":   float(res.get("Fz_RFD_stride", 0.0)),
                    },
                    "axis": {
                        "peakN":  float(res.get("Fz_peak_axis", 0.0)),
                        "peakBW": float(res.get("Fz_peakBW_axis", 0.0)),     # 0-1
                        "rfdN":   float(res.get("Fz_RFD_axis", 0.0)),
                    },
                    "impulse": float(res.get("mFz_impulse", 0.0)),
                }

                data["rot"] = {
                    "peak":     float(res.get("mTz_peak", 0.0)),
                    "peakBW":   float(res.get("mTz_peakBW", 0.0)),  # 0-1
                    "rfd":      float(res.get("mTz_RFD", 0.0)),
                    "impulse":  float(res.get("mTz_impulse", 0.0)),
                }

                log("レポートHTMLを生成しています…")
                rendered_html = render_html(
                    data=data,
                    template_dir=template_path.parent.as_posix(),
                    template_name=template_path.name,
                    out_dir=Path(tempfile.mkdtemp(prefix="report_")),
                )
                st.session_state.report_html = rendered_html
                log("レポートHTMLを生成しました。")
        except Exception as e:
            st.exception(e)
            log(f"エラー: {e!s}")

    with report_container:
        if st.session_state.report_html:
            wrapped = render_report_with_print_toolbar(st.session_state.report_html)
            components.html(wrapped, height=st.session_state.report_height, scrolling=False)
        else:
            st.info("データなし（CSVをアップロードしてください）")
