# pages/GraphViewer.py
from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import json, base64, textwrap

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objs as go
import streamlit.components.v1 as components
from string import Template

st.set_page_config(page_title="CSV Graph Viewer", layout="wide")
st.title("CSV Graph Viewer）")

# ========= Utils =========
@st.cache_data(show_spinner=False)
def _read_csv_cached(path: str) -> pd.DataFrame:
    p = Path(path)
    mtime = p.stat().st_mtime if p.exists() else 0.0
    _ = (path, mtime)  # cache key
    for enc in ("utf-8-sig", "cp932"):
        try:
            return pd.read_csv(p, encoding=enc)
        except Exception:
            continue
    return pd.read_csv(p)

def _common_columns(dfs: List[pd.DataFrame]) -> List[str]:
    cols = set(dfs[0].columns)
    for d in dfs[1:]:
        cols &= set(d.columns)
    return list(cols)

def _common_numeric_columns(dfs: List[pd.DataFrame], exclude: List[str]) -> List[str]:
    commons = [c for c in _common_columns(dfs) if c not in exclude]
    out = []
    for c in commons:
        ok = True
        for df in dfs:
            if pd.to_numeric(df[c], errors="coerce").notna().sum() == 0:
                ok = False; break
        if ok:
            out.append(c)
    return out

def _to_numeric_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def _to_x_series(s: pd.Series) -> Tuple[pd.Series, str]:
    num = pd.to_numeric(s, errors="coerce")
    if num.notna().mean() > 0.8:
        return num, "numeric"
    dt = pd.to_datetime(s, errors="coerce", infer_datetime_format=True)
    if dt.notna().mean() > 0.8:
        return dt, "datetime"
    return s.astype(str), "category"

def _downsample_xy(x: pd.Series, y: pd.Series, max_points: int = 3000):
    n = len(x)
    if n <= max_points or max_points <= 0:
        return x, y
    step = int(np.ceil(n / max_points))
    return x.iloc[::step], y.iloc[::step]

def _get_saved_range_for(path: str):
    return st.session_state.get("graph_ranges", {}).get(path)

def _set_range_for(paths: List[str], x_col: str, kind: str, start, end):
    st.session_state.setdefault("graph_ranges", {})
    for p in paths:
        st.session_state["graph_ranges"][p] = {
            "x_col": x_col, "kind": kind,
            "start": pd.to_datetime(start).isoformat() if kind=="datetime" else float(start),
            "end":   pd.to_datetime(end).isoformat()   if kind=="datetime" else float(end),
        }

def _guess_mp4_value(row: Dict) -> Optional[str]:
    # 列名に mp4 / video を含む列を優先
    for key in row.keys():
        if "mp4" in str(key).lower() or "video" in str(key).lower():
            v = str(row.get(key, "")).strip()
            if v:
                return v
    # 値が .mp4 で終わるもの
    for _, v in row.items():
        s = str(v).strip()
        if s.lower().endswith(".mp4"):
            return s
    return None

def _resolve_media_path(value: str, data_dir: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else (Path(data_dir) / p).resolve()

@st.cache_data(show_spinner=False)
def _read_file_bytes(path: str) -> bytes:
    p = Path(path)
    _ = (str(p), p.stat().st_mtime if p.exists() else 0.0)  # cache key
    return p.read_bytes()

def _b64_data_url_mp4(p: Path) -> str:
    data = _read_file_bytes(str(p))
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:video/mp4;base64,{b64}"

# ========= Home からの選択 =========
records: List[Dict] | None = st.session_state.get("selected_records")
if not records:
    st.info("メイン画面（Home）でCSVを選択してください。")
    st.page_link("Home.py", label="← メインに戻る", icon="⏪")
    st.stop()

labels: List[str] = []
path_map: Dict[str, str] = {}
rec_map: Dict[str, Dict] = {}
for i, rec in enumerate(records, start=1):
    row = rec.get("row", {})
    csv_path = rec.get("csv_path", "")
    name = row.get("name") or row.get("title") or Path(csv_path).name
    label = f"{i}. {name} ({Path(csv_path).name})"
    labels.append(label)
    path_map[label] = csv_path
    rec_map[label] = rec

# Home の既定選択を反映
default_labels = labels
sel_paths_state = st.session_state.get("selected_csv_paths")
if sel_paths_state:
    path_to_label = {v: k for k, v in path_map.items()}
    chosen = [path_to_label[p] for p in sel_paths_state if p in path_to_label]
    if chosen:
        default_labels = chosen

# ========= レイアウト =========
left, right = st.columns([1, 2], gap="large")

with left:
    st.subheader("データ & 軸の選択")
    sel = st.multiselect("表示するデータ（複数可）", options=labels, default=default_labels)
    if not sel:
        st.warning("1つ以上のデータを選択してください。"); st.stop()

    dfs: List[pd.DataFrame] = []
    sel_paths: List[str] = []
    for lab in sel:
        p = path_map[lab]
        if not Path(p).exists():
            st.error(f"CSVが見つかりません: {p}"); continue
        df = _read_csv_cached(p)
        if df is None or df.empty:
            st.warning(f"空のCSVの可能性: {p}"); continue
        dfs.append(df); sel_paths.append(p)
    if not dfs:
        st.error("有効なCSVが読み込めませんでした。"); st.stop()

    common_cols = _common_columns(dfs)
    if not common_cols:
        st.error("選択されたCSV間に共通する列がありません。"); st.stop()

    x_default = "Time" if "Time" in common_cols else common_cols[0]
    x_col = st.selectbox("横軸 (X)", options=sorted(common_cols), index=sorted(common_cols).index(x_default))

    y_candidates = _common_numeric_columns(dfs, exclude=[x_col])
    y_cols = st.multiselect(
        "縦軸 (Y)（共通して数値変換できる列）",
        options=sorted(y_candidates),
        default=[c for c in ["LFz", "RFz", "MTz"] if c in y_candidates] or (y_candidates[:1] if y_candidates else []))
    if not y_cols:
        st.warning("縦軸 (Y) を1つ以上選択してください。"); st.stop()

with right:
    # ======== 動画＋グラフ（同期） ========
    st.subheader("動画 & グラフ（動画の再生位置に赤ラインを同期）")

    # どの動画を使うか（選択行から mp4 推定）
    video_labels: List[str] = []
    video_paths: Dict[str, Path] = {}
    for lab in sel:
        rec = rec_map[lab]
        row = rec.get("row", {})
        data_dir = rec.get("data_dir", "") or str(Path(__file__).parents[1] / "data")
        mp4_val = _guess_mp4_value(row)
        if not mp4_val:
            continue
        resolved = _resolve_media_path(mp4_val, data_dir)
        video_labels.append(lab)
        video_paths[lab] = resolved

    if not video_labels:
        st.info("選択された行に mp4 の列（または .mp4 の値）が見つかりませんでした。Datalist の mp4 欄を確認してください。")
        current_video_path: Optional[Path] = None
    else:
        lab_sel = st.selectbox("表示する動画（選択CSVの中から）", options=video_labels)
        current_video_path = video_paths[lab_sel]
        if not current_video_path.exists():
            st.warning(f"動画ファイルが見つかりませんでした: {current_video_path.as_posix()}")
            current_video_path = None

    # ===== 全体レンジ =====
    x_series_first, x_kind = _to_x_series(dfs[0][x_col])
    x_min_all = x_series_first.min()
    x_max_all = x_series_first.max()

    # ===== 保存済みレンジ（代表は先頭） =====
    st.session_state.setdefault("graph_ranges", {})
    rep_path = sel_paths[0]
    saved = _get_saved_range_for(rep_path)

    if x_kind == "datetime":
        cur_start = pd.to_datetime(saved["start"]) if saved and saved.get("kind")=="datetime" else pd.to_datetime(x_min_all)
        cur_end   = pd.to_datetime(saved["end"])   if saved and saved.get("kind")=="datetime" else pd.to_datetime(x_max_all)
    else:
        x_min_f = float(pd.to_numeric(pd.Series([x_min_all]), errors="coerce").iloc[0])
        x_max_f = float(pd.to_numeric(pd.Series([x_max_all]), errors="coerce").iloc[0])
        if saved and saved.get("kind")=="numeric":
            cur_start, cur_end = float(saved["start"]), float(saved["end"])
        else:
            cur_start, cur_end = x_min_f, x_max_f

    # ===== グラフ用データを準備（JSへ渡す） =====
    # Xは「動画0秒＝CSV最小X」に合わせるため、JS内で秒に正規化して使う
    x0_for_video = x_min_all  # 動画0秒に相当するX
    traces = []
    for lab, df, p in zip(sel, dfs, sel_paths):
        x_raw = df[x_col]
        x_ser, _ = _to_x_series(x_raw)

        # 表示レンジで抽出
        if x_kind == "datetime":
            x_dt = pd.to_datetime(x_ser)
            mask = (x_dt >= pd.to_datetime(cur_start)) & (x_dt <= pd.to_datetime(cur_end))
            x_in = x_dt[mask]
            x_sec = (pd.to_datetime(x_in) - pd.to_datetime(x0_for_video)).dt.total_seconds()
        else:
            x_num = pd.to_numeric(x_ser, errors="coerce")
            mask = (x_num >= float(cur_start)) & (x_num <= float(cur_end))
            x_in = x_num[mask]
            # 数値はそのまま「秒」として扱う
            x_sec = pd.to_numeric(x_in, errors="coerce")

        for yc in y_cols:
            y = _to_numeric_series(df[yc])[mask]
            x_plot, y_plot = _downsample_xy(x_sec, y, max_points=3000)
            traces.append({
                "name": f"{Path(p).name}:{yc}",
                "x": x_plot.astype(float).fillna(method="pad").fillna(0.0).tolist(),
                "y": pd.to_numeric(y_plot, errors="coerce").fillna(method="pad").fillna(0.0).tolist(),
            })

    # X軸の初期レンジ（秒単位）
    if x_kind == "datetime":
        init_x0 = float((pd.to_datetime(cur_start) - pd.to_datetime(x0_for_video)).total_seconds())
        init_x1 = float((pd.to_datetime(cur_end)   - pd.to_datetime(x0_for_video)).total_seconds())
    else:
        init_x0 = float(cur_start) - (float(x0_for_video) if isinstance(x0_for_video, (int,float,np.floating)) else 0.0)
        init_x1 = float(cur_end)   - (float(x0_for_video) if isinstance(x0_for_video, (int,float,np.floating)) else 0.0)

    # 動画データURL（bytes→base64）
    video_data_url = _b64_data_url_mp4(current_video_path) if current_video_path else ""

    # ===== HTMLコンポーネント（video + plotly） =====
    traces_json = json.dumps(traces)

    html_template = Template("""
    <div style="display:flex; flex-direction:column; gap:10px; width:100%;">
    <video id="vid" controls style="width:100%; max-height:360px; background:#000;" src="$video_data_url"></video>
    <div id="chart" style="width:100%; height:520px;"></div>
    </div>
    <script src="https://cdn.plot.ly/plotly-2.30.0.min.js"></script>
    <script>
    const traces = $traces_json;
    const layout = {
        margin: {l: 35, r: 10, t: 10, b: 30},
        hovermode: "x unified",
        showlegend: true,
        xaxis: {
        title: "Time (s)",
        range: [$init_x0, $init_x1],
        showgrid: true
        },
        yaxis: {
        showgrid: true
        },
        shapes: [
        {
            type: 'line',
            x0: $init_x0, x1: $init_x0,
            y0: 0, y1: 1,
            xref: 'x', yref: 'paper',
            line: {color: 'red', width: 2}
        }
        ]
    };
    const data = traces.map(t => ({
        type: 'scattergl',
        mode: 'lines',
        name: t.name,
        x: t.x,
        y: t.y,
        line: {width: 2}
    }));
    const chart = document.getElementById('chart');
    Plotly.newPlot(chart, data, layout, {displaylogo:false, responsive:true});

    // 動画の再生位置で赤ラインを動かす（動画0秒 = X軸0秒）
    const vid = document.getElementById('vid');
    function updateVline(){
        const t = vid.currentTime || 0;  // 秒
        Plotly.relayout(chart, {
        'shapes[0].x0': t,
        'shapes[0].x1': t
        });
    }
    vid.addEventListener('timeupdate', updateVline);
    vid.addEventListener('seeking', updateVline);
    vid.addEventListener('seeked', updateVline);
    vid.addEventListener('play', updateVline);

    // グラフをクリックした位置に動画をジャンプ（逆同期）
    chart.on('plotly_click', function(ev){
        if (!ev || !ev.points || !ev.points.length) return;
        const x = ev.points[0].x;
        try {
        vid.currentTime = Math.max(0, Number(x));
        updateVline();
        } catch(e) {}
    });
    </script>
    """)

    html = html_template.substitute(
        traces_json=traces_json,
        init_x0=str(init_x0),
        init_x1=str(init_x1),
        video_data_url=video_data_url
    )

    components.html(html, height=920, scrolling=False)

    # ===== 下スライダー（プレビュー）＋ 再描画ボタン =====
    st.markdown("#### 解析範囲（プレビュー）— ボタンで確定して再描画")
    if x_kind == "datetime":
        preview_start, preview_end = st.slider(
            "対象区間（日時）",
            min_value=pd.to_datetime(x_min_all),
            max_value=pd.to_datetime(x_max_all),
            value=(pd.to_datetime(cur_start), pd.to_datetime(cur_end)),
            key=f"gv_preview_dt_{x_col}",
        )
        if st.button("再描画（この範囲を確定）", type="primary"):
            _set_range_for(sel_paths, x_col, "datetime", preview_start, preview_end)
            st.rerun()
        st.caption(f"プレビュー中：{preview_start} ～ {preview_end}")
    else:
        x_min_f = float(pd.to_numeric(pd.Series([x_min_all]), errors="coerce").iloc[0])
        x_max_f = float(pd.to_numeric(pd.Series([x_max_all]), errors="coerce").iloc[0])
        preview_start, preview_end = st.slider(
            "対象区間（数値）",
            min_value=x_min_f, max_value=x_max_f,
            value=(float(cur_start), float(cur_end)),
            key=f"gv_preview_num_{x_col}",
        )
        if st.button("再描画（この範囲を確定）", type="primary"):
            _set_range_for(sel_paths, x_col, "numeric", preview_start, preview_end)
            st.rerun()
        st.caption(f"プレビュー中：{preview_start:.3f} ～ {preview_end:.3f}")

    st.page_link("pages/Report.py", label="→ レポートを開く（保存済みの範囲が連動）", icon="📄")
