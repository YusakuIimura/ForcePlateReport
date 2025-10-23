# pages/GraphViewer.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import cv2
from PIL import Image

# ---- CSV 読み込み（キャッシュ） ----
@st.cache_data(show_spinner=False)
def _read_csv(p: str) -> Optional[pd.DataFrame]:
    try:
        return pd.read_csv(p)
    except Exception as e:
        st.error(f"CSV読み込みに失敗: {p}\n{e}")
        return None

# ---- 軸候補ユーティリティ ----
def _to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def _to_dt(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce")

def _numeric_cols(df: pd.DataFrame, exclude: List[str] | None = None) -> List[str]:
    exclude = set(exclude or [])
    out = []
    for c in df.columns:
        if c in exclude:
            continue
        num = _to_num(df[c])
        if num.notna().sum() > 0:
            out.append(c)
    return out

def _x_candidates(df: pd.DataFrame) -> List[str]:
    # datetimeっぽい列を優先、次に数値列
    dt_like = [c for c in df.columns if any(k in c.lower() for k in ["time", "date", "timestamp"])]
    dt_like = [c for c in dt_like if _to_dt(df[c]).notna().sum() > 0]
    nums = _numeric_cols(df)
    # 重複排除して結合
    seen, out = set(), []
    for c in dt_like + nums:
        if c not in seen:
            out.append(c); seen.add(c)
    return out or list(df.columns)

# ---- 動画関連 ----
def _guess_mp4_value(row: Dict) -> Optional[str]:
    for k in ["mp4", "video", "movie", "Video", "MP4", "path_video"]:
        if k in row and str(row[k]).strip():
            return str(row[k]).strip()
    return None

def _resolve_media_path(mp4_value: str | Path, data_dir: str | Path) -> Path:
    p = Path(str(mp4_value))
    if p.exists():
        return p
    return Path(data_dir) / p.name

def _guess_time_mapping(df: pd.DataFrame) -> Tuple[str, str]:
    """
    秒列 or 日時列 があれば使う。なければ index を time とする。
    戻り値: (列名 or "__index__", kind)  kind in {"seconds","datetime","index"}
    """
    sec_names = {"t","time","sec","seconds","elapsed","elapsed_s"}
    for c in df.columns:
        if c.lower() in sec_names and _to_num(df[c]).notna().any():
            return c, "seconds"
    for c in df.columns:
        if any(k in c.lower() for k in ["time","date","timestamp"]) and _to_dt(df[c]).notna().any():
            return c, "datetime"
    return "__index__", "index"

def _extract_frame_cv2(video_path: Path, seconds: float) -> Optional[Image.Image]:
    if not video_path or not video_path.exists():
        return None
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, seconds) * 1000.0)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return None
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame)

# ---- ページ本体 ----
def main():
    st.set_page_config(page_title="CSV Graph Viewer", layout="wide")
    st.title("動画＆グラフビュワー")
    st.caption("選択したデータごとに動画とグラフを見ながら、グラフの開始・終了位置を指定して保存できます。")

    # Home 側で保存された選択：リスト仕様
    records: List[Dict] = st.session_state.get("selected_records") or []
    if not records:
        st.info("Home でデータ行を選択してください。")
        st.stop()

    # ラベルとパスのマップを構築
    labels: List[str] = []
    path_map: Dict[str, str] = {}
    rec_map: Dict[str, Dict] = {}
    for i, rec in enumerate(records, start=1):
        row = rec.get("row", {}) or {}
        csv_path = rec.get("csv_path", "")
        name = row.get("name") or row.get("title") or Path(csv_path).name
        label = f"{i}. {name} ({Path(csv_path).name})"
        labels.append(label)
        path_map[label] = csv_path
        rec_map[label] = rec

    # 単一選択（重ね書き禁止）
    lab = st.selectbox("表示するデータ", options=labels, index=0)
    rec = rec_map[lab]
    row = rec.get("row", {}) or {}
    data_dir = rec.get("data_dir", "") or str(Path(__file__).parents[1] / "data")
    csv_path = path_map[lab]

    df = _read_csv(csv_path)
    if df is None or df.empty:
        st.error(f"空のCSVか読み込み失敗: {csv_path}")
        st.stop()

    # ---- 2カラム：左=動画、右=設定＋グラフ＋スナップ ----
    left, right = st.columns([1, 1.4])

    # 左：動画（同期しない埋め込み）
    with left:
        st.subheader("動画")
        mp4_val = _guess_mp4_value(row)
        current_video_path: Optional[Path] = None
        if mp4_val:
            resolved = _resolve_media_path(mp4_val, data_dir)
            if resolved.exists():
                current_video_path = resolved
                st.video(str(resolved))
            else:
                st.warning(f"動画が見つかりません: {resolved.as_posix()}")
        else:
            st.info("この行に mp4 情報がありません。Datalist の mp4 欄をご確認ください。")

    # 右：軸選択・グラフ・スナップ
    with right:
        st.subheader("グラフ設定")

        # X, Y の選択
        x_opts = _x_candidates(df)
        # 既定は timestamp/time/Date 系があればそれ、なければ先頭
        default_x = 0
        for cand in ["timestamp", "Timestamp", "time", "Time", "date", "Date"]:
            if cand in x_opts:
                default_x = x_opts.index(cand); break
        x_col = st.selectbox("横軸 (X)", options=x_opts, index=default_x)

        y_opts = _numeric_cols(df, exclude=[x_col])
        if not y_opts:
            st.error("数値の縦軸候補が見つかりません。"); st.stop()
        y_default = "LFz" if "LFz" in y_opts else y_opts[0]
        y_col = st.selectbox("縦軸 (Y)", options=sorted(y_opts), index=sorted(y_opts).index(y_default))

        # データの準備
        x_for_plot = df[x_col]
        y_vals = pd.to_numeric(df[y_col], errors="coerce")

        idx_max = max(0, len(df) - 1)
        col1, col2 = st.columns(2)
        with col1:
            gv_idx_start = st.slider("開始位置（赤ライン）", 0, idx_max, value=int(st.session_state.get("gv_idx_start", 0)))
        with col2:
            gv_idx_end = st.slider("終了位置（青ライン）", 0, idx_max, value=int(st.session_state.get("gv_idx_end", min(10, idx_max))))

        x_val_start = x_for_plot.iloc[gv_idx_start]
        x_val_end = x_for_plot.iloc[gv_idx_end]

        # Plotly 図：2本のラインを追加
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_for_plot, y=y_vals, mode="lines", name=f"{Path(csv_path).name}:{y_col}"))

        y_min = float(np.nanmin(y_vals)) if np.isfinite(y_vals).any() else 0.0
        y_max = float(np.nanmax(y_vals)) if np.isfinite(y_vals).any() else 1.0

        # 赤ライン（start）
        fig.add_shape(
            type="line",
            x0=x_val_start, x1=x_val_start,
            y0=y_min, y1=y_max,
            line=dict(color="red", width=2),
        )
        # 青ライン（end）
        fig.add_shape(
            type="line",
            x0=x_val_end, x1=x_val_end,
            y0=y_min, y1=y_max,
            line=dict(color="blue", width=2),
        )

        fig.update_layout(
            xaxis_title=x_col, yaxis_title=y_col,
            height=420, margin=dict(l=10, r=10, t=10, b=10),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

        # スナップショット表示
        st.subheader("スナップショット（赤=開始 / 青=終了）")

        if mp4_val:
            t_col, t_kind = _guess_time_mapping(df)

            def _time_sec(idx: int) -> float:
                """スライダーindex→動画内秒"""
                if t_kind == "seconds":
                    return float(_to_num(df[t_col].iloc[idx]) or 0.0)
                elif t_kind == "datetime":
                    dts = _to_dt(df[t_col])
                    dt0, dti = dts.iloc[0], dts.iloc[idx]
                    if pd.isna(dt0) or pd.isna(dti):
                        return float(idx)
                    return max(0.0, (dti - dt0).total_seconds())
                else:
                    return float(idx)

            # start / end それぞれの時刻に対応するフレームを抽出
            t_start = _time_sec(gv_idx_start)
            t_end = _time_sec(gv_idx_end)

            img_start = _extract_frame_cv2(_resolve_media_path(mp4_val, data_dir), t_start)
            img_end = _extract_frame_cv2(_resolve_media_path(mp4_val, data_dir), t_end)

            c1, c2 = st.columns(2)
            if img_start:
                c1.image(img_start, caption=f"Start (赤) @ {t_start:.3f}s", use_container_width=True)
            else:
                c1.warning("開始位置のフレーム取得に失敗しました。")

            if img_end:
                c2.image(img_end, caption=f"End (青) @ {t_end:.3f}s", use_container_width=True)
            else:
                c2.warning("終了位置のフレーム取得に失敗しました。")

        else:
            st.info("動画が無いのでスナップショットは表示できません。")

        # 状態保存
        st.session_state["gv_idx_start"] = int(gv_idx_start)
        st.session_state["gv_idx_end"] = int(gv_idx_end)

        # 💾 値を保持するボタン
        st.markdown("---")
        if "graph_ranges" not in st.session_state:
            st.session_state["graph_ranges"] = {}

        if st.button("💾 このデータの開始・終了位置を保持"):
            st.session_state["graph_ranges"][lab] = {
                "start": int(gv_idx_start),
                "end": int(gv_idx_end),
            }
            st.success(f"保持しました：{lab}（Start={gv_idx_start}, End={gv_idx_end}）")
    
    go_report = st.button("📝 レポートを開く", type="primary")
    if go_report:
        dest = "pages/Report.py"
        st.switch_page(dest)



if __name__ == "__main__":
    main()
