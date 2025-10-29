from pathlib import Path
import streamlit as st
import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import time

st.set_page_config(page_title="Graph Viewer", layout="wide")
st.title("動画・グラフビュワー")
st.caption("動画とグラフを確認し、開始/終了区間を指定します。")

################################
# ユーティリティ
################################
def read_csv_with_fallback(p: Path) -> pd.DataFrame:
    for enc in ("utf-8-sig", "cp932"):
        try:
            return pd.read_csv(p, encoding=enc)
        except Exception:
            continue
    return pd.read_csv(p)

def to_seconds_any(x):
    """time列を秒に変換（秒float / '00:00:00.123' どちらでもOK）"""
    try:
        if isinstance(x, (int, float, np.floating, np.integer)):
            return float(x)
        td = pd.to_timedelta(str(x))
        return td.total_seconds()
    except Exception:
        try:
            return float(x)
        except Exception:
            return np.nan

################################
# データ読み込みに使う共通準備
################################
data_dir = Path(__file__).parents[1] / "data"
csv_files_all = sorted(data_dir.glob("*.csv"))
if not csv_files_all:
    st.error("dataフォルダにCSVがありません。")
    st.stop()

################################
# セッション初期化
################################
defaults = {
    "is_playing": False,
    "marker_idx": 0,   # 赤ラインが指しているCSV上のサンプルindex
    "start_idx": None, # 区間開始(赤ラインからセット)
    "end_idx": None,   # 区間終了
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

################################
# レイアウト
################################
left_col, right_col = st.columns([0.4, 0.6])

################################
# 左カラム：ファイル選択 / 軸選択 / コントロール
################################
with left_col:

    # CSVファイル選択
    csv_file_name = st.selectbox(
        "CSVファイル",
        [f.name for f in csv_files_all],
        index=0,
    )
    csv_path = data_dir / csv_file_name
    video_path = csv_path.with_suffix(".mp4")

    df = read_csv_with_fallback(csv_path)
    cols = list(df.columns)

    # time列を特定
    time_col = next((c for c in cols if c.lower() == "time"), None)
    if not time_col:
        st.error("CSVに'time'列がありません。time列が必要です。")
        st.stop()

    # Y軸候補（time以外）
    value_cols = [c for c in cols if c != time_col]

    # Y軸を2本まで選べるようにする
    # 1本目（必須）
    y1_col = st.selectbox("Y軸(1本目)", value_cols, index=0 if value_cols else None)
    # 2本目（オプション）
    y2_col = st.selectbox(
        "Y軸(2本目・任意)",
        ["(なし)"] + value_cols,
        index=0 if value_cols else None,
    )
    y2_active = (y2_col != "(なし)")

    # CSV -> 時間軸とy列をfloat化
    x_raw = df[time_col].map(to_seconds_any)
    # 1本目
    y1_raw = pd.to_numeric(df[y1_col], errors="coerce")
    mask1 = x_raw.notna() & y1_raw.notna()
    # 2本目（もし選ばれてたら）
    if y2_active:
        y2_raw = pd.to_numeric(df[y2_col], errors="coerce")
        mask2 = x_raw.notna() & y2_raw.notna()
        mask = mask1 & mask2
    else:
        mask = mask1

    x_vals = x_raw[mask].tolist()
    y1_vals = y1_raw[mask].tolist()
    y2_vals = y2_raw[mask].tolist() if y2_active else None

    if not x_vals:
        st.error("有効なデータがありません（NaNなどで欠けている可能性があります）。")
        st.stop()

    # 動画存在確認＆動画メタ
    if not video_path.exists():
        st.error(f"{video_path.name} が見つかりません（このCSVに対応する動画がありません）。")
        st.stop()

    cap_tmp = cv2.VideoCapture(str(video_path))
    fps = cap_tmp.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap_tmp.get(cv2.CAP_PROP_FRAME_COUNT))
    video_times = np.arange(total_frames) / fps
    cap_tmp.release()

    st.markdown(" ### コントロールパネル")
    # 再生 / 停止
    row_play = st.columns(2)
    with row_play[0]:
        if st.button("▶ 再生"):
            st.session_state["is_playing"] = True
    with row_play[1]:
        if st.button("⏸ 停止"):
            st.session_state["is_playing"] = False

    # コマ送りエリア
    st.markdown("##### ⏪ / ⏩ コマ送り")

    # 1行6列で配置（左3つが戻る、右3つが進む）
    cols = st.columns(6)
    buttons = [
        ("-1.00s", -1.0),
        ("-0.30s", -0.3),
        ("-0.01s", -0.01),
        ("+0.01s", +0.01),
        ("+0.30s", +0.3),
        ("+1.00s", +1.0),
    ]

    for i, (label, delta_t) in enumerate(buttons):
        with cols[i]:
            if st.button(label):
                current_t = x_vals[st.session_state["marker_idx"]]
                new_time = current_t + delta_t
                new_idx = int(np.argmin(np.abs(np.array(x_vals) - new_time)))
                st.session_state["marker_idx"] = max(0, min(len(x_vals) - 1, new_idx))
                st.session_state["is_playing"] = False

    # 区間指定
    st.markdown("#### ⏱ 区間指定")
    seg_row1 = st.columns(2)
    with seg_row1[0]:
        if st.button("現在位置を開始時間に設定"):
            st.session_state["start_idx"] = st.session_state["marker_idx"]
    with seg_row1[1]:
        if st.button("現在位置を終了時間に設定"):
            st.session_state["end_idx"] = st.session_state["marker_idx"]

    seg_row2 = st.columns(2)
    with seg_row2[0]:
        if st.button("開始時間へ移動") and st.session_state["start_idx"] is not None:
            st.session_state["marker_idx"] = st.session_state["start_idx"]
            st.session_state["is_playing"] = False
    with seg_row2[1]:
        if st.button("終了時間へ移動") and st.session_state["end_idx"] is not None:
            st.session_state["marker_idx"] = st.session_state["end_idx"]
            st.session_state["is_playing"] = False

################################
# 右カラム：動画・グラフ・スライダー
################################
with right_col:
    # まず動画とグラフ用のエリア
    frame_slot = st.empty()

    # グラフ＋スライダーを縦にまとめるコンテナ
    timeline_area = st.container()
    with timeline_area:
        graph_slot = st.empty()

        # スライダー（赤ラインとリンク）
        max_time = float(np.nanmax(x_vals))
        current_t = x_vals[st.session_state["marker_idx"]]
        slider_val = st.slider(
            "現在位置 (秒)",
            min_value=0.0,
            max_value=max_time,
            value=float(current_t),
            step=0.01,
            key="timeline_slider",
        )
        # スライダーの操作で赤ラインを動かす
        if abs(slider_val - current_t) > 1e-6:
            nearest_idx = int(np.argmin(np.abs(np.array(x_vals) - slider_val)))
            st.session_state["marker_idx"] = nearest_idx
            st.session_state["is_playing"] = False

################################
# 描画関数
################################
def draw_graph_and_frame(marker_idx: int):
    """marker_idxに対応する時刻に赤ラインを引き、
    CSV時間に最も近い動画フレームを表示する。
    y1は左軸、y2があれば右軸。
    """
    # --- インデックスと時刻計算 ---
    marker_idx = max(0, min(marker_idx, len(x_vals) - 1))
    t_marker = x_vals[marker_idx]

    # CSV時間に最も近い動画フレームを決める
    frame_idx = int(np.argmin(np.abs(video_times - t_marker)))

    # --- Figure ---
    fig = go.Figure()

    # 左軸トレース（必須）
    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=y1_vals,
            mode="lines",
            name=y1_col,
            line=dict(color="steelblue"),
            yaxis="y",  # 左軸
        )
    )

    # 右軸トレース（オプション）
    if y2_active:
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=y2_vals,
                mode="lines",
                name=y2_col,
                line=dict(color="orange"),
                yaxis="y2",  # 右軸
        ))

    # 縦ラインの上下限（全系列のmin/maxから）
    if y2_active:
        y_all_min = min(np.nanmin(y1_vals), np.nanmin(y2_vals))
        y_all_max = max(np.nanmax(y1_vals), np.nanmax(y2_vals))
    else:
        y_all_min = np.nanmin(y1_vals)
        y_all_max = np.nanmax(y1_vals)

    fig.add_shape(
        type="line",
        x0=t_marker,
        x1=t_marker,
        y0=y_all_min,
        y1=y_all_max,
        line=dict(color="red", width=2),
        xref="x",
        yref="y",
    )

    # 区間ハイライト（開始～終了）
    if st.session_state.get("start_idx") is not None and st.session_state.get("end_idx") is not None:
        start_i = st.session_state["start_idx"]
        end_i = st.session_state["end_idx"]
        t0 = x_vals[min(start_i, end_i)]
        t1 = x_vals[max(start_i, end_i)]
        fig.add_vrect(
            x0=t0,
            x1=t1,
            fillcolor="lightgreen",
            opacity=0.3,
            line_width=0,
        )

    # ---- レイアウト ----
    # ここが肝。もとの1軸版のノリをほぼ維持しつつ、
    # yaxis2 だけ最小限のプロパティで足す。
    layout_dict = dict(
        height=240,
        margin=dict(l=40, r=40, t=20, b=30),
        dragmode=False,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            font=dict(size=10),
        ),
        xaxis=dict(
            title="Time [s]",
            fixedrange=True,
        ),
        yaxis=dict(
            title=y1_col,
            fixedrange=True,
            zeroline=False,
        ),
    )

    if y2_active:
        layout_dict["yaxis2"] = dict(
            title=y2_col,
            overlaying="y",   # 同じxで重ねる
            side="right",     # 右側に出す
            fixedrange=True,
            zeroline=False,
        )

    fig.update_layout(**layout_dict)

    # ---- Streamlit描画（グラフ）----
    graph_slot.plotly_chart(
        fig,
        use_container_width=True,
        config={"staticPlot": True},  # iPadでズーム禁止
    )

    # ---- 動画フレーム表示 ----
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if ret:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_slot.image(
            frame_rgb,
            caption=f"{t_marker:.2f}s 付近 (Frame {frame_idx}/{total_frames})",
            width=480,
        )
    cap.release()


################################
# 再生処理 / 静止描画
################################
if st.session_state["is_playing"]:
    # CSVは100 Hz、動画は約30 fps → 1フレームごとに何サンプル進めるか
    step = max(1, int(100 / fps))  # 約3〜4サンプルずつ進む想定
    while st.session_state["is_playing"]:
        idx = st.session_state["marker_idx"]
        draw_graph_and_frame(idx)

        idx += step
        if idx >= len(x_vals):
            st.session_state["is_playing"] = False
            break
        st.session_state["marker_idx"] = idx

        time.sleep(1.0 / fps)
else:
    draw_graph_and_frame(st.session_state["marker_idx"])
