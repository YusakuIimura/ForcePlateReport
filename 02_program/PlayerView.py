import streamlit as st
import pandas as pd
import numpy as np
import cv2
from pathlib import Path
import plotly.graph_objects as go
import time

# -------------------------------------------------
# ユーティリティ
# -------------------------------------------------

def read_csv_any_encoding(p: Path) -> pd.DataFrame:
    """
    cp932とかutf-8-sigとか想定して順に試す。
    """
    enc_candidates = ["utf-8-sig", "cp932", "utf-8"]
    last_err = None
    for enc in enc_candidates:
        try:
            return pd.read_csv(p, encoding=enc)
        except Exception as e:
            last_err = e
    # 最後にエンコード指定なしでもう一回
    if last_err:
        return pd.read_csv(p)

def to_seconds_any(x):
    """
    time列が "00:00:12.345" とか Timedelta っぽい / float(秒) / msなど
    -> とにかく秒(s, float)にする補助関数
    """
    try:
        # すでに数値ならそのままfloat化
        if isinstance(x, (int, float, np.integer, np.floating)):
            return float(x)
        # 文字列・オブジェクトなら pandas の to_timedelta に投げてみる
        td = pd.to_timedelta(str(x))
        return td.total_seconds()
    except Exception:
        # fallback: 強制的にfloat解釈
        try:
            return float(x)
        except Exception:
            return np.nan

def load_video_info(video_path: Path):
    """
    動画ファイルのFPSと総フレーム数、1フレーム画像取得関数を返す。
    存在しなければNoneを返す。
    """
    if not video_path.exists():
        return None

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def get_frame_bgr(frame_idx: int):
        # frame_idxのフレームを取り出して返す（BGR→RGB変換済np.array）
        if frame_idx < 0: 
            idx = 0
        elif frame_idx >= frame_count:
            idx = frame_count - 1
        else:
            idx = frame_idx
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            return None
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame_rgb

    return {
        "fps": fps,
        "frame_count": frame_count,
        "get_frame": get_frame_bgr,
    }

def build_report_summary(df: pd.DataFrame, csv_path: Path):
    """
    Reportタブ向けの簡易サマリ例。
    本来はあなたのReport.pyのロジック（計測日時、player名、所要時間とか）を入れる。
    ここでは最低限の形を書いておく。
    """
    # player列っぽいものを探す
    cand_player_cols = [c for c in df.columns if c.lower() in ["player", "name", "athlete"]]
    player_name = df[cand_player_cols[0]].iloc[0] if cand_player_cols else "(不明)"

    # 計測日時っぽいもの
    cand_date_cols = [c for c in df.columns if "date" in c.lower() or "time" in c.lower()]
    # とりあえず最初の候補を文字列で
    measure_info = ""
    if cand_date_cols:
        measure_info = str(df[cand_date_cols[0]].iloc[0])

    info = {
        "ファイル": csv_path.name,
        "選手": str(player_name),
        "計測日時らしき値": measure_info,
        "サンプル数": len(df),
    }
    return info


# -------------------------------------------------
# ページ基本設定
# -------------------------------------------------

st.set_page_config(page_title="Player View", layout="wide")

# URLパラメータから csv_path と tab を取得
params = st.query_params
csv_path_param = params.get("csv_path", "")
initial_tab = params.get("tab", "graph")

csv_path = Path(csv_path_param)

st.title("Player View (8502)")
st.caption("1つの計測データからグラフとレポートをタブで切り替えて確認")

if not csv_path.exists():
    st.error(f"指定されたCSVが見つかりません: {csv_path}")
    st.stop()

# CSVロード
df = read_csv_any_encoding(csv_path)

# time列を推定
time_col = None
for cand in df.columns:
    if cand.lower() in ["time", "t", "timestamp", "sec", "seconds"]:
        time_col = cand
        break

if time_col is None:
    # timeっぽい列がなかったらダミーでインデックスを時間にする(0,1,2,...)
    df["_time_tmp_"] = np.arange(len(df)) * 0.01  # 仮に100Hz
    time_col = "_time_tmp_"

# 2軸で見たいので、time以外の数値列を列挙
numeric_cols = []
for c in df.columns:
    if c == time_col:
        continue
    # 数値に変換できそうなら候補にする
    try:
        pd.to_numeric(df[c].dropna().head(10), errors="raise")
        numeric_cols.append(c)
    except Exception:
        pass

# 動画パスは「CSVと同じ場所/同じ名前で拡張子mp4」を仮定
video_path = csv_path.with_suffix(".mp4")
video_info = load_video_info(video_path)

# PlayerView 全体で共有する state prefix
prefix = f"pv_{csv_path.name}_"

# 初期state
defaults = {
    prefix + "frame_idx": 0,
}
for k, v in defaults.items():
    st.session_state.setdefault(k, v)

# -------------------------------------------------
# タブUI
# -------------------------------------------------

tab_graph, tab_report = st.tabs(["📈 グラフ", "📝 レポート"])

# -------------------------------------------------
# タブ1: グラフ
# -------------------------------------------------
with tab_graph:
    #
    # ====== GraphViewerタブ本体 ======
    #
    st.subheader("グラフビュー / Graph")

    # Y軸候補（time_col以外の列）
    all_cols = list(df.columns)
    value_cols = [c for c in all_cols if c != time_col]

    # UIレイアウト: 左(操作パネル) / 右(動画＋グラフ＋スライダー)
    left_col, right_col = st.columns([0.4, 0.6])

    # -------------------------------------------------
    # 左カラム：軸選択 / 再生・停止 / コマ送り / 区間指定
    # -------------------------------------------------
    with left_col:
        st.markdown("### 軸選択")

        # 1本目のY軸
        y1_col = st.selectbox(
            "Y軸(1本目)",
            value_cols,
            index=0 if value_cols else 0,
            key=prefix + "y1_col_select",
        )

        # 2本目のY軸(任意)
        y2_col = st.selectbox(
            "Y軸(2本目・任意)",
            ["(なし)"] + value_cols,
            index=0,
            key=prefix + "y2_col_select",
        )
        y2_active = (y2_col != "(なし)")

        # time列を秒に変換しておく
        x_raw = df[time_col].map(to_seconds_any)

        # y1 を数値化
        y1_raw = pd.to_numeric(df[y1_col], errors="coerce") if y1_col else None
        mask1 = x_raw.notna() & y1_raw.notna()

        # y2 もあれば数値化
        if y2_active:
            y2_raw = pd.to_numeric(df[y2_col], errors="coerce")
            mask2 = x_raw.notna() & y2_raw.notna()
            mask = mask1 & mask2
        else:
            y2_raw = None
            mask = mask1

        # 描画用に絞ったデータ
        x_vals = x_raw[mask].tolist()
        y1_vals = y1_raw[mask].tolist()
        y2_vals = y2_raw[mask].tolist() if y2_active else None

        if not x_vals:
            st.error("有効なデータがありません（NaN等で欠損している可能性があります）。")
            st.stop()

        # 動画メタデータ (GraphViewer.pyと同じロジック)
        if video_info is None:
            st.error(f"{video_path.name} が見つかりません（このCSVに対応する動画がありません）。")
            st.stop()

        fps = video_info["fps"]
        total_frames = video_info["frame_count"]
        video_times = np.arange(total_frames) / fps  # 各フレームの時刻[s]

        # ▼▼▼ セッション初期化（prefix付きに変更） ▼▼▼
        defaults = {
            prefix + "is_playing": False,       # 再生フラグ
            prefix + "marker_idx": 0,           # 赤ラインが指すサンプルindex
            prefix + "start_idx": None,         # 区間開始
            prefix + "end_idx": None,           # 区間終了
        }
        for k, v in defaults.items():
            if k not in st.session_state:
                st.session_state[k] = v

        # コントロールパネル
        st.markdown("### コントロールパネル")

        row_play = st.columns(2)
        with row_play[0]:
            if st.button("▶ 再生", key=prefix + "play_btn"):
                st.session_state[prefix + "is_playing"] = True
        with row_play[1]:
            if st.button("⏸ 停止", key=prefix + "stop_btn"):
                st.session_state[prefix + "is_playing"] = False

        st.markdown("##### ⏪ / ⏩ コマ送り")

        # コマ送りボタンを6分割で並べる
        step_cols = st.columns(6)
        buttons = [
            ("-1.00s", -1.0),
            ("-0.30s", -0.3),
            ("-0.01s", -0.01),
            ("+0.01s", +0.01),
            ("+0.30s", +0.3),
            ("+1.00s", +1.0),
        ]
        for i, (label, delta_t) in enumerate(buttons):
            with step_cols[i]:
                if st.button(label, key=f"{prefix}stepbtn_{i}"):
                    current_idx = st.session_state[prefix + "marker_idx"]
                    current_t = x_vals[current_idx]
                    new_time = current_t + delta_t
                    # 一番近いインデックスに飛ぶ
                    new_idx = int(np.argmin(np.abs(np.array(x_vals) - new_time)))
                    # 範囲チェック
                    new_idx = max(0, min(len(x_vals) - 1, new_idx))
                    st.session_state[prefix + "marker_idx"] = new_idx
                    st.session_state[prefix + "is_playing"] = False  # コマ送り時は停止

        # 区間指定UI
        st.markdown("#### ⏱ 区間指定")

        seg_row1 = st.columns(2)
        with seg_row1[0]:
            if st.button("現在位置を開始時間に設定", key=prefix + "set_start"):
                st.session_state[prefix + "start_idx"] = st.session_state[prefix + "marker_idx"]
        with seg_row1[1]:
            if st.button("現在位置を終了時間に設定", key=prefix + "set_end"):
                st.session_state[prefix + "end_idx"] = st.session_state[prefix + "marker_idx"]

        seg_row2 = st.columns(2)
        with seg_row2[0]:
            if st.button("開始時間へ移動", key=prefix + "jump_start"):
                if st.session_state[prefix + "start_idx"] is not None:
                    st.session_state[prefix + "marker_idx"] = st.session_state[prefix + "start_idx"]
                    st.session_state[prefix + "is_playing"] = False
        with seg_row2[1]:
            if st.button("終了時間へ移動", key=prefix + "jump_end"):
                if st.session_state[prefix + "end_idx"] is not None:
                    st.session_state[prefix + "marker_idx"] = st.session_state[prefix + "end_idx"]
                    st.session_state[prefix + "is_playing"] = False

    # -------------------------------------------------
    # 右カラム：動画フレーム / グラフ / タイムラインスライダー
    # -------------------------------------------------
    with right_col:
        # 右カラム内で、フレーム画像とグラフを差し替えるためのスロットを確保
        frame_slot = st.empty()

        timeline_area = st.container()
        with timeline_area:
            graph_slot = st.empty()

            # タイムラインスライダー
            max_time = float(np.nanmax(x_vals))
            marker_idx = st.session_state[prefix + "marker_idx"]
            current_t = x_vals[marker_idx]

            slider_val = st.slider(
                "現在位置 (秒)",
                min_value=0.0,
                max_value=max_time,
                value=float(current_t),
                step=0.01,
                key=prefix + "timeline_slider",
            )

            # スライダーが動いたら marker_idx を更新
            if abs(slider_val - current_t) > 1e-6:
                nearest_idx = int(np.argmin(np.abs(np.array(x_vals) - slider_val)))
                st.session_state[prefix + "marker_idx"] = nearest_idx
                st.session_state[prefix + "is_playing"] = False
                marker_idx = nearest_idx  # ローカル変数も更新

    # -------------------------------------------------
    # 描画関数（GraphViewer.pyの draw_graph_and_frame 相当をprefix対応にしたもの）
    # -------------------------------------------------
    def draw_graph_and_frame(marker_idx_now: int):
        # 安全化
        marker_idx_now = max(0, min(marker_idx_now, len(x_vals) - 1))
        t_marker = x_vals[marker_idx_now]

        # CSV時間に最も近い動画フレーム番号
        frame_idx = int(np.argmin(np.abs(video_times - t_marker)))

        # === グラフ作成 ===
        fig = go.Figure()

        # 左軸トレース
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

        # 右軸トレース
        if y2_active and y2_vals is not None:
            fig.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=y2_vals,
                    mode="lines",
                    name=y2_col,
                    line=dict(color="orange"),
                    yaxis="y2",  # 右軸
                )
            )

        # 縦線の高さ範囲
        if y2_active and y2_vals is not None:
            y_all_min = min(np.nanmin(y1_vals), np.nanmin(y2_vals))
            y_all_max = max(np.nanmax(y1_vals), np.nanmax(y2_vals))
        else:
            y_all_min = np.nanmin(y1_vals)
            y_all_max = np.nanmax(y1_vals)

        # 現在位置の赤縦線
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

        # 区間ハイライト
        start_i = st.session_state[prefix + "start_idx"]
        end_i   = st.session_state[prefix + "end_idx"]
        if start_i is not None and end_i is not None:
            t0 = x_vals[min(start_i, end_i)]
            t1 = x_vals[max(start_i, end_i)]
            fig.add_vrect(
                x0=t0,
                x1=t1,
                fillcolor="lightgreen",
                opacity=0.3,
                line_width=0,
            )

        # レイアウト（右軸あり/なし両対応）
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

        if y2_active and y2_vals is not None:
            layout_dict["yaxis2"] = dict(
                title=y2_col,
                overlaying="y",
                side="right",
                fixedrange=True,
                zeroline=False,
            )

        fig.update_layout(**layout_dict)

        # グラフ描画
        graph_slot.plotly_chart(
            fig,
            use_container_width=True,
            config={"staticPlot": True},  # ズーム/ドラッグ禁止でiPadでも誤タッチしにくく
        )

        # === 動画フレーム描画 ===
        # video_info["get_frame"](frame_idx) でRGB画像が取れるようにしてある
        rgb_img = video_info["get_frame"](frame_idx)
        if rgb_img is not None:
            frame_slot.image(
                rgb_img,
                caption=f"{t_marker:.2f}s 付近 (Frame {frame_idx}/{total_frames-1})",
                width=480,
            )
        else:
            frame_slot.error("フレームを取得できませんでした。")

    # -------------------------------------------------
    # 再生ループ / 静止表示
    # （GraphViewer.pyの while 再生ループをprefix対応にして移植）
    # -------------------------------------------------
    if st.session_state[prefix + "is_playing"]:
        # CSVサンプリングが~100Hzくらい、動画が30fpsくらい想定
        # → 1フレームあたり何サンプル進めるかざっくり決める
        step = max(1, int(100 / fps))  # 例: 3〜4サンプルずつ
        while st.session_state[prefix + "is_playing"]:
            idx_now = st.session_state[prefix + "marker_idx"]
            draw_graph_and_frame(idx_now)

            idx_next = idx_now + step
            if idx_next >= len(x_vals):
                st.session_state[prefix + "is_playing"] = False
                break
            st.session_state[prefix + "marker_idx"] = idx_next

            # フレーム間のウェイト（1/fps秒）
            time.sleep(1.0 / fps)
    else:
        # 停止中は現在位置だけ描画
        draw_graph_and_frame(st.session_state[prefix + "marker_idx"])
        
# -------------------------------------------------
# タブ2: レポート
# -------------------------------------------------
with tab_report:
    #
    # ====== Reportタブ本体 ======
    #
    from report_core import (
        load_csv_from_path,
        build_report_html_from_df,
        render_report_with_print_toolbar,
    )

    st.subheader("レポートビュー / Report")
    
    if "logs" not in st.session_state:
        st.session_state["logs"] = []

    # 1. CSVの読み込みとメタ情報取り出し（Report.pyと同じやり方）:contentReference[oaicite:3]{index=3}
    try:
        df_full, measured_at, date_str, time_str, duration_str = load_csv_from_path(csv_path)
    except Exception as e:
        st.error(f"CSV の読み込みに失敗しました: {csv_path}\n{e}")
        st.stop()

    # 2. グラフタブでユーザーが指定した区間の適用
    #    PlayerViewでは prefix+"start_idx"/"end_idx" に区間が入ってる想定。
    #    Report.pyでは graph_ranges[label]['start'/'end'] を参照してたので、
    #    それに相当するものをここで作る。
    start_idx = st.session_state.get(prefix + "start_idx", None)
    end_idx   = st.session_state.get(prefix + "end_idx", None)

    if start_idx is not None and end_idx is not None:
        s_idx = int(start_idx)
        e_idx = int(end_idx)
        # 安全なクリップ
        s_idx = max(0, min(s_idx, len(df_full) - 1))
        e_idx = max(0, min(e_idx, len(df_full) - 1))
        if e_idx < s_idx:
            e_idx = s_idx
        df_for_report = df_full.iloc[s_idx:e_idx + 1].copy()
        st.caption(f"このレポートは区間 [{s_idx}, {e_idx}] のデータで作成しています。")
    else:
        df_for_report = df_full
        st.caption("このレポートはCSV全範囲のデータで作成しています。")

    # 3. player_name 推定
    #    Report.pyでは _first_nonempty_player() でCSVからplayer列を拾っていた。:contentReference[oaicite:4]{index=4}
    def _first_nonempty_player_local(dfcandidate) -> str | None:
        cand_col = None
        for c in dfcandidate.columns:
            if str(c).strip().lower() == "player":
                cand_col = c
                break
        if not cand_col:
            return None
        try:
            s = dfcandidate[cand_col].astype(str).map(lambda x: x.strip())
            vals = [v for v in s.unique().tolist() if v]
            return vals[0] if vals else None
        except Exception:
            return None

    player_name = _first_nonempty_player_local(df_full) or ""

    # 4. basic_meta / file_meta をReport.pyと同じ形で用意する
    #    Report.pyでは row_meta (Homeで保持した行メタ) を混ぜていたけど、
    #    PlayerViewは Home側のrow_metaを持ってこないので、最低限埋められるところだけ埋める。
    basic_meta = {
        "filename": csv_path.name,
        "measured_at": measured_at,
        "date": date_str,
        "time": time_str,
        "duration_sec": duration_str,
        "player_name": player_name,
        # handedness, height_cm, weight_kg... は本来 row_meta から来てた。
        # いまのPlayerView側では持ってないので空で埋める。
        "handedness": "",
        "height_cm": "",
        "weight_kg": "",
        "foot_size_cm": "",
        "step_width_cm": "",
    }

    file_meta = {
        "filename": csv_path.name,
        "date": date_str,
        "time": time_str,
        "duration_sec": duration_str,
        "player_name": player_name,
        "title": player_name,
        "name": player_name,
    }

    # 5. レポートHTML本体を生成（report_core.build_report_html_from_df）:contentReference[oaicite:5]{index=5}
    try:
        report_html = build_report_html_from_df(
            df_for_report,
            basic_meta=basic_meta,
            file_meta=file_meta,
        )
    except Exception as e:
        st.error(f"レポートHTMLの生成に失敗しました。\n{e}")
        st.stop()

    # 6. 印刷UIでラップ（report_core.render_report_with_print_toolbar）:contentReference[oaicite:6]{index=6}
    wrapped_html = render_report_with_print_toolbar(report_html) if report_html else ""

    # 7. 表示
    if wrapped_html:
        st.components.v1.html(wrapped_html, height=1000, scrolling=False)
    else:
        st.warning("レポートHTMLが空でした。テンプレートや入力データをご確認ください。")

    # 8. CSV/メタの確認用デバッグ情報（Report.pyもデバッグ出してたので載せておく）:contentReference[oaicite:7]{index=7}
    with st.expander("デバッグ情報（CSVメタ / basic_meta / file_meta）", expanded=False):
        st.write("CSVパス:", csv_path.as_posix())
        st.json({
            "basic_meta": basic_meta,
            "file_meta": file_meta,
            "measured_at": measured_at,
            "duration": duration_str,
        }, expanded=False)
        st.dataframe(df_for_report.head(20))
