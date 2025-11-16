import streamlit as st
import pandas as pd
import numpy as np
import cv2
from pathlib import Path
import plotly.graph_objects as go
import time
import json

# -------------------------------------------------
# ユーティリティ
# -------------------------------------------------
SETTINGS_PATH = Path(__file__).parent / "settings.json"

def _load_settings() -> dict:
    if SETTINGS_PATH.exists():
        try:
            return json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    # 既定スキーマ
    return {
        "player_view": {
            "y_axes": {
                "default": {"y1": "", "y2": "(なし)"},
            }
        }
    }

def _save_settings(cfg: dict) -> None:
    SETTINGS_PATH.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")

def _get_default_axes(csv_name: str) -> tuple[str, str]:
    cfg = _load_settings()
    pv = cfg.get("player_view", {}).get("y_axes", {})
    y1 = pv.get("default", {}).get("y1") or ""
    y2 = pv.get("default", {}).get("y2") or "(なし)"
    return y1, y2

def _save_default_axes(csv_name: str, y1: str, y2: str, per_file: bool = False) -> None:
    cfg = _load_settings()
    pv = cfg.setdefault("player_view", {}).setdefault("y_axes", {})

    # 直近選択は default としても保持（次回全体の既定にする）
    pv["default"] = {"y1": y1 or "", "y2": y2 or "(なし)"}
    _save_settings(cfg)

def get_user_meta_for_csv(csv_path: Path):
    """
    datalist.csv から user を拾い、userlist.csv から身長/体重を取得。
    戻り値: dict(user, handedness, height_cm, weight_kg)
    """
    dl = load_datalist(DATALIST_PATH)
    pl = load_userlist(USERLIST_PATH)

    user = ""
    handed = ""
    height = ""
    weight = ""

    # datalist から user を解決
    row = dl[dl["csv_path"].astype(str) == csv_path.name]
    if not row.empty:
        user = str(row["user"].iloc[0] or "").strip()

    if user:
        prow = pl[pl["user"].astype(str).str.strip() == user]
        if not prow.empty:
            height = str(prow["身長"].iloc[0] or "").strip()
            weight = str(prow["体重"].iloc[0] or "").strip()

    return {
        "user": user,
        "handedness": handed,
        "height_cm": height,
        "weight_kg": weight,
    }

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
    本来はあなたのReport.pyのロジック（計測日時、user名、所要時間とか）を入れる。
    ここでは最低限の形を書いておく。
    """
    # user列っぽいものを探す
    cand_user_cols = [c for c in df.columns if c.lower() in ["user", "name", "athlete"]]
    user_name = df[cand_user_cols[0]].iloc[0] if cand_user_cols else "(不明)"

    # 計測日時っぽいもの
    cand_date_cols = [c for c in df.columns if "date" in c.lower() or "time" in c.lower()]
    # とりあえず最初の候補を文字列で
    measure_info = ""
    if cand_date_cols:
        measure_info = str(df[cand_date_cols[0]].iloc[0])

    info = {
        "ファイル": csv_path.name,
        "選手": str(user_name),
        "計測日時らしき値": measure_info,
        "サンプル数": len(df),
    }
    return info

def detect_time_and_numeric_cols(df: pd.DataFrame):
    # time候補
    time_col = None
    for cand in df.columns:
        if str(cand).lower() in ["time", "t", "timestamp", "sec", "seconds"]:
            time_col = cand; break

    # 数値列
    numeric_cols = []
    for c in df.columns:
        if c == time_col: 
            continue
        try:
            pd.to_numeric(df[c].dropna().head(10), errors="raise")
            numeric_cols.append(c)
        except Exception:
            pass
    return time_col, numeric_cols

def get_graph_range(prefix: str):
    s = st.session_state.get(prefix + "start_idx")
    e = st.session_state.get(prefix + "end_idx")
    if s is None or e is None:
        return None
    s = int(s); e = int(e)
    if e < s: e = s
    return s, e

def slice_by_range(df: pd.DataFrame, idx_range):
    if not idx_range:
        return df, None
    s, e = idx_range
    s = max(0, min(s, len(df)-1))
    e = max(0, min(e, len(df)-1))
    return df.iloc[s:e+1].copy(), (s, e)


# -------------------------------------------------
# ページ基本設定
# -------------------------------------------------

st.set_page_config(page_title="user View", layout="wide")

# URLパラメータから csv_path と tab を取得
params = st.query_params
csv_path_param = params.get("csv_path", "")
initial_tab = params.get("tab", "graph")

csv_path = Path(csv_path_param)

if not csv_path.exists():
    st.error(f"指定されたCSVが見つかりません: {csv_path}")
    st.stop()

# CSVロード
df = read_csv_any_encoding(csv_path)

time_col, numeric_cols = detect_time_and_numeric_cols(df)
value_cols = [c for c in df.columns if c != time_col]

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

# userView 全体で共有する state prefix
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

    # Y軸候補（time_col以外の列）
    all_cols = list(df.columns)

    # UIレイアウト: 左(操作パネル) / 右(動画＋グラフ＋スライダー)
    left_col, right_col = st.columns([0.3, 0.7])

    # -------------------------------------------------
    # 左カラム：軸選択 / 再生・停止 / コマ送り / 区間指定
    # -------------------------------------------------
    with left_col:
        st.markdown("### 軸選択")
        
        # 設定から取得
        saved_y1, saved_y2 = _get_default_axes(csv_path.name)
        # 選択肢
        y1_options = value_cols
        y2_options = ["(なし)"] + value_cols

        # インデックスを解決（存在しなければ先頭）
        y1_index = y1_options.index(saved_y1) if (saved_y1 in y1_options and y1_options) else 0
        y2_index = y2_options.index(saved_y2) if (saved_y2 in y2_options and y2_options) else 0

        # 1本目のY軸
        y1_col = st.selectbox(
            "Y軸（第1軸）",
            y1_options,
            index=y1_index,
            key=prefix + "y1_col_select",
        )

        # 2本目のY軸(任意)
        y2_col = st.selectbox(
            "Y軸(第2軸)",
            y2_options,
            index=y2_index,
            key=prefix + "y2_col_select",
        )
        y2_active = (y2_col != "(なし)")
        
        # 直近選択の保存（変更があれば即反映）
        if (y1_col != saved_y1) or (y2_col != saved_y2):
            _save_default_axes(csv_path.name, y1_col, y2_col)

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
        steps = [(-100, "-100"), (-10, "-10"), (-1, "-1"), (1, "+1"), (10, "+10"), (100, "+100")]

        for i, (delta, label) in enumerate(steps):
            with step_cols[i]:
                if st.button(label, key=f"{prefix}_step_{label}"):
                    idx = st.session_state[prefix + "marker_idx"]
                    new_idx = max(0, min(len(x_vals) - 1, idx + delta))
                    st.session_state[prefix + "marker_idx"] = new_idx
                    st.session_state[prefix + "is_playing"] = False

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
        frame_slot = st.container()
        timeline_area = st.container()
        with timeline_area:
            graph_slot = st.container()

            # タイムラインスライダー
            max_time = float(np.nanmax(x_vals))

            play_key   = prefix + "is_playing"
            marker_key = prefix + "marker_idx"
            slider_key = prefix + "timeline_time"

            # --- state 初期化 ---
            if marker_key not in st.session_state:
                st.session_state[marker_key] = 0
            if play_key not in st.session_state:
                st.session_state[play_key] = False
            if slider_key not in st.session_state:
                # 初期値は最初の時刻
                st.session_state[slider_key] = float(x_vals[0])

            # 現在の marker を安全にクランプ
            marker_idx = st.session_state[marker_key]
            marker_idx = max(0, min(marker_idx, len(x_vals) - 1))
            st.session_state[marker_key] = marker_idx
            current_t = float(x_vals[marker_idx])

            # 再生中フラグ
            is_playing = st.session_state[play_key]

            # 🔸再生中はスライダーを marker に追従させるだけ
            if is_playing:
                st.session_state[slider_key] = current_t

            # スライダー本体
            slider_val = st.slider(
                "現在位置 (秒)",
                min_value=0.0,
                max_value=max_time,
                step=0.01,
                key=slider_key,
            )

            # 🔸停止中のときだけ「スライダー操作」を index に反映
            if not is_playing:
                nearest_idx = int(np.argmin(np.abs(np.array(x_vals) - slider_val)))
                if nearest_idx != st.session_state[marker_key]:
                    st.session_state[marker_key] = nearest_idx
                    # 念のため再生は止めておく（手動移動扱い）
                    st.session_state[play_key] = False

    # -------------------------------------------------
    # 描画関数（グラフ＋動画フレームを1セット描画）
    # -------------------------------------------------
    def draw_graph_and_frame(marker_idx_now: int):
        # ★ container の中身を一度クリアしてから描画することで、
        #   再生中に縦に積み上がらないようにする
        graph_slot.empty()
        frame_slot.empty()

        # 安全化
        marker_idx_now = max(0, min(marker_idx_now, len(x_vals) - 1))
        t_marker = x_vals[marker_idx_now]

        # CSV時間に最も近い動画フレーム番号
        frame_idx = int(np.argmin(np.abs(video_times - t_marker)))

        # === レンジ計算（固定用） ===
        def _safe_minmax(arr):
            arr = np.asarray(arr)
            if arr.size == 0:
                return -1.0, 1.0
            vmin = float(np.nanmin(arr))
            vmax = float(np.nanmax(arr))
            if not np.isfinite(vmin) or not np.isfinite(vmax):
                vmin, vmax = -1.0, 1.0
            if abs(vmax - vmin) < 1e-12:
                vmin -= 0.5
                vmax += 0.5
            pad = 0.05 * (vmax - vmin)
            return vmin - pad, vmax + pad

        # xは全時間で固定
        x0, x1 = float(x_vals[0]), float(x_vals[-1])

        # y1固定レンジ
        y1_min, y1_max = _safe_minmax(y1_vals)

        # y2固定レンジ（使う場合）
        if y2_active and y2_vals is not None:
            y2_min, y2_max = _safe_minmax(y2_vals)
            y_all_min = min(y1_min, y2_min)
            y_all_max = max(y1_max, y2_max)
        else:
            y2_min = y2_max = None
            y_all_min, y_all_max = y1_min, y1_max

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
                yaxis="y",
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
                    yaxis="y2",
                )
            )

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
            # レポート用：選択された時刻範囲を保存
            st.session_state["report_range"] = {"t0": float(t0), "t1": float(t1)}
        else:
            st.session_state["report_range"] = None

        # レイアウト（固定レンジ）
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
                range=[x0, x1],
                autorange=False,
                fixedrange=True,
                zeroline=False,
            ),
            yaxis=dict(
                title=y1_col,
                range=[y1_min, y1_max],
                autorange=False,
                fixedrange=True,
                zeroline=False,
            ),
        )

        if y2_active and y2_vals is not None:
            layout_dict["yaxis2"] = dict(
                title=y2_col,
                overlaying="y",
                side="right",
                range=[y2_min, y2_max],
                autorange=False,
                fixedrange=True,
                zeroline=False,
            )

        fig.update_layout(**layout_dict)

        # グラフ描画
        graph_slot.plotly_chart(
            fig,
            use_container_width=True,
            config={"staticPlot": True},  # iPad での誤ドラッグ防止
        )

        # === 動画フレーム描画 ===
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
    # （1フレームずつ描画して rerun で進める方式）
    # -------------------------------------------------
    play_key      = prefix + "is_playing"
    marker_key    = prefix + "marker_idx"
    last_time_key = prefix + "last_frame_time"

    # 再生中
    if st.session_state.get(play_key, False):
        now = time.time()
        last_t = st.session_state.get(last_time_key, None)
        frame_period = 1.0 / max(fps, 1.0)  # 1フレームあたりの秒数

        # 初回は基準時間だけ保存
        if last_t is None:
            st.session_state[last_time_key] = now
        else:
            dt = now - last_t
            if dt >= frame_period:
                # 経過時間に応じて何フレーム進めるか
                n_frames = int(dt / frame_period)

                # CSV 側も 1 サンプルずつ前進させる
                step = 1

                idx = st.session_state.get(marker_key, 0)
                idx += n_frames * step

                # 終端を超えたら止める
                if idx >= len(x_vals):
                    idx = len(x_vals) - 1
                    st.session_state[play_key] = False

                st.session_state[marker_key] = idx
                st.session_state[last_time_key] = now

        # 現在位置を1回だけ描画
        draw_graph_and_frame(st.session_state.get(marker_key, 0))

        # まだ再生中なら次フレームのために rerun
        if st.session_state.get(play_key, False):
            st.rerun()

    else:
        # 🔸停止中は必ず「今の marker_idx で一度描画」する
        #    → 起動直後・スライダー操作後もここが走る
        st.session_state.pop(last_time_key, None)
        draw_graph_and_frame(st.session_state.get(marker_key, 0))

    
    

# -------------------------------------------------
# タブ2: レポート
# -------------------------------------------------
with tab_report:
    from report_core import (
        load_csv_from_path,
        build_report_html_from_df,
        render_report_with_print_toolbar,
    )
    from Home import DATALIST_PATH, USERLIST_PATH, load_datalist, load_userlist

    # --- ヘルパ ----------------------------
    def get_graph_range(prefix: str):
        """Graphタブで保存した start/end を読み出す。"""
        s = st.session_state.get(prefix + "start_idx")
        e = st.session_state.get(prefix + "end_idx")
        if s is None or e is None:
            return None
        s, e = int(s), int(e)
        if e < s:
            e = s
        return s, e

    def slice_by_range(df: pd.DataFrame, idx_range):
        """idx_range=(s,e) を df.iloc で安全にスライス。"""
        if not idx_range:
            return df, None
        s, e = idx_range
        s = max(0, min(s, len(df) - 1))
        e = max(0, min(e, len(df) - 1))
        return df.iloc[s:e + 1].copy(), (s, e)

    def pick_user_from_df(df: pd.DataFrame) -> str:
        """CSV内の 'user' 列から最初の非空値を拾う。なければ空文字。"""
        for c in df.columns:
            if str(c).strip().lower() == "user":
                try:
                    s = df[c].astype(str).str.strip()
                    vals = [v for v in s.unique().tolist() if v]
                    return vals[0] if vals else ""
                except Exception:
                    pass
        return ""

    def resolve_user_meta(csv_path: Path, df_full: pd.DataFrame):
        """
        表示名: CSVのuser列 → datalist.csvのuser の優先順で決定。
        handedness/height/weight は userlist.csv から。
        """
        # 1) CSVのuser
        user_in_csv = pick_user_from_df(df_full)

        # 2) datalist.csv から user を解決
        dl = load_datalist(DATALIST_PATH)
        user_from_dl = ""
        if "csv_path" in dl.columns:
            row = dl[dl["csv_path"].astype(str) == csv_path.name]
            if not row.empty:
                user_from_dl = str(row["user"].iloc[0] or "").strip()

        # 3) userlist.csv からプロフィール
        handedness = height_cm = weight_kg = ""
        if user_from_dl:
            pl = load_userlist(USERLIST_PATH)
            if not pl.empty and "user" in pl.columns:
                prow = pl[pl["user"].astype(str).str.strip() == user_from_dl]
                if not prow.empty:
                    height_cm  = str(prow.get("身長",  [""]).iloc[0] or "").strip()
                    weight_kg  = str(prow.get("体重",  [""]).iloc[0] or "").strip()

        # 表示名の優先度
        resolved_name = user_in_csv or user_from_dl or ""
        return resolved_name, handedness, height_cm, weight_kg
    # --------------------------------------------------------------------

    st.subheader("レポートビュー / Report")
    if "logs" not in st.session_state:
        st.session_state["logs"] = []

    # 1) CSV読込
    try:
        df_full, measured_at, date_str, time_str, duration_str = load_csv_from_path(csv_path)
    except Exception as e:
        st.error(f"CSV の読み込みに失敗しました: {csv_path}\n{e}")
        st.stop()

    # 2) Graphタブの区間を適用（start/end を反映）
    idx_range = get_graph_range(prefix)
    df_for_report, used_range = slice_by_range(df_full, idx_range)
    if used_range is None:
        st.caption("このレポートはCSV全範囲のデータで作成しています。")
    else:
        st.caption(f"このレポートは区間 [{used_range[0]}, {used_range[1]}] のデータで作成しています。")

    # 3) ユーザーメタを解決（CSV→datalist→userlist）
    user_name, handedness, height_cm, weight_kg = resolve_user_meta(csv_path, df_full)

    # 4) レポート用メタを組み立て
    report_meta = {
        "filename":     csv_path.name,
        "measured_at":  measured_at,
        "date":         date_str,
        "time":         time_str,
        "duration_sec": duration_str,
        "user_name":  user_name,
        "handedness":   handedness,
        "height_cm":    height_cm,
        "weight_kg":    weight_kg,
        # 必要なら任意項目も
        # "foot_size_cm": "", "step_width_cm": "",
    }
    
    # 5) 開始時刻サムネ（start_img_uri）を作る
    import io, base64
    from PIL import Image

    start_idx = (used_range[0] if used_range is not None else 0)

    if time_col is not None and time_col in df_full.columns and len(df_full) > 0:
        t_start_sec = float(to_seconds_any(df_full[time_col].iloc[start_idx]))
    else:
        # Time 列がない場合のフォールバック（fps から換算）
        vi_tmp = load_video_info(csv_path.with_suffix(".mp4"))
        fps_tmp = vi_tmp["fps"] if vi_tmp else 30.0
        t_start_sec = float(start_idx) / float(fps_tmp if fps_tmp > 0 else 30.0)

    start_img_uri = None
    vi = load_video_info(csv_path.with_suffix(".mp4"))
    if vi:
        frame_idx = int(round(t_start_sec * vi["fps"]))
        rgb = vi["get_frame"](frame_idx)  # 既存: RGB ndarray が返る
        if rgb is not None:
            pil = Image.fromarray(rgb)
            bio = io.BytesIO()
            pil.save(bio, format="JPEG", quality=85)
            start_img_uri = "data:image/jpeg;base64," + base64.b64encode(bio.getvalue()).decode("ascii")


    # 6) レポートHTML生成 → 印刷ツールバーでラップ
    try:
        report_html = build_report_html_from_df(df_for_report, meta=report_meta, start_img_uri=start_img_uri)
    except Exception as e:
        st.error(f"レポートHTMLの生成に失敗しました。\n{e}")
        st.stop()

    wrapped_html = render_report_with_print_toolbar(report_html) if report_html else ""

    # 6) 表示
    if wrapped_html:
        st.components.v1.html(wrapped_html, height=1000, scrolling=False)
    else:
        st.warning("レポートHTMLが空でした。テンプレートや入力データをご確認ください。")

    # 7) デバッグ表示（必要に応じて折りたたみ）
    with st.expander("デバッグ情報（CSVメタ / basic_meta / file_meta）", expanded=False):
        st.write("CSVパス:", csv_path.as_posix())
        st.json(
            {

                "measured_at": measured_at,
                "duration": duration_str,
            },
            expanded=False,
        )
        st.dataframe(df_for_report.head(20))
