# pages/graphviewer.py
import os
from typing import Tuple

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objs as go

# ---- ページ設定 -------------------------------------------------------------
st.set_page_config(page_title="CSV Graph Viewer", layout="wide")


# ---- ユーティリティ ---------------------------------------------------------
def _to_numeric_series(s: pd.Series) -> pd.Series:
    """数値化（失敗はNaN）。"""
    return pd.to_numeric(s, errors="coerce")


def _to_x_series(s: pd.Series) -> Tuple[pd.Series, str]:
    """
    X軸用の型を推定して返す。
    戻り値: (系列, kind)  where kind ∈ {"numeric", "datetime", "category"}
    """
    # 数値判定
    num = pd.to_numeric(s, errors="coerce")
    if num.notna().mean() > 0.8:
        return num, "numeric"

    # 日時判定
    dt_parsed = pd.to_datetime(s, errors="coerce", infer_datetime_format=True)
    if dt_parsed.notna().mean() > 0.8:
        return dt_parsed, "datetime"

    # 文字列カテゴリ
    return s.astype(str), "category"


def _downsample_xy(x: pd.Series, y: pd.Series, max_points: int = 5000):
    """
    単純間引き（等間隔サンプリング）。高速・低負荷。
    """
    n = len(x)
    if n <= max_points or max_points <= 0:
        return x, y
    step = int(np.ceil(n / max_points))
    return x.iloc[::step], y.iloc[::step]


# ---- サイドバー（データ入力） -----------------------------------------------
st.sidebar.header("データ")
df: pd.DataFrame | None = st.session_state.get("df")

if df is None or df.empty:
    up = st.sidebar.file_uploader("CSVを選択（ヘッダ行必須）", type=["csv"])
    if up is not None:
        try:
            df = pd.read_csv(up)
            st.session_state.df = df
            st.sidebar.success("CSVを読み込みました")
        except Exception as e:
            st.sidebar.error(f"読み込みに失敗しました: {e}")

# ---- 本体 -------------------------------------------------------------------
st.title("CSV Graph Viewer")

if df is None or df.empty:
    st.info("左のサイドバーからCSVを読み込むか、メインページでCSVを読み込んでください。")
    st.stop()

cols = list(df.columns)
default_x_idx = cols.index("Time") if "Time" in cols else 0

st.markdown("#### 軸の選択")
c1, c2 = st.columns([1, 2])
with c1:
    x_col = st.selectbox("横軸 (X)", options=cols, index=default_x_idx)
with c2:
    y_candidates = [c for c in cols if c != x_col]
    default_ys = [c for c in ["LFz", "RFz", "MTz"] if c in y_candidates] or (y_candidates[:1] if y_candidates else [])
    y_cols = st.multiselect("縦軸 (Y)（複数選択可）", options=y_candidates, default=default_ys)

if not y_cols:
    st.warning("縦軸 (Y) を1つ以上選択してください。")
    st.stop()

st.markdown("#### 表示オプション")
o1, o2, o3, o4 = st.columns(4)
with o1:
    show_grid = st.checkbox("グリッド表示", value=True)
with o2:
    x_tick_input = st.text_input("横軸の刻み幅（未入力=自動）", value="", placeholder="例: 0.1（数値のみ対応）")
with o3:
    x_range_text = st.text_input("X範囲（例: 0,10 または 2025-10-02 09:00,2025-10-02 09:10）", value="")
with o4:
    line_width = st.slider("線の太さ", min_value=1, max_value=6, value=2)

p1, p2, p3 = st.columns(3)
with p1:
    use_scattergl = st.checkbox("高速描画（Scattergl）", value=True, help="長い系列での描画を軽くします")
with p2:
    do_downsample = st.checkbox("ダウンサンプリング", value=True, help="表示点数を間引いて軽量化")
with p3:
    max_points = st.number_input("最大表示点数", min_value=500, max_value=200000, value=5000, step=500)

# ---- データ整形（X） --------------------------------------------------------
x_series_raw = df[x_col]
x_series, x_kind = _to_x_series(x_series_raw)

# ---- 図の準備 --------------------------------------------------------------
fig = go.Figure()
TraceCls = go.Scattergl if use_scattergl else go.Scatter

for yc in y_cols:
    y_raw = df[yc]
    y = _to_numeric_series(y_raw)

    x_plot, y_plot = (x_series, y)
    if do_downsample:
        x_plot, y_plot = _downsample_xy(x_plot, y_plot, max_points=max_points)

    fig.add_trace(
        TraceCls(
            x=x_plot,
            y=y_plot,
            mode="lines",
            name=yc,
            line=dict(width=line_width),
        )
    )

# ---- レイアウト／軸設定 -----------------------------------------------------
xaxis_kwargs = dict(showgrid=show_grid, title=x_col)
yaxis_kwargs = dict(showgrid=show_grid, title=(", ".join(y_cols)))

# 横軸刻み幅（数値のみ対応）
if x_tick_input.strip():
    try:
        dtick_value = float(x_tick_input.strip())
        if x_kind in ("numeric", "category"):
            xaxis_kwargs["dtick"] = dtick_value
    except Exception:
        st.toast("刻み幅は数値のみ対応（datetimeは未対応）", icon="⚠️")

# X範囲（任意）
if x_range_text.strip():
    try:
        a, b = [t.strip() for t in x_range_text.split(",", 1)]
        if x_kind == "numeric":
            xr = [float(a), float(b)]
        elif x_kind == "datetime":
            xr = [pd.to_datetime(a), pd.to_datetime(b)]
        else:
            xr = [a, b]
        xaxis_kwargs["range"] = xr
    except Exception:
        st.toast("X範囲の形式が不正です（例: 0,10  または 2025-10-02 09:00,2025-10-02 09:10）", icon="⚠️")

fig.update_layout(
    xaxis=xaxis_kwargs,
    yaxis=yaxis_kwargs,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(l=10, r=10, t=10, b=10),
    hovermode="x unified",
)

# ---- 描画（Plotlyは標準でズーム／パン対応） ----------------------------------
st.plotly_chart(
    fig,
    use_container_width=True,
    config={
        "displaylogo": False,
        # ズーム、パン、ボックスズーム、オートスケール、軸リセットなどがツールバーに含まれます
    },
)

# ---- 参考情報 ---------------------------------------------------------------
with st.expander("テーブルを確認する"):
    st.dataframe(df, use_container_width=True, height=300)

st.caption(
    "ヒント: ツールバーの『自動範囲』や『軸リセット』で簡単に見やすさを整えられます。"
)
