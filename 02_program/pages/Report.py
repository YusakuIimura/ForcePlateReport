# pages/Report.py
from pathlib import Path
import streamlit as st

from report_core import (
    log, load_csv_from_path, build_report_html_from_df, render_report_with_print_toolbar
)

st.set_page_config(page_title="レポート | 試験結果ビューア", layout="wide")
st.title("レポート表示（複数）")

# セッションから取得（複数対応）
records = st.session_state.get("selected_records")

# 後方互換：単一しか無い場合はリスト化
if not records and "selected_record" in st.session_state:
    records = [st.session_state["selected_record"]]

if not records:
    st.info("メイン画面からデータ行を選択してください。")
    st.page_link("Home.py", label="← メインに戻る", icon="⏪")
    st.stop()

# 共通の高さスライダ（必要なら個別スライダに変更可）
common_h = st.slider("各レポート枠の高さ（px）", 600, 2000, 1000, 50)

for i, rec in enumerate(records, start=1):
    row = rec.get("row", {})
    csv_col = rec.get("csv_col", "")
    csv_path_str = rec.get("csv_path", "")
    data_dir_str = rec.get("data_dir", "")

    DATA_DIR = Path(data_dir_str) if data_dir_str else (Path(__file__).parents[1] / "data")
    csv_path = Path(csv_path_str)
    if not csv_path.is_absolute():
        csv_path = (DATA_DIR / csv_path).resolve()

    st.markdown("---")
    st.subheader(f"#{i}: {row.get('name', row.get('title', csv_path.name))}")

    meta_cols = st.columns([2, 3])
    with meta_cols[0]:
        st.write("**ファイル**:", csv_path.name)
        st.write("**パス**:", f"`{csv_path.as_posix()}`")
    with meta_cols[1]:
        st.write("**選択行の主要項目**")
        preview = {k: v for k, v in row.items() if k != csv_col}

        # 解析範囲（GraphViewerで保存済みなら）
        ranges = st.session_state.get("graph_ranges", {})
        gr = ranges.get(csv_path.as_posix())
        if gr:
            if gr.get("kind") == "datetime":
                preview["解析範囲_start"] = gr.get("start")  # ISO文字列
                preview["解析範囲_end"]   = gr.get("end")
                preview["解析範囲_X列"]   = gr.get("x_col", x_col if 'x_col' in locals() else None)
            else:
                preview["解析範囲_start"] = gr.get("start")
                preview["解析範囲_end"]   = gr.get("end")
                preview["解析範囲_X列"]   = gr.get("x_col", x_col if 'x_col' in locals() else None)

        st.json(preview)
        
    if not csv_path.exists():
        st.error(f"CSVが見つかりません: {csv_path.as_posix()}")
        continue

    # ファイルメタ生成
    df, measured_at, date_str, time_str, duration_str = load_csv_from_path(csv_path)
    file_meta = {
        "filename": csv_path.name,
        "measured_at": measured_at,
        "date": date_str,
        "time": time_str,
        "duration_sec": duration_str,
    }

    basic_meta = st.session_state.get("basic", {})

    # レポートHTML生成＆埋め込み（各レポートごとに独立した印刷ボタン付き）
    html = build_report_html_from_df(df, basic_meta=basic_meta, file_meta=file_meta)
    if html:
        wrapped = render_report_with_print_toolbar(html)
        st.components.v1.html(wrapped, height=common_h, scrolling=False)

st.markdown("---")
st.page_link("Home.py", label="← メインに戻る", icon="⏪")
