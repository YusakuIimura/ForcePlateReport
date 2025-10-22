# pages/Report.py
from pathlib import Path
import streamlit as st

from report_core import (
    log, load_csv_from_path, build_report_html_from_df, render_report_with_print_toolbar
)

# セッション初期化：インライン編集（行ごと）を保持
if "report_overrides" not in st.session_state:
    st.session_state["report_overrides"] = {}

def _override_key(row: dict) -> str:
    """行を一意に識別するキー（Datafile→MP4→ID→index…の順で採用）"""
    for k in ("Datafile", "MP4", "ID", "id", "index"):
        v = row.get(k)
        if v not in (None, ""):
            return f"{k}:{v}"
    # フォールバック（衝突を避けるため長い文字列に）
    return "rowhash:" + str(hash(tuple(sorted(row.items()))))

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

for i, rec in enumerate(records, start=1):
    row = rec.get("row", {}) or {}
    csv_col = rec.get("csv_col", "") or ""
    csv_path_str = rec.get("csv_path", "") or ""
    data_dir_str = rec.get("data_dir", "") or ""

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
        # === 選択行の主要項目（インライン編集：Playerのみ可） ===
        st.write("**選択行の主要項目（編集可：レポートのみ反映）**")
        key = _override_key(row)
        ov = st.session_state["report_overrides"].setdefault(key, {})

        # 表示順
        fields = ["Player", "Date", "Time", "Datafile", "MP4"]

        # 既定値の準備
        defaults = {
            "Player": ov.get("player_name", row.get("Player", "")),
            "Date": row.get("Date", ""),
            "Time": row.get("Time", ""),
            "Datafile": row.get("Datafile", ""),
            "MP4": row.get("MP4", ""),
        }

        cols = st.columns(len(fields))
        edited_values = {}
        for idx_f, f in enumerate(fields):
            editable = (f == "Player")  # Playerのみ編集可。全項目OKにするなら True に
            edited_values[f] = cols[idx_f].text_input(
                f, value=str(defaults.get(f, "")), key=f"ie_{key}_{f}", disabled=not editable
            )

        # Playerのインライン編集値を即時セッション反映（保存ボタン無し）
        ov["player_name"] = edited_values.get("Player", "").strip()

        # プレビュー（csv_colは除外）
        preview = {k: v for k, v in row.items() if k != csv_col}
        # 解析範囲（GraphViewerで保存済みなら）
        ranges = st.session_state.get("graph_ranges", {})
        gr = ranges.get(csv_path.as_posix())
        if gr:
            if gr.get("kind") == "datetime":
                preview["解析範囲_start"] = gr.get("start")
                preview["解析範囲_end"]   = gr.get("end")
                preview["解析範囲_X列"]   = gr.get("x_col")
            else:
                preview["解析範囲_start"] = gr.get("start")
                preview["解析範囲_end"]   = gr.get("end")
                preview["解析範囲_X列"]   = gr.get("x_col")
        st.json(preview)

    if not csv_path.exists():
        st.error(f"CSVが見つかりません: {csv_path.as_posix()}")
        continue

    # --- CSVロード（テンプレ生成に使うデータ本体 + ファイル側メタ） ---
    try:
        df, measured_at, date_str, time_str, duration_str = load_csv_from_path(csv_path)
    except Exception as e:
        st.error(f"CSV の読み込みに失敗しました: {csv_path}\n{e}")
        continue

    file_meta_from_loader = {
        "filename": csv_path.name,
        "measured_at": measured_at,
        "date": date_str,
        "time": time_str,
        "duration_sec": duration_str,
    }
    # 既存の file_meta（セッション上の共通メタ）があればマージ（loader優先を保ちつつ上書き可）
    session_file_meta = st.session_state.get("file_meta", {}) or {}
    file_meta = {**file_meta_from_loader, **session_file_meta}

    # --- basic_meta の構築：CSVのPlayer → インライン編集の順に反映（編集値が優先） ---
    basic_meta = dict(st.session_state.get("basic", {}) or {})
    player_from_csv = row.get("Player") or row.get("player") or ""
    player_inline = ov.get("player_name", "").strip()
    player_for_report = player_inline or player_from_csv
    if player_for_report:
        basic_meta["player_name"] = player_for_report

    # --- レポートHTML生成＆埋め込み（各レポートごとに独立した印刷ボタン付き） ---
    try:
        html = build_report_html_from_df(df, basic_meta=basic_meta, file_meta=file_meta)
    except Exception as e:
        st.error(f"レポートHTMLの生成に失敗しました。\n{e}")
        continue

    if html:
        wrapped = render_report_with_print_toolbar(html)
        st.components.v1.html(wrapped, height=1000, scrolling=False)
    else:
        st.warning("レポートHTMLが空でした。テンプレや入力データをご確認ください。")

st.markdown("---")
st.page_link("Home.py", label="← メインに戻る", icon="⏪")
