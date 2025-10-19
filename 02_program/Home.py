# Home.py
from pathlib import Path
import pandas as pd
import streamlit as st
import re
from pathlib import Path

def _records_from_selected(df_sel, csv_col: str):
    recs = []
    for _, r in df_sel.iterrows():
        row_dict = r.to_dict()
        csv_value = str(row_dict.get(csv_col, "")).strip()
        csv_path = Path(csv_value)
        if not csv_path.is_absolute():
            csv_path = (DATA_DIR / csv_path).resolve()
        recs.append({
            "row": row_dict,
            "csv_col": csv_col,
            "csv_path": csv_path.as_posix(),
            "data_dir": DATA_DIR.as_posix(),
            "datalist_path": DATALIST_PATH.as_posix(),
        })
    return recs

st.set_page_config(page_title="メイン | 試験結果ビューア", layout="wide")

st.title("メイン（Datalist から選択）")
st.caption("data フォルダ内の Datalist.csv を読み込みます。複数行選択して『レポートを開く』で縦に並べて表示します。")

# ---- パス前提：このファイルと同階層に data/ フォルダ ----
APP_DIR = Path(__file__).parent.resolve()
DATA_DIR = APP_DIR / "data"
DATALIST_PATH = DATA_DIR / "Datalist.csv"   # ← 固定

# ---- セッション初期化 ----
st.session_state.setdefault("logs", [])
st.session_state.setdefault("basic", {
    "player_name":"", "height_cm":"", "weight_kg":"",
    "foot_size_cm":"", "handedness":"右", "step_width_cm":""
})

# ---- data/ の存在確認 ----
if not DATA_DIR.exists():
    st.error(f"data フォルダが見つかりません: {DATA_DIR.as_posix()}")
    st.stop()

# ---- Datalist.csv 読み込み ----
if not DATALIST_PATH.exists():
    st.error(f"Datalist.csv が見つかりません: {DATALIST_PATH.as_posix()}")
    st.info("data/ フォルダに Datalist.csv と、各行が参照する CSV / MP4 を置いてください。")
    st.stop()

try:
    df_raw = pd.read_csv(DATALIST_PATH)
except Exception as e:
    st.error(f"Datalist.csv の読み込みに失敗: {e}")
    st.stop()


def guess_csv_col(df: pd.DataFrame) -> str:
    cand = [c for c in df.columns if "csv" in c.lower() or "path" in c.lower()]
    if cand:
        return cand[0]
    for c in df.columns:
        try:
            if df[c].astype(str).str.contains(r"\.csv$", case=False, regex=True).any():
                return c
        except Exception:
            pass
    return df.columns[0]

st.subheader("データ一覧")
csv_col = guess_csv_col(df_raw)

# 選択列を付与
SELECT_COL = "選択"
df_show = df_raw.copy()
if SELECT_COL not in df_show.columns:
    df_show.insert(0, SELECT_COL, False)

edited = st.data_editor(
    df_show,
    use_container_width=True,
    hide_index=True,
    column_config={
        SELECT_COL: st.column_config.CheckboxColumn(required=False, help="複数選択できます"),
    },
    disabled=False,
    height=520,
    key="datalist_editor",
)

sel_mask = edited[SELECT_COL] == True
selected_rows = edited[sel_mask].drop(columns=[SELECT_COL], errors="ignore")

st.session_state["selected_records"] = _records_from_selected(selected_rows, csv_col)
st.session_state["selected_csv_paths"] = [r["csv_path"] for r in st.session_state["selected_records"]]

if st.button("レポートを開く", type="primary"):
    if selected_rows.empty:
        st.warning("1行以上選択してください。")
        st.stop()

    # 複数行をそのままセッションへ
    records = []
    for _, r in selected_rows.iterrows():
        row_dict = r.to_dict()
        csv_value = str(row_dict.get(csv_col, "")).strip()
        csv_path = Path(csv_value)
        if not csv_path.is_absolute():
            csv_path = (DATA_DIR / csv_path).resolve()
        records.append({
            "row": row_dict,
            "csv_col": csv_col,
            "csv_path": csv_path.as_posix(),
            "data_dir": DATA_DIR.as_posix(),
            "datalist_path": DATALIST_PATH.as_posix(),
        })

    st.session_state["selected_records"] = records

    st.switch_page("pages/Report.py")
