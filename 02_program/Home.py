import streamlit as st
import pandas as pd
from pathlib import Path

# ===== アプリ基本設定 =====
st.set_page_config(page_title="Home", layout="wide")

# ===== ここはあなたの環境に合わせて下さい =====
DATA_DIR = Path("data")          # CSVが相対で置いてあるベースディレクトリ等あれば
DATALIST_PATH = Path("data/datalist.csv")  # Homeで一覧表示に使っている一覧CSVなどがあるならそれを読む想定

SELECT_COL = "_select_"  # 選択用チェックボックス列の名前

# ===== データ読み込み部分 =====
@st.cache_data
def load_datalist(path: Path) -> pd.DataFrame:
    # あなたの元コードにあった読み方に合わせてOK
    # ここでは仮にUTF-8で読む
    df = pd.read_csv(path)
    return df

st.title("データ一覧 / Home (8501)")
st.caption("対象データを選んで解析ビュー(PlayerView)を開きます")

# datalist.csv想定:
#   - csv_path: 計測データCSV（相対 or 絶対パス）
#   - player: 選手名
#   - date/timeなど色々
# などが入っている前提で進める
if not DATALIST_PATH.exists():
    st.error(f"一覧ファイルが見つかりません: {DATALIST_PATH}")
    st.stop()

df_list = load_datalist(DATALIST_PATH).copy()

# チェックボックス列を追加
df_list[SELECT_COL] = False

st.subheader("計測データ一覧")
st.write("解析したい行をチェックしてください。複数チェックもOKです。")

edited = st.data_editor(
    df_list,
    hide_index=True,
    key="datalist_editor",
    column_config={
        SELECT_COL: st.column_config.CheckboxColumn("選択", default=False),
    },
)

# 次の画面へ
st.subheader("解析ビューを開く")

go_graph = st.button("📈 グラフタブを開く (8502)")
go_report = st.button("📝 レポートタブを開く (8502)")

if go_graph or go_report:
    # チェックされた行だけ抽出
    sel_mask = edited[SELECT_COL] == True
    selected_rows = edited[sel_mask].drop(columns=[SELECT_COL], errors="ignore")

    if selected_rows.empty:
        st.warning("少なくとも1行チェックしてください。")
        st.stop()

    # Streamlit間(同ポート内)の保持用: 選ばれたやつ全部
    records = []
    for _, r in selected_rows.iterrows():
        row_dict = r.to_dict()

        # datalist.csvの csv_path が相対とかだったら DATA_DIR と結合して絶対にする
        raw_csv_path = str(row_dict.get("csv_path", "")).strip()
        full_csv_path = (DATA_DIR / raw_csv_path).resolve() if raw_csv_path else ""

        records.append({
            "row": row_dict,
            "csv_path": str(full_csv_path),
            "data_dir": str(DATA_DIR.resolve()),
            "datalist_path": str(DATALIST_PATH.resolve()),
        })

    st.session_state["selected_records"] = records

    # 代表として最初の1件だけURLに埋めてPlayerViewを開かせる
    first_csv_abs = records[0]["csv_path"]

    # 8502 側へのリンクを作る
    base_url = "http://localhost:8502"
    # go_graphなら最初にgraphタブを開きたい / go_reportならreportタブを開きたい
    initial_tab = "graph" if go_graph else "report"

    # クエリパラメータとして csv_path と tab を渡す
    # 注意：パスにスペースなどが入るとURLエンコード必要になるが、まずは素直に埋める
    url = f"{base_url}/?csv_path={first_csv_abs}&tab={initial_tab}"

    st.markdown("以下をクリックしてください👇")
    st.markdown(f"[➡ 解析ビューを開く]({url})")
    st.info("PlayerViewはポート8502で起動しておいてください。")
