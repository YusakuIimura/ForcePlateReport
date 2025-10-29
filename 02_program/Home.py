from pathlib import Path
import pandas as pd
import streamlit as st
import re
from datetime import date

# -------------------------------------------------
# ページ設定
# -------------------------------------------------
st.set_page_config(page_title="メイン | 計測データ管理", layout="wide")

st.title("メイン画面（計測データ管理）")
st.caption("data/ 内の *_FP.csv を自動検出して台帳(datalist.csv)に反映します。")
st.caption("右側で動画を確認しながら1件ずつ player / 身長 / 体重 などを確定して保存できます。")

# -------------------------------------------------
# パス定義
# -------------------------------------------------
APP_DIR = Path(__file__).parent.resolve()
DATA_DIR = APP_DIR / "data"

# datalist は小文字優先、なければ Datalist.csv を救済
DATALIST_PATH = DATA_DIR / "datalist.csv"
if not DATALIST_PATH.exists():
    legacy = DATA_DIR / "Datalist.csv"
    if legacy.exists():
        DATALIST_PATH = legacy

# -------------------------------------------------
# dataディレクトリ存在チェック
# -------------------------------------------------
if not DATA_DIR.exists():
    st.error(f"data フォルダが見つかりません: {DATA_DIR.as_posix()}")
    st.stop()

# -------------------------------------------------
# datalist.csv の読み込み or 新規作成
# -------------------------------------------------
if DATALIST_PATH.exists():
    try:
        df_list = pd.read_csv(DATALIST_PATH)
    except Exception as e:
        st.error(f"{DATALIST_PATH.name} の読み込みに失敗: {e}")
        st.stop()
else:
    df_list = pd.DataFrame(
        columns=[
            "Date",      # ex: 2025-10-22
            "Time",      # ex: 00:00:03
            "player",    # 選手名
            "利き手",     # 右/左
            "身長",       # cm
            "体重",       # kg
            "csv_path",  # ex: 20251022_000003_FP.csv
        ]
    )

# -------------------------------------------------
# data/ 内の *_FP.csv / *_fp.csv をスキャンし、未登録のものを df_list に追加
# -------------------------------------------------
fp_files = list(DATA_DIR.glob("*_FP.csv")) + list(DATA_DIR.glob("*_fp.csv"))

existing_paths = set()
if "csv_path" in df_list.columns:
    existing_paths = set(df_list["csv_path"].astype(str).str.strip())

new_rows = []
for p in fp_files:
    name_only = p.name  # ex: 20251022_000003_FP.csv

    if name_only in existing_paths:
        continue

    # ファイル名から Date / Time を推測（YYYYMMDD_HHMMSS_FP.csv）
    m = re.match(r"(\d{8})_(\d{6})_?FP\.csv$", name_only, flags=re.IGNORECASE)
    if m:
        ymd = m.group(1)  # "20251022"
        hms = m.group(2)  # "000003"
        date_str = f"{ymd[0:4]}-{ymd[4:6]}-{ymd[6:8]}"  # "2025-10-22"
        time_str = f"{hms[0:2]}:{hms[2:4]}:{hms[4:6]}"  # "00:00:03"
    else:
        date_str = ""
        time_str = ""

    new_rows.append({
        "Date":     date_str,
        "Time":     time_str,
        "player":   "",
        "利き手":     "",
        "身長":       "",
        "体重":       "",
        "csv_path": name_only,
    })

if new_rows:
    df_list = pd.concat([df_list, pd.DataFrame(new_rows)], ignore_index=True)

# ユニーク保証
if "csv_path" not in df_list.columns:
    df_list["csv_path"] = ""
df_list = df_list.drop_duplicates(subset=["csv_path"]).reset_index(drop=True)

# -------------------------------------------------
# 日付フィルタ用のdatetimeを作る（NaTはのちほどFalse扱いにする）
# -------------------------------------------------
dt_series = pd.to_datetime(
    df_list["Date"].astype(str).str.strip() + " " + df_list["Time"].astype(str).str.strip(),
    errors="coerce"
)

valid_dt = dt_series.dropna()
if len(valid_dt) > 0:
    min_d = valid_dt.dt.date.min()
    max_d = valid_dt.dt.date.max()
else:
    today = date.today()
    min_d = today
    max_d = today

# -------------------------------------------------
# レイアウト全体を左右2カラムに
# 左: フィルタ＋表＋遷移
# 右: プレビュー＋1件保存
# -------------------------------------------------
left_col, right_col = st.columns([2, 1], vertical_alignment="top")

# =================================================
# 左カラム
# =================================================
with left_col:
    # フィルタUIを表の上に
    raw_value = st.date_input(
        "Date 範囲を選択",
        value=(min_d, max_d),
        min_value=min_d,
        max_value=max_d,
        format="YYYY-MM-DD",
        key="date_range"
    )

    # date_inputの戻りを正規化
    if isinstance(raw_value, tuple):
        if len(raw_value) == 2:
            start_date, end_date = raw_value
        elif len(raw_value) == 1:
            start_date = end_date = raw_value[0]
        else:
            start_date, end_date = min_d, max_d
    else:
        start_date = end_date = raw_value

    if start_date is None and end_date is None:
        start_date, end_date = min_d, max_d
    elif start_date is None:
        start_date = end_date
    elif end_date is None:
        end_date = start_date

    if start_date > end_date:
        start_date, end_date = end_date, start_date

    # NaT行は除外、範囲に入った行だけマスク
    mask_valid = dt_series.notna()
    date_only = dt_series.dt.date
    mask_range = (date_only >= start_date) & (date_only <= end_date)
    mask = mask_valid & mask_range

    df_filtered = df_list.loc[mask].copy().reset_index(drop=True)

    st.subheader("計測データ一覧")

    SELECT_COL = "選択"
    if SELECT_COL not in df_filtered.columns:
        df_filtered.insert(0, SELECT_COL, False)

    edited = st.data_editor(
        df_filtered,
        use_container_width=True,
        hide_index=True,
        column_config={
            SELECT_COL: st.column_config.CheckboxColumn(
                required=False,
                help="レポートやグラフ表示に使いたい行をチェック",
            ),
            "player": st.column_config.TextColumn(
                "選手名",
                help="動画を見て確定させてください（unknownの場合は修正）",
            ),
            "利き手": st.column_config.TextColumn(
                "利き手",
                help="右 / 左 など",
            ),
            "身長": st.column_config.NumberColumn(
                "身長[cm]",
                help="身長(cm)",
            ),
            "体重": st.column_config.NumberColumn(
                "体重[kg]",
                help="体重(kg)",
            ),
            "csv_path": st.column_config.TextColumn(
                "計測CSVファイル",
                disabled=True,
                help="data/ 内の元CSVファイル名",
            ),
        },
        disabled=False,
        height=520,
        key="datalist_editor",
    )

    st.markdown("---")

    # 次の画面へ（GraphViewer / Report）
    st.subheader("次の画面へ")
    go_graph = st.button("📈 グラフビュワーへ")
    go_report = st.button("📝 レポートを開く")

    if go_graph or go_report:
        sel_mask = edited[SELECT_COL] == True
        selected_rows = edited[sel_mask].drop(columns=[SELECT_COL], errors="ignore")

        if selected_rows.empty:
            st.warning("1行以上チェックしてください。")
            st.stop()

        # records 構築
        records = []
        for _, r in selected_rows.iterrows():
            row_dict = r.to_dict()
            csv_val = str(row_dict.get("csv_path", "")).strip()
            full_path = (DATA_DIR / csv_val).resolve()
            records.append({
                "row": row_dict,
                "csv_path": full_path.as_posix(),
                "data_dir": DATA_DIR.as_posix(),
                "datalist_path": DATALIST_PATH.as_posix(),
            })

        st.session_state["selected_records"] = records
        st.session_state["selected_csv_paths"] = {
            f"{i+1}. {Path(rec['csv_path']).name}": rec["csv_path"]
            for i, rec in enumerate(records)
        }

        dest = "pages/GraphViewer.py" if go_graph else "pages/Report.py"
        st.switch_page(dest)

# =================================================
# 右カラム
# =================================================
with right_col:
    st.subheader("選手情報記入欄")

    if 'df_filtered' not in locals() or len(df_filtered) == 0:
        st.info("この期間内に該当データがありません。")
    else:
        # プレビュー対象を選択（今フィルタで表示中のものだけ対象）
        preview_key = st.selectbox(
            "編集するデータ",
            df_filtered["csv_path"].tolist(),
        )

        # 対応行を df_list から取得（フィルタ前の元データを信頼する）
        row_current = df_list.loc[df_list["csv_path"] == preview_key].copy()
        if len(row_current) > 0:
            row_current = row_current.iloc[0]
        else:
            row_current = pd.Series({
                "player": "",
                "利き手": "",
                "身長": "",
                "体重": "",
            })

        # 対応動画推定: 例 "xxx_FP.csv" -> "xxx_FP.mp4"
        mp4_path = DATA_DIR / (Path(preview_key).stem + ".mp4")
        if mp4_path.exists():
            st.video(str(mp4_path))
        else:
            st.info("対応する動画(.mp4)が見つかりませんでした。")

        st.markdown("#### この計測の情報を確定して保存")

        with st.form("single_row_update"):
            new_player = st.text_input("選手名", row_current.get("player", ""))
            new_handed = st.text_input("利き手(右/左)", row_current.get("利き手", ""))
            col_h, col_w = st.columns(2)
            new_height = col_h.text_input("身長[cm]", str(row_current.get("身長", "")))
            new_weight = col_w.text_input("体重[kg]", str(row_current.get("体重", "")))

            apply_single = st.form_submit_button("保存")

        if apply_single:
            # df_list 内の該当行だけ更新
            idx_match = df_list["csv_path"] == preview_key
            df_list.loc[idx_match, "player"] = new_player
            df_list.loc[idx_match, "利き手"] = new_handed
            df_list.loc[idx_match, "身長"] = new_height
            df_list.loc[idx_match, "体重"] = new_weight

            # datalist.csv を即上書き
            df_list.to_csv(DATALIST_PATH, index=False, encoding="utf-8-sig")
            st.success(f"{preview_key} の情報を更新して保存しました。")
            st.rerun()
